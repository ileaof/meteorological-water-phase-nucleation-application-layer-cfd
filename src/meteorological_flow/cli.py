"""Command-line interface for the meteorological_flow solver.

Usage::

    python -m meteorological_flow.cli --config configs/cold_dry_vs_warm_moist.yaml \
        --output outputs/flow_reference --grid-resolution 20 --duration 60 --one-way-coupling

Returns an int exit code (0 = success).  ``__main__.py`` calls ``sys.exit(main())``.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import config as cfgmod
from .backend import BackendError, get_backend
from .simulation import Simulation, _grid_from_config


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meteorological_flow",
        description="3D Boussinesq flow + one-way water-phase nucleation (CPU).")
    p.add_argument("--config", default=None, help="YAML scenario file")
    p.add_argument("--output", default=None, help="output directory")
    p.add_argument("--grid-resolution", type=int, default=None, metavar="N",
                   help="isotropic cell count nx=ny=nz (e.g. 20, 24, 40); for "
                        "--storm-scale, omit to use the tuned 24x24x40 grid")
    # explicit domain/grid (CLI precedence over YAML/preset/grid-resolution)
    p.add_argument("--Lx", type=float, default=None, help="domain length x [m]")
    p.add_argument("--Ly", type=float, default=None, help="domain length y [m]")
    p.add_argument("--Lz", type=float, default=None, help="domain length z [m]")
    p.add_argument("--Nx", type=int, default=None, help="cells in x")
    p.add_argument("--Ny", type=int, default=None, help="cells in y")
    p.add_argument("--Nz", type=int, default=None, help="cells in z")
    p.add_argument("--preset", default=None, choices=sorted(cfgmod.PRESETS),
                   help="named CPU mesh preset. Chamber: fast/light/recommended/"
                        "advanced/convective-column. Deep-convection storm (imply the "
                        "storm setup + anelastic core): storm-quick/storm/storm-refined/"
                        "storm-fine/storm-hires")
    p.add_argument("--cfl", type=float, default=None, help="CFL target (0,1]")
    p.add_argument("--dt-max", type=float, default=None, dest="dt_max",
                   help="maximum timestep [s]")
    p.add_argument("--sgs", type=float, default=None,
                   help="subgrid eddy viscosity=diffusivity nu=kappa [m^2/s] "
                        "(raise to damp grid-scale 2-delta noise; storm default ~80)")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--pressure-drop", type=float, default=None, dest="pressure_drop",
                     help="total pressure drop across x [Pa]")
    grp.add_argument("--pressure-gradient", type=float, default=None, dest="pressure_gradient",
                     help="pressure gradient [Pa/m] (drop = gradient * Lx)")
    p.add_argument("--float32", action="store_true",
                   help="performance mode: store the prognostic state in float32 "
                        "(equivalent to --precision float32; see --precision)")
    p.add_argument("--precision", choices=("float64", "float32"), default=None,
                   help="numerical precision. float64 is the scientific default "
                        "and required for validated results; float32 is an "
                        "explicit performance opt-in. Supersedes --float32.")
    p.add_argument("--device", choices=("auto", "cpu", "gpu"), default=None,
                   help="execution backend: auto-detect (default), force CPU, or "
                        "force GPU (fails loudly if unavailable -- never silently "
                        "falls back to CPU)")
    p.add_argument("--compute-threads", type=int, default=None, dest="compute_threads",
                   help="BLAS/OpenMP thread cap for the per-step solver (CPU path). "
                        "Distinct from --threads, which is for the offline "
                        "nucleation lookup-table build only.")
    p.add_argument("--max-memory-gb", type=float, default=16.0, dest="max_memory_gb",
                   help="refuse to run if the estimated field memory exceeds this "
                        "(override with --force)")
    p.add_argument("--force", action="store_true",
                   help="run even if the memory estimate exceeds --max-memory-gb")
    p.add_argument("--duration", type=float, default=None, help="simulation duration [s]")
    p.add_argument("--output-interval", type=int, default=None,
                   help="output + nucleation cadence [steps]")
    p.add_argument("--threads", type=int, default=None,
                   help="threads for the OFFLINE nucleation lookup-table build "
                        "(not the per-step solver; see --compute-threads)")
    p.add_argument("--no-microphysics", action="store_true",
                   help="pure flow; no nucleation evaluation")
    p.add_argument("--one-way-coupling", action="store_true",
                   help="diagnostic nucleation; state not modified (Batch 1)")
    p.add_argument("--diagnostic-only", action="store_true",
                   help="alias for one-way coupling")
    p.add_argument("--two-way-coupling", "--hydrometeors", dest="two_way_coupling",
                   action="store_true",
                   help="two-way microphysics: hydrometeor growth + latent-heat "
                        "feedback + sedimentation (Increment 2)")
    p.add_argument("--storm-scale", "--deep-convection", dest="storm_scale",
                   action="store_true",
                   help="km-scale deep-convection storm: stratified sounding + "
                        "warm-bubble trigger + two-way microphysics (demonstration; "
                        "Boussinesq-stretched over a deep column)")
    p.add_argument("--z-stretch", type=float, default=None, dest="z_stretch",
                   help="vertical grid stretching ratio (>1 clusters levels near the "
                        "surface: finer dz low, coarser aloft; 1.0 = uniform)")
    p.add_argument("--dynamics", choices=("boussinesq", "anelastic"), default=None,
                   help="dynamical core: boussinesq (constant density, test mode) or "
                        "anelastic (rho0(z) reference density, div(rho0 u)=0 -- the "
                        "deep-convection core that captures updraft mass expansion)")
    p.add_argument("--tecplot", action="store_true",
                   help="also write Tecplot 360 ASCII flow.dat (ORDERED/POINT zones, "
                        "STRANDID time animation) alongside the NetCDF output")
    p.add_argument("--animate", action="store_true",
                   help="after the run, build one MP4 per figure field plus a combined "
                        "side-by-side panel (MP4+GIF, default fields w/S_w/q_v) from the "
                        "figures/ snapshots -- requires ffmpeg and PNG snapshots enabled "
                        "(config output.figures includes 'slices', the default); if it "
                        "fails, the exact manual commands are printed so it can be redone "
                        "by hand (scripts/make_anim.py / scripts/make_panel.py)")
    p.add_argument("--periodic", action="store_true",
                   help="periodic lateral (x,y) boundaries: ingest the environmental "
                        "mean wind so vertical shear can tilt/organise the storm "
                        "(pair with a sheared sounding). Milestone follow-up.")
    p.add_argument("--kernel-nucleation", action="store_true", dest="kernel_nucleation",
                   help="two-way stage: feed the validated 2nd-order kernel rate J as the "
                        "microphysics embryo source (eq39 pathway) instead of CCN/IN "
                        "activation -- builds/uses the nucleation lookup table (M7)")
    p.add_argument("--method", choices=("lookup", "direct"), default=None,
                   help="nucleation evaluation method")
    p.add_argument("--restart", default=None, help="restart from .npz checkpoint")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (grid, dt estimate, table size) and exit")
    p.add_argument("--validate", action="store_true",
                   help="run the flow validation suite and exit 0/1")
    return p


def _default_config() -> cfgmod.SimulationConfig:
    return cfgmod.SimulationConfig()


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    if args.validate:
        return _run_validation()

    if args.config:
        cfg = cfgmod.from_yaml(args.config)
    else:
        cfg = _default_config()
    cfg = cfgmod.apply_overrides(
        cfg, grid_resolution=args.grid_resolution, duration=args.duration,
        output_interval=args.output_interval, output=args.output,
        no_microphysics=args.no_microphysics, one_way=args.one_way_coupling,
        diagnostic_only=args.diagnostic_only, two_way=args.two_way_coupling,
        storm_scale=args.storm_scale, preset=args.preset,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz,
        z_stretch=args.z_stretch,
        cfl=args.cfl, dt_max=args.dt_max, sgs=args.sgs,
        pressure_drop=args.pressure_drop,
        pressure_gradient=args.pressure_gradient, float32=args.float32,
        dynamics=args.dynamics, tecplot=args.tecplot, periodic=args.periodic,
        kernel_nucleation=args.kernel_nucleation,
        method=args.method, threads=args.threads,
        device=args.device, compute_threads=args.compute_threads,
        precision=args.precision, animate=args.animate)

    # geometry + memory report (always shown so the run records its geometry)
    print("=== meteorological_flow geometry ===")
    print(cfgmod.format_geometry(cfg))
    mem = cfgmod.estimate_memory_gb(cfg)
    if mem > args.max_memory_gb and not args.force:
        print(f"\nERROR: estimated memory ~{mem:.2f} GB exceeds --max-memory-gb "
              f"{args.max_memory_gb:.1f} GB. Reduce N, use --float32, or pass --force.")
        return 2

    if args.dry_run:
        return _dry_run(cfg)

    try:
        compute_backend = get_backend(cfg.performance.device, required_gb=mem)
    except BackendError as e:
        print(f"\nERROR [{e.category}]: {e}")
        return 3
    except NotImplementedError as e:
        print(f"\nERROR [not_implemented]: {e}")
        return 3
    print(f"=== backend: {compute_backend.name} ({compute_backend.device_info()['label']}) "
          f"precision={cfg.physics.precision} "
          f"threads={cfg.performance.compute_threads or 'default'} ===")

    import time as _time
    _t0 = _time.perf_counter()

    def _prog(t, dur, step):
        el = _time.perf_counter() - _t0
        sps = step / el if el > 0 else 0.0
        frac = (t / dur) if dur > 0 else 0.0
        # ETA from the simulated-time progress rate (robust to a varying dt)
        eta_min = ((dur - t) * (el / t) / 60.0) if t > 1e-9 and el > 0 else float("inf")
        print("  step %5d  t=%7.1f/%.0fs (%4.1f%%)  wall=%6.1fs  %4.2f steps/s  "
              "ETA~%5.1f min" % (step, t, dur, 100 * frac, el, sps, eta_min),
              flush=True)

    if cfg.performance.compute_threads:
        from threadpoolctl import threadpool_limits
        thread_ctx = threadpool_limits(limits=cfg.performance.compute_threads)
    else:
        from contextlib import nullcontext
        thread_ctx = nullcontext()
    with thread_ctx:
        sim = Simulation(cfg, restart=args.restart, backend=compute_backend)
        report = sim.run(progress=_prog)
    _print_report(report)
    if cfg.output.animate:
        _maybe_animate(cfg)
    return 0


def _maybe_animate(cfg) -> None:
    """Best-effort post-run animation (--animate): the simulation itself has
    already completed and been reported by this point, so a failure here
    (missing ffmpeg, no figures/ snapshots, ...) must never look like the run
    failed -- print the exact manual commands as a fallback instead."""
    from . import animate as an
    outdir = cfg.output.outdir
    print("\n=== building animations (--animate) ===")
    try:
        result = an.animate_run(outdir, panel_fields=an.DEFAULT_PANEL_FIELDS)
    except (FileNotFoundError, ValueError) as e:
        print(f"  could not build animations automatically: {e}")
        print("  run these manually instead:")
        for cmd in an.manual_commands(outdir):
            print(f"    {cmd}")
        return

    print(f"  ffmpeg: {result['ffmpeg']}")
    failed = []
    for field, outcome in result["fields"].items():
        if isinstance(outcome, Exception):
            failed.append(field)
        else:
            print(f"  {field}: {outcome}")
    panel = result["panel"]
    if isinstance(panel, Exception):
        failed.append("panel")
    elif panel:
        if panel["mp4"]:
            print(f"  panel: {panel['mp4']}")
        if panel["gif"]:
            print(f"  panel: {panel['gif']}")

    if failed:
        print(f"  {len(failed)} item(s) failed ({', '.join(failed)}); to redo by hand:")
        for cmd in an.manual_commands(outdir):
            print(f"    {cmd}")


def _dry_run(cfg) -> int:
    g = _grid_from_config(cfg)
    n = g.nx * g.ny * g.nz
    lk = cfg.nucleation.lookup
    ntab = (lk.n_T * lk.n_pv * lk.n_grad * 2) if cfg.nucleation.method == "lookup" else 0
    rho0 = cfg.physics.P0 / (287.058 * 293.0)
    dt_adv = cfg.time.cfl * min(g.dx, g.dy, g.dz) / 2.0
    dt_diff = 0.5 * min(g.dx, g.dy, g.dz) ** 2 / (3.0 * max(cfg.flow.nu, cfg.flow.kappa))
    print("=== meteorological_flow dry-run ===")
    print(f"  grid     : {g.nx}x{g.ny}x{g.nz} = {n} cells  (dx={g.dx:g} m)")
    print(f"  domain   : {g.Lx}x{g.Ly}x{g.Lz} m")
    print(f"  duration : {cfg.time.duration}s  cfl={cfg.time.cfl}")
    print(f"  dt est.  : adv~{dt_adv:.3f}s  diff~{dt_diff:.3f}s  cap={cfg.time.dt_max}s")
    print(f"  rho0     : {rho0:.3f} kg/m3  (T_ref~293K)")
    print(f"  stage    : {cfg.nucleation.stage}  method={cfg.nucleation.method}")
    mem = cfgmod.estimate_memory_gb(cfg)
    try:
        cb = get_backend(cfg.performance.device, required_gb=mem, log=lambda m: None)
        print(f"  device   : {cb.name} ({cb.device_info()['label']})  "
              f"precision={cfg.physics.precision}")
    except (BackendError, NotImplementedError) as e:
        print(f"  device   : requested={cfg.performance.device}  "
              f"unavailable ({getattr(e, 'category', 'not_implemented')}): {e}")
    if ntab:
        print(f"  lookup   : {ntab} table points "
              f"(T:{lk.n_T} x pv:{lk.n_pv} x grad:{lk.n_grad} x 2 phases)")
    print("  (no run performed)")
    return 0


def _run_validation() -> int:
    """Run the flow validation suite (pytest) and return 0/1.

    Only the meteorological_flow test files are run here; the engine's own
    validation (`met_h2o_nucleation.py --validate`) covers the guarded core.
    """
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tests_dir = os.path.join(here, "tests")
    flow_files = [
        "test_grid.py", "test_advection.py", "test_pressure_projection.py",
        "test_scalar_conservation.py", "test_boundary_conditions.py",
        "test_nucleation_adapter.py", "test_lookup_accuracy.py",
        "test_reference_scenario.py",
    ]
    flow_tests = [os.path.join(tests_dir, f) for f in flow_files
                  if os.path.exists(os.path.join(tests_dir, f))]
    if not flow_tests:
        print("No flow validation tests found.")
        return 1
    print(f"Running {len(flow_tests)} flow validation test files...")
    rc = subprocess.call([sys.executable, "-m", "pytest", "-q"] + flow_tests)
    return 0 if rc == 0 else 1


def _print_report(report: dict) -> None:
    print("\n=== meteorological_flow run complete ===")
    print(f"  wall clock   : {report['wall_clock_s']:.2f}s")
    if report.get("memory_max_kb"):
        print(f"  memory (max) : {report['memory_max_kb'] / 1024:.1f} MB")
    print(f"  steps        : {report['n_steps']}  final t={report['final_time']:.2f}s")
    print(f"  max CFL      : {report['max_cfl']:.3f}")
    s = report["final_stats"]
    print(f"  T range      : {s['T_min']:.2f} .. {s['T_max']:.2f} K")
    print(f"  max |u|      : {s['umax']:.3f} m/s   max |w|: {s['wmax']:.3f} m/s")
    print(f"  max S_w/S_i  : {s['S_w_max']:.3f} / {s['S_i_max']:.3f}")
    import math as _m
    _lq, _ic = s.get('log10I_liq_max', float('-inf')), s.get('log10I_ice_max', float('-inf'))
    if _m.isfinite(_lq) or _m.isfinite(_ic):
        print(f"  max log10I   : liq={_lq:.2f}  ice={_ic:.2f}")
        print(f"  liq nuc cells: {s['n_liq_nucleation_cells']}  "
              f"ice nuc cells: {s['n_ice_nucleation_cells']}")
    if report.get("stage_microphysics"):
        prec = report.get("surface_precip_mm", {})
        print(f"  microphysics : two-way (hydrometeors + latent heat + sedimentation)")
        print(f"  surface precip [mm]: rain={prec.get('rain', 0):.3e} "
              f"snow={prec.get('snow', 0):.3e} graupel={prec.get('graupel', 0):.3e} "
              f"hail={prec.get('hail', 0):.3e}  total={prec.get('total_mm', 0):.3e}")
    b = report["final_budgets"]
    print(f"  water rel err: {b['total_water_rel_err']:.2e}")
    print(f"  energy rel err: {b['total_energy_rel_err']:.2e}")
    print(f"  solver resid : {report['final_solver_residual']:.2e} "
          f"(iters {report['final_solver_iters']})")
    print("  limitations  :")
    for lim in report["limitations"]:
        print(f"    - {lim}")
    print(f"  outputs      : {report['config']}")


if __name__ == "__main__":
    sys.exit(main())