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
from .simulation import Simulation, _grid_from_config


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meteorological_flow",
        description="3D Boussinesq flow + one-way water-phase nucleation (CPU).")
    p.add_argument("--config", default=None, help="YAML scenario file")
    p.add_argument("--output", default=None, help="output directory")
    p.add_argument("--grid-resolution", type=int, default=None,
                   choices=(20, 40, 50), help="isotropic cell count (20/40/50)")
    p.add_argument("--duration", type=float, default=None, help="simulation duration [s]")
    p.add_argument("--output-interval", type=int, default=None,
                   help="output + nucleation cadence [steps]")
    p.add_argument("--threads", type=int, default=None,
                   help="threads for lookup-table build")
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
        storm_scale=args.storm_scale, method=args.method, threads=args.threads)

    if args.dry_run:
        return _dry_run(cfg)

    def _prog(t, dur, step):
        print(f"  step {step:5d}  t={t:7.2f}/{dur:.1f}s")

    sim = Simulation(cfg, restart=args.restart)
    report = sim.run(progress=_prog)
    _print_report(report)
    return 0


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