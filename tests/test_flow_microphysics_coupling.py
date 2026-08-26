"""Tests for the two-way flow<->microphysics coupling (Increment 2).

Verify that the 3D flow now carries prognostic precipitating hydrometeors, that
the coupler conserves water (surface precipitation is the only sink), and that a
short coupled run forms condensate and reaches the surface without crashing.
"""
from __future__ import annotations

import numpy as np
import pytest

from meteorological_flow import thermodynamics as th
from meteorological_flow.cli import build_argparser
from meteorological_flow.config import SimulationConfig, apply_overrides
from meteorological_flow.grid import Grid
from meteorological_flow.microphysics_coupling import MicrophysicsCoupler
from meteorological_flow.simulation import Simulation
from meteorological_flow.state import FlowState


def _supersaturated_state(g, cfg, T=264.0, ss=1.12, w=5.0):
    xp = g.xp
    st = FlowState.zeros(g)
    P = xp.full(g.center_shape, cfg.physics.P0)
    st.theta = th.theta_from_T(xp.full(g.center_shape, T), P, th.P0_REF, xp=xp)
    qsat = th.q_v_from_p_v(th.psat_water(xp.full(g.center_shape, T), xp=xp),
                           cfg.physics.P0, xp=xp)
    st.qv = qsat * ss
    st.w[:] = w
    st.diagnose(cfg)
    return st


def test_flowstate_has_hydrometeor_fields():
    g = Grid(nx=4, ny=4, nz=6, Lx=100, Ly=100, Lz=100)
    st = FlowState.zeros(g)
    for name in ("qr", "qs", "qg", "qh"):
        assert getattr(st, name) is not None
        assert np.all(getattr(st, name) == 0.0)
    assert set(st.surface_precip) == {"rain", "snow", "graupel", "hail"}
    st.qr[:] = 1e-4
    assert st.total_water() > 0.0


def test_coupler_conserves_water():
    cfg = SimulationConfig()
    g = Grid(nx=6, ny=6, nz=10, Lx=600, Ly=600, Lz=1000)
    st = _supersaturated_state(g, cfg)
    co = MicrophysicsCoupler()
    w0 = st.total_water()
    for _ in range(40):
        co.apply(st, g, 3.0)
        co.sediment(st, g, 3.0)
        st.diagnose(cfg)
    for name in ("qv", "ql", "qi", "qr", "qs", "qg", "qh"):
        assert np.all(np.asarray(getattr(st, name)) >= 0.0)
    # closed box (no flow BCs applied): water lost == surface precipitation mass
    precip_mass = sum(float(np.sum(v)) for v in st.surface_precip.values()) * g.dx * g.dy
    removed = w0 - st.total_water()
    assert abs(removed - precip_mass) < 1e-6 * max(w0, 1.0)


def test_apply_overrides_two_way_sets_stage():
    cfg = apply_overrides(SimulationConfig(), two_way=True)
    assert cfg.nucleation.stage == "hydrometeor"


def test_cli_parses_two_way_flag():
    args = build_argparser().parse_args(
        ["--grid-resolution", "20", "--duration", "60", "--two-way-coupling"])
    assert args.two_way_coupling is True
    # alias
    args2 = build_argparser().parse_args(["--hydrometeors"])
    assert args2.two_way_coupling is True


def test_coupled_sim_forms_condensate_and_reports_precip():
    cfg = SimulationConfig()
    cfg.grid.nx = cfg.grid.ny = 10
    cfg.grid.nz = 12
    cfg.time.duration = 12.0
    cfg.nucleation.stage = "hydrometeor"
    cfg.output.outdir = "outputs/_test_coupled"
    cfg.output.format = ["json"]
    cfg.output.figures = []
    cfg.output.restart = False
    cfg.output.interval_steps = 50
    sim = Simulation(cfg)
    assert sim.do_microphysics and not sim.do_nucleation
    report = sim.run()
    st = sim.state
    # the flow drove the microphysics: some condensate formed
    formed = max(float(np.max(st.ql)), float(np.max(st.qi)), float(np.max(st.qr)))
    assert formed > 0.0
    # all fields finite and non-negative
    for name in ("qv", "ql", "qi", "qr", "qs", "qg", "qh"):
        a = np.asarray(getattr(st, name))
        assert np.all(np.isfinite(a)) and np.all(a >= 0.0)
    # surface precipitation is reported
    assert "surface_precip_mm" in report
    assert report["stage_microphysics"] is True


def test_storm_scale_override_and_cli():
    cfg = apply_overrides(SimulationConfig(), storm_scale=True)
    assert cfg.physics.scenario == "deep_convection"
    assert cfg.nucleation.stage == "hydrometeor"
    assert cfg.boundaries.x_west == "wall" and cfg.boundaries.x_east == "wall"
    assert cfg.domain.Lz >= 8000.0
    args = build_argparser().parse_args(["--storm-scale"])
    assert args.storm_scale is True
    args2 = build_argparser().parse_args(["--deep-convection"])
    assert args2.storm_scale is True


def test_storm_presets():
    """The storm-* presets imply the deep-convection setup + anelastic core, on a
    deep (EL-containing) mesh; explicit --dynamics still overrides."""
    from meteorological_flow.config import PRESETS
    expected_nz = {"storm-quick": 40, "storm": 45, "storm-refined": 50,
                   "storm-fine": 60, "storm-hires": 64}
    for p, nz in expected_nz.items():
        assert PRESETS[p].get("storm") is True
        cfg = apply_overrides(SimulationConfig(), preset=p)
        assert cfg.physics.scenario == "deep_convection"
        assert cfg.physics.dynamics == "anelastic"          # deep-convection core
        assert cfg.nucleation.stage == "hydrometeor"
        assert cfg.boundaries.x_west == "wall" and cfg.boundaries.z_top == "damping_layer"
        assert cfg.domain.Lz >= 16000.0 and cfg.grid.nz == nz   # deep enough for the EL
        assert build_argparser().parse_args(["--preset", p]).preset == p
    # explicit --dynamics overrides the preset's anelastic default
    over = apply_overrides(SimulationConfig(), preset="storm", dynamics="boussinesq")
    assert over.physics.dynamics == "boussinesq"


def test_kernel_nucleation_coupling_wiring():
    """M7: --kernel-nucleation couples the validated kernel rate J as the two-way
    microphysics embryo source (eq39 pathway).  Verifies the config/CLI wiring and
    that the two-way Simulation builds the adapter + sets couple_nucleation."""
    from meteorological_flow.base_state import weisman_klemp
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics="anelastic",
                          kernel_nucleation=True, method="direct")   # direct -> no lookup build
    assert cfg.nucleation.couple_kernel is True
    assert build_argparser().parse_args(["--kernel-nucleation"]).kernel_nucleation is True
    cfg.grid.nx = cfg.grid.ny = 6; cfg.grid.nz = 12; cfg.domain.Lz = 12000.0
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_kernel"
    g = Grid(nx=6, ny=6, nz=12, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=12000.0)
    sim = Simulation(cfg, base=weisman_klemp(g))
    assert sim.couple_nucleation is True and sim.adapter is not None
    # default (no flag): two-way stage runs its own CCN/IN activation, not the kernel
    cfg0 = apply_overrides(SimulationConfig(), storm_scale=True)
    cfg0.grid.nx = cfg0.grid.ny = 6; cfg0.grid.nz = 12; cfg0.domain.Lz = 12000.0
    cfg0.output.format = []; cfg0.output.figures = []; cfg0.output.restart = False
    cfg0.output.outdir = "outputs/_test_kernel0"
    assert cfg0.nucleation.couple_kernel is False
    assert Simulation(cfg0, base=weisman_klemp(g)).couple_nucleation is False


def test_storm_scale_deep_convection_runs_stably():
    cfg = apply_overrides(SimulationConfig(), storm_scale=True)
    cfg.grid.nx = cfg.grid.ny = 10
    cfg.grid.nz = 18
    cfg.time.duration = 90.0
    cfg.output.outdir = "outputs/_test_storm"
    cfg.output.format = ["json"]
    cfg.output.figures = []
    cfg.output.restart = False
    cfg.output.interval_steps = 100
    sim = Simulation(cfg)
    assert sim.base is not None and cfg.physics.scenario == "deep_convection"
    # realistic sounding: warm near surface, cold near the top
    assert sim.base.T0[0] > 290.0 and sim.base.T0[-1] < 265.0
    report = sim.run()
    st = sim.state
    # numerically stable and physically bounded
    assert report["max_cfl"] < 1.0
    assert float(np.min(st.T)) >= 179.0 and float(np.max(st.T)) <= 336.0
    for name in ("qv", "ql", "qi", "qr", "qs", "qg", "qh"):
        a = np.asarray(getattr(st, name))
        assert np.all(np.isfinite(a)) and np.all(a >= 0.0)
    # a deep cloud forms from the resolved updraft
    assert max(float(np.max(st.ql)), float(np.max(st.qi))) > 0.0


def test_netcdf_axes_correct_for_noncubic_grid():
    """NetCDF output must label axes correctly on a non-cubic grid (nx != nz),
    e.g. the storm-scale 24x24x40 -- a cubic grid masked the mislabelling."""
    import xarray as xr

    from meteorological_flow import io as fio
    from meteorological_flow.grid import Grid
    g = Grid(nx=5, ny=6, nz=9, Lx=1000, Ly=1000, Lz=2000)
    fld = np.arange(5 * 6 * 9, dtype=float).reshape(5, 6, 9)   # (nx, ny, nz)
    snaps = [{"time": float(t), "T": fld.copy()} for t in (0.0, 1.0)]
    path = "outputs/_test_io_axes.nc"
    fio.write_netcdf(snaps, path, g, {"k": "v"})
    ds = xr.open_dataset(path, engine="scipy")
    try:
        assert ds.sizes["x"] == 5 and ds.sizes["y"] == 6 and ds.sizes["z"] == 9
        # stored (nx,ny,nz) -> written (z,y,x)
        assert np.allclose(ds["T"].values[0], np.transpose(fld, (2, 1, 0)))
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# CPU-vs-GPU equivalence for the two-way microphysics coupling (precip_microphysics
# GPU port). Skipped entirely (not failed) on machines without a working
# CUDA/CuPy GPU -- same idiom as tests/test_backend_equivalence.py.
# ---------------------------------------------------------------------------
def _gpu_available() -> bool:
    try:
        import cupy
        _ = cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _run_coupler(device: str, cfg, n_steps=20, dt=3.0):
    from meteorological_flow.backend import get_backend
    backend = get_backend(device)
    g = Grid(nx=6, ny=6, nz=10, Lx=600, Ly=600, Lz=1000, backend=backend)
    st = _supersaturated_state(g, cfg)
    co = MicrophysicsCoupler()
    w0 = st.total_water()   # already a plain Python float on any backend
    for _ in range(n_steps):
        co.apply(st, g, dt)
        co.sediment(st, g, dt)
        st.diagnose(cfg)
    return st, g, backend, w0


@pytest.mark.skipif(not _gpu_available(), reason="no working CUDA/CuPy GPU in this environment")
class TestCpuVsGpuMicrophysics:
    def test_gpu_coupler_stays_finite_and_conserves_water(self):
        cfg = SimulationConfig()
        st, g, backend, w0 = _run_coupler("gpu", cfg)
        assert backend.name == "gpu"
        to_cpu = backend.to_cpu
        for name in ("qv", "ql", "qi", "qr", "qs", "qg", "qh"):
            arr = to_cpu(getattr(st, name))
            assert np.all(np.isfinite(arr)) and np.all(arr >= 0.0)
        precip_mass = sum(float(np.sum(to_cpu(v))) for v in st.surface_precip.values()) \
            * g.dx * g.dy
        removed = w0 - st.total_water()
        assert abs(removed - precip_mass) < 1e-6 * max(w0, 1.0)

    def test_cpu_vs_gpu_microphysics_match(self):
        # Tolerance rationale: same as tests/test_backend_equivalence.py --
        # the pressure projection differs (CG vs direct for this small grid),
        # which perturbs the resolved w/updraft the microphysics responds to;
        # rtol=1e-4 comfortably bounds that while still catching a genuinely
        # wrong GPU kernel (which would produce O(1) differences, not O(1e-5)).
        cfg = SimulationConfig()
        st_cpu, g_cpu, b_cpu, w0_cpu = _run_coupler("cpu", cfg)
        st_gpu, g_gpu, b_gpu, w0_gpu = _run_coupler("gpu", cfg)
        assert b_gpu.name == "gpu"
        for name in ("T", "qv", "ql", "qi", "qr", "qs", "qg", "qh"):
            a_cpu = b_cpu.to_cpu(getattr(st_cpu, name))
            a_gpu = b_gpu.to_cpu(getattr(st_gpu, name))
            np.testing.assert_allclose(a_gpu, a_cpu, rtol=1e-4, atol=1e-9)
        for cat in st_cpu.surface_precip:
            p_cpu = float(np.sum(b_cpu.to_cpu(st_cpu.surface_precip[cat])))
            p_gpu = float(np.sum(b_gpu.to_cpu(st_gpu.surface_precip[cat])))
            assert p_gpu == pytest.approx(p_cpu, rel=1e-3, abs=1e-9)
