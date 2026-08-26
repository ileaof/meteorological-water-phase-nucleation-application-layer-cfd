"""CPU-vs-GPU numeric-equivalence tests for the meteorological_flow solver.

Runs an identical small case once per backend and compares the reported
diagnostics within explicitly justified tolerances -- NOT bit-for-bit
equality, since the GPU backend deliberately uses an iterative CG pressure
solve (see pressure_solver.py) where the CPU path would use a direct
factorisation (splu) for grids this small; the two are different, individually
correct discretisations of the same Poisson problem, so their solutions agree
only up to the CG residual tolerance, not machine precision.

Skipped entirely (not failed) on machines without a working CUDA/CuPy GPU, so
this file passes unchanged on CPU-only CI (see tests/test_backend.py for the
same skip idiom).
"""
from __future__ import annotations

import numpy as np
import pytest

from meteorological_flow.backend import BackendError, get_backend
from meteorological_flow.base_state import weisman_klemp
from meteorological_flow.config import SimulationConfig, apply_overrides
from meteorological_flow.grid import Grid
from meteorological_flow.simulation import Simulation


def _gpu_available() -> bool:
    try:
        import cupy
        _ = cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(),
                                  reason="no working CUDA/CuPy GPU in this environment")


# Small storm-scale case: exercises the anelastic core, the deep-convection
# scalar transport, and the pressure projection -- the same template as
# tests/test_conservation.py's _storm_cfg(), just smaller/shorter for speed.
#
# NOTE: `storm_scale=True` defaults nucleation.stage to "hydrometeor" (two-way
# microphysics coupling, via `precip_microphysics`). `precip_microphysics` IS
# backend-aware (ported alongside the core solver -- see
# tests/test_flow_microphysics_coupling.py's TestCpuVsGpuMicrophysics for its
# dedicated equivalence coverage). This helper still forces "one_way" so the
# tests below isolate the dynamical core specifically; see
# test_cpu_vs_gpu_storm_two_way for the combined (dynamics + microphysics)
# GPU path.
def _small_storm_cfg(duration=20.0):
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, dynamics="anelastic")
    cfg.nucleation.stage = "one_way"
    cfg.domain.Lz = 16000.0
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.grid.nz = 16
    cfg.time.duration = duration
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_backend_equiv_storm"
    return cfg


def _small_storm_two_way_cfg(duration=6.0):
    cfg = _small_storm_cfg(duration=duration)
    cfg.nucleation.stage = "hydrometeor"   # two-way microphysics coupling
    return cfg


# Small mixing-chamber case: exercises the default (Boussinesq, one-way) path.
def _small_chamber_cfg(duration=15.0):
    cfg = SimulationConfig()
    cfg = apply_overrides(cfg, grid_resolution=8, duration=duration)
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_backend_equiv_chamber"
    return cfg


def _run(cfg, device: str) -> dict:
    backend = get_backend(device)
    base = weisman_klemp(Grid(nx=8, ny=8, nz=16, Lx=cfg.domain.Lx, Ly=cfg.domain.Ly,
                              Lz=cfg.domain.Lz)) if cfg.physics.scenario == "deep_convection" else None
    sim = Simulation(cfg, base=base, backend=backend)
    report = sim.run()
    return report, sim


def _assert_finite(sim):
    to_cpu = sim.backend.to_cpu
    for name in ("u", "v", "w", "theta", "qv", "T", "P_total", "rho", "S_w", "S_i"):
        arr = to_cpu(getattr(sim.state, name))
        assert np.all(np.isfinite(arr)), f"non-finite values in state.{name}"


def _run_default_base(cfg, device: str) -> dict:
    """Like _run(), but WITHOUT an explicit base= -- exercises Simulation's
    own default base_state.build_base_state(self.grid) call internally
    (caught a real bug: build_base_state indexed grid.zc directly without a
    to_cpu() conversion, which broke under a GPU-resolved grid)."""
    backend = get_backend(device)
    sim = Simulation(cfg, backend=backend)
    report = sim.run()
    return report, sim


def _stretched_storm_cfg(duration=20.0):
    """Stretched vertical grid (z_stretch != 1.0) + anelastic: caught a real
    bug where the GPU backend forced the iterative CG pressure solve even
    for a stretched grid, whose vertical operator is asymmetric under
    stretching (see pressure_solver.py) -- CG diverged (max CFL ~1e12, all
    fields NaN) where the CPU heuristic in _pressure_method() would always
    have picked the direct solver instead. Fixed by forcing the direct
    (host) solve for stretched grids regardless of backend."""
    cfg = _small_storm_cfg(duration=duration)
    cfg.grid.z_stretch = 1.08
    return cfg


class TestCpuOnly:
    """Sanity checks that don't require a GPU: run twice on CPU, confirm
    reproducibility, so the equivalence tests below have a known-stable
    CPU baseline to compare a GPU run against."""

    def test_cpu_chamber_reproducible(self):
        cfg = _small_chamber_cfg()
        rep1, _ = _run(cfg, "cpu")
        rep2, _ = _run(cfg, "cpu")
        assert rep1["final_stats"] == rep2["final_stats"]

    def test_cpu_storm_reproducible(self):
        cfg = _small_storm_cfg()
        rep1, _ = _run(cfg, "cpu")
        rep2, _ = _run(cfg, "cpu")
        assert rep1["final_stats"] == rep2["final_stats"]
        assert rep1["conservation"] == rep2["conservation"]


@requires_gpu
class TestCpuVsGpu:
    """Same case, same seed, run once per backend; compare within tolerances.

    Tolerance rationale:
    - Field extrema / velocity / thermodynamic stats: rtol=1e-4. The GPU CG
      solve converges to `tol=1e-6` (PressureSolver default) vs the CPU
      direct solve's ~1e-14 residual -- a real, expected, bounded numerical
      difference in the projected velocity that propagates (damped, since
      the projection is a small correction each step) into the transported
      scalars over the run. 1e-4 comfortably bounds this for a short run
      while still catching a genuinely wrong GPU kernel (which would produce
      O(1) differences, not O(1e-5)).
    - Conservation/budget relative errors: atol=1e-3 (absolute, since these
      are themselves small numbers close to zero) for the same reason.
    - No tolerance is bit-exact; see module docstring.
    """

    def test_cpu_vs_gpu_chamber(self):
        cfg = _small_chamber_cfg()
        rep_cpu, sim_cpu = _run(cfg, "cpu")
        rep_gpu, sim_gpu = _run(cfg, "gpu")
        assert sim_gpu.backend.name == "gpu"
        _assert_finite(sim_cpu)
        _assert_finite(sim_gpu)

        s_cpu, s_gpu = rep_cpu["final_stats"], rep_gpu["final_stats"]
        for key in ("T_min", "T_max", "qv_min", "qv_max", "S_w_max", "S_i_max",
                   "S_w_min", "umax", "wmax", "gradT_max"):
            assert s_gpu[key] == pytest.approx(s_cpu[key], rel=1e-4, abs=1e-8), key

        b_cpu, b_gpu = rep_cpu["final_budgets"], rep_gpu["final_budgets"]
        assert b_gpu["total_water_rel_err"] == pytest.approx(
            b_cpu["total_water_rel_err"], abs=1e-3)
        assert b_gpu["total_energy_rel_err"] == pytest.approx(
            b_cpu["total_energy_rel_err"], abs=1e-3)

    def test_cpu_vs_gpu_storm_anelastic(self):
        cfg = _small_storm_cfg()
        rep_cpu, sim_cpu = _run(cfg, "cpu")
        rep_gpu, sim_gpu = _run(cfg, "gpu")
        assert sim_gpu.backend.name == "gpu"
        _assert_finite(sim_cpu)
        _assert_finite(sim_gpu)

        s_cpu, s_gpu = rep_cpu["final_stats"], rep_gpu["final_stats"]
        for key in ("T_min", "T_max", "qv_min", "qv_max", "umax", "wmax"):
            assert s_gpu[key] == pytest.approx(s_cpu[key], rel=1e-4, abs=1e-8), key

        c_cpu, c_gpu = rep_cpu["conservation"], rep_gpu["conservation"]
        assert c_gpu["mass_continuity_residual_norm"] < 1e-2
        assert c_gpu["total_water_rel_err"] == pytest.approx(
            c_cpu["total_water_rel_err"], abs=1e-3)
        assert c_gpu["total_energy_rel_err"] == pytest.approx(
            c_cpu["total_energy_rel_err"], abs=1e-3)

    def test_gpu_reproducible(self):
        cfg = _small_chamber_cfg()
        rep1, _ = _run(cfg, "gpu")
        rep2, _ = _run(cfg, "gpu")
        assert rep1["final_stats"] == rep2["final_stats"]

    def test_cpu_vs_gpu_default_base_state(self):
        # regression test for a real bug: build_base_state(grid) (used when
        # Simulation gets no explicit base=) indexed grid.zc directly without
        # converting a GPU-resident grid to host first.
        cfg = _small_storm_cfg(duration=10.0)
        rep_cpu, sim_cpu = _run_default_base(cfg, "cpu")
        rep_gpu, sim_gpu = _run_default_base(cfg, "gpu")
        assert sim_gpu.backend.name == "gpu"
        _assert_finite(sim_cpu)
        _assert_finite(sim_gpu)
        s_cpu, s_gpu = rep_cpu["final_stats"], rep_gpu["final_stats"]
        assert s_gpu["T_min"] == pytest.approx(s_cpu["T_min"], rel=1e-4, abs=1e-8)
        assert s_gpu["T_max"] == pytest.approx(s_cpu["T_max"], rel=1e-4, abs=1e-8)

    def test_gpu_stretched_grid_uses_direct_solve_and_stays_finite(self):
        # regression test for a real bug: the GPU backend forced iterative CG
        # for the pressure solve even on a stretched grid, whose vertical
        # operator is asymmetric under stretching -- CG diverged (CFL ~1e12,
        # all fields NaN). Must use the direct solve instead, matching the
        # CPU heuristic, and stay numerically sane.
        cfg = _stretched_storm_cfg()
        rep_cpu, sim_cpu = _run(cfg, "cpu")
        rep_gpu, sim_gpu = _run(cfg, "gpu")
        assert sim_gpu.backend.name == "gpu"
        assert sim_gpu.pressure.method == "direct"
        assert rep_gpu["max_cfl"] < 10.0
        _assert_finite(sim_cpu)
        _assert_finite(sim_gpu)
        s_cpu, s_gpu = rep_cpu["final_stats"], rep_gpu["final_stats"]
        assert s_gpu["T_min"] == pytest.approx(s_cpu["T_min"], rel=1e-4, abs=1e-8)
        assert s_gpu["T_max"] == pytest.approx(s_cpu["T_max"], rel=1e-4, abs=1e-8)

    def test_gpu_backend_forces_cg_pressure_solver(self):
        # documents/pins the deliberate GPU-vs-CPU behavioural difference
        # from pressure_solver.py rather than leaving it implicit.
        cfg = _small_chamber_cfg(duration=1.0)
        _, sim = _run(cfg, "gpu")
        assert sim.pressure.method == "cg"

    def test_cpu_vs_gpu_storm_two_way(self):
        # combined path: anelastic dynamical core + two-way microphysics
        # coupling (hydrometeor growth, latent-heat feedback, sedimentation)
        # together on GPU -- the exact configuration `--storm-scale` defaults
        # to. See tests/test_flow_microphysics_coupling.py for microphysics-
        # only equivalence coverage in isolation.
        cfg = _small_storm_two_way_cfg()
        rep_cpu, sim_cpu = _run(cfg, "cpu")
        rep_gpu, sim_gpu = _run(cfg, "gpu")
        assert sim_gpu.backend.name == "gpu"
        _assert_finite(sim_cpu)
        _assert_finite(sim_gpu)
        s_cpu, s_gpu = rep_cpu["final_stats"], rep_gpu["final_stats"]
        assert s_gpu["T_min"] == pytest.approx(s_cpu["T_min"], rel=1e-4, abs=1e-8)
        assert s_gpu["T_max"] == pytest.approx(s_cpu["T_max"], rel=1e-4, abs=1e-8)
        b_cpu, b_gpu = rep_cpu["final_budgets"], rep_gpu["final_budgets"]
        assert b_gpu["total_water_rel_err"] == pytest.approx(
            b_cpu["total_water_rel_err"], abs=1e-3)
