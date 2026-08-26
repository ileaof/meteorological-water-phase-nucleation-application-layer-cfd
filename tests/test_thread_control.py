"""--compute-threads must change speed, never numerics.

threadpoolctl caps the BLAS/OpenMP thread pool at runtime around the solver
(see cli.py); this pins the "no numerics change" requirement for that
specific knob, separate from the broader CPU/GPU equivalence tests.
"""
from __future__ import annotations

from threadpoolctl import threadpool_limits

from meteorological_flow.config import SimulationConfig, apply_overrides
from meteorological_flow.simulation import Simulation


def _small_cfg():
    cfg = SimulationConfig()
    # no_microphysics=True -> nucleation.stage="none": skips building the
    # nucleation lookup table entirely (~minutes, uncached) -- this test only
    # needs to check flow-solver numerics, not exercise nucleation.
    cfg = apply_overrides(cfg, grid_resolution=8, duration=10.0, no_microphysics=True)
    cfg.output.format = []; cfg.output.figures = []; cfg.output.restart = False
    cfg.output.outdir = "outputs/_test_thread_control"
    return cfg


def test_compute_threads_flag_does_not_change_results():
    cfg = _small_cfg()
    with threadpool_limits(limits=1):
        rep_1 = Simulation(cfg).run()
    with threadpool_limits(limits=None):
        rep_default = Simulation(cfg).run()
    assert rep_1["final_stats"] == rep_default["final_stats"]
    assert rep_1["conservation"] == rep_default["conservation"]
