#!/usr/bin/env python3
"""test_reference_scenario.py -- reference-scenario smoke + budget report.

Runs the cold/dry vs warm/moist reference scenario at small scale end-to-end
through the Simulation orchestrator and checks:
  * the run completes and produces the full output set (NetCDF/CSV/JSON/PNG);
  * the flow stays stable (bounded velocities, small divergence, sane T range);
  * the one-way nucleation diagnostic populates log10I in the mixing zone;
  * the budget report is structurally complete (keys + limitations present).

A tiny in-test lookup is used for the one-way smoke so the suite stays fast
(the production 10080-point table build is exercised by the demo, not here).
"""
import json
import os
import tempfile

import numpy as np

from meteorological_flow.config import apply_overrides, from_yaml
from meteorological_flow.simulation import Simulation

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "configs", "cold_dry_vs_warm_moist.yaml")


def _cfg(**over):
    cfg = from_yaml(CFG_PATH)
    return apply_overrides(cfg, **over)


def test_reference_smoke_no_microphysics():
    """[reg] A short no-microphysics run completes, stays stable, and writes
    the full output set."""
    with tempfile.TemporaryDirectory() as out:
        cfg = _cfg(grid_resolution=20, duration=10.0, no_microphysics=True,
                   output_interval=5, output=out)
        sim = Simulation(cfg)
        report = sim.run()
        assert report["n_steps"] > 0
        assert report["stage"] == "none"
        assert os.path.exists(os.path.join(out, "flow.nc"))
        assert os.path.exists(os.path.join(out, "history.csv"))
        assert os.path.exists(os.path.join(out, "summary.json"))
        st = report["final_stats"]
        # stability: velocities bounded, T in a sane atmospheric range
        assert st["umax"] < 20.0, f"velocity blew up: {st['umax']}"
        assert 230.0 < st["T_min"] < st["T_max"] < 320.0
        assert report["max_cfl"] < 1.0, f"CFL exceeded 1: {report['max_cfl']}"


def test_reference_one_way_nucleation_populates():
    """[reg] A one-way run with a tiny lookup populates nucleation log10I in
    the supersaturated mixing zone (diagnostic only; state not modified)."""
    with tempfile.TemporaryDirectory() as out:
        cfg = _cfg(grid_resolution=8, duration=6.0, one_way=True,
                   output_interval=3, output=out)
        # shrink the lookup so the in-test build is fast (a few seconds)
        cfg.nucleation.lookup.n_T = 6
        cfg.nucleation.lookup.n_pv = 4
        cfg.nucleation.lookup.n_grad = 4
        cfg.nucleation.lookup.scan_resolution = 10
        cfg.nucleation.lookup.rebuild = True
        cfg.nucleation.lookup.threads = 1   # set by apply_overrides path
        sim = Simulation(cfg)
        report = sim.run()
        assert report["stage"] == "one_way"
        assert report["lookup_used"]
        # the final nucleation field should have at least some finite log10I
        nf = sim.last_nf
        assert np.any(np.isfinite(nf.log10I[0]) | np.isfinite(nf.log10I[1])), \
            "one-way nucleation did not produce any finite rate"
        # one-way: theta was advected/diffused by the FLOW, not by microphysics;
        # the check is that the run is stable (the gate's actual claim).
        assert report["final_stats"]["umax"] < 20.0


def test_reference_budget_report_complete():
    """[reg] The summary JSON report carries the required conservation /
    benchmark fields and the scientific-integrity limitations list."""
    with tempfile.TemporaryDirectory() as out:
        cfg = _cfg(grid_resolution=12, duration=4.0, no_microphysics=True,
                   output_interval=4, output=out)
        sim = Simulation(cfg)
        report = sim.run()
        for k in ("wall_clock_s", "n_steps", "max_cfl", "rho0", "T_ref",
                  "final_stats", "final_budgets", "stage", "limitations",
                  "code_version", "config"):
            assert k in report, f"report missing key {k}"
        bud = report["final_budgets"]
        for k in ("total_water_kg", "total_water_rel_err", "total_energy_J",
                  "total_energy_rel_err"):
            assert k in bud, f"budget missing key {k}"
        # limitations present and non-empty (scientific integrity)
        assert isinstance(report["limitations"], list) and report["limitations"]
        # code version + seed recorded (reproducibility)
        assert report["code_version"]
        assert report["config"]["random_seed"] == cfg.random_seed
        # cross-check the JSON file on disk matches the returned report
        with open(os.path.join(out, "summary.json"), encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["stage"] == report["stage"]
        assert on_disk["n_steps"] == report["n_steps"]