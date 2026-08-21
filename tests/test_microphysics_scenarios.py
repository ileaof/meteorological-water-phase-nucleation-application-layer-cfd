"""Integration tests for the four reference scenarios and the config loader.

These lock in the central claim: a high nucleation rate alone (scenario 1)
confirms nothing, while scenarios 2-4 confirm precipitation only because the
growth / sedimentation / survival chain actually ran.
"""
from __future__ import annotations

import os

from precip_microphysics import scenarios
from precip_microphysics.config import from_dict, from_yaml


def _best(diag):
    return max(diag["categories"], key=lambda c: (c["diagnostic_level"], c["confidence"]))


def test_scenario1_nucleation_only_confirms_nothing():
    d = scenarios.high_nucleation_no_microphysics(use_kernel=False)
    assert d["overall"]["max_diagnostic_level"] <= 1
    assert not d["overall"]["any_confirmed"]


def test_scenario2_warm_rain_reaches_surface():
    d = scenarios.warm_rain(use_kernel=False)
    rain = next(c for c in d["categories"] if c["category"] == "rain")
    assert rain["diagnostic_level"] == 4
    assert rain["confirmed"]
    assert rain["accumulation_mm"] > 0.0


def test_scenario3_mixed_phase_makes_snow():
    d = scenarios.mixed_phase(use_kernel=False)
    snow = next(c for c in d["categories"] if c["category"] == "snow")
    assert snow["diagnostic_level"] >= 2          # at least production aloft
    assert snow["mixing_ratio_kg_kg"] > 0.0


def test_scenario4_hail_grows_and_partially_survives():
    d = scenarios.deep_convective_hail(use_kernel=False)
    hail = next(c for c in d["categories"] if c["category"] == "hail")
    assert hail["diagnostic_level"] == 4
    assert hail["confirmed"]
    assert 0.0 < hail["surface_survival_probability"] <= 1.0
    assert hail["growth_regime"] == "wet_growth"


def test_scenarios_distinguish_the_stages():
    res = scenarios.run_all(use_kernel=False)
    lvl1 = res["1_high_nucleation_no_microphysics"]["overall"]["max_diagnostic_level"]
    assert lvl1 <= 1
    for key in ("2_warm_rain", "3_mixed_phase", "4_deep_convective_hail"):
        assert res[key]["overall"]["max_diagnostic_level"] >= 3


def test_config_from_dict_and_yaml():
    cfg = from_dict({"processes": {"riming": False}, "threshold_hail": 0.75})
    assert cfg.processes.riming is False
    assert cfg.threshold("hail") == 0.75
    path = os.path.join(os.path.dirname(__file__), "..", "configs",
                        "microphysics_reference.yaml")
    if os.path.exists(path):
        c2 = from_yaml(path)
        assert c2.threshold("rain") == 0.50
        assert c2.processes.sedimentation is True
