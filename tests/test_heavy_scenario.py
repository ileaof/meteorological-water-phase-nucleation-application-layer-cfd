"""Regression test for the ~100 mm rain+hail severe-storm example.

Locks in that the two-core storm example produces order-100 mm of rain plus
confirmed surface hail, with water conserved.
"""
from __future__ import annotations

import importlib.util
import os

from precip_microphysics import diagnostics as dg
from precip_microphysics.config import MicrophysicsConfig


def _load_example():
    path = os.path.join(os.path.dirname(__file__), "..", "examples",
                        "heavy_rain_hail_scenario.py")
    spec = importlib.util.spec_from_file_location("heavy_rain_hail_scenario", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cat(diag, name):
    return next(c for c in diag["categories"] if c["category"] == name)


def test_heavy_storm_about_100mm_rain_and_hail():
    hs = _load_example()
    cfg = MicrophysicsConfig()

    rain_st, rain_cum, rain_env, rain_prov = hs.simulate_rain_core()
    rain_mm = rain_st.accumulation.get("rain", 0.0)
    rdiag = dg.diagnose(rain_st, rain_cum, dict(rain_st.surface_flux), cfg,
                        env=rain_env, provenance=rain_prov, dt=3600.0)
    rain = _cat(rdiag, "rain")

    peak, hail_cum, hail_sf, hail_acc, hail_env, hail_prov = hs.simulate_hail_core()
    hdiag = dg.diagnose(peak, hail_cum, hail_sf, cfg,
                        env=hail_env, provenance=hail_prov, dt=1000.0)
    hail = _cat(hdiag, "hail")

    total_rain = rain_mm + hail_acc.get("rain", 0.0)
    total_hail = hail_acc.get("hail", 0.0)

    # order ~100 mm of rain, plus some hail reaching the ground
    assert 60.0 <= total_rain <= 170.0
    assert total_hail > 0.0
    # both categories confirmed at the surface
    assert rain["confirmed"] and rain["diagnostic_level"] == 4
    assert hail["confirmed"] and hail["diagnostic_level"] == 4
    # water conserved by the internal microphysics
    assert abs(rain_cum["_water_rel_err"]) < 1e-9
    assert abs(hail_cum["_water_rel_err"]) < 1e-9
