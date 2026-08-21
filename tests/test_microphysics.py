"""Validation suite for precip_microphysics (Increment 1).

Twenty tests demonstrating the required behaviour: a high nucleation rate can
never, by itself, confirm precipitation; hydrometeor growth, sedimentation and
survival are what promote a category to Levels 3-4; water is conserved; latent
heating has the correct sign; and the confidence/level/caveat/reason-code model
behaves exactly as specified.
"""
from __future__ import annotations

import numpy as np
import pytest

from precip_microphysics import diagnostics as dg
from precip_microphysics import sedimentation as sed
from precip_microphysics.column import ColumnModel
from precip_microphysics.config import MicrophysicsConfig, ProcessSwitches
from precip_microphysics.evidence import CAVEAT, Reason
from precip_microphysics.scheme import BulkMicrophysics
from precip_microphysics.state import MicrophysicsState
from precip_microphysics import thermo as th


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _cfg(**over):
    return MicrophysicsConfig(**over)


def _only(**enabled):
    """A config with every process off except those named True."""
    sw = ProcessSwitches().all_off()
    for k, v in enabled.items():
        setattr(sw, k, v)
    return MicrophysicsConfig(processes=sw)


def _prov(**over):
    p = {"w_supplied": True, "dz_supplied": True,
         "residence_time_supplied": True, "freezing_level_supplied": True}
    p.update(over)
    return p


def _cat(diag, name):
    return next(c for c in diag["categories"] if c["category"] == name)


def _confirmed_inputs(category):
    """Craft (st, budget, surface_flux, env, provenance) that SHOULD confirm the
    given category at Level 4 (used to exercise the diagnostic logic directly)."""
    base = dict(T=257.0, P=60000.0, rho=0.8, w=30.0, dz=2000.0,
                qc=2.5e-3, qi=6e-4, qs=1e-3, qg=1.5e-3)
    st = MicrophysicsState(qv=1e-3, **base)
    budget = {
        "condensation": 3e-4, "autoconversion": 2e-4, "accretion": 2e-4,
        "nucleation_ice": 2e-5, "deposition_snow": 1e-4, "deposition_ice": 1e-4,
        "aggregation": 1e-4, "riming_snow": 1e-4, "riming_graupel": 1e-4,
        "snow_to_graupel": 1e-4, "riming_hail": 1e-4, "hail_embryo": 5e-5,
        "_water_rel_err": 0.0,
    }
    surface_flux = {"rain": 1e-3, "snow": 8e-4, "graupel": 6e-4, "hail": 5e-4}
    env = {"Sw": 1.15, "Si": 1.30, "wmax": 30.0, "rho": 0.8,
           "cloud_depth": 4000.0, "residence_time": 900.0, "freezing_level": 3000.0}
    if category == "rain":
        st.T = np.asarray(280.0)                    # warm rain
        env["wmax"] = 2.0
    st.qr = np.asarray(1.2e-3)
    st.qh = np.asarray(1.2e-3)
    return st, budget, surface_flux, env, _prov()


# ==========================================================================
# 1-2  nucleation alone cannot confirm precipitation; Level <= 1
# ==========================================================================
def test_01_nucleation_without_growth_not_confirmed():
    col = ColumnModel(_cfg())
    d = col.run_parcel(T=260.0, P=70000.0, RH=112.0, w=None,
                       microphysics_enabled=False, use_kernel=True)
    assert d["nucleation"]["log10I_liquid"] > 20.0        # nucleation is favourable
    assert not d["overall"]["any_confirmed"]
    for c in d["categories"]:
        assert c["confirmed"] is False
        assert c["caveat_required"] is True


def test_02_high_nucleation_is_level_one_at_most():
    col = ColumnModel(_cfg())
    d = col.run_parcel(T=258.0, P=70000.0, RH=130.0,
                       microphysics_enabled=False, use_kernel=True)
    assert d["overall"]["max_diagnostic_level"] <= 1
    assert Reason.THERMODYNAMICS_ONLY in _cat(d, "rain")["reason_codes"]


# ==========================================================================
# 3-7  each category needs its own growth/transport evidence
# ==========================================================================
def test_03_rain_needs_collision_coalescence_and_sedimentation():
    st, b, sf, env, prov = _confirmed_inputs("rain")
    # baseline confirms
    ok = _cat(dg.diagnose(st, b, sf, _cfg(), env=env, provenance=prov), "rain")
    assert ok["confirmed"]
    # remove collision-coalescence evidence
    b2 = dict(b); b2["autoconversion"] = 0.0; b2["accretion"] = 0.0
    r = _cat(dg.diagnose(st, b2, sf, _cfg(), env=env, provenance=prov), "rain")
    assert not r["confirmed"]
    assert Reason.NO_COLLISION_COALESCENCE in r["reason_codes"]
    # remove surface flux
    r2 = _cat(dg.diagnose(st, b, {"rain": 0.0}, _cfg(), env=env, provenance=prov), "rain")
    assert r2["diagnostic_level"] < 4 and not r2["confirmed"]
    assert Reason.NO_SURFACE_FLUX in r2["reason_codes"]


def test_04_snow_needs_deposition_aggregation_and_survival():
    st, b, sf, env, prov = _confirmed_inputs("snow")
    b2 = dict(b); b2["deposition_snow"] = 0.0; b2["deposition_ice"] = 0.0; b2["aggregation"] = 0.0
    s = _cat(dg.diagnose(st, b2, sf, _cfg(), env=env, provenance=prov), "snow")
    assert not s["confirmed"]
    assert Reason.NO_DEPOSITION_GROWTH in s["reason_codes"]
    assert Reason.NO_AGGREGATION in s["reason_codes"]


def test_05_graupel_needs_riming():
    st, b, sf, env, prov = _confirmed_inputs("graupel")
    b2 = dict(b); b2["riming_snow"] = b2["riming_graupel"] = b2["riming_hail"] = 0.0
    b2["snow_to_graupel"] = 0.0
    g = _cat(dg.diagnose(st, b2, sf, _cfg(), env=env, provenance=prov), "graupel")
    assert not g["confirmed"]
    assert Reason.NO_RIMING_MODEL in g["reason_codes"]


def test_06_hail_needs_embryo_slw_updraft_residence():
    st, b, sf, env, prov = _confirmed_inputs("hail")
    assert _cat(dg.diagnose(st, b, sf, _cfg(), env=env, provenance=prov), "hail")["confirmed"]
    # remove the updraft and residence-time evidence
    h = _cat(dg.diagnose(st, b, sf, _cfg(),
                         env={**env, "wmax": 0.0},
                         provenance=_prov(w_supplied=False, residence_time_supplied=False)),
             "hail")
    assert not h["confirmed"]
    assert Reason.MISSING_VERTICAL_VELOCITY in h["reason_codes"]
    assert Reason.INSUFFICIENT_RESIDENCE_TIME in h["reason_codes"]


def test_07_hail_aloft_distinct_from_surface():
    st, b, sf, env, prov = _confirmed_inputs("hail")
    aloft = _cat(dg.diagnose(st, b, {"hail": 0.0}, _cfg(), env=env, provenance=prov), "hail")
    surface = _cat(dg.diagnose(st, b, sf, _cfg(), env=env, provenance=prov), "hail")
    assert aloft["diagnostic_level"] < 4 and not aloft["confirmed"]
    assert Reason.NO_SURFACE_FLUX in aloft["reason_codes"]
    assert surface["diagnostic_level"] == 4 and surface["confirmed"]


# ==========================================================================
# 8-10  thresholds, caveat text, reason codes
# ==========================================================================
def test_08_thresholds_exactly_050_and_075():
    cfg = _cfg()
    assert cfg.threshold("rain") == 0.50
    assert cfg.threshold("snow") == 0.50
    assert cfg.threshold("graupel") == 0.50
    assert cfg.threshold("hail") == 0.75
    # a confidence in [0.5, 0.75) confirms rain but NOT hail (differential bar)
    st, b, sf, env, prov = _confirmed_inputs("hail")
    # drop two of six hail data vars -> completeness 4/6 = 0.667 in [0.5,0.75)
    h = _cat(dg.diagnose(st, b, sf, _cfg(),
                         env={**env, "wmax": 0.0},
                         provenance=_prov(residence_time_supplied=False,
                                          freezing_level_supplied=False)), "hail")
    assert 0.5 <= h["confidence"] < 0.75
    assert not h["confirmed"] and h["caveat_required"]


def test_09_exact_caveat_below_threshold():
    st, b, sf, env, prov = _confirmed_inputs("rain")
    conf = _cat(dg.diagnose(st, b, sf, _cfg(), env=env, provenance=prov), "rain")
    assert conf["caveat"] == ""                              # confirmed -> no caveat
    unconf = _cat(dg.diagnose(st, b, {"rain": 0.0}, _cfg(), env=env, provenance=prov), "rain")
    assert unconf["caveat"] == CAVEAT
    assert CAVEAT == ("Thermodynamically favourable to nucleation, but the dynamic and "
                      "microphysical data are insufficient to confirm precipitation or hail.")


def test_10_reason_codes_identify_missing_evidence():
    st, b, sf, env, prov = _confirmed_inputs("rain")
    r = _cat(dg.diagnose(st, b, sf, _cfg(),
                         env=env, provenance=_prov(w_supplied=False)), "rain")
    assert Reason.MISSING_VERTICAL_VELOCITY in r["reason_codes"]
    assert "updraft" in r["missing_variables"]


# ==========================================================================
# 11-16  conservation, latent heat, positivity, sedimentation, phase changes
# ==========================================================================
def test_11_water_mass_conserved():
    mp = BulkMicrophysics(_cfg())
    qv = th.qsat_water(268.0, 70000.0) * 1.2
    st = MicrophysicsState(T=268.0, P=70000.0, rho=0.9, w=5.0, dz=1000.0,
                           qv=float(qv), qc=1e-3, qr=5e-4, qi=3e-4, qs=2e-4)
    b = mp.step(st, 5.0, cell_volume=1e9, J_liquid=1e40, J_ice=1e40)
    assert abs(b["_water_rel_err"]) < 1e-10


def test_12_latent_heating_signs():
    # condensation warms
    mp = BulkMicrophysics(_only(condensation=True))
    qv = float(th.qsat_water(285.0, 90000.0) * 1.05)
    st = MicrophysicsState(T=285.0, P=90000.0, rho=1.1, qv=qv, qc=1e-4)
    T0 = float(st.T); mp.step(st, 2.0); assert float(st.T) > T0
    # cloud evaporation cools
    qv = float(th.qsat_water(285.0, 90000.0) * 0.9)
    st = MicrophysicsState(T=285.0, P=90000.0, rho=1.1, qv=qv, qc=1e-3)
    T0 = float(st.T); mp.step(st, 2.0); assert float(st.T) < T0
    # graupel melting cools
    mp = BulkMicrophysics(_only(graupel_melting=True))
    st = MicrophysicsState(T=280.0, P=90000.0, rho=1.1, qv=1e-3, qg=1e-3)
    T0 = float(st.T); mp.step(st, 5.0); assert float(st.T) < T0
    # freezing (immersion) warms
    mp = BulkMicrophysics(_only(ice_nucleation=True))
    st = MicrophysicsState(T=258.0, P=70000.0, rho=0.9, qv=1e-3, qc=1e-3)
    T0 = float(st.T); mp.step(st, 5.0); assert float(st.T) >= T0


def test_13_fields_stay_nonnegative():
    mp = BulkMicrophysics(_cfg())
    qv = float(th.qsat_water(265.0, 70000.0) * 1.15)
    st = MicrophysicsState(T=265.0, P=70000.0, rho=0.9, w=10.0, dz=1000.0,
                           qv=qv, qc=1e-3, qi=5e-4)
    for _ in range(50):
        mp.step(st, 5.0, cell_volume=1e9, J_liquid=1e40, J_ice=1e40)
        sed.sediment(st, mp.cfg, 5.0)
        for s in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
            assert np.all(np.asarray(getattr(st, s)) >= 0.0)


def test_14_sedimentation_surface_flux_mass_balance():
    cfg = _only(sedimentation=True)
    st = MicrophysicsState(T=283.0, P=95000.0, rho=1.2, dz=1000.0, qr=1e-3)
    q0 = float(st.qr)
    sf = sed.sediment(st, cfg, 10.0)
    q1 = float(st.qr)
    assert sf["rain"] > 0.0
    mass_lost = (q0 - q1) * 1.2 * 1000.0            # kg/m^2
    assert st.accumulation["rain"] == pytest.approx(mass_lost, rel=1e-6)


def test_15_melting_reduces_frozen_mass():
    mp = BulkMicrophysics(_only(graupel_melting=True))
    st = MicrophysicsState(T=282.0, P=90000.0, rho=1.1, qv=1e-3, qg=2e-3)
    qg0, qr0 = float(st.qg), float(st.qr)
    mp.step(st, 10.0)
    assert float(st.qg) < qg0
    assert float(st.qr) > qr0


def test_16_evaporation_reduces_rain_in_subsaturated_air():
    mp = BulkMicrophysics(_only(rain_evaporation=True))
    qv = float(th.qsat_water(295.0, 95000.0) * 0.6)     # dry
    st = MicrophysicsState(T=295.0, P=95000.0, rho=1.1, qv=qv, qr=1e-3)
    qr0, qv0 = float(st.qr), float(st.qv)
    mp.step(st, 10.0)
    assert float(st.qr) < qr0
    assert float(st.qv) > qv0


# ==========================================================================
# 17-20  disable-microphysics, numerical/validity confidence, reproducibility
# ==========================================================================
def test_17_disable_microphysics_restores_thermodynamic_only():
    col = ColumnModel(_cfg())
    d = col.run_parcel(T=288.0, P=95000.0, RH=101.0, w=1.0, dz=1500.0,
                       duration=1200.0, dt=5.0,
                       microphysics_enabled=False, use_kernel=False)
    assert d["overall"]["max_diagnostic_level"] <= 1
    assert not d["overall"]["any_confirmed"]
    for c in d["categories"]:
        assert Reason.THERMODYNAMICS_ONLY in c["reason_codes"]
        assert c["mixing_ratio_kg_kg"] == 0.0


def test_18_numerical_failure_lowers_confidence():
    st, b, sf, env, prov = _confirmed_inputs("rain")
    good = _cat(dg.diagnose(st, b, sf, _cfg(), env=env, provenance=prov), "rain")
    bad_budget = dict(b); bad_budget["_water_rel_err"] = 1e-2
    bad = _cat(dg.diagnose(st, bad_budget, sf, _cfg(), env=env, provenance=prov), "rain")
    assert bad["confidence"] < good["confidence"]
    assert Reason.NUMERICAL_CONSERVATION_FAILURE in bad["reason_codes"]


def test_19_model_validity_violation_lowers_confidence():
    st, b, sf, env, prov = _confirmed_inputs("rain")
    good = _cat(dg.diagnose(st, b, sf, _cfg(), env=env, provenance=prov), "rain")
    st.T = np.asarray(200.0)                            # below the validity envelope
    bad = _cat(dg.diagnose(st, b, sf, _cfg(), env=env, provenance=prov), "rain")
    assert bad["confidence"] < good["confidence"]
    assert Reason.OUTSIDE_MODEL_VALIDITY in bad["reason_codes"]


def test_20_reproducible_outputs():
    col = ColumnModel(_cfg())
    kw = dict(T=290.0, P=95000.0, RH=100.5, w=1.0, dz=1500.0,
              duration=900.0, dt=5.0, use_kernel=False)
    a = _cat(col.run_parcel(**kw), "rain")
    b = _cat(col.run_parcel(**kw), "rain")
    assert a["confidence"] == b["confidence"]
    assert a["accumulation_mm"] == b["accumulation_mm"]
    assert a["diagnostic_level"] == b["diagnostic_level"]
