"""Assemble the evidence context and emit the precipitation-diagnostic schema.

:func:`diagnose` reduces a (0-D or column) :class:`MicrophysicsState`, the
process budget from :meth:`scheme.BulkMicrophysics.step`, the sedimentation
surface fluxes and the supplied-variable provenance into a scalar evidence
context, then calls :func:`evidence.evaluate_category` for rain, snow, graupel
and hail.  It also returns a cloud-formation diagnostic and hail-specific extras
(embryo source, supercooled LWC, updraft, residence time, wet/dry regime,
maximum diameter, melting fraction, surface-survival probability).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import constants as C
from . import evidence as ev
from . import size_distributions as sd

_CAT_SP = {"rain": "qr", "snow": "qs", "graupel": "qg", "hail": "qh"}

# which budget process names deposit mass INTO each category
_INTO = {
    "qc": ("nucleation_liquid", "condensation"),
    "qi": ("nucleation_ice", "deposition_ice", "immersion_freezing_cloud"),
    "qr": ("autoconversion", "accretion", "melting_snow", "melting_graupel", "melting_hail"),
    "qs": ("aggregation", "riming_snow", "deposition_snow"),
    "qg": ("snow_to_graupel", "riming_graupel", "freezing_rain"),
    "qh": ("hail_embryo", "riming_hail"),
}


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _redmax(x):
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return 0.0
    finite = a[np.isfinite(a)]
    return float(finite.max()) if finite.size else 0.0


def _budget(budget, name):
    return float(budget.get(name, 0.0))


def diagnose(st, budget, surface_flux, cfg, *, env, provenance,
             microphysics_enabled=True, dt=1.0):
    """Return the full diagnostic dict for one parcel/column."""
    rho_rep = float(np.mean(np.asarray(st.rho)))
    Tmin = float(np.min(np.asarray(st.T)))
    Tmax = float(np.max(np.asarray(st.T)))

    # reduced category mixing ratios / kinematics
    q = {c: _redmax(getattr(st, sp)) for c, sp in _CAT_SP.items()}
    q_qc = _redmax(st.qc)
    q_qi = _redmax(st.qi)
    vt = {c: _redmax(sd.mass_weighted_vt(getattr(st, sp), st.rho, c))
          for c, sp in _CAT_SP.items()}
    number = {c: float(np.nan_to_num(_redmax(sd.number_conc(getattr(st, sp), st.rho, c))))
              for c, sp in _CAT_SP.items()}
    radius = {c: float(np.nan_to_num(_redmax(sd.characteristic_radius(getattr(st, sp), st.rho, c))))
              for c, sp in _CAT_SP.items()}

    # production rates (kg/kg/s and kg/m3/s) from the budget
    prod_mix = {c: sum(_budget(budget, n) for n in _INTO[sp]) / max(dt, C.TINY)
                for c, sp in _CAT_SP.items()}
    prod_si = {c: prod_mix[c] * rho_rep for c in _CAT_SP}

    # --- thermodynamic (Level-1) favourability ---
    Sw, Si = env["Sw"], env["Si"]
    wmax = env.get("wmax", 0.0)
    cold = _clip01((C.T0 - Tmin) / 40.0)
    supw = _clip01((Sw - 1.0) / 0.20)
    supi = _clip01((Si - 1.0) / 0.20)
    favourability = {
        "rain": supw,
        "snow": supi * cold,
        "graupel": cold * supi,
        "hail": cold * supw,
    }
    supersaturated = (Sw > 1.0) or (Si > 1.0)

    # --- evidence "has" flags (data completeness) ---
    supercooled_lwc = q_qc * rho_rep if Tmin < C.T0 else 0.0
    has = {
        "cloud_liquid": q_qc > C.QSMALL,
        "updraft": provenance.get("w_supplied", False),
        "strong_updraft": provenance.get("w_supplied", False) and wmax > C.HAIL_UPDRAFT_CRIT,
        "layer_depth": provenance.get("dz_supplied", False),
        "dt": True,
        "ice_water": q_qi > C.QSMALL,
        "snow": q["snow"] > C.QSMALL,
        "subfreezing": Tmin < C.T0,
        "supercooled_liquid": (q_qc > C.QSMALL) and (Tmin < C.T0),
        "graupel": q["graupel"] > C.QSMALL,
        "residence_time": provenance.get("residence_time_supplied", False),
        "freezing_level": provenance.get("freezing_level_supplied", False),
    }

    sed_on = cfg.processes.sedimentation

    def _sed_contrib(cat):
        sp = _CAT_SP[cat]
        return sed_on and (_redmax(getattr(st, sp)) > C.QSMALL or surface_flux.get(cat, 0.0) > 0.0)

    contrib = {
        "condensation": _budget(budget, "condensation") > 0.0,
        "warm_collection": (_budget(budget, "autoconversion") + _budget(budget, "accretion")) > 0.0,
        "ice_nucleation": (_budget(budget, "nucleation_ice")
                           + _budget(budget, "immersion_freezing_cloud")) > 0.0,
        "deposition": (_budget(budget, "deposition_snow") + _budget(budget, "deposition_ice")) > 0.0,
        "aggregation": _budget(budget, "aggregation") > 0.0,
        "riming": (_budget(budget, "riming_snow") + _budget(budget, "riming_graupel")
                   + _budget(budget, "riming_hail")) > 0.0,
        "graupel_conversion": _budget(budget, "snow_to_graupel") > 0.0,
        "hail_embryo": _budget(budget, "hail_embryo") > 0.0,
        "hail_survival_evaluated": cfg.processes.hail_melting,
        "sedimentation_rain": _sed_contrib("rain"),
        "sedimentation_snow": _sed_contrib("snow"),
        "sedimentation_graupel": _sed_contrib("graupel"),
        "sedimentation_hail": _sed_contrib("hail"),
    }

    # --- validity / numerics ---
    T_lo, T_hi = cfg.T_valid
    T_in_range = bool(Tmin >= T_lo and Tmax <= T_hi)
    water_rel_err = float(budget.get("_water_rel_err", 0.0))
    conservation_ok = abs(water_rel_err) <= max(cfg.conservation_tol, 1e-8)
    numerics_ok = bool(np.all(np.isfinite(np.asarray(st.T))))
    for sp in _CAT_SP.values():
        numerics_ok = numerics_ok and bool(np.all(np.isfinite(np.asarray(getattr(st, sp)))))

    ctx = SimpleNamespace(
        q=q, vt=vt, number=number, radius=radius,
        production_rate=prod_mix, production_rate_si=prod_si,
        favourability=favourability, supersaturated=supersaturated,
        surface_flux={c: float(surface_flux.get(c, 0.0)) for c in _CAT_SP},
        accumulation={c: float(st.accumulation.get(c, 0.0)) for c in _CAT_SP},
        has=has, contrib=contrib,
        microphysics_enabled=microphysics_enabled,
        T_in_range=T_in_range, conservation_ok=conservation_ok,
        numerics_ok=numerics_ok, water_rel_err=water_rel_err,
    )

    categories = []
    for cat in ("rain", "snow", "graupel", "hail"):
        rec = ev.evaluate_category(cat, ctx, cfg.threshold(cat))
        if cat == "hail":
            rec.update(_hail_extras(st, budget, surface_flux, env, q, radius,
                                    supercooled_lwc, wmax, rho_rep, dt))
        categories.append(rec)

    cloud = _cloud_diagnostic(q_qc, q_qi, rho_rep, Tmin, supersaturated)
    overall = _overall(categories, water_rel_err, conservation_ok)
    return {"categories": categories, "cloud_formation": cloud, "overall": overall}


def _hail_extras(st, budget, surface_flux, env, q, radius, slw, wmax, rho, dt):
    produced = sum(float(budget.get(n, 0.0)) for n in _INTO["qh"])
    melted = float(budget.get("melting_hail", 0.0))
    melt_frac = _clip01(melted / (produced + q["hail"] + C.TINY))
    regime = "wet_growth" if slw > C.HAIL_LWC_CRIT else "dry_growth"
    surf = float(surface_flux.get("hail", 0.0))
    survival = _clip01(1.0 - melt_frac) if q["hail"] > C.QSMALL or surf > 0 else 0.0
    return {
        "embryo_source": "graupel" if float(budget.get("hail_embryo", 0.0)) > 0 else "none",
        "supercooled_liquid_water_kg_m3": float(slw),
        "max_updraft_m_s": float(wmax),
        "growth_region_depth_m": float(env.get("cloud_depth") or 0.0),
        "residence_time_s": float(env.get("residence_time") or 0.0),
        "growth_regime": regime,
        "max_diameter_m": float(2.0 * radius["hail"]),
        "melting_fraction": round(melt_frac, 4),
        "surface_survival_probability": round(survival, 4),
    }


def _cloud_diagnostic(qc, qi, rho, Tmin, supersaturated):
    lwc = qc * rho
    iwc = qi * rho
    level = 0
    if supersaturated:
        level = 1
    if qc > C.QSMALL or qi > C.QSMALL:
        level = 2
    return {
        "liquid_cloud_present": qc > C.QSMALL,
        "ice_cloud_present": qi > C.QSMALL,
        "LWC_kg_m3": float(lwc),
        "IWC_kg_m3": float(iwc),
        "diagnostic_level": level,
        "diagnostic_level_name": ev.LEVEL_NAMES[level],
    }


def _overall(categories, water_rel_err, conservation_ok):
    max_level = max(c["diagnostic_level"] for c in categories)
    confirmed = [c["category"] for c in categories if c["confirmed"]]
    reasons = sorted({r for c in categories for r in c["reason_codes"]})
    return {
        "max_diagnostic_level": max_level,
        "max_diagnostic_level_name": ev.LEVEL_NAMES[max_level],
        "confirmed_categories": confirmed,
        "any_confirmed": bool(confirmed),
        "reason_codes": reasons,
        "water_rel_err": float(water_rel_err),
        "conservation_ok": bool(conservation_ok),
    }


__all__ = ["diagnose"]
