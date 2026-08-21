"""Evidence-based confidence, diagnostic levels, reason codes and caveat logic.

This module resolves the limitation that "a high nucleation rate never by itself
implies rain or hail".  It never confirms a precipitation category from the
nucleation rate: confirmation requires *physical evidence* that the growth,
sedimentation and (for surface precipitation) survival chain actually ran.

Confidence is an auditable product of four components, each in [0, 1]:

    confidence = data_completeness * model_validity
               * process_evidence * numerical_quality

* **data_completeness** -- fraction of the dynamic/microphysical variables the
  phenomenon needs that were actually supplied;
* **model_validity**    -- 1 inside the correlation envelope, reduced when the
  state is extrapolated;
* **process_evidence**  -- fraction of the required growth processes that were
  enabled *and contributed mass* (this is what makes nucleation-only evidence
  insufficient);
* **numerical_quality** -- 1 when water is conserved and no numerical failure
  occurred, sharply reduced otherwise.

Diagnostic levels (0-4) separate favourability from production, precipitation
and surface precipitation.  Only Levels 3-4 are "precipitation"; a category is
``confirmed`` only at **Level 4** with confidence at/above its threshold and no
hard blocking reason -- so a high nucleation rate alone can reach Level 1 only.
"""
from __future__ import annotations

CAVEAT = ("Thermodynamically favourable to nucleation, but the dynamic and "
          "microphysical data are insufficient to confirm precipitation or hail.")


class Reason:
    THERMODYNAMICS_ONLY = "THERMODYNAMICS_ONLY"
    MISSING_VERTICAL_VELOCITY = "MISSING_VERTICAL_VELOCITY"
    MISSING_SUPERCOOLED_LIQUID_WATER = "MISSING_SUPERCOOLED_LIQUID_WATER"
    NO_COLLISION_COALESCENCE = "NO_COLLISION_COALESCENCE"
    NO_DEPOSITION_GROWTH = "NO_DEPOSITION_GROWTH"
    NO_AGGREGATION = "NO_AGGREGATION"
    NO_RIMING_MODEL = "NO_RIMING_MODEL"
    NO_SEDIMENTATION = "NO_SEDIMENTATION"
    NO_SURFACE_FLUX = "NO_SURFACE_FLUX"
    INSUFFICIENT_RESIDENCE_TIME = "INSUFFICIENT_RESIDENCE_TIME"
    HAIL_SURVIVAL_NOT_EVALUATED = "HAIL_SURVIVAL_NOT_EVALUATED"
    OUTSIDE_MODEL_VALIDITY = "OUTSIDE_MODEL_VALIDITY"
    NUMERICAL_CONSERVATION_FAILURE = "NUMERICAL_CONSERVATION_FAILURE"


# reasons that block confirmation outright (independent of level/confidence)
HARD_BLOCK = {
    Reason.THERMODYNAMICS_ONLY,
    Reason.OUTSIDE_MODEL_VALIDITY,
    Reason.NUMERICAL_CONSERVATION_FAILURE,
}

LEVEL_NAMES = {
    0: "insufficient_information",
    1: "thermodynamic_favourability",
    2: "hydrometeor_production",
    3: "precipitation_development",
    4: "surface_precipitation",
}

# level thresholds
Q_PRODUCTION = 1.0e-9      # kg/kg, category exists aloft (Level 2)
Q_PRECIP = 1.0e-5         # kg/kg, enough mass to precipitate (Level 3)
FLUX_SURFACE = 1.0e-8     # kg m^-2 s^-1, positive surface flux (Level 4)
VT_PRECIP = {"rain": 1.0, "snow": 0.3, "graupel": 1.0, "hail": 5.0}   # m/s


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# Per-category evidence specification. Each entry:
#   data_vars    : (label, ctx-key-present, reason-if-missing)
#   proc_evidence: (label, ctx-key-contributed, reason-if-absent)
def _rain_spec(ctx):
    data = [
        ("cloud_liquid_water", ctx.has["cloud_liquid"], Reason.MISSING_SUPERCOOLED_LIQUID_WATER),
        ("updraft", ctx.has["updraft"], Reason.MISSING_VERTICAL_VELOCITY),
        ("layer_depth", ctx.has["layer_depth"], None),
        ("sedimentation_dt", ctx.has["dt"], None),
    ]
    proc = [
        ("condensation_growth", ctx.contrib["condensation"], Reason.NO_DEPOSITION_GROWTH),
        ("collision_coalescence", ctx.contrib["warm_collection"], Reason.NO_COLLISION_COALESCENCE),
        ("sedimentation", ctx.contrib["sedimentation_rain"], Reason.NO_SEDIMENTATION),
    ]
    return data, proc


def _snow_spec(ctx):
    data = [
        ("ice_water", ctx.has["ice_water"], None),
        ("subfreezing", ctx.has["subfreezing"], None),
        ("layer_depth", ctx.has["layer_depth"], None),
        ("sedimentation_dt", ctx.has["dt"], None),
    ]
    proc = [
        ("ice_nucleation", ctx.contrib["ice_nucleation"], Reason.THERMODYNAMICS_ONLY),
        ("deposition_growth", ctx.contrib["deposition"], Reason.NO_DEPOSITION_GROWTH),
        ("aggregation", ctx.contrib["aggregation"], Reason.NO_AGGREGATION),
        ("sedimentation", ctx.contrib["sedimentation_snow"], Reason.NO_SEDIMENTATION),
    ]
    return data, proc


def _graupel_spec(ctx):
    data = [
        ("ice_or_snow_embryo", ctx.has["ice_water"] or ctx.has["snow"], None),
        ("supercooled_liquid_water", ctx.has["supercooled_liquid"], Reason.MISSING_SUPERCOOLED_LIQUID_WATER),
        ("updraft", ctx.has["updraft"], Reason.MISSING_VERTICAL_VELOCITY),
        ("layer_depth", ctx.has["layer_depth"], None),
    ]
    proc = [
        ("ice_nucleation", ctx.contrib["ice_nucleation"], Reason.THERMODYNAMICS_ONLY),
        ("riming", ctx.contrib["riming"], Reason.NO_RIMING_MODEL),
        ("graupel_conversion", ctx.contrib["graupel_conversion"], Reason.NO_RIMING_MODEL),
        ("sedimentation", ctx.contrib["sedimentation_graupel"], Reason.NO_SEDIMENTATION),
    ]
    return data, proc


def _hail_spec(ctx):
    data = [
        ("graupel_embryo", ctx.has["graupel"], None),
        ("supercooled_liquid_water", ctx.has["supercooled_liquid"], Reason.MISSING_SUPERCOOLED_LIQUID_WATER),
        ("strong_deep_updraft", ctx.has["strong_updraft"], Reason.MISSING_VERTICAL_VELOCITY),
        ("residence_time", ctx.has["residence_time"], Reason.INSUFFICIENT_RESIDENCE_TIME),
        ("freezing_level", ctx.has["freezing_level"], None),
        ("layer_depth", ctx.has["layer_depth"], None),
    ]
    proc = [
        ("hail_embryo", ctx.contrib["hail_embryo"], Reason.MISSING_SUPERCOOLED_LIQUID_WATER),
        ("wet_dry_growth", ctx.contrib["riming"], Reason.NO_RIMING_MODEL),
        ("sedimentation", ctx.contrib["sedimentation_hail"], Reason.NO_SEDIMENTATION),
        ("melting_survival", ctx.contrib["hail_survival_evaluated"], Reason.HAIL_SURVIVAL_NOT_EVALUATED),
    ]
    return data, proc


_SPEC = {"rain": _rain_spec, "snow": _snow_spec, "graupel": _graupel_spec, "hail": _hail_spec}


def evaluate_category(category: str, ctx, threshold: float) -> dict:
    """Return the full per-category diagnostic dict (the output schema)."""
    data_vars, proc_ev = _SPEC[category](ctx)
    reasons: set[str] = set()
    supporting: dict = {}
    missing: list[str] = []

    # --- data completeness ---
    n_present = 0
    for label, present, rc in data_vars:
        supporting[label] = bool(present)
        if present:
            n_present += 1
        else:
            missing.append(label)
            if rc:
                reasons.add(rc)
    data_completeness = n_present / len(data_vars) if data_vars else 1.0

    # --- process evidence ---
    n_contrib = 0
    for label, contributed, rc in proc_ev:
        supporting[f"process:{label}"] = bool(contributed)
        if contributed:
            n_contrib += 1
        elif rc:
            reasons.add(rc)
    process_evidence = n_contrib / len(proc_ev) if proc_ev else 0.0
    if not ctx.microphysics_enabled:
        process_evidence = 0.0
        reasons.add(Reason.THERMODYNAMICS_ONLY)

    # --- model validity ---
    model_validity = 1.0
    if not ctx.T_in_range:
        model_validity = 0.4
        reasons.add(Reason.OUTSIDE_MODEL_VALIDITY)

    # --- numerical quality ---
    numerical_quality = 1.0
    if not ctx.conservation_ok:
        numerical_quality *= 0.3
        reasons.add(Reason.NUMERICAL_CONSERVATION_FAILURE)
    if not ctx.numerics_ok:
        numerical_quality *= 0.3

    confidence = _clip01(data_completeness * model_validity
                         * process_evidence * numerical_quality)

    # --- diagnostic level ---
    q = ctx.q[category]
    prod = ctx.production_rate[category]
    vt = ctx.vt[category]
    flux = ctx.surface_flux[category]
    level = 0
    if ctx.favourability[category] > 0.0 or ctx.supersaturated:
        level = 1
    if prod > 0.0 or q > Q_PRODUCTION:
        level = max(level, 2)
    if q > Q_PRECIP and vt > VT_PRECIP[category]:
        level = max(level, 3)
    if flux > FLUX_SURFACE:
        level = max(level, 4)

    if flux <= FLUX_SURFACE:
        reasons.add(Reason.NO_SURFACE_FLUX)

    # Confirmation requires the COMPLETE growth-and-transport chain: every
    # required process must have contributed (process_evidence == 1), the
    # category must have reached the surface (Level 4), confidence must clear
    # its threshold, and no hard blocking reason may be present.  A missing link
    # in the chain (e.g. no collision-coalescence, no riming, no sedimentation)
    # therefore blocks confirmation regardless of the numeric confidence.
    full_chain = process_evidence >= 1.0 - 1e-9
    confirmed = (level >= 4 and confidence >= threshold
                 and full_chain and not (reasons & HARD_BLOCK))
    caveat_required = not confirmed

    return {
        "category": category,
        "diagnostic_level": level,
        "diagnostic_level_name": LEVEL_NAMES[level],
        "thermodynamic_favourability": round(float(ctx.favourability[category]), 4),
        "production_rate_kg_m3_s": float(ctx.production_rate_si[category]),
        "mixing_ratio_kg_kg": float(q),
        "number_concentration_m3": float(ctx.number[category]),
        "characteristic_radius_m": float(ctx.radius[category]),
        "terminal_velocity_m_s": float(vt),
        "surface_flux_kg_m2_s": float(flux),
        "accumulation_mm": float(ctx.accumulation[category]),
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "confirmed": bool(confirmed),
        "caveat_required": bool(caveat_required),
        "caveat": CAVEAT if caveat_required else "",
        "reason_codes": sorted(reasons),
        "supporting_evidence": supporting,
        "missing_variables": missing,
        "model_validity": {
            "T_in_range": bool(ctx.T_in_range),
            "score": round(model_validity, 3),
        },
        "numerical_quality": {
            "conservation_ok": bool(ctx.conservation_ok),
            "water_rel_err": float(ctx.water_rel_err),
            "score": round(numerical_quality, 3),
        },
        "confidence_components": {
            "data_completeness": round(data_completeness, 3),
            "model_validity": round(model_validity, 3),
            "process_evidence": round(process_evidence, 3),
            "numerical_quality": round(numerical_quality, 3),
        },
    }


__all__ = ["CAVEAT", "Reason", "HARD_BLOCK", "LEVEL_NAMES", "evaluate_category"]
