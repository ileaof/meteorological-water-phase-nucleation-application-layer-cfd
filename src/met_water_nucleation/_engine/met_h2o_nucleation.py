#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 met_h2o_nucleation.py
 Meteorological water-phase nucleation APPLICATION layer for
        H2O(vapour) -> H2O(liquid)      (condensation)
        H2O(vapour) -> H2O(hexagonal ice) (deposition)
================================================================================

This module is an APPLICATION / DIAGNOSIS layer built ON TOP of the validated
core `unified_h2o_nucleation_climate/unified_h2o_nucleation_climate.py`
(Ferreira Eq.39a/39b shifted-equilibrium nucleation, Physica B 695 (2024)
416494; MRS Meeting 2026).  It imports that core READ-ONLY and does NOT modify
it: the core closure, the 1st/2nd-order critical radius, the 2nd-order
heterogeneous parabola, the surface-stress law, the nucleation rate and the
validation suite [1]-[21] (incl. the ice-reference SHA-256 guard) all stay
exactly as validated.

What this layer ADDS (the parts the core deliberately does not own):

  * Free-energy decomposition  dG_V / dG_bulk / dG_surface / dG_config / dG_total
    at the evaluated radius, using the core's own constitutive hooks.
  * Precipitation diagnosis: a diagnostic_class plus transparent, documented
    0..1 favourability indices for rain / snow / graupel / hail, each with the
    contributing variables, the missing variables, a confidence and a short
    physical explanation.  Crucially, a high nucleation rate NEVER by itself
    implies rain or hail -- growth processes (condensation/deposition,
    collision-coalescence, accretion, riming, melting/refreezing) are NOT
    modelled here; when the dynamic/microphysical data are absent the index
    reflects thermodynamic favourability only, the confidence is LOW, and the
    caveat "Thermodynamically favourable to nucleation, but the dynamic and
    microphysical data are insufficient to confirm precipitation or hail" is
    attached.
  * xarray / NetCDF (scipy engine, NetCDF3) / GRIB ingestion adapters and
    structured xarray + JSON + CSV outputs, with graceful "undetermined"
    degradation when a field or a backend is unavailable.
  * The full mandatory 50-field report schema with a metadata block
    (units, sign conventions, correlation sources, validity ranges), an
    assumptions list, a warnings list and validity flags.
  * Visualisation (optional PNG figures).
  * A 20-test validation suite is provided separately in
    `test_met_nucleation.py`.

CONVENTIONS
  * All internal quantities are SI.
  * Heterogeneous geometry uses the core's convention
        f(theta) = 2 - 3 cos(theta) + cos^3(theta)   (un-normalised, 0..4),
        heterogeneous factor = f(theta)/4            (normalised, 0..1).
    The configurational free-energy contribution is the heterogeneous
    correction  dG_config = (f(theta)/4 - 1) * (dG_bulk + dG_surface), so that
    dG_total = (f/4)*(dG_bulk + dG_surface) and the homogeneous limit
    theta = pi  gives  f/4 = 1  ->  dG_config = 0.
  * The Gibbs-Thomson coefficient follows the core convention  GT = r_C * dT/2.
  * P_total  is the total atmospheric pressure; p_v the water-vapour partial
    pressure; P_eq the phase equilibrium (saturation) pressure.  P_eq,shift
    is the SHIFTED equilibrium pressure  P_sat,phase(T_local).

Author: generated for Prof. I. L. Ferreira (UFPa / ITEC / FEM).
================================================================================
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
#  Load the validated core READ-ONLY (importlib, same pattern as the repo's
#  test_gt_2nd_het_parabola.py loader).  The core lives at the REPO ROOT under
#  unified_h2o_nucleation_climate/; this module may sit either at the repo root
#  or inside a subfolder, so search the nearest candidate location rather than
#  hard-coding one.  The core file itself is NEVER modified (SHA-256 guarded).
# -----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "unified_h2o_nucleation_climate",
                 "unified_h2o_nucleation_climate.py"),
    os.path.join(_HERE, "..", "unified_h2o_nucleation_climate",
                 "unified_h2o_nucleation_climate.py"),
    os.path.join(_HERE, "..", "..", "unified_h2o_nucleation_climate",
                 "unified_h2o_nucleation_climate.py"),
]
_CORE = next((p for p in _CANDIDATES if os.path.isfile(p)), None)
if _CORE is None:
    raise FileNotFoundError(
        "Could not locate the validated core "
        "'unified_h2o_nucleation_climate.py' relative to " + _HERE)
_spec = importlib.util.spec_from_file_location("un_core_met", _CORE)
un = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(un)

# -----------------------------------------------------------------------------
#  Load the sibling application module `het_contact_angle` (repo root), which
#  supplies the substrate / Young / line-tension / chemistry contact-angle
#  models.  It imports the same core read-only.  The repo root is the parent of
#  the core's folder.
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_CORE)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import het_contact_angle as hca

SaturationProperties = un.SaturationProperties
UnifiedNucleationSimulator = un.UnifiedNucleationSimulator
AtmosphericInput = un.AtmosphericInput
LiquidNucleationModel = un.LiquidNucleationModel
IceNucleationModel = un.IceNucleationModel
ftheta = un.ftheta
PHASE_LIQUID = un.PHASE_LIQUID
PHASE_ICE = un.PHASE_ICE
Tt = un.Tt
Pt = un.Pt
THETA0 = un.THETA0
R_REF_DEFAULT = un.R_REF_DEFAULT
T_MIN_LOCAL = un.T_MIN_LOCAL

# epsilon for water vapour mass mixing ratio (vs dry air)
EPS_MW = 0.622        # M_H2O / M_dry_air

# A sentinel string for "could not be determined" (spec requirement).
NA = "undetermined"

# Canonical mandatory-output field order (the 48 scalar fields of the spec).
# `contact_angle_deg` is the nucleation angle theta SOLVED by the core via
# Ferreira Eq. 17 (r_C,Het/r_C,Hom self-consistency); it is an OUTPUT of the
# nucleation solution, not the --theta solver fallback.
MANDATORY_FIELDS = [
    "status", "phase", "nucleation_mode", "contact_angle_deg",
    "T_ambient_K", "T_local_K", "P_total_Pa", "p_v_Pa",
    "RH_water_percent", "RH_ice_percent", "S_water", "S_ice",
    "gradT_K_m", "DeltaT_K",
    "P_eq_classical_Pa", "P_eq_shift_Pa", "DeltaP_eq_Pa",
    "gamma_J_m2", "dgamma_dr_J_m3", "surface_stress_N_m",
    "DeltaS_bulk", "DeltaG_V_J_m3",
    "DeltaG_bulk_J", "DeltaG_surface_J", "DeltaG_config_J", "DeltaG_total_J",
    "Gamma_1st", "Gamma_2nd",
    "r_critical_1st_m", "r_critical_2nd_m",
    "DeltaG_critical_1st_J", "DeltaG_critical_2nd_J",
    "nucleation_rate_m3_s", "log10_nucleation_rate", "expected_events",
    "dominant_phase",
    "rain_favorability", "snow_favorability",
    "graupel_favorability", "hail_favorability",
    "diagnostic_class", "confidence",
    "assumptions", "warnings", "validity_flags",
    "solver_iterations", "closure_residual", "critical_radius_residual",
    "theta_model",
]

UNITS = {
    "T_ambient_K": "K", "T_local_K": "K", "P_total_Pa": "Pa", "p_v_Pa": "Pa",
    "RH_water_percent": "%", "RH_ice_percent": "%", "S_water": "1", "S_ice": "1",
    "gradT_K_m": "K/m", "DeltaT_K": "K",
    "P_eq_classical_Pa": "Pa", "P_eq_shift_Pa": "Pa", "DeltaP_eq_Pa": "Pa",
    "gamma_J_m2": "J/m^2", "dgamma_dr_J_m3": "J/m^3", "surface_stress_N_m": "N/m",
    "DeltaS_bulk": "J/(m^3.K)", "DeltaG_V_J_m3": "J/m^3",
    "DeltaG_bulk_J": "J", "DeltaG_surface_J": "J",
    "DeltaG_config_J": "J", "DeltaG_total_J": "J",
    "Gamma_1st": "K.m", "Gamma_2nd": "K.m",
    "r_critical_1st_m": "m", "r_critical_2nd_m": "m",
    "DeltaG_critical_1st_J": "J", "DeltaG_critical_2nd_J": "J",
    "nucleation_rate_m3_s": "1/(m^3.s)", "log10_nucleation_rate": "log10(1/(m^3.s))",
    "expected_events": "1",
    "rain_favorability": "0..1", "snow_favorability": "0..1",
    "graupel_favorability": "0..1", "hail_favorability": "0..1",
    "confidence": "0..1",
    "contact_angle_deg": "deg",
    "solver_iterations": "1", "closure_residual": "K/m",
    "critical_radius_residual": "J",
    "theta_model": "-",
}


# =============================================================================
#  INPUTS
# =============================================================================
@dataclass
class MetInput:
    """Meteorological + microphysical inputs (all SI).  Holds the dynamic /
    microphysical fields the core `AtmosphericInput` does not carry, plus the
    thermo fields shared with the core.  Scalars, 1-D arrays (profiles / time
    series) or callables are accepted; the runner iterates arrays elementwise.

    The dynamic/microphysical fields default to None -> "undetermined" in
    the report when absent.  No silent empirical tuning is performed.
    """
    # --- thermodynamic (shared with the core) ---
    T: Union[float, np.ndarray, Callable] = 258.15            # ambient temperature [K]
    P: Union[float, np.ndarray, Callable] = 70000.0          # total pressure [Pa]
    RH: Optional[Union[float, np.ndarray, Callable]] = None  # relative humidity [%]
    rh_reference: str = "water"                              # "water" | "ice"
    y_v: Optional[Union[float, np.ndarray, Callable]] = None # vapour mole fraction [0..1]
    p_v: Optional[Union[float, np.ndarray, Callable]] = None  # vapour partial pressure [Pa]
    q_v: Optional[Union[float, np.ndarray, Callable]] = None  # specific humidity [kg/kg]
    r_mix: Optional[Union[float, np.ndarray, Callable]] = None  # mixing ratio [kg/kg]
    grad_T: Optional[Union[float, np.ndarray, Callable]] = None  # |dT/dr| normal to interface [K/m]
    # --- continuation / heterogeneous ---
    r_ref: float = R_REF_DEFAULT                             # continuation radius [m]
    theta: float = THETA0                                    # contact angle [rad]
    mode: str = "homogeneous"                                # homogeneous | heterogeneous
    phase_mode: str = "auto"                                 # auto|liquid|ice|both
    # --- heterogeneous contact-angle model (LIQUID only; see het_contact_angle) ---
    # ferreira_eq17 (default) = core self-consistent Eq.17, no override;
    # young_constant / young_line_tension / young_chemistry = substrate-angle
    # override applied post-solve in the met layer (classical theta-independent r*,
    # barrier via f_F(theta), rate via the core Eq.21 kernel).  Ice always uses
    # the core Eq.17 theta regardless of this setting.
    theta_model: str = "ferreira_eq17"
    substrate: Optional[str] = None                          # name in hca.SUBSTRATES
    tau: Optional[float] = None                             # line tension [N]
    gamma_sv: Optional[float] = None                        # solid-vapour energy [J/m^2]
    gamma_sl: Optional[float] = None                        # solid-liquid energy [J/m^2]
    theta_explicit: bool = False                            # --theta was given (precedence)
    # --- dynamic / microphysical (NOT in the core) ---
    w: Optional[Union[float, np.ndarray, Callable]] = None  # vertical velocity [m/s]
    LWC: Optional[Union[float, np.ndarray, Callable]] = None  # liquid water content [kg/m^3]
    IWC: Optional[Union[float, np.ndarray, Callable]] = None  # ice water content [kg/m^3]
    N_ccn: Optional[Union[float, np.ndarray, Callable]] = None  # CCN number conc. [1/m^3]
    N_inp: Optional[Union[float, np.ndarray, Callable]] = None  # INP number conc. [1/m^3]
    cooling_rate: Optional[Union[float, np.ndarray, Callable]] = None  # dT/dt [K/s] (<0 cooling)
    dt_micro: Optional[float] = None                         # microphysics timestep [s]
    cell_volume: Optional[float] = None                      # grid-cell volume [m^3]
    freezing_level: Optional[float] = None                   # altitude of 0 deg C [m]
    # --- coordinates / metadata ---
    z: Optional[Union[float, np.ndarray, Callable]] = None  # geopotential altitude [m]
    lat: Optional[Union[float, np.ndarray, Callable]] = None
    lon: Optional[Union[float, np.ndarray, Callable]] = None
    time: Optional[Union[float, np.ndarray, Callable]] = None  # seconds since reference
    # --- provenance of derived humidity (filled by the builder) ---
    _humidity_source: str = "unset"

    def __post_init__(self):
        if self.phase_mode not in ("auto", "liquid", "ice", "both"):
            raise ValueError("phase_mode must be auto|liquid|ice|both")
        if self.mode not in ("homogeneous", "heterogeneous"):
            raise ValueError("mode must be homogeneous|heterogeneous")
        if self.rh_reference not in ("water", "ice"):
            raise ValueError("rh_reference must be 'water' or 'ice'")
        if self.theta_model not in hca.VALID_MODELS:
            raise ValueError(
                "theta_model must be one of %s" % (hca.VALID_MODELS,))


# -----------------------------------------------------------------------------
#  Humidity conversions (re-use the core's saturation correlations; record origin)
# -----------------------------------------------------------------------------
def _to_float(x):
    if callable(x):
        return float(x())
    if isinstance(x, np.ndarray):
        return float(np.asarray(x).reshape(-1)[0])
    return float(x)


def resolve_humidity(met: "MetInput", T: float, P: float) -> Tuple[float, str, List[str]]:
    """Resolve p_v [Pa] from whichever humidity input is given, verifying
    consistency when more than one is provided.  Returns (p_v, source, warnings).

    The humidity closure re-uses the core's SaturationProperties correlations
    (IAPWS Wagner liquid, Goff-Gratch ice).  Mixing ratio r and specific
    humidity q are inter-converted with the standard moist-air relations
    (eps = M_H2O / M_dry = 0.622):
        p_v = r * P / (r + eps)        (from r)
        r   = eps * p_v / (P - p_v)     (from p_v)
        q   = r / (1 + r)              ;  r = q / (1 - q)
    """
    warns: List[str] = []
    given = []
    if met.p_v is not None:
        given.append(("p_v", _to_float(met.p_v)))
    if met.RH is not None:
        given.append(("RH", SaturationProperties.RH_to_p_v(
            _to_float(met.RH), T, met.rh_reference)))
    if met.y_v is not None:
        given.append(("y_v", _to_float(met.y_v) * P))
    if met.r_mix is not None:
        r = _to_float(met.r_mix)
        given.append(("r_mix", r * P / (r + EPS_MW)))
    if met.q_v is not None:
        q = _to_float(met.q_v)
        r = q / (1.0 - q) if q < 1.0 else float("inf")
        given.append(("q_v", r * P / (r + EPS_MW) if math.isfinite(r) else P))

    if not given:
        raise ValueError("Provide at least one of p_v, RH, y_v, r_mix, q_v.")
    if len(given) == 1:
        return max(given[0][1], 0.0), given[0][0], warns
    # cross-check consistency (1% relative tolerance)
    base = given[0][1]
    for name, v in given[1:]:
        if base > 0 and abs(v - base) / max(base, 1e-30) > 1e-2:
            warns.append(
                f"Humidity inputs inconsistent: {given[0][0]}={base:.4e} Pa vs "
                f"{name}={v:.4e} Pa (using {given[0][0]}).")
    return max(base, 0.0), given[0][0], warns


def mixing_ratio_from_p_v(p_v: float, P: float) -> float:
    """r = eps * p_v / (P - p_v)."""
    return EPS_MW * p_v / max(P - p_v, 1e-30)


def specific_humidity_from_p_v(p_v: float, P: float) -> float:
    r = mixing_ratio_from_p_v(p_v, P)
    return r / (1.0 + r)


# =============================================================================
#  FREE-ENERGY DECOMPOSITION  (at the evaluated radius, using core hooks)
# =============================================================================
def free_energy_decomposition(model, st: dict, theta: float) -> Dict[str, float]:
    """Decompose the nucleation free energy at the evaluated radius st['r'].

        dG_V      = Delta_S_V * Delta_T                       [J/m^3]
        dG_bulk   = (4 pi/3) r^3 dG_V                         [J]
        dG_surface= 4 pi r^2 gamma(r, T_local)                [J]
        dG_config = (f(theta)/4 - 1) * (dG_bulk + dG_surface) [J]   (het correction)
        dG_total  = dG_bulk + dG_surface + dG_config
                  = (f(theta)/4) * (dG_bulk + dG_surface)

    Homogeneous limit theta = pi -> f/4 = 1 -> dG_config = 0.

    Uses the core model's own hooks (surface_energy, bulk_entropy_change) and
    the core state dict (dGv, gam) -- no re-derivation of the physics.  The
    *critical* barriers dG_C come from the validated core (r_C_1st/r_C_2nd
    parabolic stationarity), reported separately; this decomposition is a
    diagnostic at the representative continuation radius r.
    """
    r = st["r"]
    dGv = st["dGv"]                       # Delta_S_V * Delta_T  [J/m^3]
    gam = st["gam"]                       # gamma(r, T_local)    [J/m^2]
    f_norm = ftheta(theta) / 4.0
    dG_bulk = (4.0 * math.pi / 3.0) * r ** 3 * dGv
    dG_surface = 4.0 * math.pi * r ** 2 * gam
    dG_config = (f_norm - 1.0) * (dG_bulk + dG_surface)
    dG_total = dG_bulk + dG_surface + dG_config
    return {
        "DeltaG_V_J_m3": dGv,
        "DeltaS_bulk": st["dsv"],
        "DeltaG_bulk_J": dG_bulk,
        "DeltaG_surface_J": dG_surface,
        "DeltaG_config_J": dG_config,
        "DeltaG_total_J": dG_total,
        "f_theta": ftheta(theta),
        "f_theta_normalised": f_norm,
    }


# =============================================================================
#  PRECIPITATION DIAGNOSIS
# =============================================================================
def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x, x0=6.0, width=1.5):
    """Smooth 0..1 nucleation-tendency mapping from log10(I) [log10 1/(m^3 s)].
    Threshold log10I = 6 (I ~ 1e6 /m^3/s) taken as 'active'; width 1.5 decades.
    Documented, no hidden tuning -- this is a transparent diagnostic mapping,
    not a physical rate law."""
    if not math.isfinite(x):
        return 0.0
    return 1.0 / (1.0 + math.exp(-(x - x0) / width))


@dataclass
class Favorability:
    value: float                  # 0..1
    contributing_vars: List[str]
    missing_vars: List[str]
    confidence: float             # 0..1
    explanation: str
    caveat: str = ""


class PrecipitationDiagnosis:
    """Combines nucleation results with available dynamic/microphysical
    information into transparent 0..1 favourability indices and a diagnostic
    class.  HONESTY GUARD: a high nucleation rate never by itself implies rain
    or hail; hydrometeor growth is not modelled.  When dynamic/microphysical
    data are absent, the index is thermodynamic-only, confidence is low, and
    the standard caveat is attached.
    """

    CAVEAT = ("Thermodynamically favourable to nucleation, but the dynamic and "
              "microphysical data are insufficient to confirm precipitation or hail.")

    def __init__(self, T_ambient, S_w, S_i, log10I, phase,
                 w=None, LWC=None, IWC=None, cooling_rate=None,
                 freezing_level=None, N_ccn=None, N_inp=None, z=None):
        self.T = T_ambient
        self.S_w = S_w
        self.S_i = S_i
        self.log10I = log10I if math.isfinite(log10I) else float("-inf")
        self.phase = phase
        self.w = w
        self.LWC = LWC
        self.IWC = IWC
        self.cool = cooling_rate
        self.fz = freezing_level
        self.N_ccn = N_ccn
        self.N_inp = N_inp
        self.z = z

    # -- elementary normalised factors --
    def _f_sup_w(self): return _clip01((self.S_w - 1.0) / 0.20)      # sat at 20% SS
    def _f_sup_i(self): return _clip01((self.S_i - 1.0) / 0.20)
    def _f_nuc(self):   return _sigmoid(self.log10I)
    def _f_cold(self):  return _clip01((273.15 - self.T) / 40.0)     # 0 at 0C, 1 at -40C
    def _f_warm(self):  return _clip01((self.T - 273.15) / 20.0)     # 0 at 0C, 1 at 20C
    def _f_updraft(self): return _clip01(self.w / 5.0) if self.w is not None else None
    def _f_hail_updraft(self): return _clip01((self.w - 5.0) / 15.0) if self.w is not None else None
    def _f_lwc(self): return _clip01(self.LWC / 1.0e-3) if self.LWC is not None else None  # 1 g/m3
    def _f_iwc(self): return _clip01(self.IWC / 1.0e-3) if self.IWC is not None else None
    def _f_cool(self): return _clip01(abs(self.cool) / 5.0e-4) if self.cool is not None else None  # ~1.8 K/h
    def _f_ccn(self): return _clip01(self.N_ccn / 1.0e9) if self.N_ccn is not None else None  # 1000 /cm3
    def _f_inp(self): return _clip01(self.N_inp / 1.0e6) if self.N_inp is not None else None  # 1 /cm3

    def _combine(self, factors):
        """Weighted mean over PRESENT factors (equal weights).  Returns
        (value, contributing, missing).  Absent factors do not penalise the
        value (renormalised) but lower the confidence downstream."""
        present = [(n, v) for n, v in factors if v is not None]
        missing = [n for n, v in factors if v is None]
        if not present:
            return 0.0, [], [n for n, _ in factors]
        val = sum(v for _, v in present) / len(present)
        return _clip01(val), [n for n, _ in present], missing

    def _confidence(self, contributing, missing, ideal):
        """Confidence = fraction of the ideal (physics-relevant) factors that
        are present, scaled to 0..1.  Low when < 0.5 of the ideal factors
        available."""
        have = sum(1 for c in contributing if c in ideal)
        return have / max(len(ideal), 1)

    # -- the four indices --
    def rain(self) -> Favorability:
        # Rain needs supersaturation wrt water + nucleation tendency (always),
        # plus warm-cloud growth dynamics (updraft, LWC, CCN) when present,
        # and a cold-rain (Bergeron + melt) term when BOTH ice and a freezing
        # level are available.  Dynamics are blended in whenever present,
        # regardless of T -- at sub-freezing they still gauge growth potential.
        f_supw = self._f_sup_w()
        f_nuc = self._f_nuc()
        f_w = self._f_updraft()
        f_lwc = self._f_lwc()
        f_ccn = self._f_ccn()
        f_iwc = self._f_iwc()
        factors = [("thermo_supw", f_supw), ("thermo_nuc", f_nuc),
                   ("updraft", f_w), ("LWC", f_lwc), ("CCN", f_ccn)]
        ideal = ["thermo_supw", "thermo_nuc", "updraft", "LWC", "CCN"]
        # cold-rain (ice aloft melting below the freezing level): only when
        # BOTH IWC and a freezing level are available -- otherwise this term
        # is "undetermined" (not silently zero).
        if f_iwc is not None and self.fz is not None:
            factors.append(("cold_rain_melt", 1.0))
            ideal.append("cold_rain_melt")
        val, contr, miss = self._combine(factors)
        if self.T > 273.15:
            expl = "Warm rain: warm cloud (T>0C) with supersaturation wrt water, updraft, LWC, CCN."
        elif f_iwc is not None and self.fz is not None:
            expl = "Cold rain: ice formed aloft melting below the freezing level (Bergeron/cold-rain)."
        else:
            expl = ("Rain favourability: supersaturation wrt water + nucleation tendency "
                    "blended with available warm-cloud dynamics (cold-rain melt term "
                    "needs IWC + freezing level).")
        conf = self._confidence(contr, miss, ideal)
        caveat = self.CAVEAT if conf < 0.5 else ""
        return Favorability(val, contr, miss, conf, expl, caveat)

    def snow(self) -> Favorability:
        f_supi = self._f_sup_i()
        f_cold = self._f_cold()
        f_nuc = self._f_nuc()
        f_iwc = self._f_iwc()
        f_inp = self._f_inp()
        factors = [("sup_ice", f_supi), ("cold", f_cold), ("nuc", f_nuc),
                   ("IWC", f_iwc), ("INP", f_inp)]
        val, contr, miss = self._combine(factors)
        ideal = ["sup_ice", "cold", "nuc", "IWC", "INP"]
        conf = self._confidence(contr, miss, ideal)
        expl = "Snow: sub-freezing, supersaturation wrt ice, vapour deposition / INP activity, IWC growth."
        caveat = self.CAVEAT if conf < 0.5 else ""
        return Favorability(val, contr, miss, conf, expl, caveat)

    def graupel(self) -> Favorability:
        f_cold = self._f_cold()
        f_supi = self._f_sup_i()
        f_nuc = self._f_nuc()
        f_lwc = self._f_lwc()
        f_iwc = self._f_iwc()
        f_w = self._f_updraft()
        factors = [("cold", f_cold), ("sup_ice", f_supi), ("nuc", f_nuc),
                   ("LWC", f_lwc), ("IWC", f_iwc), ("updraft", f_w)]
        val, contr, miss = self._combine(factors)
        ideal = ["cold", "sup_ice", "nuc", "LWC", "IWC", "updraft"]
        conf = self._confidence(contr, miss, ideal)
        expl = "Graupel: supercooled LWC + ice (riming) in sub-freezing cloud with moderate updraft."
        caveat = self.CAVEAT if conf < 0.5 else ""
        return Favorability(val, contr, miss, conf, expl, caveat)

    def hail(self) -> Favorability:
        # Hail has the HIGHEST data bar: needs strong updraft (w>~10 m/s),
        # supercooled LWC, large supercooled depth, and a melting layer below.
        f_cold = self._f_cold()
        f_nuc = self._f_nuc()
        f_lwc = self._f_lwc()
        f_w = self._f_hail_updraft()
        f_iwc = self._f_iwc()
        factors = [("cold", f_cold), ("nuc", f_nuc), ("LWC", f_lwc),
                   ("hail_updraft", f_w), ("IWC", f_iwc)]
        val, contr, miss = self._combine(factors)
        ideal = ["cold", "nuc", "LWC", "hail_updraft", "supercooled_depth", "melt_below"]
        conf = self._confidence(contr, miss, ideal)
        expl = ("Hail: requires strong updraft (w>~10 m/s), large supercooled LWC, deep "
                "supercooled region and a warm lower layer -- the highest data bar.")
        # hail almost always lacks the full data here -> low confidence + caveat
        caveat = self.CAVEAT if conf < 0.75 else ""
        return Favorability(val, contr, miss, conf, expl, caveat)

    # -- diagnostic class --
    def diagnostic_class(self) -> str:
        T = self.T
        if not (math.isfinite(self.S_w) and math.isfinite(self.S_i)):
            return "insufficient_data"
        sub_w = self.S_w < 1.0
        sub_i = self.S_i < 1.0
        sat_w = abs(self.S_w - 1.0) < 0.02
        sat_i = abs(self.S_i - 1.0) < 0.02
        sup_w = self.S_w > 1.02
        sup_i = self.S_i > 1.02
        if sub_w and sub_i:
            return "subsaturated"
        if sat_w and sat_i:
            return "saturated_water" if T > 273.15 else "saturated_ice"
        if T > 273.15:
            if sup_w:
                return "warm_rain" if sup_w else "condensation_favorable"
            return "condensation_favorable"
        # sub-freezing
        if sup_i and sup_w:
            return "mixed_phase"
        if sup_w and not sup_i:
            return "supercooled_liquid"
        if sup_i and not sup_w:
            return "deposition_favorable"
        # below 0C but near-saturated
        if T < 273.15 and (sat_w or sat_i):
            return "supercooled_liquid" if sat_w else "deposition_favorable"
        return "insufficient_data"


# =============================================================================
#  REPORT
# =============================================================================
@dataclass
class MetNucleationReport:
    """Full mandatory-output record for one phase at one ambient point."""
    # identity
    status: str
    phase: str
    nucleation_mode: str
    # nucleation angle theta SOLVED by the core via Ferreira Eq. 17
    # (r_C,Het / r_C,Hom self-consistency); degrees.  == 180 for the
    # homogeneous / no-substrate limit.  --theta / THETA0 is only the
    # brentq fallback, not this value.
    contact_angle_deg: float
    # atmosphere
    T_ambient_K: float
    T_local_K: float
    P_total_Pa: float
    p_v_Pa: float
    RH_water_percent: float
    RH_ice_percent: float
    S_water: float
    S_ice: float
    # thermal field
    gradT_K_m: float
    DeltaT_K: float
    # equilibrium pressures
    P_eq_classical_Pa: float
    P_eq_shift_Pa: float
    DeltaP_eq_Pa: float
    # surface
    gamma_J_m2: float
    dgamma_dr_J_m3: float
    surface_stress_N_m: float
    # bulk / free-energy
    DeltaS_bulk: float
    DeltaG_V_J_m3: float
    DeltaG_bulk_J: float
    DeltaG_surface_J: float
    DeltaG_config_J: float
    DeltaG_total_J: float
    # Gibbs-Thomson / critical radii / barriers
    Gamma_1st: float
    Gamma_2nd: float
    r_critical_1st_m: float
    r_critical_2nd_m: float
    DeltaG_critical_1st_J: float
    DeltaG_critical_2nd_J: float
    # rate
    nucleation_rate_m3_s: float
    log10_nucleation_rate: float
    expected_events: Optional[float]
    # competition / diagnosis
    dominant_phase: str
    rain_favorability: float
    snow_favorability: float
    graupel_favorability: float
    hail_favorability: float
    diagnostic_class: str
    confidence: float
    # provenance / quality
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validity_flags: List[str] = field(default_factory=list)
    solver_iterations: Optional[int] = None
    closure_residual: float = float("nan")
    critical_radius_residual: float = float("nan")
    # contact-angle model provenance (49th mandatory field).
    # ferreira_eq17 = core self-consistent Eq.17 (default, no override);
    # young_constant / young_line_tension / young_chemistry = substrate-angle
    # override (liquid only; rate recomputed via the core Eq.21 kernel).
    theta_model: str = "ferreira_eq17"
    # richer diagnosis detail (not in the 47-field list but useful)
    favorability_detail: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


# -----------------------------------------------------------------------------
#  small helpers for the substrate-angle override provenance
# -----------------------------------------------------------------------------
def _fmt_override_info(info: Optional[Dict[str, Any]]) -> str:
    """Compact one-line summary of the substrate-angle solve for the assumptions list."""
    if not info:
        return ""
    parts = []
    for k in ("substrate", "theta_Y_deg", "tau_N", "gamma_LV_J_m2",
              "r_used_m", "delta_theta_deg", "g_SV", "g_SL"):
        if k in info and info[k] is not None:
            parts.append(f"{k}={info[k]}")
    return "(" + ", ".join(parts) + ")" if parts else ""


def _merge_override(detail: Dict[str, Any],
                    info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Attach the substrate-angle override provenance to the favourability detail."""
    out = dict(detail)
    if info:
        out["theta_override"] = info
    return out


# =============================================================================
#  RUNNER
# =============================================================================
class MetNucleationRunner:
    """Builds a core `AtmosphericInput` from a `MetInput`, evaluates the
    validated core, then augments with the free-energy decomposition, the
    expected event count and the precipitation diagnosis to produce a
    `MetNucleationReport` per phase.
    """

    def __init__(self, met: MetInput):
        self.met = met

    # -- core AtmosphericInput builder (no modification of the core class) --
    def _atm(self, T, P, p_v, grad_T=None) -> "AtmosphericInput":
        m = self.met
        return AtmosphericInput(
            T=T, P=P, p_v=p_v, grad_T=grad_T,
            theta=m.theta, mode=m.mode, phase_mode=m.phase_mode,
            scenario="single_state",
        )

    def _solver_iterations(self, model, r, T, P, p_v) -> Optional[int]:
        """Re-run the SAME bracket + brentq the core uses, but with
        `full_output=True` to capture iteration count.  Deterministic, so the
        captured count corresponds to the core's own solve.  Guarded."""
        try:
            br = model._find_bracket(r, T, p_v)
            if br is None:
                return None
            lo, hi = br
            _g, r_obj = brentq_local(model._residual, lo, hi,
                                     args=(r, T, p_v),
                                     xtol=model.cfg.xtol, rtol=model.cfg.rtol,
                                     maxiter=model.cfg.maxiter)
            return getattr(r_obj, "iterations", None)
        except Exception:
            return None

    def _atm_value(self, x, idx=None):
        if x is None:
            return None
        if callable(x):
            return float(x() if idx is None else x(idx))
        if isinstance(x, np.ndarray):
            arr = np.asarray(x).reshape(-1)
            return float(arr[idx if idx is not None else 0])
        return float(x)

    def evaluate_point(self, T: float, P: float, p_v: float,
                       grad_T: Optional[float] = None,
                       dynamics: Optional[Dict[str, float]] = None,
                       ) -> Dict[str, MetNucleationReport]:
        """Evaluate one ambient point.  Returns {phase: MetNucleationReport}.
        `dynamics` carries the per-point dynamic/microphysical scalars
        (w, LWC, IWC, cooling_rate, freezing_level, N_ccn, N_inp, z)."""
        m = self.met
        atm = self._atm(T, P, p_v, grad_T)
        sim = UnifiedNucleationSimulator(atm)
        nr = sim.evaluate_point(T, P, p_v, r_ref=m.r_ref,
                                grad_T_req=grad_T if grad_T else None)
        dyn = dynamics or {}

        out: Dict[str, MetNucleationReport] = {}
        for ph, res in nr.items():
            out[ph] = self._build_report(ph, res, sim, T, P, p_v, grad_T, dyn)
        return out

    def _build_report(self, ph, res, sim, T, P, p_v, grad_T, dyn) -> MetNucleationReport:
        m = self.met
        model = sim.liquid if ph == PHASE_LIQUID else sim.ice
        assumptions: List[str] = []
        warnings: List[str] = []
        flags: List[str] = []

        # ambient humidity
        RH_w = 100.0 * p_v / SaturationProperties.Psat_water(T, extended=True)
        RH_i = 100.0 * p_v / SaturationProperties.Psat_ice(T)
        S_w = p_v / SaturationProperties.Psat_water(T, extended=True)
        S_i = p_v / SaturationProperties.Psat_ice(T)

        if res.status != "ok":
            # subsaturated / no_solution: fill what we can, NaN the rest
            d = self._diagnosis(ph, T, S_w, S_i, float("-inf"), dyn)
            rpt = MetNucleationReport(
                status=res.status, phase=ph, nucleation_mode=m.mode,
                contact_angle_deg=float("nan"),
                T_ambient_K=T, T_local_K=float("nan"), P_total_Pa=P, p_v_Pa=p_v,
                RH_water_percent=RH_w, RH_ice_percent=RH_i, S_water=S_w, S_ice=S_i,
                gradT_K_m=float("nan"), DeltaT_K=float("nan"),
                P_eq_classical_Pa=model.Psat(T),
                P_eq_shift_Pa=float("nan"), DeltaP_eq_Pa=float("nan"),
                gamma_J_m2=float("nan"), dgamma_dr_J_m3=float("nan"),
                surface_stress_N_m=float("nan"),
                DeltaS_bulk=float("nan"), DeltaG_V_J_m3=float("nan"),
                DeltaG_bulk_J=float("nan"), DeltaG_surface_J=float("nan"),
                DeltaG_config_J=float("nan"), DeltaG_total_J=float("nan"),
                Gamma_1st=float("nan"), Gamma_2nd=float("nan"),
                r_critical_1st_m=float("nan"), r_critical_2nd_m=float("nan"),
                DeltaG_critical_1st_J=float("nan"), DeltaG_critical_2nd_J=float("nan"),
                nucleation_rate_m3_s=float("nan"),
                log10_nucleation_rate=float("nan"),
                expected_events=None,
                dominant_phase=res.dominant or "none",
                rain_favorability=d["rain"].value, snow_favorability=d["snow"].value,
                graupel_favorability=d["graupel"].value, hail_favorability=d["hail"].value,
                diagnostic_class=d["class"], confidence=d["confidence_mean"],
                assumptions=assumptions, warnings=warnings, validity_flags=flags,
                solver_iterations=None, closure_residual=float("nan"),
                critical_radius_residual=float("nan"),
                theta_model=m.theta_model,
                favorability_detail=d["detail"],
                metadata=self._metadata(),
            )
            if res.status == "subsaturated":
                flags.append("subsaturated")
                warnings.append("Subsaturated: no nucleation closure solved (S<1).")
            else:
                flags.append("no_solution")
                warnings.append("No physical closure solution at the requested radius.")
            rpt.validity_flags = flags
            return rpt

        # ---- converged: pull from the core result + its state dict ----
        # recover the state dict the core solved (re-solve deterministically
        # to access internal quantities + iteration count; identical inputs).
        r_eval = m.r_ref
        st = model.solve(r_eval, T, P, p_v, theta_default=m.theta)
        if st is None:  # defensive: should not happen since res.status == ok
            warnings.append("Internal: core re-solve returned None; using result fields only.")
            st = {}
        iters = self._solver_iterations(model, r_eval, T, P, p_v)

        # ---- report-locals (default to the core result; overridden below for a
        #      non-default LIQUID contact-angle model) ----
        theta_report = res.theta
        I_report = res.I
        log10I_report = res.log10I
        dGc_crit_report = res.DeltaG_2nd            # homogeneous 2nd-order barrier
        parab_resid_report = res.parabolic_resid
        override_info = None

        # ---- substrate-angle override (LIQUID only; ice keeps the core Eq.17) ----
        # For a non-default theta_model the liquid contact angle is set by the
        # substrate (Young / line-tension / chemistry), NOT by the core's
        # self-consistent Eq.17.  Classical theta-independent r* is retained
        # (st["rC_2nd"] is the homogeneous critical radius); the barrier is
        # f_F(theta) * dGc_hom and the rate is recomputed through the core's OWN
        # validated `_rate` kernel (read-only call, no core edit).  The exact core
        # relation  dGc_het(theta) = -ftheta(theta) * dGc_hom  is used, so feeding
        # the core's solved theta back in reproduces res.I to machine precision.
        if ph == PHASE_LIQUID and m.theta_model != "ferreira_eq17" and st:
            if m.mode != "heterogeneous":
                warnings.append(
                    "theta_model=%s implies a substrate (heterogeneous); with "
                    "--mode homogeneous the override is inactive and the core "
                    "Eq.17 homogeneous angle/rate are retained."
                    % m.theta_model)
            else:
                # NOTE: the angle and the rate are sourced from the CORE RESULT
                # `res`, NOT from the met re-solve `st`.  When a gradient is
                # requested the core scans a radius grid and INTERPOLATES to
                # grad_T_req (core evaluate_point), so `res` lives at the requested
                # gradient while `st` (a single solve at r_ref) does not -- using
                # `st` here would recompute the rate at the wrong gradient.
                try:
                    gLV = hca.gamma_LV(model, res.rC_2nd, res.T_eq_shift)
                    thY, gsv, gsl, _src = hca.resolve_substrate(
                        m.substrate,
                        theta_Y_deg=(math.degrees(m.theta) if m.theta_explicit else None),
                        g_SV=m.gamma_sv, g_SL=m.gamma_sl)
                    theta_sub, fF_sub, oinfo = hca.solve_theta(
                        m.theta_model, r=res.rC_2nd, T=res.T_eq_shift,
                        gamma_LV=gLV, theta_Y_rad=thY, tau=(m.tau or 0.0),
                        g_SV=gsv, g_SL=gsl, substrate=m.substrate)
                except Exception as exc:  # pragma: no cover - defensive
                    theta_sub, fF_sub, oinfo = None, None, {"error": str(exc)}
                    warnings.append(f"Substrate-angle model {m.theta_model} failed "
                                    f"({exc}); core theta retained.")

                if theta_sub is not None and math.isfinite(theta_sub):
                    theta_report = theta_sub
                    # recompute the het barrier + rate from the core's OWN result
                    # state (res.rC_2nd, res.T_eq_shift, res.Delta_T, res.Gamma2),
                    # using the core's OWN _dGc_* functions (read-only call) so
                    # the result is identical in form to what produced res.I.
                    r_c = res.rC_2nd
                    rec_ok = (math.isfinite(r_c) and r_c > 0.0
                              and math.isfinite(res.T_eq_shift)
                              and math.isfinite(res.Delta_T)
                              and math.isfinite(res.Gamma2)
                              and abs(ftheta(res.theta)) > 1e-300)
                    if rec_ok:
                        g_c = res.Gamma2 / (4.0 * math.pi * r_c * r_c)
                        try:
                            dGc_het_core = model._dGc_het_2nd(
                                r_c, res.T_eq_shift, res.Delta_T, g_c, res.theta)
                            dGc_hom = model._dGc_hom_2nd(
                                r_c, res.T_eq_shift, res.Delta_T, g_c)
                            if not (math.isfinite(dGc_het_core)
                                    and math.isfinite(dGc_hom)):
                                # core took the classical-expr fallback (solve
                                # lines 1152-1156); use the matching expr form.
                                dGc_hom = model._dGc_hom_expr(
                                    r_c, res.T_eq_shift, res.Delta_T)
                                dGc_het_core = model._dGc_het_expr(
                                    r_c, res.T_eq_shift, res.Delta_T, res.theta)
                        except Exception as exc:  # pragma: no cover
                            dGc_het_core = float("nan"); dGc_hom = float("nan")
                            warnings.append(f"Barrier reconstruction failed "
                                            f"({exc}); core rate retained.")
                        if (math.isfinite(dGc_het_core) and math.isfinite(dGc_hom)
                                and abs(ftheta(res.theta)) > 1e-300):
                            # Both barrier forms are LINEAR in ftheta(theta), so the
                            # substrate barrier scales the core het barrier by the
                            # ftheta ratio.  At theta_sub == res.theta the ratio is 1
                            # and res.I is reproduced to machine precision
                            # (the faithfulness invariant).
                            dGc_het_sub = (dGc_het_core * ftheta(theta_sub)
                                           / ftheta(res.theta))
                            dGc_eq = abs(dGc_hom)
                            try:
                                I_sub, log10I_sub = model._rate(
                                    r_c, res.T_eq_shift, P,
                                    dGc_het_sub, dGc_eq, theta=theta_sub, het=True)
                                I_report, log10I_report = I_sub, log10I_sub
                                dGc_crit_report = abs(dGc_het_sub)
                            except Exception as exc:  # pragma: no cover
                                warnings.append(f"Rate recompute failed ({exc}); "
                                                f"core rate retained.")
                    # the het parabola residual is not meaningful under the
                    # substrate model (the core's self-consistent Eq.17 parabola
                    # is bypassed)
                    parab_resid_report = float("nan")
                    override_info = oinfo
                    assumptions.append(
                        "theta from %s: substrate=%s theta=%.4g deg f_F=%.4g %s"
                        % (m.theta_model, m.substrate, math.degrees(theta_sub),
                           fF_sub if fF_sub is not None else float("nan"),
                           _fmt_override_info(oinfo)))
                    warnings.append(
                        "Substrate-angle override active (liquid): r* "
                        "theta-independent (classical); barrier via f_F(theta); "
                        "rate via core Eq.21 kernel with cap "
                        "A=2*pi*r^2*(1-cos theta).")
        elif ph != PHASE_LIQUID and m.theta_model != "ferreira_eq17":
            warnings.append(
                "theta_model=%s applies to LIQUID only; ice retains the core "
                "Eq.17 contact angle." % m.theta_model)

        # free-energy decomposition at the evaluated radius (uses the reported
        # theta, which may be the overridden substrate angle)
        fe = free_energy_decomposition(model, st, theta_report) if st else {
            "DeltaG_V_J_m3": res.DeltaG_V, "DeltaS_bulk": res.DeltaS_V,
            "DeltaG_bulk_J": float("nan"), "DeltaG_surface_J": float("nan"),
            "DeltaG_config_J": float("nan"), "DeltaG_total_J": float("nan"),
            "f_theta": ftheta(theta_report), "f_theta_normalised": ftheta(theta_report) / 4.0}

        # equilibrium pressures
        P_eq_classical = model.Psat(T)
        P_eq_shift = res.P_eq_shift
        DeltaP_eq = P_eq_classical - P_eq_shift

        # expected events (use the possibly-overridden rate)
        if (m.dt_micro is not None and m.cell_volume is not None
                and math.isfinite(I_report)):
            expected_events = I_report * m.dt_micro * m.cell_volume
        else:
            expected_events = None
            if m.dt_micro is None or m.cell_volume is None:
                warnings.append(
                    "expected_events = undetermined: microphysics timestep "
                    "and/or cell volume not provided.")

        # precipitation diagnosis (use the possibly-overridden log10I)
        d = self._diagnosis(ph, T, S_w, S_i, log10I_report, dyn)

        # validity flags / stable-metastable-extrapolated
        if res.in_valid_range:
            flags.append("in_valid_range")
        else:
            flags.append("out_of_range")
        if T < 273.15 and S_w >= 1.0:
            flags.append("supercooled_liquid_meta")
        if math.isfinite(res.T_eq_shift) and res.T_eq_shift < T_MIN_LOCAL + 1e-9:
            flags.append("T_local_near_lower_bound_extrapolated")
            warnings.append("T_local near the deep-supercooling lower bound (extrapolated).")
        if T > Tt + 5.0 and ph == PHASE_LIQUID:
            flags.append("above_triple_point_liquid_stable")
        if not res.in_valid_range:
            assumptions.append("Result reported though outside the constitutive validity range.")

        assumptions.append("Gibbs-Thomson coefficient follows GT = r_C * dT/2.")
        assumptions.append("Heterogeneous geometry: f(theta)=2-3cos+cos^3; factor=f/4.")

        rpt = MetNucleationReport(
            status=res.status, phase=ph, nucleation_mode=m.mode,
            contact_angle_deg=math.degrees(theta_report) if math.isfinite(theta_report) else float("nan"),
            T_ambient_K=T, T_local_K=res.T_eq_shift, P_total_Pa=P, p_v_Pa=p_v,
            RH_water_percent=RH_w, RH_ice_percent=RH_i, S_water=S_w, S_ice=S_i,
            gradT_K_m=res.grad_T, DeltaT_K=res.Delta_T,
            P_eq_classical_Pa=P_eq_classical, P_eq_shift_Pa=P_eq_shift,
            DeltaP_eq_Pa=DeltaP_eq,
            gamma_J_m2=res.gamma_r, dgamma_dr_J_m3=res.dgamma_dr,
            surface_stress_N_m=res.surface_stress,
            DeltaS_bulk=fe["DeltaS_bulk"], DeltaG_V_J_m3=fe["DeltaG_V_J_m3"],
            DeltaG_bulk_J=fe["DeltaG_bulk_J"], DeltaG_surface_J=fe["DeltaG_surface_J"],
            DeltaG_config_J=fe["DeltaG_config_J"], DeltaG_total_J=fe["DeltaG_total_J"],
            Gamma_1st=res.Gamma1, Gamma_2nd=res.Gamma2,
            r_critical_1st_m=res.rC_1st, r_critical_2nd_m=res.rC_2nd,
            DeltaG_critical_1st_J=res.DeltaG_1st, DeltaG_critical_2nd_J=dGc_crit_report,
            nucleation_rate_m3_s=I_report, log10_nucleation_rate=log10I_report,
            expected_events=expected_events,
            dominant_phase=res.dominant or ph,
            rain_favorability=d["rain"].value, snow_favorability=d["snow"].value,
            graupel_favorability=d["graupel"].value, hail_favorability=d["hail"].value,
            diagnostic_class=d["class"], confidence=d["confidence_mean"],
            assumptions=assumptions, warnings=warnings, validity_flags=flags,
            solver_iterations=iters,
            closure_residual=res.closure_resid,
            critical_radius_residual=parab_resid_report,
            theta_model=m.theta_model,
            favorability_detail=_merge_override(d["detail"], override_info),
            metadata=self._metadata(),
        )
        return rpt

    # -- diagnosis helper --
    def _diagnosis(self, ph, T, S_w, S_i, log10I, dyn) -> Dict[str, Any]:
        pdiag = PrecipitationDiagnosis(
            T, S_w, S_i, log10I, ph,
            w=dyn.get("w"), LWC=dyn.get("LWC"), IWC=dyn.get("IWC"),
            cooling_rate=dyn.get("cooling_rate"),
            freezing_level=dyn.get("freezing_level"),
            N_ccn=dyn.get("N_ccn"), N_inp=dyn.get("N_inp"), z=dyn.get("z"))
        rain = pdiag.rain()
        snow = pdiag.snow()
        graupel = pdiag.graupel()
        hail = pdiag.hail()
        confs = [rain.confidence, snow.confidence, graupel.confidence, hail.confidence]
        # overall confidence = mean of the per-index confidences for the present phase
        conf_mean = sum(confs) / len(confs)
        detail = {
            "rain": _fav_dict(rain), "snow": _fav_dict(snow),
            "graupel": _fav_dict(graupel), "hail": _fav_dict(hail),
        }
        return {"rain": rain, "snow": snow, "graupel": graupel, "hail": hail,
                "class": pdiag.diagnostic_class(),
                "confidence_mean": conf_mean, "detail": detail}

    def _metadata(self) -> Dict[str, Any]:
        return {
            "units": UNITS,
            "sign_conventions": {
                "DeltaS_bulk": "volumetric entropy change (liquid/ice minus vapour), <0",
                "DeltaG_V_J_m3": "Delta_S_V * Delta_T, <0 drives nucleation",
                "DeltaP_eq_Pa": "P_eq_classical - P_eq_shift (positive under cooling)",
                "dgamma_dr_J_m3": "radial derivative of surface energy at r",
                "cooling_rate": "dT/dt, <0 means cooling",
            },
            "sources": {
                "Psat_water": "IAPWS Wagner saturation, extended below triple point",
                "Psat_ice": "Goff-Gratch sublimation, anchored at the triple point",
                "surface_liquid": "Tolman curvature  gamma(r)=gamma_inf/(1+2 dTol/r)",
                "surface_ice": "Shuttleworth/Gurtin-Murdoch  tau=gamma+r dgamma/dr",
                "nucleation_rate": "Ferreira shifted-equilibrium rate (D A N_v / lambda^4 exp(dGc/dGc_eq))",
                "framework": "Ferreira, Physica B 695 (2024) 416494; MRS Meeting 2026",
            },
            "validity_ranges": {
                "T_ambient": "233..373 K (ice 233..273; liquid 233..647)",
                "gradT": "1..1e4 K/m validated; beyond is extrapolation",
                "r_continuation": "1e-9..1e-2 m",
            },
            "f_theta_convention": "f=2-3cos+cos^3 (0..4); normalised factor f/4 (0..1)",
            "GT_convention": "Gibbs-Thomson coefficient = r_C * dT/2",
            "note": "Hydrometeor growth (condensation/deposition, collision-coalescence, "
                    "accretion, riming, melting/refreezing) is NOT modelled; "
                    "favourability indices are diagnostic only.",
        }

    # -- array drivers --
    def evaluate_profile(self, T_arr, P_arr, p_v_arr, z_arr,
                         dyn_arrs: Optional[Dict[str, np.ndarray]] = None,
                         ) -> List[Dict[str, MetNucleationReport]]:
        """Vertical profile: elementwise over the arrays."""
        n = len(T_arr)
        results = []
        for i in range(n):
            dyn = {k: (v[i] if v is not None and i < len(v) else None)
                   for k, v in (dyn_arrs or {}).items()}
            results.append(self.evaluate_point(
                float(T_arr[i]), float(P_arr[i]), float(p_v_arr[i]),
                dynamics=dyn))
        return results

    def evaluate_series(self, T_arr, P_arr, p_v_arr, t_arr,
                        dyn_arrs: Optional[Dict[str, np.ndarray]] = None,
                        ) -> List[Dict[str, MetNucleationReport]]:
        n = len(T_arr)
        results = []
        for i in range(n):
            dyn = {k: (v[i] if v is not None and i < len(v) else None)
                   for k, v in (dyn_arrs or {}).items()}
            results.append(self.evaluate_point(
                float(T_arr[i]), float(P_arr[i]), float(p_v_arr[i]),
                dynamics=dyn))
        return results


# -- brentq with full_output (local import to avoid touching the core) --
def brentq_local(func, a, b, args=(), xtol=2e-16, rtol=8.88e-16, maxiter=500):
    from scipy.optimize import brentq
    return brentq(func, a, b, args=args, xtol=xtol, rtol=rtol,
                  maxiter=maxiter, full_output=True)


def _fav_dict(f: Favorability) -> Dict[str, Any]:
    return {"value": f.value, "contributing_vars": f.contributing_vars,
            "missing_vars": f.missing_vars, "confidence": f.confidence,
            "explanation": f.explanation, "caveat": f.caveat}


# =============================================================================
#  I/O ADAPTERS  (xarray / NetCDF / GRIB)
# =============================================================================
# Name-tolerant field mapping: canonical name -> list of accepted input names.
FIELD_ALIASES = {
    "T": ["T", "tair", "air_temperature", "temp", "temperature", "T_ambient"],
    "P": ["P", "ps", "surface_pressure", "pressure", "air_pressure", "sp"],
    "RH": ["RH", "relative_humidity", "hur", "r"],
    "p_v": ["p_v", "vapour_pressure", "vapor_pressure", "e"],
    "q_v": ["q_v", "q", "specific_humidity", "hus", "humidity_mixing_ratio"],
    "r_mix": ["r_mix", "mr", "mixing_ratio"],
    "grad_T": ["grad_T", "gradT", "thermal_gradient", "dTdz"],
    "w": ["w", "vertical_velocity", "updraft", "va"],
    "LWC": ["LWC", "lwc", "liquid_water_content", "clw", "qc"],
    "IWC": ["IWC", "iwc", "ice_water_content", "cli", "qi"],
    "N_ccn": ["N_ccn", "ccn", "CCN"],
    "N_inp": ["N_inp", "inp", "INP"],
    "cooling_rate": ["cooling_rate", "dTdt", "dtdt"],
    "z": ["z", "altitude", "height", "h", "geopotential_height", "gph"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
    "time": ["time", "t"],
}


def _xr_var(ds, canonical):
    """Return the xarray variable matching a canonical name, or None."""
    import xarray as xr  # local
    for name in FIELD_ALIASES.get(canonical, [canonical]):
        if name in ds.variables:
            return ds[name]
        if name in ds.coords:
            return ds[name]
    return None


def from_xarray(ds) -> MetInput:
    """Build a MetInput from an xarray.Dataset, mapping common variable names.
    Missing variables -> None (reported as 'undetermined' downstream).
    """
    import xarray as xr  # noqa
    def get(name):
        v = _xr_var(ds, name)
        return None if v is None else np.asarray(v.values)

    met = MetInput(
        T=get("T") if get("T") is not None else 258.15,
        P=get("P") if get("P") is not None else 70000.0,
        RH=get("RH"), p_v=get("p_v"), q_v=get("q_v"), r_mix=get("r_mix"),
        grad_T=get("grad_T"),
        w=get("w"), LWC=get("LWC"), IWC=get("IWC"),
        N_ccn=get("N_ccn"), N_inp=get("N_inp"),
        cooling_rate=get("cooling_rate"), z=get("z"),
        lat=get("lat"), lon=get("lon"), time=get("time"),
    )
    met._humidity_source = "xarray"
    return met


def from_netcdf(path: str) -> MetInput:
    """Read a NetCDF file via xarray.  Tries backends netcdf4, h5netcdf, scipy
    in order; if none is available for the file format, raises a clear error
    naming the missing dependency (so the caller can degrade to
    'undetermined')."""
    import xarray as xr
    last = None
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds = xr.open_dataset(path, engine=eng)
            return from_xarray(ds)
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(
        f"Could not open {path!r} with xarray (tried netcdf4/h5netcdf/scipy). "
        f"Last error: {last}.  Install netCDF4 or h5netcdf for HDF5/NetCDF4; "
        f"GRIB needs cfgrib.")


def from_grib(path: str) -> MetInput:
    """Read a GRIB file via cfgrib.  cfgrib is NOT installed in this
    environment; this degrades to 'undetermined' with a clear message."""
    try:
        import cfgrib  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"GRIB ingestion requires the 'cfgrib' backend (not installed: "
            f"{e}).  Install it with `pip install cfgrib eccodes`.")
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib")
    return from_xarray(ds)


def reports_to_records(reports) -> List[Dict[str, Any]]:
    """Flatten a (dict per point) structure to a list of JSON records (one per
    phase per point), keeping the mandatory 47 fields + detail + metadata."""
    flat = []
    # accept: dict[phase]->report, or list of those
    if isinstance(reports, MetNucleationReport):
        reports = {"_": reports}
    if isinstance(reports, dict) and any(
            isinstance(v, MetNucleationReport) for v in reports.values()):
        for ph, r in reports.items():
            if isinstance(r, MetNucleationReport):
                flat.append(r.to_dict())
    elif isinstance(reports, (list, tuple)):
        for item in reports:
            flat.extend(reports_to_records(item))
    return flat


def to_json(reports, path: str) -> str:
    recs = reports_to_records(reports)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def to_csv(reports, path: str) -> str:
    import csv
    recs = reports_to_records(reports)
    if not recs:
        open(path, "w", encoding="utf-8").close()
        return path
    cols = MANDATORY_FIELDS
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            w.writerow([_csv_cell(r.get(c)) for c in cols])
    return path


def _csv_cell(v):
    if v is None:
        return NA
    if isinstance(v, float) and not math.isfinite(v):
        return NA
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, default=_json_default)
    return v


def to_xarray(reports, path: Optional[str] = None):
    """Collect per-phase scalar reports into an xarray.Dataset over the
    mandatory scalar fields, indexed by an UNNAMED 'phase' dimension (one
    entry per phase, in sorted phase-name order).  An unnamed dimension is
    used because string/coord variables do not round-trip cleanly through
    NetCDF3/scipy; the phase-name mapping is stored in `ds.attrs['phase_names']`
    (comma-separated, same order as the dimension).  Writes NetCDF
    (scipy engine -> NetCDF3) if a path is given.  Returns the Dataset."""
    import xarray as xr
    recs = reports_to_records(reports)
    if not recs:
        return xr.Dataset()
    phase_names = sorted({r["phase"] for r in recs})
    data_vars = {}
    # String/categorical mandatory fields are kept in JSON/CSV but skipped in
    # the numeric NetCDF table (they would become nan columns, and "phase"
    # would collide with the dimension name).  The phase identity is preserved
    # in `ds.attrs['phase_names']`; the categories in JSON/CSV.
    STRING_FIELDS = {"phase", "status", "nucleation_mode",
                      "dominant_phase", "diagnostic_class"}
    for c in MANDATORY_FIELDS:
        if c in ("assumptions", "warnings", "validity_flags") or c in STRING_FIELDS:
            continue
        vals = []
        for ph in phase_names:
            match = next((r for r in recs if r["phase"] == ph), {})
            v = match.get(c)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                vals.append(float("nan"))
            elif isinstance(v, (int, float)):
                vals.append(float(v))
            else:
                vals.append(float("nan"))
        data_vars[c] = (("phase",), vals)
    ds = xr.Dataset(data_vars)               # unnamed 'phase' dim (len = #phases)
    ds.attrs["title"] = "Meteorological H2O phase-nucleation report"
    ds.attrs["source"] = "met_h2o_nucleation.py (Ferreira Eq.39a/39b framework)"
    ds.attrs["conventions"] = "GT=r_C*dT/2; f(theta)=2-3cos+cos^3; factor=f/4"
    ds.attrs["phase_names"] = ",".join(phase_names)
    if path:
        _write_netcdf(ds, path)
    return ds


def _write_netcdf(ds, path: str):
    """Write NetCDF, trying engines netcdf4/h5netcdf/scipy (scipy -> NetCDF3)."""
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds.to_netcdf(path, engine=eng)
            return eng
        except Exception:
            continue
    raise RuntimeError("No NetCDF backend available (need netCDF4 or h5netcdf or scipy).")


def to_netcdf(reports, path: str) -> str:
    to_xarray(reports, path)
    return path


# =============================================================================
#  VISUALISATION
# =============================================================================
class MetNucleationPlotter:
    """Optional PNG figures, written to an output directory."""

    def __init__(self, outdir: str = "out_met_nucleation"):
        import matplotlib
        matplotlib.use("Agg")
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def _save(self, fig, name):
        path = os.path.join(self.outdir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path

    def plot_peq_shift_surface(self, phase="liquid",
                               T_range=(240.0, 285.0), nT=20,
                               g_range=(1.0, 1.0e4), ng=20):
        """P_eq,shift(T, gradT) surface for one phase (uses the core directly)."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        Ts = np.linspace(*T_range, nT)
        logg = np.linspace(np.log10(g_range[0]), np.log10(g_range[1]), ng)
        model = (LiquidNucleationModel(AtmosphericInput(phase_mode="liquid"))
                 if phase == "liquid"
                 else IceNucleationModel(AtmosphericInput(phase_mode="ice")))
        Tt_loc = Tt
        Psat_T = np.array([model.Psat(T) for T in Ts])
        G, TT = np.meshgrid(10.0 ** logg, Ts)
        Pshift = np.full_like(G, np.nan)
        for i, T in enumerate(Ts):
            pv = model.Psat(T)
            for j, g in enumerate(10.0 ** logg):
                # local shifted pressure: find r such that closure gives g
                rscan = np.geomspace(1e-2, 1e-9, 60)
                best = None
                for r in rscan:
                    st = model.solve(r, T, 70000.0, pv)
                    if st is not None and math.isfinite(st["g"]):
                        if best is None or abs(st["g"] - g) < abs(best[1] - g):
                            best = (r, st["g"], st["T_local"])
                if best is not None:
                    Pshift[i, j] = model.Psat(best[2])
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(TT, G, Pshift, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.set_xlabel("T [K]"); ax.set_ylabel("gradT [K/m]"); ax.set_zlabel("P_eq,shift [Pa]")
        ax.set_title(f"P_eq,shift(T, gradT)  phase={phase}")
        return self._save(fig, f"peq_shift_surface_{phase}.png")

    def plot_gibbs_thomson_and_radii(self, reports_by_gradT, phase="liquid"):
        """Gamma_1/2 and r_C1/r_C2 vs gradT from a list of reports."""
        import matplotlib.pyplot as plt
        gs, g1, g2, r1, r2, dT = [], [], [], [], [], []
        for g, reps in reports_by_gradT:
            r = reps.get(phase)
            if r is None or r.status != "ok":
                continue
            gs.append(r.gradT_K_m); g1.append(r.Gamma_1st); g2.append(r.Gamma_2nd)
            r1.append(r.r_critical_1st_m); r2.append(r.r_critical_2nd_m)
            dT.append(r.DeltaT_K)
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        if gs:
            axs[0, 0].loglog(gs, np.abs(g1), "o-", label="|Gamma_1st|")
            axs[0, 0].loglog(gs, np.abs(g2), "s-", label="|Gamma_2nd|")
            axs[0, 1].loglog(gs, np.abs(r1), "o-", label="r_C,1st")
            axs[0, 1].loglog(gs, np.abs(r2), "s-", label="r_C,2nd")
            axs[1, 0].loglog(gs, dT, "o-")
            axs[1, 0].set_ylabel("DeltaT [K]")
        axs[0, 0].set(xlabel="gradT [K/m]", ylabel="|Gamma| [K.m]", title="Gibbs-Thomson")
        axs[0, 0].legend(); axs[0, 0].grid(True, ls="--", alpha=0.4)
        axs[0, 1].set(xlabel="gradT [K/m]", ylabel="r_C [m]", title="Critical radii")
        axs[0, 1].legend(); axs[0, 1].grid(True, ls="--", alpha=0.4)
        axs[1, 0].set(xlabel="gradT [K/m]"); axs[1, 0].grid(True, ls="--", alpha=0.4)
        axs[1, 1].axis("off")
        fig.suptitle(f"Gibbs-Thomson & critical radii  phase={phase}")
        fig.tight_layout()
        return self._save(fig, f"gt_and_radii_{phase}.png")

    def plot_free_energy(self, model, T, P, p_v, theta=math.pi, n=40):
        """dG_bulk / dG_surface / dG_config / dG_total vs radius."""
        import matplotlib.pyplot as plt
        rs = np.geomspace(1e-8, 1e-5, n)
        bulk, surf, conf, tot = [], [], [], []
        for r in rs:
            st = model.solve(r, T, P, p_v, theta_default=theta)
            if st is None:
                bulk.append(np.nan); surf.append(np.nan); conf.append(np.nan); tot.append(np.nan)
                continue
            fe = free_energy_decomposition(model, st, theta)
            bulk.append(fe["DeltaG_bulk_J"]); surf.append(fe["DeltaG_surface_J"])
            conf.append(fe["DeltaG_config_J"]); tot.append(fe["DeltaG_total_J"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rs, np.array(bulk), label="dG_bulk")
        ax.plot(rs, np.array(surf), label="dG_surface")
        ax.plot(rs, np.array(conf), label="dG_config")
        ax.plot(rs, np.array(tot), label="dG_total", lw=2, color="k")
        ax.set_xscale("log"); ax.set_xlabel("r [m]"); ax.set_ylabel("dG [J]")
        ax.set_title("Free-energy decomposition vs radius")
        ax.legend(); ax.grid(True, ls="--", alpha=0.4)
        return self._save(fig, "free_energy_vs_r.png")

    def plot_rates(self, reports_liquid, reports_ice):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for reps, lab, c in [(reports_liquid, "liquid", "#1f4fbf"),
                            (reports_ice, "ice", "#9a8cd6")]:
            xs, ys = [], []
            for r in reps if isinstance(reps, list) else []:
                rr = r.get(lab)
                if rr and rr.status == "ok" and math.isfinite(rr.log10_nucleation_rate):
                    xs.append(rr.T_ambient_K); ys.append(rr.log10_nucleation_rate)
            if xs:
                ax.plot(xs, ys, "o-", color=c, label=lab)
        ax.set_xlabel("T [K]"); ax.set_ylabel("log10 I  [1/(m^3 s)]")
        ax.set_title("Nucleation rate vs temperature"); ax.legend(); ax.grid(True, ls="--", alpha=0.4)
        return self._save(fig, "rates_vs_T.png")

    def plot_vertical_profile(self, profile_reports, z_arr):
        import matplotlib.pyplot as plt
        z = np.asarray(z_arr)
        fig, axs = plt.subplots(1, 3, figsize=(13, 7), sharey=True)
        for ph, c in [(PHASE_LIQUID, "#1f4fbf"), (PHASE_ICE, "#9a8cd6")]:
            logI, rC, fav = [], [], []
            for reps in profile_reports:
                r = reps.get(ph)
                if r and r.status == "ok":
                    logI.append(r.log10_nucleation_rate); rC.append(r.r_critical_2nd_m)
                    fav.append((r.rain_favorability + r.snow_favorability) / 2)
                else:
                    logI.append(np.nan); rC.append(np.nan); fav.append(np.nan)
            axs[0].plot(np.array(logI), z, "o-", color=c, label=ph)
            axs[1].semilogx(np.abs(np.array(rC)), z, "o-", color=c, label=ph)
            axs[2].plot(np.array(fav), z, "o-", color=c, label=ph)
        axs[0].set(xlabel="log10 I", ylabel="z [m]", title="Nucleation rate")
        axs[1].set(xlabel="|r_C,2nd| [m]", title="Critical radius")
        axs[2].set(xlabel="fav (rain+snow)/2", title="Favourability")
        for a in axs:
            a.legend(); a.grid(True, ls="--", alpha=0.4)
        fig.suptitle("Vertical nucleation profile")
        fig.tight_layout()
        return self._save(fig, "vertical_profile.png")

    def plot_favorability_bars(self, report):
        import matplotlib.pyplot as plt
        labels = ["rain", "snow", "graupel", "hail"]
        vals = [report.rain_favorability, report.snow_favorability,
                report.graupel_favorability, report.hail_favorability]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, vals, color=["#1f4fbf", "#9a8cd6", "#39b6d4", "#d23b3b"])
        ax.set_ylim(0, 1); ax.set_ylabel("favourability [0..1]")
        ax.set_title(f"Precipitation favourability  T={report.T_ambient_K:.1f} K phase={report.phase}")
        return self._save(fig, "favorability_bars.png")


# =============================================================================
#  SELF-VALIDATION  (core untouched + a few met-layer checks)
# =============================================================================
def run_self_checks(verbose=True) -> bool:
    """Run the CORE validation suite (proves it is untouched and still passes)
    plus a few met-layer consistency checks."""
    pr = print if verbose else (lambda *a, **k: None)
    pr("=" * 78); pr("SELF-CHECKS  met_h2o_nucleation.py"); pr("=" * 78)
    ok = True
    # 1) core validation suite
    try:
        core_ok = un.run_validation_tests(verbose=verbose)
        ok &= bool(core_ok)
        pr(f"[core validation] -> {'PASS' if core_ok else 'FAIL'}")
    except Exception as e:
        ok = False
        pr(f"[core validation] EXCEPTION: {e}")
    # 2) met-layer free-energy identity (dG_total = bulk + surface + config)
    try:
        cfg = AtmosphericInput(T=260.0, P=70000.0, p_v=400.0, phase_mode="liquid")
        model = LiquidNucleationModel(cfg)
        st = model.solve(1e-7, 260.0, 70000.0, 400.0)
        fe = free_energy_decomposition(model, st, math.pi)
        s = fe["DeltaG_bulk_J"] + fe["DeltaG_surface_J"] + fe["DeltaG_config_J"]
        ok &= abs(s - fe["DeltaG_total_J"]) < 1e-18
        pr(f"[decomposition identity] dG_total==sum: "
           f"{'PASS' if abs(s-fe['DeltaG_total_J'])<1e-18 else 'FAIL'}")
    except Exception as e:
        ok = False; pr(f"[decomposition identity] EXCEPTION: {e}")
    # 3) runner end-to-end at one point
    try:
        met = MetInput(T=260.0, P=70000.0, RH=110.0, phase_mode="both",
                       w=2.0, LWC=5e-4, IWC=1e-4, cooling_rate=-2e-4,
                       dt_micro=60.0, cell_volume=1e6)
        runner = MetNucleationRunner(met)
        T = 260.0; P = 70000.0
        pv, src, _ = resolve_humidity(met, T, P)
        reps = runner.evaluate_point(T, P, pv, dynamics={
            "w": 2.0, "LWC": 5e-4, "IWC": 1e-4, "cooling_rate": -2e-4})
        ok &= bool(reps)
        for ph, r in reps.items():
            ok &= r.status in ("ok", "subsaturated", "no_solution")
            ok &= 0.0 <= r.rain_favorability <= 1.0
            ok &= 0.0 <= r.confidence <= 1.0
        pr(f"[runner end-to-end] {len(reps)} phases: "
           f"{'PASS' if reps else 'FAIL'}")
    except Exception as e:
        ok = False; pr(f"[runner end-to-end] EXCEPTION: {e}")
    pr("-" * 78)
    pr(f"SELF-CHECKS {'PASS' if ok else 'FAIL'}")
    return ok


# =============================================================================
#  CLI
# =============================================================================
# --summary: compact one-row-per-phase table (the layout shown in the manual).
# The full 48-field vertical report remains the default; --summary replaces it
# with this 14-column at-a-glance table so the documented output is real output.
SUMMARY_COLS = [
    ("phase",      "phase",                   "s"),
    ("status",     "status",                  "s"),
    ("S_w",        "S_water",                 "sat"),
    ("S_i",        "S_ice",                   "sat"),
    ("gradT",      "gradT_K_m",               "grad"),
    ("rC2nd",      "r_critical_2nd_m",         "sci"),
    ("log10I",     "log10_nucleation_rate",   "f2"),
    ("theta_deg",  "contact_angle_deg",       "ang"),
    ("dominant",   "dominant_phase",           "s"),
    ("rain",       "rain_favorability",       "f3"),
    ("snow",       "snow_favorability",       "f3"),
    ("graup",      "graupel_favorability",    "f3"),
    ("hail",       "hail_favorability",       "f3"),
    ("class",      "diagnostic_class",        "s"),
    ("theta_model", "theta_model",            "s"),
    ("exp_events", "expected_events",         "exp"),
]


def _fmt_cell(v, kind):
    """Format one summary cell. None/NaN -> 'undet.' (numeric) or 'undetermined'."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "undetermined" if kind == "exp" else "undet."
    if kind == "s":
        return str(v)
    if kind == "sat":    # saturation ratio
        return f"{v:.2f}"
    if kind == "grad":   # thermal gradient K/m
        return f"{v:.4g}"
    if kind == "sci":    # critical radius m
        return f"{v:.2e}"
    if kind == "f2":     # log10 rate
        return f"{v:.2f}"
    if kind == "ang":    # contact angle [deg]
        return f"{v:.2f}"
    if kind == "f3":     # favourability 0..1
        return f"{v:.3f}"
    if kind == "exp":    # expected events
        return f"{v:.2e}"
    return str(v)


def _print_summary(reps):
    """Print the compact horizontal summary table (one row per phase)."""
    rows = []
    for ph, r in reps.items():
        d = r.to_dict()
        rows.append([_fmt_cell(d.get(f), k) for (_, f, k) in SUMMARY_COLS])
    headers = [c[0] for c in SUMMARY_COLS]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print("  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    for row in rows:
        print("  " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


def build_argparser():
    p = argparse.ArgumentParser(
        description="Meteorological H2O phase-nucleation application layer "
                    "(wraps the validated Ferreira Eq.39a/39b core).")
    p.add_argument("--validate", action="store_true",
                   help="run the core validation suite + met-layer self-checks")
    p.add_argument("--T", type=float, default=260.0, help="ambient temperature [K]")
    p.add_argument("--P", type=float, default=70000.0, help="total pressure [Pa]")
    p.add_argument("--RH", type=float, default=None, help="relative humidity [%]")
    p.add_argument("--p-v", type=float, default=None, help="vapour partial pressure [Pa]")
    p.add_argument("--phase-mode", default="auto",
                   choices=["auto", "liquid", "ice", "both"])
    p.add_argument("--mode", default="homogeneous",
                   choices=["homogeneous", "heterogeneous"])
    p.add_argument("--theta", type=float, default=None, help="contact angle [deg]")
    p.add_argument("--theta-model", default="ferreira_eq17",
                   choices=list(hca.VALID_MODELS),
                   help="LIQUID contact-angle model: ferreira_eq17 (core default, "
                        "no override) | young_constant | young_line_tension | "
                        "young_chemistry.  Ice always uses the core Eq.17.")
    p.add_argument("--substrate", default=None,
                   help="substrate name (see het_contact_angle.SUBSTRATES: "
                        + ", ".join(sorted(hca.SUBSTRATES)) + ")")
    p.add_argument("--tau", type=float, default=None,
                   help="line tension [N] (used by young_line_tension)")
    p.add_argument("--gamma-sv", type=float, default=None,
                   help="solid-vapour interfacial energy [J/m^2] (young_chemistry)")
    p.add_argument("--gamma-sl", type=float, default=None,
                   help="solid-liquid interfacial energy [J/m^2] (young_chemistry)")
    p.add_argument("--r-ref", type=float, default=R_REF_DEFAULT, help="continuation radius [m]")
    p.add_argument("--gradT", type=float, default=None, help="requested gradT [K/m]")
    p.add_argument("--w", type=float, default=None, help="vertical velocity [m/s]")
    p.add_argument("--LWC", type=float, default=None, help="liquid water content [kg/m^3]")
    p.add_argument("--IWC", type=float, default=None, help="ice water content [kg/m^3]")
    p.add_argument("--dt", type=float, default=None, help="microphysics timestep [s]")
    p.add_argument("--cell-volume", "--microphysics-volume", "--Vcell", dest="Vcell",
                   type=float, default=None,
                   help="subgrid control (parcel) volume [m^3] for "
                        "expected_events = I*dt*V_cell; this is the LOCAL cell/parcel "
                        "volume, NOT the domain volume (a 0-D parcel is a single cell). "
                        "'--Vcell' is a deprecated alias of '--cell-volume'.")
    p.add_argument("--summary", action="store_true",
                   help="print the compact one-row-per-phase table (the manual "
                        "layout) instead of the full 48-field vertical report")
    p.add_argument("--outdir", default="out_met_nucleation")
    p.add_argument("--json", default=None, help="write JSON report to this path")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.validate:
        return 0 if run_self_checks(verbose=True) else 1

    theta = math.radians(args.theta) if args.theta is not None else THETA0
    met = MetInput(T=args.T, P=args.P, RH=args.RH, p_v=args.p_v,
                   phase_mode=args.phase_mode, mode=args.mode,
                   theta=theta, r_ref=args.r_ref, grad_T=args.gradT,
                   w=args.w, LWC=args.LWC, IWC=args.IWC,
                   dt_micro=args.dt, cell_volume=args.Vcell,
                   theta_model=args.theta_model, substrate=args.substrate,
                   tau=args.tau, gamma_sv=args.gamma_sv, gamma_sl=args.gamma_sl,
                   theta_explicit=(args.theta is not None))
    pv, src, warns = resolve_humidity(met, args.T, args.P)
    runner = MetNucleationRunner(met)
    dyn = {"w": args.w, "LWC": args.LWC, "IWC": args.IWC}
    reps = runner.evaluate_point(args.T, args.P, pv, grad_T=args.gradT, dynamics=dyn)

    # console (utf-8 safe)
    try:
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    if args.summary:
        print("=" * 78)
        print("METEOROLOGICAL NUCLEATION REPORT  (summary)")
        print("=" * 78)
        _print_summary(reps)
        if warns:
            print("\nhumidity warnings: " + "; ".join(warns))
        if args.json:
            to_json(reps, args.json)
            print(f"\nJSON -> {args.json}")
        return 0

    print("=" * 78)
    print("METEOROLOGICAL NUCLEATION REPORT")
    print("=" * 78)
    for ph, r in reps.items():
        d = r.to_dict()
        print(f"\n--- phase={ph}  status={r.status}  mode={r.nucleation_mode} ---")
        for c in MANDATORY_FIELDS:
            v = d.get(c)
            if isinstance(v, float) and not math.isfinite(v):
                v = NA
            print(f"  {c:28s} = {v}")
        if r.warnings:
            print("  warnings: " + "; ".join(r.warnings))
    if warns:
        print("\nhumidity warnings: " + "; ".join(warns))
    if args.json:
        to_json(reps, args.json)
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())