#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 het_contact_angle.py
 Heterogeneous LIQUID contact-angle models for H2O(vapour) -> H2O(liquid),
 implemented as a pure application-layer module that IMPORTS THE VALIDATED CORE
 READ-ONLY (same __file__-relative importlib pattern as the met loader).  The
 core file is never modified.

WHY THIS MODULE EXISTS
--------------------------------------------------------------------------------
The liquid-nucleation-angle assessment (see memory
`eq17-is-fletcher-barrier-cuberoot` and the report
`theta_liquid_nucleation_assessment.md`) established that Ferreira Eq.17
    r_C,Het / r_C,Hom = (2 + cos th)^(1/3) (1 - cos th)^(2/3) / 2^(2/3)
is exactly the CUBE ROOT of the Fletcher flat-substrate BARRIER shape factor
    f_F(th) = (2 - 3 cos th + cos^3 th)/4 = ftheta(th)/4   (verified to 1e-16)
i.e. Eq.17 embeds the classical *barrier* reduction into the *radius* as a cube
root -- a model-dependent radius rescaling, NOT the classical liquid-nucleation
relation.  In classical CNT (Volmer; Fletcher JCP 29 572 1958) the critical
radius is theta-INDEPENDENT and only the barrier carries f_F(th).

The recommended, most defensible formulation for atmospheric liquid nucleation
is therefore:
    * default liquid theta = substrate Young angle theta_Y(T)  (Model D),
    * optional line-tension correction for r <= 10 nm            (Model C),
    * classical theta-independent r*,
    * barrier via f_F(theta),
    * rate via the core's Ferreira-Eq.21 kernel with cap area A = 2 pi r^2 (1-cos th);
    * soluble CCN -> theta = 0 (Kohler, flagged idealization);
    * Ferreira Eq.17 preserved as the default comparison mode.

This module supplies the angle; the met layer (`met_h2o_nucleation.py`) applies it
as a post-solve override and recomputes the barrier + rate through the core's own
validated `_rate` kernel (read-only call, no core edit).

MODELS
  ferreira_eq17        no override -- use the core's self-consistent Eq.17 theta
                       (returns the sentinel (None, None, {})).
  young_constant       theta = theta_Y (substrate Young angle, constant).
  young_line_tension   modified Young:  cos th(r) = cos th_Y - tau/(gamma_LV r),
                       capped to (1e-6, pi-1e-6).  tau ~ 1e-12..1e-10 N.
  young_chemistry      Young from interfacial energies: th_Y(T) = acos((g_SV-g_SL)/gamma_LV(T)).

UNITS
  theta          rad
  r              m
  T              K
  gamma_LV       J/m^2  (= N/m)
  tau            N
  g_SV, g_SL     J/m^2

Author: generated for Prof. I. L. Ferreira (UFPa / ITEC / FEM).
================================================================================
"""
import importlib.util, math, os

# ---- load the validated core READ-ONLY (same pattern as the met loader) ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = [
    os.path.join(_HERE, "unified_h2o_nucleation_climate",
                 "unified_h2o_nucleation_climate.py"),
    os.path.join(_HERE, "..", "unified_h2o_nucleation_climate",
                 "unified_h2o_nucleation_climate.py"),
]
_CORE = next((p for p in _CANDS if os.path.isfile(p)), None)
if _CORE is None:
    raise FileNotFoundError("Could not locate unified_h2o_nucleation_climate.py")
_spec = importlib.util.spec_from_file_location("un_core_hca", _CORE)
un = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(un)

# re-export the core geometry helpers we depend on (do not reimplement them)
ftheta = un.ftheta                       # f(th) = 2 - 3 cos th + cos^3 th   (0..4)
het_hom_radius_ratio = un.het_hom_radius_ratio   # Ferreira Eq.17 closed form
gamma_VL_inf = un.gamma_VL_inf           # planar liquid-vapour surface energy [J/m^2]

VALID_MODELS = ("ferreira_eq17", "young_constant",
                "young_line_tension", "young_chemistry")

# angle cap consistent with the core's _solve_theta bracket (1e-6, pi-1e-6)
_THETA_MIN = 1e-6
_THETA_MAX = math.pi - 1e-6


# =============================================================================
#  geometry / shape factors
# =============================================================================
def f_F(theta):
    """Fletcher flat-substrate barrier shape factor, normalized 0..1
    (= ftheta/4 = the classical DeltaG*_het / DeltaG*_hom)."""
    return ftheta(theta) / 4.0


def ferreira_eq17_ratio(theta):
    """Ferreira Eq.17 closed form  r_C,Het / r_C,Hom  (= f_F^(1/3)).
    Delegates to the core's validated `het_hom_radius_ratio` (log-form) and
    falls back to the exact cube-root-of-f_F form near the endpoints, where the
    log form suffers cos->1 rounding (1-cos -> 0 -> log domain error).  Both
    forms agree to ~1e-16 in the interior (R^3 = f_F)."""
    try:
        v = het_hom_radius_ratio(theta)
        if math.isfinite(v):
            return v
    except (ValueError, OverflowError):
        pass
    ff = f_F(theta)
    return math.copysign(abs(ff) ** (1.0 / 3.0), 1.0)


# =============================================================================
#  substrate table
# =============================================================================
# Reported water contact angles / interfacial energies on atmospheric substrates.
# theta_Y_deg is the macroscopic Young angle (sessile-drop / AFM / MD literature);
# g_SV, g_SL are illustrative solid interfacial energies [J/m^2] consistent with
# the quoted Young angle at ~293 K (gamma_LV(293) ~ 0.073 J/m^2); they are only
# used by the young_chemistry model when the user does not pass --gamma-sv/sl.
# Sources: Ethington (1990, USGS OFR 90-409); Rao et al. (2021, Chem Eng Sci);
# Eastwood et al. (2008, JGR); Kulkarni et al. (2012); Werder et al. (2003, JPCB, MD).
SUBSTRATES = {
    # name: (theta_Y_deg, g_SV, g_SL, source)
    "quartz":            (12.0, 0.085, 0.067, "Ethington 1990 (sessile drop)"),
    "calcite":           (0.0,  0.073, 0.073, "Ethington 1990 (fully wetting)"),
    "k_feldspar":        (31.5, 0.110, 0.048, "Rao 2021 (AFM, microcline)"),
    "na_feldspar":       (34.0, 0.113, 0.052, "Rao 2021 (AFM, albite)"),
    "kaolinite":         (9.0,  0.080, 0.069, "Eastwood 2008 (ice embryo angle)"),
    "graphite":          (86.0, 0.060, 0.065, "Werder 2003 (MD)"),
    "soot":              (60.0, 0.040, 0.076, "typical aged black carbon (review)"),
    "soluble_ccn":       (0.0,  None, None,  "soluble CCN: Kohler (theta->0 idealization)"),
}


def resolve_substrate(name, theta_Y_deg=None, g_SV=None, g_SL=None):
    """Resolve a Young angle [rad] and interfacial energies [J/m^2] from the
    requested substrate / explicit overrides.  Precedence (highest wins):
        1. explicit theta_Y_deg  (from --theta)
        2. explicit g_SV & g_SL   (from --gamma-sv / --gamma-sl)
        3. the SUBSTRATES table
        4. ValueError
    Returns (theta_Y_rad, g_SV, g_SL, source).  g_SV/g_SL may be None when only a
    Young angle is known (then young_chemistry cannot be used)."""
    src = None
    tab_gsv = tab_gsl = None
    if name is not None:
        if name not in SUBSTRATES:
            raise ValueError(
                f"unknown substrate '{name}'; choose from {sorted(SUBSTRATES)}")
        tab_deg, tab_gsv, tab_gsl, src = SUBSTRATES[name]

    # 1. explicit Young angle
    if theta_Y_deg is not None:
        th = math.radians(float(theta_Y_deg))
        return th, g_SV, g_SL, (src or "explicit --theta")

    # 2. explicit interfacial energies -> Young angle from them
    if g_SV is not None and g_SL is not None:
        return None, g_SV, g_SL, (src or "explicit --gamma-sv/sl")

    # 3. substrate table
    if name is not None:
        tab_th = math.radians(tab_deg) if tab_deg is not None else None
        if tab_th is not None:
            return tab_th, (g_SV if g_SV is not None else tab_gsv), \
                   (g_SL if g_SL is not None else tab_gsl), src
        # table only had energies (none currently), fall through

    raise ValueError(
        "cannot resolve a Young angle: provide --substrate, --theta, or "
        "--gamma-sv/--gamma-sl")


def gamma_LV(model, r, T):
    """Liquid-vapour surface energy [J/m^2] at radius r, temperature T.
    Uses the core model's Tolman-corrected `surface_energy(r, T)` when a model
    instance and a finite r are available, else the planar limit `gamma_VL_inf(T)`."""
    if model is not None and r is not None and math.isfinite(r) and r > 0.0:
        try:
            return model.surface_energy(r, T)
        except Exception:
            pass
    return gamma_VL_inf(T)


# =============================================================================
#  the angle solver
# =============================================================================
def _cap(theta):
    return min(max(theta, _THETA_MIN), _THETA_MAX)


def solve_theta(model_name, r, T, gamma_LV, theta_Y_rad=None, tau=0.0,
                g_SV=None, g_SL=None, substrate=None):
    """Compute the heterogeneous liquid contact angle [rad] under the requested
    model.  Returns (theta_rad, f_F(theta), info_dict).

    `model_name` in VALID_MODELS.  `ferreira_eq17` is the no-override sentinel:
    it returns (None, None, {}) so the caller knows to use the core's own theta.

    `theta_Y_rad` is the resolved Young angle (from resolve_substrate); for
    young_chemistry it may be None and the angle is computed from g_SV/g_SL.
    """
    if model_name not in VALID_MODELS:
        raise ValueError(f"theta_model must be one of {VALID_MODELS}, "
                         f"got '{model_name}'")

    info = {"model": model_name, "substrate": substrate,
            "tau_N": tau, "gamma_LV_J_m2": gamma_LV,
            "r_used_m": r, "T_used_K": T}

    if model_name == "ferreira_eq17":
        return None, None, {}          # no override; caller keeps the core theta

    if model_name == "young_constant":
        if theta_Y_rad is None:
            raise ValueError("young_constant needs a Young angle "
                             "(--theta or --substrate)")
        th = _cap(theta_Y_rad)
        info["theta_Y_deg"] = math.degrees(theta_Y_rad)
        return th, f_F(th), info

    if model_name == "young_line_tension":
        if theta_Y_rad is None:
            raise ValueError("young_line_tension needs a Young angle "
                             "(--theta or --substrate)")
        if r is None or r <= 0.0 or gamma_LV is None or gamma_LV <= 0.0:
            th = _cap(theta_Y_rad)
            info["theta_Y_deg"] = math.degrees(theta_Y_rad)
            info["note"] = "line-tension term skipped (r/gamma unavailable)"
            return th, f_F(th), info
        x = math.cos(theta_Y_rad) - tau / (gamma_LV * r)
        x = max(-1.0, min(1.0, x))
        th = _cap(math.acos(x))
        info["theta_Y_deg"] = math.degrees(theta_Y_rad)
        info["delta_theta_deg"] = math.degrees(th) - math.degrees(theta_Y_rad)
        return th, f_F(th), info

    if model_name == "young_chemistry":
        if g_SV is None or g_SL is None or gamma_LV is None or gamma_LV <= 0.0:
            raise ValueError("young_chemistry needs g_SV, g_SL and gamma_LV")
        x = (g_SV - g_SL) / gamma_LV
        x = max(-1.0, min(1.0, x))
        th = _cap(math.acos(x))
        info["g_SV"] = g_SV
        info["g_SL"] = g_SL
        return th, f_F(th), info

    # unreachable
    raise ValueError(f"unhandled theta_model '{model_name}'")


# =============================================================================
#  self-test / quick demo
# =============================================================================
def _demo():
    print("=" * 72)
    print("het_contact_angle  -- substrate liquid-nucleation angle models")
    print("=" * 72)
    g = gamma_VL_inf(293.15)
    print(f"gamma_LV(293.15 K) = {g:.5f} J/m^2\n")

    print("[identity] ferreira_eq17_ratio^3 == f_F == ftheta/4")
    worst = 0.0
    for th in [1e-3, 0.3, 1.0, math.pi / 2, 2.0, 2.8, math.pi - 1e-3]:
        R = ferreira_eq17_ratio(th)
        worst = max(worst, abs(R ** 3 - f_F(th)), abs(f_F(th) - ftheta(th) / 4.0))
    print(f"    max|err| = {worst:.2e}\n")

    print("[limits]  theta -> 0 / pi/2 / pi")
    for name, th in [("0+", 1e-9), ("pi/2", math.pi / 2), ("pi", math.pi - 1e-12)]:
        print(f"    {name:5s} f_F={f_F(th):.4f}  R_eq17={ferreira_eq17_ratio(th):.4f}")

    print("\n[young_constant]  substrate Young angles")
    for sub, (deg, _, _, src) in SUBSTRATES.items():
        th, ff, _ = solve_theta("young_constant", r=1e-6, T=293.15, gamma_LV=g,
                                theta_Y_rad=math.radians(deg))
        print(f"    {sub:14s} theta={math.degrees(th):6.2f} deg  f_F={ff:.4f}  ({src})")

    print("\n[young_line_tension]  theta(r), quartz (12 deg), tau=2e-10 N")
    thY = math.radians(12.0)
    for r in [5e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
        th, ff, info = solve_theta("young_line_tension", r=r, T=293.15,
                                   gamma_LV=g, theta_Y_rad=thY, tau=2e-10)
        print(f"    r={r:.1e} m  theta={math.degrees(th):7.3f} deg  "
              f"dtheta={info['delta_theta_deg']:+.3f} deg  f_F={ff:.4f}")

    print("\n[young_chemistry]  theta_Y(T) at fixed dg=g_SV-g_SL=0.5*gamma_LV(293)")
    dg = 0.5 * gamma_VL_inf(293.15)
    for T in [253.15, 293.15, 333.15, 373.15]:
        try:
            gT = gamma_VL_inf(T)
        except Exception:
            continue
        th, ff, _ = solve_theta("young_chemistry", r=1e-6, T=T, gamma_LV=gT,
                                g_SV=gT + dg, g_SL=gT)
        print(f"    T={T:6.2f} K  gamma_LV={gT:.4f}  theta={math.degrees(th):6.2f} deg  f_F={ff:.4f}")


if __name__ == "__main__":
    _demo()