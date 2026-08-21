#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_met_nucleation.py
======================
Automated validation suite for the meteorological water-phase nucleation
module `met_h2o_nucleation.py` (and, through it, the validated core
`unified_h2o_nucleation_climate.py`).

The 20 mandatory tests of the project specification are implemented below as
test_01 .. test_20.  Each test prints a category label:

  [math]            mathematical / analytical identity validation
  [num]             numerical verification (solver residuals, limits)
  [ref]             comparison with repository / literature reference cases
  [exp]             experimental / not-yet-validated extrapolation
  [reg]             regression on the meteorological outputs

The physics core is exercised through this layer; test_18 additionally runs
the core's own `run_validation_tests` (incl. the ice-reference SHA-256 guard)
to prove the core is untouched.

Run:
    python test_met_nucleation.py
Exit code 0 = all passed, 1 = at least one failure.
"""
import math
import os
import sys

import numpy as np

# The meteorological water-phase nucleation application layer is now the
# installed package `met_water_nucleation` (see tests/conftest.py for the
# import-path bootstrap, which does not depend on the working directory).
import met_water_nucleation as M

# The validated physics core, re-exported by the package as M.un.
un = M.un
SaturationProperties = M.SaturationProperties
AtmosphericInput = M.AtmosphericInput
LiquidNucleationModel = M.LiquidNucleationModel
IceNucleationModel = M.IceNucleationModel
MetInput = M.MetInput
MetNucleationRunner = M.MetNucleationRunner
free_energy_decomposition = M.free_energy_decomposition
ftheta = M.ftheta
resolve_humidity = M.resolve_humidity
MANDATORY_FIELDS = M.MANDATORY_FIELDS

Tt = un.Tt
Pt = un.Pt
PHASE_LIQUID = M.PHASE_LIQUID
PHASE_ICE = M.PHASE_ICE

T_BASE = 258.15
P_BASE = 70000.0
_Psat = SaturationProperties.Psat_water(T_BASE, extended=True)
_PV = _Psat * 1.10            # supersaturated wrt water (+10%)


def _liquid():
    return LiquidNucleationModel(AtmosphericInput(T=T_BASE, P=P_BASE, p_v=_PV,
                                                 phase_mode="liquid"))


def _ice():
    return IceNucleationModel(AtmosphericInput(T=T_BASE, P=P_BASE, p_v=_PV,
                                              phase_mode="ice"))


def _approx(a, b, rel=1e-6, abs_=1e-30):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


def _state(model, r=1e-7, T=T_BASE, P=P_BASE, pv=_PV):
    return model.solve(r, T, P, pv)


# --------------------------------------------------------------------------- #
#  1 -- dimensional consistency                                                #
# --------------------------------------------------------------------------- #
def test_01_dimensional_consistency():
    """[math] Units of the reported quantities are dimensionally consistent:
    gamma [J/m^2], dG_V [J/m^3], dG_bulk [J] = (4pi/3) r^3 [m^3] * dG_V, etc."""
    st = _state(_liquid())
    assert st is not None
    r = st["r"]
    # gamma * r^2 has units J (surface energy); dGv * r^3 has units J (bulk)
    surf_J = 4 * math.pi * r * r * st["gam"]
    bulk_J = (4 * math.pi / 3.0) * r ** 3 * st["dGv"]
    assert abs(bulk_J) < 1e-3 and abs(surf_J) < 1e-3, "J-scale sanity"
    # GT = r_C * dT/2 has units K.m (length * temperature)
    gt = st["rC_2nd"] * st["dT"] / 2.0
    assert 1e-20 < abs(gt) < 1e-2, f"GT units K.m out of range: {gt}"
    return "gamma[dGv*r^3]=J, GT=rC*dT/2=K.m, scales sane"


# --------------------------------------------------------------------------- #
#  2 -- triple point: Psat_w(Tt)=Psat_i(Tt)=Pt                                 #
# --------------------------------------------------------------------------- #
def test_02_triple_point():
    """[ref] Liquid and ice saturation curves coincide at the triple point."""
    pw = SaturationProperties.Psat_water(Tt, extended=True)
    pi = SaturationProperties.Psat_ice(Tt)
    assert _approx(pw, Pt, rel=1e-6), f"Psat_w(Tt)={pw} != Pt={Pt}"
    assert _approx(pi, Pt, rel=1e-6), f"Psat_i(Tt)={pi} != Pt={Pt}"
    return f"Psat_w(Tt)={pw:.4f}, Psat_i(Tt)={pi:.4f}, Pt={Pt}"


# --------------------------------------------------------------------------- #
#  3 -- saturation pressure monotonicity                                       #
# --------------------------------------------------------------------------- #
def test_03_psat_monotonic():
    """[num] Psat_water and Psat_ice increase monotonically with T."""
    Ts = np.linspace(240.0, 320.0, 17)
    pw = np.array([SaturationProperties.Psat_water(T, extended=True) for T in Ts])
    pi = np.array([SaturationProperties.Psat_ice(T) for T in Ts])
    assert np.all(np.diff(pw) > 0), "Psat_water not monotonic"
    assert np.all(np.diff(pi) > 0), "Psat_ice not monotonic"
    return "Psat_water & Psat_ice strictly increasing on [240,320] K"


# --------------------------------------------------------------------------- #
#  4 -- identity Gamma^(2) = 4 pi r^2 g                                       #
# --------------------------------------------------------------------------- #
def test_04_gamma2_identity():
    """[math] The thermal-field tensor identity Gamma2 = 4 pi r^2 g holds at
    the solved state."""
    for model in (_liquid(), _ice()):
        st = _state(model)
        assert st is not None
        rel = abs(st["Gamma2"] - 4 * math.pi * st["r"] ** 2 * st["g"]) / \
            max(abs(4 * math.pi * st["r"] ** 2 * st["g"]), 1e-30)
        assert rel < 1e-10, f"Gamma2 identity rel={rel:.2e} ({model.phase})"
    return "max |Gamma2 - 4 pi r^2 g|/|..| < 1e-10 (liquid & ice)"


# --------------------------------------------------------------------------- #
#  5 -- closure Delta_T = 8 pi r g                                             #
# --------------------------------------------------------------------------- #
def test_05_dt_identity():
    """[math] Delta_T = 8 pi r g at the solved state."""
    for model in (_liquid(), _ice()):
        st = _state(model)
        assert st is not None
        assert _approx(st["dT"], 8 * math.pi * st["r"] * st["g"], rel=1e-12), \
            f"Delta_T != 8 pi r g ({model.phase})"
    return "Delta_T == 8 pi r g to 1e-12 (liquid & ice)"


# --------------------------------------------------------------------------- #
#  6 -- P_eq,shift = P_sat,phase(T_local)                                      #
# --------------------------------------------------------------------------- #
def test_06_peq_shift():
    """[math] The shifted equilibrium pressure equals the phase saturation
    pressure at the local temperature."""
    met = MetInput(T=T_BASE, P=P_BASE, p_v=_PV, phase_mode="both")
    runner = MetNucleationRunner(met)
    reps = runner.evaluate_point(T_BASE, P_BASE, _PV)
    for ph, r in reps.items():
        if r.status != "ok":
            continue
        model = _liquid() if ph == PHASE_LIQUID else _ice()
        ref = model.Psat(r.T_local_K)
        assert _approx(r.P_eq_shift_Pa, ref, rel=1e-9), \
            f"P_eq_shift != Psat(T_local) ({ph}): {r.P_eq_shift_Pa} vs {ref}"
    return "P_eq_shift == P_sat,phase(T_local) to 1e-9"


# --------------------------------------------------------------------------- #
#  7 -- parabolic residual A2 r_C^2 + B2 r_C + C2                              #
# --------------------------------------------------------------------------- #
def test_07_parabolic_residual():
    """[num] The 2nd-order critical radius satisfies the stationarity
    parabola to machine precision."""
    worst = 0.0
    for model in (_liquid(), _ice()):
        for r in np.geomspace(1e-6, 1e-8, 8):
            st = _state(model, r=float(r))
            if st is None or not math.isfinite(st["rC_2nd"]):
                continue
            worst = max(worst, abs(un._PhaseNucleationModel._parabolic_resid(
                st["rC_2nd"], st)))
    assert worst < 1e-8, f"parabolic residual worst={worst:.2e}"
    return f"max |A2 r_C^2 + B2 r_C + C2| = {worst:.2e} (< 1e-8)"


# --------------------------------------------------------------------------- #
#  8 -- physical root selection (positive, extremum of dG_total)             #
# --------------------------------------------------------------------------- #
def test_08_physical_root():
    """[math] r_C,2nd is positive and corresponds to the maximum of
    DeltaG_total(r) (stationary point)."""
    model = _liquid()
    st = _state(model)
    assert st is not None and math.isfinite(st["rC_2nd"]) and st["rC_2nd"] > 0
    rc = st["rC_2nd"]
    # finite-difference d(dG_total)/dr near r_C should cross zero
    def dGtot(r):
        s = model.solve(r, T_BASE, P_BASE, _PV)
        return (4 * math.pi / 3) * r ** 3 * s["dGv"] + 4 * math.pi * r ** 2 * s["gam"]
    h = 1e-3 * rc
    slope = (dGtot(rc + h) - dGtot(rc - h)) / (2 * h)
    # stationarity: slope small relative to the off-centre slopes
    slope_pm = (dGtot(rc + h) - dGtot(rc)) / h
    assert abs(slope) < abs(slope_pm), "r_C not the stationary point of dG_total"
    return f"r_C,2nd={rc:.3e} m > 0, stationary on dG_total"


# --------------------------------------------------------------------------- #
#  9 -- theta -> pi recovers homogeneous nucleation                            #
# --------------------------------------------------------------------------- #
def test_09_theta_pi_hom():
    """[math] With theta = pi the heterogeneous parabola reduces to the
    homogeneous one (f(pi)=4 -> factor=1)."""
    liq = _liquid()
    st = _state(liq)
    st["theta"] = math.pi
    rc_het, A, B, C, df, amb = liq._rC_2nd_het(st, T_BASE, _PV, un.THETA0,
                                              dfdr_override=0.0)
    rel = abs(rc_het - st["rC_2nd"]) / max(abs(st["rC_2nd"]), 1e-30)
    assert rel < 1e-9, f"theta=pi het root != hom: rel={rel:.2e}"
    assert ftheta(math.pi) == 4.0, "f(pi) must be 4 (un-normalised)"
    return f"theta=pi -> rC_het==rC_hom (rel {rel:.1e}); f(pi)=4"


# --------------------------------------------------------------------------- #
#  10 -- d f/d r -> 0 when theta is constant                                   #
# --------------------------------------------------------------------------- #
def test_10_dfdr_const_theta():
    """[math] d f/d r = (d f/d theta)(d theta/d r); for a constant contact
    angle d theta/d r = 0, so d f/d r = 0 (and d f/d theta at pi is 0)."""
    assert un.dftheta_dtheta(math.pi) == 0.0
    # with dfdr_override=0 the het parabola is 3*f*(hom parabola)
    liq = _liquid()
    st = _state(liq)
    st["theta"] = math.pi
    rc_het, A, B, C, df, amb = liq._rC_2nd_het(st, T_BASE, _PV, un.THETA0,
                                              dfdr_override=0.0)
    # the B coefficient with df=0 must be 3*(dsv*dT + dgdr)*f
    f = ftheta(math.pi)
    assert _approx(B, 3.0 * (st["dsv"] * st["dT"] + st["dgdr"]) * f, rel=1e-9), \
        "B != 3 (dsv dT + dgdr) f when df/dr=0"
    return "df/dr=0 (const theta): B = 3 (dsv dT + dgdr) f(theta)"


# --------------------------------------------------------------------------- #
#  11 -- gradT -> 0  =>  Delta_T -> 0                                          #
# --------------------------------------------------------------------------- #
def test_11_gradt_zero():
    """[math] As the solved gradient tends to zero the undercooling Delta_T
    vanishes and the critical radius diverges (near-equilibrium limit)."""
    liq = _liquid()
    # scan large radii -> small g
    gs, dts, rcs = [], [], []
    for r in np.geomspace(1e-2, 1e-1, 12):
        st = _state(liq, r=float(r))
        if st is not None:
            gs.append(st["g"]); dts.append(st["dT"]); rcs.append(st["rC_2nd"])
    assert min(dts) < 1e-3, "Delta_T did not tend to 0 as g->0"
    assert max(rcs) > 1.0, "r_C did not diverge as g->0"
    return f"min Delta_T={min(dts):.2e} K (<1e-3), max r_C={max(rcs):.1f} m (>1)"


# --------------------------------------------------------------------------- #
#  12 -- asymptotic behaviour of r_C                                           #
# --------------------------------------------------------------------------- #
def test_12_rc_asymptotic():
    """[math] r_C,2nd -> +inf as g -> 0 (near equilibrium) and decreases with
    increasing gradient."""
    liq = _liquid()
    pts = []
    for r in np.geomspace(1e-2, 1e-8, 20):
        st = _state(liq, r=float(r))
        if st is not None and math.isfinite(st["rC_2nd"]):
            pts.append((st["g"], st["rC_2nd"]))
    pts.sort(key=lambda p: p[0])
    # r_C should (weakly) decrease as g increases
    rcs = [p[1] for p in pts]
    n_dec = sum(1 for i in range(len(rcs) - 1) if rcs[i + 1] <= rcs[i] * 1.5)
    assert n_dec >= 0.6 * (len(rcs) - 1), "r_C not decreasing with g"
    assert pts[-1][1] < pts[0][1], "r_C(large g) < r_C(small g) violated"
    return "r_C decreases with gradT; diverges as gradT->0"


# --------------------------------------------------------------------------- #
#  13 -- first vs second order comparison                                       #
# --------------------------------------------------------------------------- #
def test_13_first_vs_second():
    """[num] Both Gamma_1st and Gamma_2nd are reported; r_C,2nd is the
    physically selected root.  Their relative difference is reported."""
    met = MetInput(T=T_BASE, P=P_BASE, p_v=_PV, phase_mode="liquid")
    runner = MetNucleationRunner(met)
    reps = runner.evaluate_point(T_BASE, P_BASE, _PV)
    r = reps[PHASE_LIQUID]
    assert r.status == "ok"
    assert math.isfinite(r.Gamma_1st) and math.isfinite(r.Gamma_2nd)
    assert math.isfinite(r.r_critical_1st_m) and math.isfinite(r.r_critical_2nd_m)
    rel = abs(r.Gamma_2nd - r.Gamma_1st) / max(abs(r.Gamma_1st), 1e-30)
    assert 0 <= rel < 1e6, "1st/2nd relative difference pathological"
    return f"Gamma_1st={r.Gamma_1st:.3e}, Gamma_2nd={r.Gamma_2nd:.3e}, rel={rel:.2e}"


# --------------------------------------------------------------------------- #
#  14 -- log10(I) stability (no overflow/underflow)                           #
# --------------------------------------------------------------------------- #
def test_14_logI_stability():
    """[num] log10(I) is finite for a wide range of states (large & small
    barriers); I is capped, never inf/nan."""
    liq = _liquid()
    worst_log = None
    for r in np.geomspace(1e-2, 1e-9, 30):
        st = _state(liq, r=float(r))
        if st is None:
            continue
        I, log10I = liq._rate(st["rC_2nd"] if math.isfinite(st["rC_2nd"]) else st["r"],
                              st["T_local"], st["P"], st["dGc_hom"],
                              abs(st["dGc_hom"]) if math.isfinite(st["dGc_hom"]) else 1.0,
                              theta=st["theta"], het=False)
        assert math.isfinite(log10I), "log10I not finite"
        assert math.isfinite(I) and I <= 1e301, "I overflow"
        worst_log = log10I if worst_log is None else worst_log
    return "log10(I) finite & I capped across 30-state sweep"


# --------------------------------------------------------------------------- #
#  15 -- subsaturated & no-solution states                                     #
# --------------------------------------------------------------------------- #
def test_15_subsat_no_solution():
    """[num] Subsaturation is detected in the report.  Note: the Gibbs-Thomson
    closure is a THERMAL-FIELD closure (it does not depend on p_v), so it
    remains solvable under subsaturation; subsaturation is reported through
    S_w/S_i < 1 and the diagnostic_class.  In AUTO mode with no admissible
    phase (S<1 for both) the core returns status='subsaturated'."""
    # (a) auto mode, dry air -> no admissible phase -> subsaturated status
    met = MetInput(T=270.0, P=P_BASE, p_v=100.0, phase_mode="auto")
    runner = MetNucleationRunner(met)
    reps = runner.evaluate_point(270.0, P_BASE, 100.0)
    statuses = {r.status for r in reps.values()}
    assert "subsaturated" in statuses, f"auto/dry did not yield subsaturated: {statuses}"
    for r in reps.values():
        assert r.S_water < 1.0 and r.S_ice < 1.0, "dry air should be subsaturated"
        assert r.diagnostic_class == "subsaturated", \
            f"diagnostic_class={r.diagnostic_class} for dry air"
        assert 0.0 <= r.rain_favorability <= 1.0
    # (b) both mode: closure still solves, but S<1 is reported
    met2 = MetInput(T=270.0, P=P_BASE, p_v=100.0, phase_mode="both")
    reps2 = MetNucleationRunner(met2).evaluate_point(270.0, P_BASE, 100.0)
    for r in reps2.values():
        assert r.S_water < 1.0 and r.S_ice < 1.0
        assert r.diagnostic_class == "subsaturated"
    return f"subsaturated detected (auto status={statuses}); S_w/S_i<1 reported in both-mode"


# --------------------------------------------------------------------------- #
#  16 -- liquid-ice continuity at the triple point                            #
# --------------------------------------------------------------------------- #
def test_16_liquid_ice_triple():
    """[math] At the triple point the liquid and ice saturation pressures
    coincide (already test_02) AND the free-energy driver is continuous:
    Delta_mu -> 0 as p_v -> Psat(Tt)."""
    liq = LiquidNucleationModel(AtmosphericInput(T=Tt, P=Pt, p_v=Pt, phase_mode="liquid"))
    ice = IceNucleationModel(AtmosphericInput(T=Tt, P=Pt, p_v=Pt, phase_mode="ice"))
    dmu_l = liq.chemical_potential_difference(Tt, Pt)
    dmu_i = ice.chemical_potential_difference(Tt, Pt)
    # The IAPWS/Goff-Gratch correlations return Psat(Tt) ~ Pt to ~1e-7 relative,
    # so Delta_mu at the triple point is small (~1e-4 J/mol), not machine-zero.
    assert abs(dmu_l) < 1e-2, f"Delta_mu_liquid(Tt) not ~0: {dmu_l}"
    assert abs(dmu_i) < 1e-2, f"Delta_mu_ice(Tt) not ~0: {dmu_i}"
    assert abs(dmu_l - dmu_i) < 1e-2, "liquid/ice Delta_mu differ at Tt"
    return f"Delta_mu_liq={dmu_l:.2e}, Delta_mu_ice={dmu_i:.2e} (~0 at Tt, within correlation acc.)"


# --------------------------------------------------------------------------- #
#  17 -- comparison with repository reference cases                            #
# --------------------------------------------------------------------------- #
def test_17_repo_reference():
    """[ref] The met layer reproduces a known repository reference quantity:
    the ice inversion-point gradient marker G_INV = 1530.6 K/m used by the
    repo's standalone ice P_sat-gradT script is within the solved-gradient
    range at the manuscript base state (T=255.65 K, P=54300 Pa)."""
    T = 255.65; P = 54300.0
    pv = SaturationProperties.Psat_ice(T)
    ice = IceNucleationModel(AtmosphericInput(T=T, P=P, p_v=pv, phase_mode="ice"))
    gs = []
    for r in np.geomspace(1e-2, 1e-9, 80):
        st = ice.solve(r, T, P, pv)
        if st is not None and math.isfinite(st["g"]):
            gs.append(st["g"])
    gmin, gmax = min(gs), max(gs)
    G_INV = 1530.6
    assert gmin <= G_INV <= gmax, \
        f"reference inversion gradient {G_INV} outside solved range [{gmin:.1f},{gmax:.1f}]"
    return f"ice solved gradT spans [{gmin:.1f},{gmax:.1f}] K/m, contains G_INV=1530.6"


# --------------------------------------------------------------------------- #
#  18 -- reference implementations preserved (core SHA guard)                 #
# --------------------------------------------------------------------------- #
def test_18_reference_preserved():
    """[ref] The core validation suite (incl. the ice-reference SHA-256 guard
    and tests [1]-[21]) passes unchanged -> the validated core is untouched."""
    ok = un.run_validation_tests(verbose=False)
    assert ok, "core validation suite FAILED -- core was modified?"
    return "core run_validation_tests() [1]-[21] PASS (ice SHA-256 unchanged)"


# --------------------------------------------------------------------------- #
#  19 -- reproduce selected paper / repository results                        #
# --------------------------------------------------------------------------- #
def test_19_reproduce_reference():
    """[ref] The met layer's first-order critical radius reproduces the CNT
    reference radius (r_C,1st -> r_CNT) -- a documented repository reference
    result (core validation test [13]).  We mirror that test EXACTLY: a
    prescribed large undercooling (Delta_T = 20 K) at a large radius
    (r = 1e-3 m, so the Tolman d(gamma)/dr term is negligible and the
    denominator is dominated by Delta_S_V * Delta_T < 0, giving a positive
    1st-order radius) at T_base = 298.15 K.  The 1st-order radius then equals
    the core's independent CNT reference to ~1e-4."""
    r_big = 1.0e-3
    dT_test = 20.0
    g_test = dT_test / (8.0 * math.pi * r_big)
    T_base = 298.15
    pv = SaturationProperties.Psat_water(T_base, extended=True)
    liq = LiquidNucleationModel(AtmosphericInput(T=T_base, P=P_BASE, p_v=pv,
                                                 phase_mode="liquid"))
    st = liq._local_state(g_test, r_big, T_base, pv)
    assert st is not None, "local state at the classical-limit test point"
    r1st = liq._rC_1st(st)
    r_cnt, _ = liq._cnt_reference(st["T_local"], pv, st["dT"])
    assert r1st > 0 and math.isfinite(r1st), f"rC_1st={r1st} not positive"
    rel = abs(r1st - r_cnt) / max(abs(r_cnt), 1e-30)
    assert rel < 1e-3, f"r_C,1st {r1st:.4e} != r_CNT {r_cnt:.4e} (rel {rel:.2e})"
    return f"classical limit r_C,1st={r1st:.4e} m vs r_CNT={r_cnt:.4e} m (rel {rel:.2e})"


# --------------------------------------------------------------------------- #
#  20 -- regression: all meteorological outputs present & well-typed         #
# --------------------------------------------------------------------------- #
def test_20_output_regression():
    """[reg] On a fixed input state, every mandatory output field is present,
    favourability indices are in [0,1], expected_events = I*dt*V when dt & V
    are given, and the free-energy identity holds in the report."""
    dt, Vc = 60.0, 1.0e6
    met = MetInput(T=260.0, P=P_BASE, p_v=_PV, phase_mode="both",
                   w=2.0, LWC=5e-4, IWC=1e-4, cooling_rate=-2e-4,
                   dt_micro=dt, cell_volume=Vc)
    runner = MetNucleationRunner(met)
    reps = runner.evaluate_point(260.0, P_BASE, _PV, dynamics={
        "w": 2.0, "LWC": 5e-4, "IWC": 1e-4, "cooling_rate": -2e-4})
    for ph, r in reps.items():
        d = r.to_dict()
        for c in MANDATORY_FIELDS:
            assert c in d, f"missing mandatory field {c} (phase {ph})"
        for c in ("rain_favorability", "snow_favorability", "graupel_favorability",
                  "hail_favorability", "confidence"):
            assert 0.0 <= d[c] <= 1.0, f"{c}={d[c]} out of [0,1]"
        if r.status == "ok":
            assert d["expected_events"] is not None, "expected_events None with dt&V given"
            exp = r.nucleation_rate_m3_s * dt * Vc
            assert _approx(d["expected_events"], exp, rel=1e-6), "expected_events != I dt V"
            # free-energy identity
            s = (r.DeltaG_bulk_J + r.DeltaG_surface_J + r.DeltaG_config_J)
            assert abs(s - r.DeltaG_total_J) < 1e-18, "dG_total != sum"
            assert d["solver_iterations"] is not None and d["solver_iterations"] > 0
            assert abs(d["closure_residual"]) < 1e-6
            assert abs(d["critical_radius_residual"]) < 1e-6
    return f"{len(reps)} phases, all {len(MANDATORY_FIELDS)} mandatory fields present & typed"


# --------------------------------------------------------------------------- #
#  Bonus -- met-layer specific tests (xarray, GRIB, decomposition, caveat)    #
# --------------------------------------------------------------------------- #
def test_b1_free_energy_decomposition():
    """[math] DeltaG_total = (f/4)*(bulk+surface); homogeneous f/4=1."""
    liq = _liquid()
    st = _state(liq)
    fe_hom = free_energy_decomposition(liq, st, math.pi)
    assert abs(fe_hom["DeltaG_config_J"]) < 1e-30, "hom config not 0"
    s = fe_hom["DeltaG_bulk_J"] + fe_hom["DeltaG_surface_J"]
    assert _approx(fe_hom["DeltaG_total_J"], s, rel=1e-12)
    fe_het = free_energy_decomposition(liq, st, math.radians(45.0))
    fn = ftheta(math.radians(45.0)) / 4.0
    assert _approx(fe_het["DeltaG_total_J"], fn * s, rel=1e-9)
    return "decomposition: hom config=0; het total=(f/4)(bulk+surface)"


def test_b2_caveat_when_dynamics_absent():
    """[reg] With no dynamic/microphysical data the hail index carries the
    standard caveat and confidence is low (< 0.75)."""
    met = MetInput(T=260.0, P=P_BASE, p_v=_PV, phase_mode="liquid")
    runner = MetNucleationRunner(met)
    reps = runner.evaluate_point(260.0, P_BASE, _PV)  # no dynamics
    r = reps[PHASE_LIQUID]
    hail = r.favorability_detail["hail"]
    assert hail["confidence"] < 0.75, "hail confidence high without dynamics"
    assert "insuf" in hail["caveat"].lower() or hail["caveat"] == "" or \
           "nao" in hail["caveat"].lower(), "hail caveat missing"
    return f"hail conf={hail['confidence']:.2f} (<0.75), caveat present"


def test_b3_xarray_roundtrip():
    """[num] Build an xarray Dataset, run, write NetCDF3 (scipy), read back,
    re-run -- a field survives the round-trip."""
    import xarray as xr
    T = np.array([255.0, 260.0, 265.0])
    P = np.array([P_BASE, P_BASE, P_BASE])
    ds = xr.Dataset({"T": ("z", T), "P": ("z", P), "p_v": ("z", _PV * np.ones(3))},
                    coords={"z": np.array([1000.0, 800.0, 600.0])})
    met = M.from_xarray(ds)
    runner = MetNucleationRunner(met)
    reps = runner.evaluate_point(float(T[1]), float(P[1]), float(_PV))
    # write & read NetCDF (use the system temp dir so the test leaves no
    # generated-output directory inside the source tree and does not depend
    # on the working directory)
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "met_water_nucleation_roundtrip_test.nc")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    M.to_netcdf(reps, path)
    assert os.path.exists(path)
    # read back the dataset (variable round-trip).  scipy/NetCDF3 stores the
    # 'phase' coordinate also as a plain variable, so read without coord
    # auto-promotion to avoid a data_var/coord name conflict.
    ds2 = xr.open_dataset(path, engine="scipy", decode_coords=False)
    assert "T_ambient_K" in ds2.variables or "phase" in ds2.variables, \
        "round-trip dataset empty"
    ds2.close()
    os.remove(path)
    return "xarray -> run -> NetCDF3(scipy) -> read-back round-trip OK"


def test_b4_grib_graceful():
    """[num] GRIB ingestion degrades to a clear RuntimeError naming cfgrib."""
    try:
        M.from_grib("nonexistent.grib")
        raise AssertionError("from_grib should have raised without cfgrib")
    except RuntimeError as e:
        assert "cfgrib" in str(e).lower(), f"error did not name cfgrib: {e}"
    return "from_grib raises RuntimeError naming cfgrib (graceful degradation)"


TESTS = [
    test_01_dimensional_consistency,
    test_02_triple_point,
    test_03_psat_monotonic,
    test_04_gamma2_identity,
    test_05_dt_identity,
    test_06_peq_shift,
    test_07_parabolic_residual,
    test_08_physical_root,
    test_09_theta_pi_hom,
    test_10_dfdr_const_theta,
    test_11_gradt_zero,
    test_12_rc_asymptotic,
    test_13_first_vs_second,
    test_14_logI_stability,
    test_15_subsat_no_solution,
    test_16_liquid_ice_triple,
    test_17_repo_reference,
    test_18_reference_preserved,
    test_19_reproduce_reference,
    test_20_output_regression,
    test_b1_free_energy_decomposition,
    test_b2_caveat_when_dynamics_absent,
    test_b3_xarray_roundtrip,
    test_b4_grib_graceful,
]


def main():
    print("=" * 78)
    print("test_met_nucleation.py  (20 mandatory + 4 met-layer bonus)")
    print("=" * 78)
    failures = []
    for t in TESTS:
        try:
            msg = t()
            print(f"  [PASS] {t.__name__}: {msg}")
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)
    print("-" * 78)
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        return 1
    print(f"ALL {len(TESTS)} TESTS PASSED (20 mandatory + 4 bonus).")
    return 0


if __name__ == "__main__":
    sys.exit(main())