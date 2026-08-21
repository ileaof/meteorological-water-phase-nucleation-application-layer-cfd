#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_met_frontal_collision.py
================================
Scenario: a **warm, moist** air mass collides with a **cold, dry** air mass
(a classic front / mixing-fog situation) -> meteorological nucleation report.

The tool evaluates a single thermodynamic *state*, so a "collision" is encoded
as the parcel that the mixing produces.  Two effects are modelled here:

  1. ISOBARIC MIXING (Rogers & Yau mixing-cloud construction).
     Mix temperature and vapour pressure linearly between the two air masses:
         T_mix   = f*T_warm  + (1-f)*T_cold
         p_v_mix = f*pv_warm + (1-f)*pv_cold      (f = warm-air mass fraction)
     Because e_sat(T) is convex, the straight mixing line can rise ABOVE the
     saturation curve: the mixture is supersaturated (S_water > 1) even when
     BOTH parents were subsaturated.  That is the frontal cloud / mixing fog.
     We scan f in [0,1] and pick the most supersaturated mixture.

  2. FORCED ASCENT of the warm air over the cold wedge -> adiabatic cooling,
     fed in as `w` (updraft) and `cooling_rate = -w * lapse`, plus a modest
     LWC/IWC and CCN/INP population for the mixed-phase cloud.

Then `evaluate_point` returns the full nucleation diagnosis for both phases
(supersaturations, critical radii, log10 nucleation rate, precip class).

NOTE: this script needs the full environment (the physics core
`unified_h2o_nucleation_climate` and `het_contact_angle` alongside this module).
Edit the two air-mass blocks below to set YOUR scenario.
"""
import io
import os
import sys

import numpy as np

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

# Bootstrap the package import when running from a source checkout without
# `pip install -e .` (path resolved relative to this file, not to CWD).
try:
    import met_water_nucleation as M
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    import met_water_nucleation as M

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "frontal_collision")
os.makedirs(OUTDIR, exist_ok=True)

Sat = M.SaturationProperties


# ---------------------------------------------------------------------------
#  1. Define the two colliding air masses  (edit these for your scenario)
# ---------------------------------------------------------------------------
P = 90000.0            # near-surface frontal pressure [Pa] (~900 hPa)

# WARM, MOIST air mass (e.g. subtropical warm sector)
T_warm, RH_warm = 293.15, 95.0     # 20 C, 95% RH wrt water

# COLD, DRY air mass (e.g. polar / continental outbreak)
T_cold, RH_cold = 268.15, 40.0     # -5 C, 40% RH wrt water

pv_warm = Sat.RH_to_p_v(RH_warm, T_warm, "water")
pv_cold = Sat.RH_to_p_v(RH_cold, T_cold, "water")


# ---------------------------------------------------------------------------
#  2. Isobaric mixing: find the most supersaturated mixture
# ---------------------------------------------------------------------------
f = np.linspace(0.0, 1.0, 501)                 # warm-air mass fraction
T_mix = f * T_warm + (1.0 - f) * T_cold
pv_mix = f * pv_warm + (1.0 - f) * pv_cold
esat_w = np.array([Sat.Psat_water(t, extended=True) for t in T_mix])
S_mix = pv_mix / esat_w                         # supersaturation wrt water

k = int(np.argmax(S_mix))                        # peak-supersaturation mixture
Tk, pvk, fk, Sk = float(T_mix[k]), float(pv_mix[k]), float(f[k]), float(S_mix[k])

print("=" * 78)
print("WARM-MOIST  x  COLD-DRY  AIR-MASS COLLISION  (isobaric mixing)")
print("=" * 78)
print(f"  warm mass : T={T_warm:6.2f} K  RH={RH_warm:5.1f}%  p_v={pv_warm:8.2f} Pa")
print(f"  cold mass : T={T_cold:6.2f} K  RH={RH_cold:5.1f}%  p_v={pv_cold:8.2f} Pa")
print(f"  peak-S mix: f_warm={fk:4.2f}  T={Tk:6.2f} K  p_v={pvk:8.2f} Pa  "
      f"S_water={Sk:5.3f}  ({'SUPERSATURATED' if Sk > 1 else 'subsaturated'})")
if Sk <= 1.0:
    print("  --> mixing did not reach saturation; make the warm mass moister/warmer.")


# ---------------------------------------------------------------------------
#  3. Forced ascent of the warm air over the cold wedge + microphysics
# ---------------------------------------------------------------------------
w = 1.5                                  # updraft over the frontal surface [m/s]
lapse = 6.5e-3                            # environmental lapse [K/m]
cooling_rate = -w * lapse                 # dT/dt from ascent [K/s]
LWC = 5.0e-4 if Tk <= 273.15 else 8.0e-4  # supercooled liquid if below 0 C
IWC = 1.0e-4 if Tk <= 273.15 else 0.0

met = M.MetInput(
    T=Tk, P=P, p_v=pvk,                   # the mixed, supersaturated parcel
    phase_mode="both", mode="homogeneous",
    r_ref=M.R_REF_DEFAULT,
    w=w, LWC=LWC, IWC=IWC, cooling_rate=cooling_rate,
    N_ccn=3.0e8, N_inp=1.0e4,
    dt_micro=60.0, cell_volume=1.0e6,     # so expected_events is determined
)
runner = M.MetNucleationRunner(met)
pv_res, src, warns = M.resolve_humidity(met, Tk, P)
if warns:
    print("  humidity warnings:", warns)

reps = runner.evaluate_point(Tk, P, pv_res, dynamics={
    "w": w, "LWC": LWC, "IWC": IWC, "cooling_rate": cooling_rate,
    "N_ccn": 3.0e8, "N_inp": 1.0e4})

print("-" * 78)
print("NUCLEATION DIAGNOSIS AT THE MIXED (FRONTAL-CLOUD) STATE")
print("-" * 78)
for ph, r in reps.items():
    print(f"\n--- phase = {ph}  (status={r.status}, mode={r.nucleation_mode}) ---")
    print(f"  T_ambient        = {r.T_ambient_K:.2f} K")
    print(f"  S_water / S_ice  = {r.S_water:.3f} / {r.S_ice:.3f}")
    print(f"  gradT (solved)   = {r.gradT_K_m:.4e} K/m")
    print(f"  r_critical_2nd   = {r.r_critical_2nd_m:.4e} m")
    print(f"  log10 I          = {r.log10_nucleation_rate:.3f}")
    print(f"  diagnostic_class = {r.diagnostic_class}")
    print(f"  rain/snow/graupel/hail fav = "
          f"{r.rain_favorability:.2f} / {r.snow_favorability:.2f} / "
          f"{r.graupel_favorability:.2f} / {r.hail_favorability:.2f}")

# save the report
M.to_json(reps, os.path.join(OUTDIR, "frontal_collision_report.json"))
M.to_csv(reps, os.path.join(OUTDIR, "frontal_collision_report.csv"))
print(f"\nSaved: {os.path.join(OUTDIR, 'frontal_collision_report.json')} (+ .csv)")

if __name__ == "__main__":
    sys.exit(0)
