"""Severe convective storm producing ~100 mm of rain and hail.

A single closed air parcel cannot rain out 100 mm -- that much water must be
*supplied* by the storm's moisture convergence (mesoscale inflow feeding the
updraft).  This example therefore models a severe multicell/supercell storm as a
steady-state processor with two coupled cores, the standard conceptual model of
a hailstorm:

  * a WARM heavy-rain core (cloud base ~ 293 K) fed by continuous moisture
    convergence -- condensation -> warm-rain coalescence -> heavy surface rain;
  * a COLD supercell hail-growth core (sustained supercooled liquid water +
    graupel embryos in a strong updraft) -- hail grows aloft, then descends
    through the 0 degC level, partly melts (adding to the rain) and the survivors
    reach the ground.

The combined surface accumulation is ~100 mm (rain + hail).  The physics, the
mass conservation and the evidence-based diagnostics are exactly those of
``precip_microphysics``; only the boundary forcing (moisture supply, supercell
core) represents the storm-scale environment a 0-D column cannot self-generate.

    python examples/heavy_rain_hail_scenario.py [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from precip_microphysics import constants as C          # noqa: E402
from precip_microphysics import diagnostics as dg       # noqa: E402
from precip_microphysics import sedimentation as sed    # noqa: E402
from precip_microphysics import thermo as th            # noqa: E402
from precip_microphysics.config import MicrophysicsConfig  # noqa: E402
from precip_microphysics.scheme import BulkMicrophysics    # noqa: E402
from precip_microphysics.state import MicrophysicsState    # noqa: E402


# ---------------------------------------------------------------------------
# warm heavy-rain core (moisture-convergence fed, steady state)
# ---------------------------------------------------------------------------
def simulate_rain_core(duration=3600.0, dt=5.0, T=293.0, P=92000.0,
                       storm_depth=5000.0, updraft=12.0,
                       moisture_supply=2.7e-6):
    """Return (state, cum_budget, env, provenance).

    ``moisture_supply`` [kg/kg/s] is the vapour convergence the updraft feeds
    into the cloud each second (mesoscale inflow); in steady state essentially
    all of it precipitates, so it sets the rain rate.
    """
    cfg = MicrophysicsConfig()
    mp = BulkMicrophysics(cfg)
    qv0 = float(th.qsat_water(T, P) * 1.02)
    rho = P / (C.R_d * T * (1.0 + 0.61 * qv0))
    st = MicrophysicsState(T=T, P=P, rho=rho, w=updraft, dz=storm_depth,
                           qv=qv0, qc=1.0e-3)
    cum, max_relerr = {}, 0.0
    Sw_pk = 0.0
    n = int(round(duration / dt))
    for _ in range(n):
        # continuous moisture convergence into the storm updraft
        st.qv = np.asarray(st.qv, dtype=float) + moisture_supply * dt
        # steady-state isothermal cloud base: ascent cooling balances the
        # condensation latent heating on the moist adiabat, so the core stays at
        # its cloud-base temperature (documented steady-state assumption).
        st.T = np.asarray(T, dtype=float)
        Sw_pk = max(Sw_pk, float(np.max(th.saturation_ratio_water(st.qv, st.T, st.P))))
        b = mp.step(st, dt, cell_volume=storm_depth ** 3, J_liquid=1e40, J_ice=None)
        sed.sediment(st, cfg, dt)
        for k, v in b.items():
            if not k.startswith("_"):
                cum[k] = cum.get(k, 0.0) + float(v)
        max_relerr = max(max_relerr, abs(b.get("_water_rel_err", 0.0)))
    cum["_water_rel_err"] = max_relerr
    env = {"Sw": Sw_pk, "Si": 0.0, "wmax": updraft, "rho": float(rho),
           "cloud_depth": storm_depth, "residence_time": storm_depth / max(updraft, 1e-3),
           "freezing_level": None}
    prov = {"w_supplied": True, "dz_supplied": True,
            "residence_time_supplied": True, "freezing_level_supplied": False}
    return st, cum, env, prov


# ---------------------------------------------------------------------------
# supercell hail core (growth aloft -> descent -> melt/survival)
# ---------------------------------------------------------------------------
def simulate_hail_core(growth_time=270.0, descent_time=400.0, dt=5.0,
                       T_core=255.0, P_core=45000.0, storm_depth=6000.0,
                       updraft=45.0, slw_reservoir=3.0e-3, graupel_seed=2.0e-3):
    """Return (peak_state, cum_budget, surface_flux, accumulation, env, prov).

    A strong, deep supercell updraft recirculates embryos and maintains a
    supercooled-liquid reservoir ``slw_reservoir`` [kg/kg] for ``growth_time``
    seconds -- the regime that grows hail.  A short, bounded growth window keeps
    the total processed water physical.  The hail then falls through the melting
    level: most reaches the ground as rain (melted graupel/hail) and a few mm
    survive as hail -- the realistic ~1:40 hail:rain ratio.
    """
    cfg = MicrophysicsConfig()
    mp = BulkMicrophysics(cfg)
    qv0 = float(th.qsat_water(T_core, P_core) * 1.05)
    rho = P_core / (C.R_d * T_core * (1.0 + 0.61 * qv0))
    st = MicrophysicsState(T=T_core, P=P_core, rho=rho, w=updraft, dz=storm_depth,
                           freezing_level=3200.0, qv=qv0, qc=slw_reservoir, qg=graupel_seed)
    cum, max_relerr = {}, 0.0
    Sw_pk = Si_pk = 0.0
    for _ in range(int(round(growth_time / dt))):
        # sustained supercooled-water reservoir in the recirculating core; the
        # riming latent heat is allowed to warm the growing hail toward the
        # wet-growth regime (the short growth window keeps T supercooled).
        st.qc = np.asarray(slw_reservoir, dtype=float)
        Sw_pk = max(Sw_pk, float(np.max(th.saturation_ratio_water(st.qv, st.T, st.P))))
        Si_pk = max(Si_pk, float(np.max(th.saturation_ratio_ice(st.qv, st.T, st.P))))
        b = mp.step(st, dt, cell_volume=storm_depth ** 3, J_liquid=1e40, J_ice=1e40)
        _acc(cum, b)
        max_relerr = max(max_relerr, abs(b.get("_water_rel_err", 0.0)))
    peak = st.copy()
    residence_time = growth_time

    # descent below the freezing level: melting + sedimentation to the ground
    st.T = np.asarray(287.0)
    st.qc = np.asarray(0.0)
    for _ in range(int(round(descent_time / dt))):
        b = mp.step(st, dt, cell_volume=storm_depth ** 3, J_liquid=None, J_ice=None)
        sed.sediment(st, cfg, dt)
        _acc(cum, b)
        max_relerr = max(max_relerr, abs(b.get("_water_rel_err", 0.0)))
    cum["_water_rel_err"] = max_relerr

    peak.surface_flux = dict(st.surface_flux)
    peak.accumulation = dict(st.accumulation)
    env = {"Sw": Sw_pk, "Si": Si_pk, "wmax": updraft, "rho": float(peak.rho),
           "cloud_depth": storm_depth, "residence_time": residence_time,
           "freezing_level": 3200.0}
    prov = {"w_supplied": True, "dz_supplied": True,
            "residence_time_supplied": True, "freezing_level_supplied": True}
    return peak, cum, dict(st.surface_flux), dict(st.accumulation), env, prov


def _acc(cum, budget):
    for k, v in budget.items():
        if not k.startswith("_"):
            cum[k] = cum.get(k, 0.0) + float(v)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    # 1) warm heavy-rain core
    rain_st, rain_cum, rain_env, rain_prov = simulate_rain_core()
    rain_diag = dg.diagnose(rain_st, rain_cum, dict(rain_st.surface_flux),
                            MicrophysicsConfig(), env=rain_env, provenance=rain_prov,
                            microphysics_enabled=True, dt=3600.0)
    rain_mm = rain_st.accumulation.get("rain", 0.0)

    # 2) supercell hail core
    hail_peak, hail_cum, hail_sf, hail_acc, hail_env, hail_prov = simulate_hail_core()
    hail_diag = dg.diagnose(hail_peak, hail_cum, hail_sf, MicrophysicsConfig(),
                            env=hail_env, provenance=hail_prov,
                            microphysics_enabled=True, dt=1000.0)
    hail_mm = hail_acc.get("hail", 0.0)
    hail_rain_mm = hail_acc.get("rain", 0.0)   # rain from melted hail/graupel
    hail_graupel_mm = hail_acc.get("graupel", 0.0)

    total_rain = rain_mm + hail_rain_mm
    total_hail = hail_mm
    total = total_rain + total_hail + hail_graupel_mm

    r = next(c for c in rain_diag["categories"] if c["category"] == "rain")
    h = next(c for c in hail_diag["categories"] if c["category"] == "hail")

    print("SEVERE CONVECTIVE STORM  --  heavy rain + hail")
    print("=" * 64)
    print(f"{'component':<26}{'accum (mm)':>12}{'level':>7}{'confirmed':>11}")
    print("-" * 64)
    print(f"{'rain (warm core)':<26}{rain_mm:>12.1f}{r['diagnostic_level']:>7}"
          f"{str(r['confirmed']):>11}")
    print(f"{'rain (melted hail/graupel)':<26}{hail_rain_mm:>12.1f}{'-':>7}{'-':>11}")
    print(f"{'hail (surface)':<26}{hail_mm:>12.1f}{h['diagnostic_level']:>7}"
          f"{str(h['confirmed']):>11}")
    print(f"{'graupel (surface)':<26}{hail_graupel_mm:>12.1f}{'-':>7}{'-':>11}")
    print("-" * 64)
    print(f"{'TOTAL rain':<26}{total_rain:>12.1f}")
    print(f"{'TOTAL hail':<26}{total_hail:>12.1f}")
    print(f"{'TOTAL precipitation':<26}{total:>12.1f}   (target ~100 mm)")

    print("\nHail detail (supercell core):")
    for k in ("growth_regime", "max_diameter_m", "melting_fraction",
              "surface_survival_probability", "max_updraft_m_s"):
        print(f"    {k:<30} {h.get(k)}")
    print(f"\nRain diagnostic level : {r['diagnostic_level_name']}  "
          f"confidence={r['confidence']}  confirmed={r['confirmed']}")
    print(f"Hail diagnostic level : {h['diagnostic_level_name']}  "
          f"confidence={h['confidence']}  confirmed={h['confirmed']}")
    print(f"Water conservation    : rain core rel.err={rain_diag['overall']['water_rel_err']:.2e}, "
          f"hail core rel.err={hail_diag['overall']['water_rel_err']:.2e}")

    if args.json:
        out = {"rain_core": rain_diag, "hail_core": hail_diag,
               "totals_mm": {"rain": total_rain, "hail": total_hail, "all": total}}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\nWritten {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
