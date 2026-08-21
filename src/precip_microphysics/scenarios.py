"""Four reference scenarios demonstrating that nucleation and precipitation are
distinct physical stages.

    1. high_nucleation_no_microphysics -- thermodynamic favourability only;
    2. warm_rain                       -- condensation -> collision-coalescence
                                          -> rain sedimentation -> surface flux;
    3. mixed_phase                     -- ice deposition -> snow aggregation
                                          (+ some riming to graupel);
    4. deep_convective_hail            -- a supercell updraft core (sustained
                                          supercooled liquid + graupel embryos)
                                          growing hail, then descent, melting and
                                          a surface-survival assessment.

Each returns the full diagnostic dict.  Scenario 4 is an idealisation: a 0-D
parcel cannot self-consistently sustain supercooled liquid water against the
Bergeron process, so the supercell core condition (continuous condensate supply
in a strong updraft) is imposed -- exactly the regime real hail requires and
which distinguishes it from ordinary cold-cloud precipitation.
"""
from __future__ import annotations

import numpy as np

from . import constants as C
from . import diagnostics as dg
from . import sedimentation as sed
from . import thermo as th
from .column import ColumnModel, _kernel_rates
from .config import MicrophysicsConfig
from .scheme import BulkMicrophysics
from .state import MicrophysicsState


def high_nucleation_no_microphysics(use_kernel=True):
    """Strongly supersaturated, very high nucleation rate, but growth unresolved."""
    col = ColumnModel(MicrophysicsConfig())
    return col.run_parcel(T=260.0, P=70000.0, RH=115.0, w=None,
                          microphysics_enabled=False, use_kernel=use_kernel)


def warm_rain(use_kernel=False):
    """Warm maritime cloud: gentle updraft, condensation, warm-rain coalescence,
    rain falling to the surface."""
    col = ColumnModel(MicrophysicsConfig())
    return col.run_parcel(T=291.0, P=95000.0, RH=100.5, w=1.2, dz=1600.0,
                          duration=1800.0, dt=5.0, use_kernel=use_kernel)


def mixed_phase(use_kernel=False):
    """Mixed-phase cloud: cold updraft, ice nucleation, vapour deposition and
    snow aggregation; snow reaches the surface."""
    col = ColumnModel(MicrophysicsConfig())
    return col.run_parcel(T=264.0, P=68000.0, RH=112.0, w=2.0, dz=2500.0,
                          duration=1500.0, dt=5.0, freezing_level=1500.0,
                          use_kernel=use_kernel)


def deep_convective_hail(use_kernel=False):
    """Deep convective hail: a supercell updraft core sustaining supercooled
    liquid water and graupel embryos, growing hail; then descent below the
    freezing level with melting and a surface-survival assessment."""
    cfg = MicrophysicsConfig()
    mp = BulkMicrophysics(cfg)
    T0, P0 = 258.0, 60000.0
    qv0 = float(th.qsat_water(T0, P0) * 1.05)
    rho0 = P0 / (C.R_d * T0 * (1.0 + 0.61 * qv0))
    st = MicrophysicsState(T=T0, P=P0, rho=rho0, w=35.0, dz=4000.0,
                           freezing_level=3000.0, qv=qv0, qc=3.0e-3, qg=2.0e-3)

    Jl = Ji = None
    Ll = Li = float("nan")
    if use_kernel:
        try:
            pv = float(th.p_v_from_qv(qv0, P0))
            Jl, Ji, Ll, Li = _kernel_rates(T0, P0, pv)
        except Exception:
            Jl = Ji = None

    cum, max_relerr, dt = {}, 0.0, 5.0
    Sw_pk = Si_pk = 0.0
    growth_steps, growth_dt = 24, 5.0     # ~120 s in the core
    for _ in range(growth_steps):
        st.qc = np.asarray(3.0e-3)        # steady supercooled-water supply in the core
        Sw_pk = max(Sw_pk, float(np.max(th.saturation_ratio_water(st.qv, st.T, st.P))))
        Si_pk = max(Si_pk, float(np.max(th.saturation_ratio_ice(st.qv, st.T, st.P))))
        b = mp.step(st, growth_dt, cell_volume=st.dz ** 3, J_liquid=Jl, J_ice=Ji)
        _accumulate(cum, b)
        max_relerr = max(max_relerr, abs(b.get("_water_rel_err", 0.0)))
    peak = st.copy()                       # snapshot with hail aloft + SLW evidence
    residence_time = growth_steps * growth_dt

    # descent below the freezing level: melt + sediment
    st.T = np.asarray(283.0)
    st.qc = np.asarray(0.0)
    for _ in range(40):
        b = mp.step(st, dt, cell_volume=st.dz ** 3, J_liquid=None, J_ice=None)
        sed.sediment(st, cfg, dt)
        _accumulate(cum, b)
        max_relerr = max(max_relerr, abs(b.get("_water_rel_err", 0.0)))

    # diagnose on the growth-phase state (carries SLW/updraft/subfreezing
    # evidence) but with the surface flux + accumulation from the descent phase.
    peak.surface_flux = dict(st.surface_flux)
    peak.accumulation = dict(st.accumulation)
    cum["_water_rel_err"] = max_relerr
    env = {"Sw": Sw_pk, "Si": Si_pk, "wmax": 35.0, "rho": float(peak.rho),
           "cloud_depth": 4000.0, "residence_time": residence_time,
           "freezing_level": 3000.0}
    prov = {"w_supplied": True, "dz_supplied": True,
            "residence_time_supplied": True, "freezing_level_supplied": True}
    total_t = residence_time + 40 * dt
    diag = dg.diagnose(peak, cum, peak.surface_flux, cfg, env=env, provenance=prov,
                       microphysics_enabled=True, dt=total_t)
    diag["nucleation"] = {"log10I_liquid": Ll, "log10I_ice": Li,
                          "kernel_used": bool(use_kernel and Jl is not None)}
    diag["final_state"] = {s: float(np.asarray(getattr(st, s)).max())
                           for s in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")}
    return diag


def _accumulate(cum, budget):
    for k, v in budget.items():
        if not k.startswith("_"):
            cum[k] = cum.get(k, 0.0) + float(v)


def run_all(use_kernel=False):
    return {
        "1_high_nucleation_no_microphysics": high_nucleation_no_microphysics(use_kernel),
        "2_warm_rain": warm_rain(use_kernel),
        "3_mixed_phase": mixed_phase(use_kernel),
        "4_deep_convective_hail": deep_convective_hail(use_kernel),
    }


__all__ = ["high_nucleation_no_microphysics", "warm_rain", "mixed_phase",
           "deep_convective_hail", "run_all"]
