"""Standalone parcel / column driver.

Ties the pipeline together for an externally supplied meteorological state (a
single parcel, 0-D) or vertical column, without needing the 3D flow solver:

    meteorological state (T, P, humidity, w, ...)
      -> second-order vapour-liquid / vapour-ice nucleation rate (kernel)
      -> embryo source (q_c / q_i)
      -> bulk microphysical growth & conversion (scheme)
      -> sedimentation & surface flux
      -> evidence-based precipitation diagnostics.

Setting ``microphysics_enabled=False`` runs *thermodynamics only*: the
nucleation favourability is still evaluated but no hydrometeors are grown, so
every precipitation category is reported at Level 1 with the caveat -- exactly
the behaviour the honesty guard requires when growth is unresolved.

The validated nucleation kernel is used **read-only** via
``import met_water_nucleation as M`` (never modified).
"""
from __future__ import annotations

import numpy as np

from . import constants as C
from . import diagnostics as dg
from . import sedimentation as sed
from . import thermo as th
from .config import MicrophysicsConfig
from .scheme import BulkMicrophysics
from .state import MicrophysicsState

_LOG10I_CAP = 300.0     # avoid 10**overflow (rate is vapour-limited anyway)


def _kernel_rates(T, P, pv, grad_T=1.0):
    """Second-order nucleation rate J [m^-3 s^-1] for liquid and ice at a
    single state, from the validated kernel (read-only).  Returns
    (J_liquid, J_ice, log10I_liquid, log10I_ice); NaN where not solved."""
    import met_water_nucleation as M
    atm = M.un.AtmosphericInput(theta=np.pi, mode="homogeneous",
                                phase_mode="both", scenario="single_state")
    sim = M.un.UnifiedNucleationSimulator(atm)
    res = sim.evaluate_point(float(T), float(P), float(pv),
                             r_ref=1.0e-7, grad_T_req=float(grad_T))

    def _rate(ph):
        r = res.get(ph)
        if r is None:
            return np.nan, np.nan
        l = getattr(r, "log10I", None)
        if l is None or not np.isfinite(l):
            return np.nan, np.nan
        return 10.0 ** min(float(l), _LOG10I_CAP), float(l)

    Jl, Ll = _rate("liquid")
    Ji, Li = _rate("ice")
    return Jl, Ji, Ll, Li


class ColumnModel:
    def __init__(self, micro_cfg: MicrophysicsConfig | None = None):
        self.cfg = micro_cfg or MicrophysicsConfig()
        self.scheme = BulkMicrophysics(self.cfg)

    # ------------------------------------------------------------------
    def run_parcel(self, T, P, *, RH=None, qv=None, rh_reference="water",
                   w=None, dz=1000.0, duration=600.0, dt=5.0,
                   freezing_level=None, residence_time=None, cloud_depth=None,
                   ascent=None, microphysics_enabled=True, use_kernel=True):
        """Evolve a single air parcel and return its diagnostics.

        Provide humidity as ``RH`` [%] or ``qv`` [kg/kg].  ``w`` is the updraft
        [m/s] (``None`` => not supplied, lowers hail confidence).  ``dz`` is the
        parcel depth for sedimentation [m].  When ``ascent`` is enabled (default:
        whenever ``w > 0``) the parcel is lifted adiabatically at ``w`` each
        step, cooling it and continuously generating the supersaturation that
        drives condensation -> collision-coalescence -> rain; the condensation
        latent heat then yields the moist adiabat automatically.
        """
        do_ascent = (ascent if ascent is not None else (w is not None and w > 0.0)) \
            and microphysics_enabled
        if qv is None:
            if RH is None:
                raise ValueError("provide RH [%] or qv [kg/kg]")
            es = th.psat_water(T) if rh_reference == "water" else th.psat_ice(T)
            pv0 = (RH / 100.0) * es
            qv = float(th.qv_from_pv(pv0, P))
        rho = float(P / (C.R_d * T * (1.0 + 0.61 * qv)))
        w_val = 0.0 if w is None else float(w)

        st = MicrophysicsState(T=T, P=P, rho=rho, w=w_val, dz=dz,
                               freezing_level=freezing_level, qv=qv)

        # nucleation favourability (kernel, read-only)
        pv = float(th.p_v_from_qv(qv, P))
        Jl = Ji = None
        Ll = Li = float("nan")
        if use_kernel:
            try:
                Jl, Ji, Ll, Li = _kernel_rates(T, P, pv)
            except Exception:
                Jl = Ji = None

        env = {
            "Sw": float(th.saturation_ratio_water(qv, T, P)),
            "Si": float(th.saturation_ratio_ice(qv, T, P)),
            "wmax": w_val, "rho": rho,
            "cloud_depth": cloud_depth, "residence_time": residence_time,
            "freezing_level": freezing_level,
        }
        provenance = {
            "w_supplied": w is not None,
            "dz_supplied": True,
            "residence_time_supplied": residence_time is not None,
            "freezing_level_supplied": freezing_level is not None,
        }

        cum, max_relerr, last_nuc, env_dyn = self._integrate(
            st, dt, duration, Jl, Ji, microphysics_enabled, do_ascent, w_val)
        cum["_water_rel_err"] = max_relerr
        # favourability reflects the PEAK supersaturation reached during ascent
        env["Sw"] = max(env["Sw"], env_dyn.get("Sw_max", env["Sw"]))
        env["Si"] = max(env["Si"], env_dyn.get("Si_max", env["Si"]))
        if cloud_depth is None and env_dyn.get("z_top", 0.0) > 0.0:
            env["cloud_depth"] = env_dyn["z_top"]
        surface_flux = {c: st.surface_flux.get(c, 0.0) for c in ("rain", "snow", "graupel", "hail")}

        total_t = max(duration, dt)
        diag = dg.diagnose(st, cum, surface_flux, self.cfg, env=env,
                           provenance=provenance,
                           microphysics_enabled=microphysics_enabled, dt=total_t)
        diag["nucleation"] = {
            "log10I_liquid": Ll, "log10I_ice": Li,
            "kernel_used": bool(use_kernel and Jl is not None),
            "N_expected": last_nuc,
        }
        diag["final_state"] = {s: float(np.asarray(getattr(st, s)).max())
                               for s in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")}
        return diag

    # ------------------------------------------------------------------
    def _integrate(self, st, dt, duration, Jl, Ji, enabled, do_ascent, w):
        """Time-integrate; return (cumulative budget, max |water rel err|,
        last nucleation diag, dynamic-env maxima)."""
        cum: dict = {}
        max_relerr = 0.0
        last_nuc: dict = {}
        env_dyn = {"Sw_max": 0.0, "Si_max": 0.0, "z_top": 0.0}
        if not enabled:
            return cum, max_relerr, last_nuc, env_dyn
        cellvol = float(np.asarray(st.dz).mean()) ** 3 if np.ndim(st.dz) else float(st.dz) ** 3
        n = max(1, int(round(duration / dt)))
        z = 0.0
        for _ in range(n):
            if do_ascent and w > 0.0:
                dz_step = w * dt
                # dry-adiabatic cooling on ascent; latent heat from the ensuing
                # condensation (in scheme.step) restores the moist adiabat.
                Ttot = np.asarray(st.T, dtype=float)
                rho = np.asarray(st.rho, dtype=float)
                st.P = np.asarray(st.P, dtype=float) - rho * C.g0 * dz_step
                st.T = Ttot - (C.g0 / C.cp_d) * dz_step
                st.rho = st.P / (C.R_d * st.T * (1.0 + 0.61 * np.asarray(st.qv, dtype=float)))
                z += dz_step
                env_dyn["z_top"] = z
            # peak supersaturation (pre-condensation) for favourability
            env_dyn["Sw_max"] = max(env_dyn["Sw_max"],
                                    float(np.max(th.saturation_ratio_water(st.qv, st.T, st.P))))
            env_dyn["Si_max"] = max(env_dyn["Si_max"],
                                    float(np.max(th.saturation_ratio_ice(st.qv, st.T, st.P))))
            budget = self.scheme.step(st, dt, cell_volume=cellvol,
                                      J_liquid=Jl, J_ice=Ji)
            sed.sediment(st, self.cfg, dt)
            for k, v in budget.items():
                if k.startswith("_"):
                    continue
                cum[k] = cum.get(k, 0.0) + float(v)
            max_relerr = max(max_relerr, abs(budget.get("_water_rel_err", 0.0)))
            last_nuc = budget.get("_nucleation", {})
        return cum, max_relerr, last_nuc, env_dyn


__all__ = ["ColumnModel"]
