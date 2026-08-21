"""Two-way coupling of the bulk microphysics into the 3D flow (Increment 2).

The validated ``precip_microphysics`` scheme is applied to the whole grid at once
(its process rates are numpy-vectorised, so a :class:`MicrophysicsState` whose
fields are 3-D arrays steps exactly like the 0-D/1-D case).  Each flow step, over
the resolved circulation:

1. **growth & conversion + embryo source** — build a microphysics state from the
   flow's diagnosed ``T, P, rho, w`` and its mixing ratios, run one scheme step
   (condensation/deposition, collision-coalescence, riming, aggregation,
   freezing/melting, hail growth), then write the updated mixing ratios back;
2. **latent-heat feedback** — the scheme updates the actual temperature ``T``;
   that heating is baked back into the transported potential temperature
   ``theta = T (P0/P)^(R_d/c_p)`` so buoyancy feels it on the next step;
3. **sedimentation** — rain/snow/graupel/hail fall along ``-z`` (upwind column
   flux on the last grid axis, index 0 = bottom), leaving the domain as surface
   precipitation accumulated per column [kg m^-2 == mm].

Transport of the precipitating fields is done by the flow's own conservative
advection (in :mod:`simulation`).  Water is conserved by construction: the scheme
only moves mass between species, and sedimentation is the sole sink (booked as
surface precipitation).  The embryo source is driven by the flow's own
supersaturation (CCN/IN activation); passing the kernel nucleation field is
supported for kernel-rate embryo counts.
"""
from __future__ import annotations

import numpy as np

from precip_microphysics import size_distributions as sd
from precip_microphysics.config import MicrophysicsConfig
from precip_microphysics.scheme import BulkMicrophysics
from precip_microphysics.state import MicrophysicsState

from . import thermodynamics as th

_CATS = (("rain", "qr"), ("snow", "qs"), ("graupel", "qg"), ("hail", "qh"))


class MicrophysicsCoupler:
    def __init__(self, micro_cfg: MicrophysicsConfig | None = None, max_dT: float = 8.0):
        self.cfg = micro_cfg or MicrophysicsConfig()
        self.scheme = BulkMicrophysics(self.cfg)
        self.max_dT = float(max_dT)   # max per-step latent heating [K] (stability)

    # ---- growth/conversion + embryo source + latent-heat feedback ----
    def apply(self, flow, grid, dt, nf=None) -> dict:
        """Run one microphysics step over the grid; update ``flow`` in place and
        return the process budget (mass moved per process + conservation)."""
        flow.ensure_hydrometeors()
        # cell-centre vertical velocity (updraft) for the hail gate
        Wc = 0.5 * (flow.w[:, :, :-1] + flow.w[:, :, 1:])
        micro = MicrophysicsState(
            T=flow.T.copy(), P=flow.P_total.copy(), rho=flow.rho.copy(),
            w=Wc, dz=float(grid.dz),
            qv=flow.qv.copy(), qc=flow.ql.copy(), qi=flow.qi.copy(),
            qr=flow.qr.copy(), qs=flow.qs.copy(), qg=flow.qg.copy(), qh=flow.qh.copy())
        Jl = None if nf is None else nf.I[0]
        Ji = None if nf is None else nf.I[1]
        budget = self.scheme.step(micro, dt, cell_volume=float(grid.cell_vol),
                                  J_liquid=Jl, J_ice=Ji)
        # write back mixing ratios (qc -> ql)
        flow.qv = np.asarray(micro.qv, dtype=float)
        flow.ql = np.asarray(micro.qc, dtype=float)
        flow.qi = np.asarray(micro.qi, dtype=float)
        flow.qr = np.asarray(micro.qr, dtype=float)
        flow.qs = np.asarray(micro.qs, dtype=float)
        flow.qg = np.asarray(micro.qg, dtype=float)
        flow.qh = np.asarray(micro.qh, dtype=float)
        # latent-heat feedback: bake the updated actual T back into theta.
        # The per-step heating is bounded (stability safeguard for the explicit
        # scheme in an under-resolved updraft core; only bites at extreme cells).
        T_new = np.asarray(micro.T, dtype=float)
        dT = np.clip(T_new - np.asarray(flow.T, dtype=float), -self.max_dT, self.max_dT)
        flow.theta = th.theta_from_T(np.asarray(flow.T, dtype=float) + dT,
                                     flow.P_total, th.P0_REF)
        return budget

    # ---- gravitational sedimentation (column fall along -z) ----
    def sediment(self, flow, grid, dt) -> dict:
        """Fall rain/snow/graupel/hail along -z; accumulate surface precip.
        Returns {category: domain-mean surface flux [kg m^-2 s^-1]}."""
        flow.ensure_hydrometeors()
        out = {}
        if not self.cfg.processes.sedimentation:
            return {c: 0.0 for c, _ in _CATS}
        rho = flow.rho
        dz = float(grid.dz)
        for cat, sp in _CATS:
            q = np.asarray(getattr(flow, sp), dtype=float)
            vt = sd.mass_weighted_vt(q, rho, cat)
            vmax = float(np.max(vt)) if vt.size else 0.0
            nsub = max(1, int(np.ceil(vmax * dt / max(dz, 1e-9))))
            dts = dt / nsub
            surf = flow.surface_precip[cat]           # 2-D (nx,ny), accumulates mm
            step_out = np.zeros_like(surf)
            for _ in range(nsub):
                vt = sd.mass_weighted_vt(q, rho, cat)
                F = rho * q * vt                       # downward flux at each cell
                F_above = np.zeros_like(F)
                F_above[:, :, :-1] = F[:, :, 1:]       # inflow from the cell above
                dq = (F_above - F) / (rho * dz) * dts
                q = np.maximum(q + dq, 0.0)
                step_out += F[:, :, 0] * dts           # bottom-face outflux [kg/m^2]
            setattr(flow, sp, q)
            surf += step_out                           # accumulate (mm)
            out[cat] = float(np.mean(step_out)) / max(dt, 1e-12)
        return out

    # ---- clean inflow: precipitating air enters the x-inflow faces empty ----
    @staticmethod
    def zero_inflow_hydrometeors(flow) -> None:
        for sp in ("qr", "qs", "qg", "qh"):
            a = getattr(flow, sp)
            if a is not None:
                a[0, :, :] = 0.0
                a[-1, :, :] = 0.0


__all__ = ["MicrophysicsCoupler"]
