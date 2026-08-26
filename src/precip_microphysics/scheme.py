"""Bulk-microphysics orchestrator.

:class:`BulkMicrophysics` applies the nucleation embryo source and the ordered
process list (:data:`processes.PROCESS_ORDER`) to a :class:`MicrophysicsState`
for one time step, conserving total water and accumulating latent heating.

Conservation is structural: every process returns mass *transfers* (src -> dst),
and :meth:`_apply` moves mass (re-capped to what is actually available at
application time, which also makes the operator-split ordering safe) and updates
the temperature from the phase-rank change:

    vapour(0) -> liquid(1) : +L_v/c_p   (condensation, warming)
    vapour(0) -> ice(2)    : +L_s/c_p   (deposition,   warming)
    liquid(1) -> ice(2)    : +L_f/c_p   (freezing,     warming)

and the reverse transfers cool.  The returned budget carries the per-process
mass moved, the net latent heating, and the total-water error (which should be
at round-off since only transfers and the booked clip act on the state).
"""
from __future__ import annotations

import numpy as np

from . import constants as C
from . import nucleation_source as ns
from . import processes as proc

# phase rank: vapour < liquid < ice
_RANK = {"qv": 0, "qc": 1, "qr": 1, "qi": 2, "qs": 2, "qg": 2, "qh": 2}
_LKIND = {frozenset({0, 1}): ("vapor_liquid", C.Lv),
          frozenset({0, 2}): ("vapor_ice", C.Ls),
          frozenset({1, 2}): ("liquid_ice", C.Lf)}


class BulkMicrophysics:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ---- apply one transfer with conservation + latent heat ----
    def _apply(self, st, tr):
        xp = st.xp
        src = xp.asarray(getattr(st, tr.src), dtype=float)
        dq = xp.clip(xp.asarray(tr.dq, dtype=float), 0.0, None)
        dq = xp.minimum(dq, xp.maximum(src, 0.0))          # re-cap at apply time
        if not xp.any(dq > 0):
            return 0.0
        setattr(st, tr.src, src - dq)
        setattr(st, tr.dst, xp.asarray(getattr(st, tr.dst), dtype=float) + dq)
        rs, rd = _RANK[tr.src], _RANK[tr.dst]
        if rs != rd:
            _, L = _LKIND[frozenset({rs, rd})]
            sign = 1.0 if rd > rs else -1.0                # denser => warming
            st.T = xp.asarray(st.T, dtype=float) + sign * (L / C.cp_d) * dq
            self._latent += float(xp.sum(sign * (L / C.cp_d) * dq))
        return float(xp.sum(dq))

    def step(self, st, dt, cell_volume=None, J_liquid=None, J_ice=None):
        """Advance the microphysical state by ``dt`` (no sedimentation here;
        that is a separate column operator).  Returns a budget dict."""
        cfg = self.cfg
        if cell_volume is None:
            cell_volume = float(st.dz.mean()) ** 3 if np.ndim(st.dz) else st.dz ** 3
        water0 = st.water_path()
        self._latent = 0.0
        budget = {}

        # 1. nucleation embryo source (favourability from the kernel)
        src_transfers, nuc_diag = ns.embryo_source(
            st, cfg, dt, cell_volume, J_liquid=J_liquid, J_ice=J_ice, rng=self.rng)
        for tr in src_transfers:
            moved = self._apply(st, tr)
            budget[tr.name] = budget.get(tr.name, 0.0) + moved

        # 2. growth / conversion processes (ordered operator split)
        for fn in proc.PROCESS_ORDER:
            for tr in fn(st, cfg, dt):
                moved = self._apply(st, tr)
                budget[tr.name] = budget.get(tr.name, 0.0) + moved

        st.clip()
        water1 = st.water_path()
        denom = max(abs(water0), C.TINY)
        budget["_latent_heating_J_per_kg_air"] = self._latent
        budget["_water_before"] = water0
        budget["_water_after"] = water1
        budget["_water_rel_err"] = (water1 - water0) / denom
        budget["_clip_loss"] = st.clip_loss
        budget["_nucleation"] = nuc_diag
        return budget


__all__ = ["BulkMicrophysics"]
