"""Gravitational sedimentation of the precipitating categories.

Rain, snow, graupel and hail fall at their mass-weighted terminal velocities
(:func:`size_distributions.mass_weighted_vt`).  This is the operator that turns
"production aloft" into a "surface flux" -- the physical distinction the
diagnostics rely on (Level 3 vs Level 4).

Two geometries are supported by the same code:

* **0-D box** (scalar state, depth ``dz``): a cell loses mass at rate
  ``V_t q / dz``; the flux ``rho q V_t`` [kg m^-2 s^-1] leaves the lower face.
* **1-D column** (arrays of shape ``(nz,)``, index 0 = surface, increasing
  upward): first-order upwind downward flux,
  ``d(rho q dz)/dt = F_above - F_here`` with ``F_here = rho q V_t`` and the
  domain-bottom flux accumulated as surface precipitation.

An explicit sub-step keeps the fall Courant number ``V_t dt / dz <= 1``.
Accumulation is in mm of liquid water (1 kg m^-2 == 1 mm).
"""
from __future__ import annotations

import math

import numpy as np

from . import constants as C
from . import size_distributions as sd

_CATS = (("rain", "qr"), ("snow", "qs"), ("graupel", "qg"), ("hail", "qh"))


def _sediment_box(q, rho, vt, dz, dt):
    """0-D box of depth ``dz``: mass falls out of the lower face at flux
    ``rho q V_t``.  Returns (q_new, surface_flux_mean, mass_out)."""
    n = max(1, int(np.ceil(vt * dt / max(dz, C.TINY))))
    dts = dt / n
    q = float(q)
    mass_out = 0.0
    for _ in range(n):
        if q <= C.QSMALL:
            break
        flux = rho * q * vt                # kg m^-2 s^-1 (recomputed as q decays)
        dq = min(flux * dts / (rho * dz), q)
        q -= dq
        mass_out += rho * dq * dz          # kg m^-2 this sub-step ( = flux*dts )
    surf_flux = mass_out / dt              # mean over the step
    return q, surf_flux, mass_out


def _sediment_column(q, rho, dz, cat, dt, xp=np):
    """1-D/N-D column upwind sedimentation. Returns (q_new, surface_flux, mass_out)."""
    q = xp.array(q, dtype=float)
    rho = xp.asarray(rho, dtype=float)
    dz = xp.asarray(dz, dtype=float) if np.ndim(dz) else xp.full_like(q, float(dz))
    vt = sd.mass_weighted_vt(q, rho, cat, xp)
    vmax = float(xp.max(vt)) if vt.size else 0.0
    dzmin = float(xp.min(dz))
    n = max(1, int(math.ceil(vmax * dt / max(dzmin, C.TINY))))
    dts = dt / n
    mass_out = 0.0
    for _ in range(n):
        vt = sd.mass_weighted_vt(q, rho, cat, xp)
        F = rho * q * vt                    # downward flux at each cell [kg m^-2 s^-1]
        F_above = xp.empty_like(F)
        F_above[:-1] = F[1:]                # flux entering from the cell above
        F_above[-1] = 0.0                   # top: no inflow
        dq = (F_above - F) / (rho * dz) * dts
        q = xp.maximum(q + dq, 0.0)
        mass_out += float(F[0]) * dts       # domain-bottom (surface) outflux
    surf_flux = mass_out / dt
    return q, surf_flux, mass_out


def sediment(st, cfg, dt):
    """Apply sedimentation to all precip categories; update surface flux and
    accumulation on the state.  Returns {category: surface_flux_kg_m2_s}."""
    xp = st.xp
    out = {}
    if not cfg.processes.sedimentation:
        for _, sp in _CATS:
            out[sp] = 0.0
        return out
    column = st.T.ndim >= 1
    for cat, sp in _CATS:
        q = getattr(st, sp)
        if column:
            q_new, surf, mout = _sediment_column(q, st.rho, st.dz, cat, dt, xp)
        else:
            vt = float(sd.mass_weighted_vt(q, st.rho, cat, xp))
            q_new, surf, mout = _sediment_box(float(q), float(st.rho), vt,
                                              float(st.dz), dt)
            q_new = xp.asarray(q_new, dtype=float)
        setattr(st, sp, xp.asarray(q_new, dtype=float))
        st.surface_flux[cat] = surf
        st.accumulation[cat] = st.accumulation.get(cat, 0.0) + mout   # kg/m^2 == mm
        out[cat] = surf
    return out


__all__ = ["sediment"]
