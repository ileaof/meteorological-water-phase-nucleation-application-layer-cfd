"""Prognostic microphysical state (framework-agnostic).

A :class:`MicrophysicsState` holds the seven water mixing ratios and the local
environment for a single air parcel (0-D) or a vertical column (1-D numpy
arrays of shape ``(nz,)``).  The same object is used by the standalone column
driver and, in Increment 2, by the 3D flow coupling -- it carries no dependence
on either framework.

Mixing ratios are mass of the water species per mass of moist air [kg/kg],
matching the convention of ``meteorological_flow.thermodynamics`` (so
``p_v = q_v P / (eps + (1-eps) q_v)``).  Number concentrations are per cubic
metre [m^-3]; when ``None`` they are diagnosed from the mixing ratio and the
assumed Marshall-Palmer distribution (single-moment closure).

All fields are kept non-negative; :meth:`clip` is the last-resort guard and its
mass loss is booked so conservation can be audited.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Union

import numpy as np

Array = Union[float, np.ndarray]

# canonical category order used everywhere (vapour first, then the six condensed)
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
CONDENSED = ("qc", "qr", "qi", "qs", "qg", "qh")
FROZEN = ("qi", "qs", "qg", "qh")
LIQUID = ("qc", "qr")
PRECIP = ("qr", "qs", "qg", "qh")            # sedimenting categories
NUMBER = ("Nc", "Nr", "Ni", "Ns", "Ng", "Nh")


@dataclass
class MicrophysicsState:
    # --- environment ---
    T: Array                      # temperature [K]
    P: Array                      # total pressure [Pa]
    rho: Array                    # moist-air density [kg/m^3]
    w: Array = 0.0                # vertical velocity / updraft [m/s]
    dz: Array = 1.0               # layer thickness [m] (column) or cell dz
    z: Array = 0.0                # height above lower boundary [m]
    freezing_level: Optional[float] = None   # height of 0 degC [m]

    # --- prognostic mixing ratios [kg/kg] ---
    qv: Array = 0.0
    qc: Array = 0.0
    qr: Array = 0.0
    qi: Array = 0.0
    qs: Array = 0.0
    qg: Array = 0.0
    qh: Array = 0.0

    # --- optional number concentrations [m^-3] (None -> diagnosed) ---
    Nc: Optional[Array] = None
    Nr: Optional[Array] = None
    Ni: Optional[Array] = None
    Ns: Optional[Array] = None
    Ng: Optional[Array] = None
    Nh: Optional[Array] = None

    # --- accumulators (surface, per category) ---
    surface_flux: dict = field(default_factory=dict)   # kg m^-2 s^-1
    accumulation: dict = field(default_factory=dict)   # mm (liquid-equivalent)
    clip_loss: float = 0.0        # booked mass created/destroyed by clip [kg/kg]
    t: float = 0.0

    def __post_init__(self) -> None:
        for name in SPECIES:
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        for name in ("T", "P", "rho", "w", "dz", "z"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))

    # ---- shape helpers ----
    @property
    def is_column(self) -> bool:
        return np.asarray(self.T).ndim >= 1

    @property
    def shape(self):
        return np.asarray(self.T).shape

    # ---- water bookkeeping ----
    def total_water_mixing(self) -> Array:
        """Sum of all seven water mixing ratios [kg/kg] (pointwise)."""
        return sum(np.asarray(getattr(self, s)) for s in SPECIES)

    def water_path(self) -> float:
        """Column-integrated total water mass per unit area [kg/m^2]
        (0-D: mass per unit area over the single layer of thickness dz)."""
        q_t = self.total_water_mixing()
        col = np.asarray(self.rho) * q_t * np.asarray(self.dz)
        return float(np.sum(col))

    def condensed_mixing(self) -> Array:
        return sum(np.asarray(getattr(self, s)) for s in CONDENSED)

    def frozen_mixing(self) -> Array:
        return sum(np.asarray(getattr(self, s)) for s in FROZEN)

    # ---- positivity ----
    def clip(self) -> None:
        """Floor every mixing ratio at zero; book the (tiny) created mass."""
        created = 0.0
        for s in SPECIES:
            arr = np.asarray(getattr(self, s))
            neg = np.minimum(arr, 0.0)
            created += float(-np.sum(neg))
            setattr(self, s, np.maximum(arr, 0.0))
        self.clip_loss += created

    def copy(self) -> "MicrophysicsState":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                kw[f.name] = v.copy()
            elif isinstance(v, dict):
                kw[f.name] = dict(v)
            else:
                kw[f.name] = v
        return MicrophysicsState(**kw)


__all__ = [
    "MicrophysicsState",
    "SPECIES", "CONDENSED", "FROZEN", "LIQUID", "PRECIP", "NUMBER",
]
