"""Thermodynamic helpers for the microphysics scheme.

Saturation vapour pressures come from the validated engine
(``met_water_nucleation.SaturationProperties`` -- IAPWS Wagner for liquid,
extended below the triple point; Goff-Gratch for ice).  They are **not**
reimplemented here: the engine is used read-only so the microphysics stays
consistent with the nucleation kernel.  The engine's ``Psat_*`` take scalars,
so they are vectorised (the underlying validated equations are unchanged).

Latent-heat exchange is expressed as a temperature tendency; the sign
convention is: condensation/deposition/freezing warm the air (+), and
evaporation/sublimation/melting cool it (-).  ``dT = (L/c_p) dq`` with dq the
mass converted to the denser phase.
"""
from __future__ import annotations

import numpy as np

import met_water_nucleation as M

from . import constants as C

_SP = M.SaturationProperties

_psat_water_v = np.vectorize(lambda t: _SP.Psat_water(float(t), extended=True), otypes=[float])
_psat_ice_v = np.vectorize(lambda t: _SP.Psat_ice(float(t)), otypes=[float])


def psat_water(T):
    """Saturation vapour pressure over liquid water [Pa] (engine, vectorised)."""
    T = np.asarray(T, dtype=float)
    if T.ndim == 0:
        return float(_SP.Psat_water(float(T), extended=True))
    return _psat_water_v(T)


def psat_ice(T):
    """Saturation vapour pressure over ice [Pa] (engine, vectorised)."""
    T = np.asarray(T, dtype=float)
    if T.ndim == 0:
        return float(_SP.Psat_ice(float(T)))
    return _psat_ice_v(T)


def p_v_from_qv(qv, P):
    qv = np.asarray(qv, dtype=float)
    P = np.asarray(P, dtype=float)
    return qv * P / (C.EPS + (1.0 - C.EPS) * qv)


def qv_from_pv(pv, P):
    pv = np.asarray(pv, dtype=float)
    P = np.asarray(P, dtype=float)
    return C.EPS * pv / (P - (1.0 - C.EPS) * pv)


def qsat_water(T, P):
    """Saturation mixing ratio over liquid water [kg/kg]."""
    return qv_from_pv(psat_water(T), P)


def qsat_ice(T, P):
    """Saturation mixing ratio over ice [kg/kg]."""
    return qv_from_pv(psat_ice(T), P)


def saturation_ratio_water(qv, T, P):
    return p_v_from_qv(qv, P) / psat_water(T)


def saturation_ratio_ice(qv, T, P):
    return p_v_from_qv(qv, P) / psat_ice(T)


def latent_heating(dq_to_denser, kind):
    """Temperature tendency [K] from converting ``dq_to_denser`` [kg/kg] of
    water to a denser phase.  Positive dq releases latent heat (warming).

    kind in {'vapor_liquid', 'vapor_ice', 'liquid_ice'} selects L_v / L_s / L_f.
    """
    L = {"vapor_liquid": C.Lv, "vapor_ice": C.Ls, "liquid_ice": C.Lf}[kind]
    return (L / C.cp_d) * np.asarray(dq_to_denser, dtype=float)


def ventilation_factor(D, vt):
    """Ventilation coefficient f = a + b Sc^(1/3) Re^(1/2) (Rutledge & Hobbs
    1983) for evaporation/sublimation of a falling particle of diameter ``D``
    [m] falling at ``vt`` [m/s]."""
    Re = np.maximum(np.asarray(vt) * np.asarray(D), 0.0) / C.NU_AIR
    return C.VENT_A + C.VENT_B * (C.SC ** (1.0 / 3.0)) * np.sqrt(Re)


__all__ = [
    "psat_water", "psat_ice", "p_v_from_qv", "qv_from_pv",
    "qsat_water", "qsat_ice", "saturation_ratio_water", "saturation_ratio_ice",
    "latent_heating", "ventilation_factor",
]
