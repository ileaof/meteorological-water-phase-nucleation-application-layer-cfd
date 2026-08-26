"""Thermodynamic helpers for the microphysics scheme.

Saturation vapour pressures come from the validated engine
(``met_water_nucleation.SaturationProperties`` -- IAPWS Wagner for liquid,
extended below the triple point; Goff-Gratch for ice).  ``psat_water``/
``psat_ice`` below are a true vectorised (ufunc) re-expression of the exact
same closed-form equations the engine computes scalar-wise -- not a new
derivation -- reproduced here (rather than imported) to avoid this standalone
package depending on ``meteorological_flow``, which already depends on it
(see ``meteorological_flow/thermodynamics.py`` for the sibling copy of this
same fix, and ``tests/test_precip_microphysics_thermo_vectorization.py`` for
the equivalence pin against the engine's own per-element output).  The
previous ``np.vectorize`` wrapper was a hidden Python-level per-element loop
into the engine -- slow, and unable to accept a CuPy array (would force a
silent GPU<->host round trip every call).

Every function accepts an optional ``xp=`` array-module override (default
``numpy``) so the flow-coupled path (GPU-resident ``MicrophysicsState``) can
call these without a host round-trip.

Latent-heat exchange is expressed as a temperature tendency; the sign
convention is: condensation/deposition/freezing warm the air (+), and
evaporation/sublimation/melting cool it (-).  ``dT = (L/c_p) dq`` with dq the
mass converted to the denser phase.
"""
from __future__ import annotations

import math

import numpy as np

import met_water_nucleation as M

from . import constants as C

_SP = M.SaturationProperties

# Same published coefficients the engine uses (see
# src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/
# unified_h2o_nucleation_climate.py:240-249,258-298) -- copied, not
# re-derived; see meteorological_flow/thermodynamics.py for the sibling copy.
_WAGNER_A = (-7.85951783, 1.84408259, -11.7866497, 22.6807411,
             -15.9618719, 1.80122502)                          # IAPWS Wagner
_WAGNER_B = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
_TC = 647.096          # K,  critical temperature
_PC = 22.064e6         # Pa, critical pressure
_PT = 611.657          # Pa, triple-point pressure
_GG_TT = 273.16        # K,  triple-point temperature / Goff-Gratch anchor
_GG_LOGE_REF = math.log10(_PT / 100.0)   # log10(Pt in hPa)


def psat_water(T, xp=np):
    """Saturation vapour pressure over liquid water [Pa] (IAPWS Wagner,
    extended below the triple point). Fully vectorised."""
    T = xp.asarray(T, dtype=float)
    tau = 1.0 - T / _TC
    s = sum(a * tau ** b for a, b in zip(_WAGNER_A, _WAGNER_B))
    return _PC * xp.exp((_TC / T) * s)


def psat_ice(T, xp=np):
    """Saturation vapour pressure over ice [Pa] (Goff-Gratch). Fully vectorised."""
    T = xp.asarray(T, dtype=float)
    ratio = _GG_TT / T
    log10e = (-9.09718 * (ratio - 1.0)
              - 3.56654 * xp.log10(ratio)
              + 0.876793 * (1.0 - T / _GG_TT)
              + _GG_LOGE_REF)
    return 100.0 * xp.power(10.0, log10e)   # hPa -> Pa


def p_v_from_qv(qv, P, xp=np):
    qv = xp.asarray(qv, dtype=float)
    P = xp.asarray(P, dtype=float)
    return qv * P / (C.EPS + (1.0 - C.EPS) * qv)


def qv_from_pv(pv, P, xp=np):
    pv = xp.asarray(pv, dtype=float)
    P = xp.asarray(P, dtype=float)
    return C.EPS * pv / (P - (1.0 - C.EPS) * pv)


def qsat_water(T, P, xp=np):
    """Saturation mixing ratio over liquid water [kg/kg]."""
    return qv_from_pv(psat_water(T, xp=xp), P, xp=xp)


def qsat_ice(T, P, xp=np):
    """Saturation mixing ratio over ice [kg/kg]."""
    return qv_from_pv(psat_ice(T, xp=xp), P, xp=xp)


def saturation_ratio_water(qv, T, P, xp=np):
    return p_v_from_qv(qv, P, xp=xp) / psat_water(T, xp=xp)


def saturation_ratio_ice(qv, T, P, xp=np):
    return p_v_from_qv(qv, P, xp=xp) / psat_ice(T, xp=xp)


def latent_heating(dq_to_denser, kind, xp=np):
    """Temperature tendency [K] from converting ``dq_to_denser`` [kg/kg] of
    water to a denser phase.  Positive dq releases latent heat (warming).

    kind in {'vapor_liquid', 'vapor_ice', 'liquid_ice'} selects L_v / L_s / L_f.
    """
    L = {"vapor_liquid": C.Lv, "vapor_ice": C.Ls, "liquid_ice": C.Lf}[kind]
    return (L / C.cp_d) * xp.asarray(dq_to_denser, dtype=float)


def ventilation_factor(D, vt, xp=np):
    """Ventilation coefficient f = a + b Sc^(1/3) Re^(1/2) (Rutledge & Hobbs
    1983) for evaporation/sublimation of a falling particle of diameter ``D``
    [m] falling at ``vt`` [m/s]."""
    Re = xp.maximum(xp.asarray(vt) * xp.asarray(D), 0.0) / C.NU_AIR
    return C.VENT_A + C.VENT_B * (C.SC ** (1.0 / 3.0)) * xp.sqrt(Re)


__all__ = [
    "psat_water", "psat_ice", "p_v_from_qv", "qv_from_pv",
    "qsat_water", "qsat_ice", "saturation_ratio_water", "saturation_ratio_ice",
    "latent_heating", "ventilation_factor",
]
