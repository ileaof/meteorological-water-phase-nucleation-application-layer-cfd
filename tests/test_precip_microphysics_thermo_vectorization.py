"""Pins precip_microphysics.thermo.psat_water/psat_ice -- rewritten as true
vectorised (ufunc) closed-form expressions instead of the previous
np.vectorize wrapper -- against the engine's own scalar per-element output.

Sibling of tests/test_thermodynamics_vectorization.py (same fix, duplicated
independently in precip_microphysics to avoid it depending on
meteorological_flow -- see precip_microphysics/thermo.py's module docstring).
"""
import numpy as np
import pytest

from met_water_nucleation import SaturationProperties
from precip_microphysics import thermo as th


def _reference_psat_water(T_array):
    return np.array([SaturationProperties.Psat_water(float(t), extended=True)
                      for t in T_array])


def _reference_psat_ice(T_array):
    return np.array([SaturationProperties.Psat_ice(float(t)) for t in T_array])


def test_psat_water_matches_engine_scalar_over_grid():
    T = np.linspace(200.0, 320.0, 200)
    got = th.psat_water(T)
    want = _reference_psat_water(T)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_psat_ice_matches_engine_scalar_over_grid():
    T = np.linspace(180.0, 273.15, 200)
    got = th.psat_ice(T)
    want = _reference_psat_ice(T)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_psat_water_scalar_input():
    T = 273.16
    got = float(th.psat_water(T))
    want = SaturationProperties.Psat_water(T, extended=True)
    assert got == pytest.approx(want, rel=1e-12)


def test_psat_ice_scalar_input():
    T = 250.0
    got = float(th.psat_ice(T))
    want = SaturationProperties.Psat_ice(T)
    assert got == pytest.approx(want, rel=1e-12)


def test_qsat_and_saturation_ratio_still_vectorised_and_finite():
    T = np.linspace(230.0, 305.0, 200)
    P = 90000.0 * np.ones_like(T)
    qv = 0.01 * np.ones_like(T)
    assert np.all(np.isfinite(th.qsat_water(T, P)))
    assert np.all(np.isfinite(th.qsat_ice(T, P)))
    assert np.all(np.isfinite(th.saturation_ratio_water(qv, T, P)))
    assert np.all(np.isfinite(th.saturation_ratio_ice(qv, T, P)))
