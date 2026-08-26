"""Pins meteorological_flow.thermodynamics.psat_water/psat_ice -- rewritten as
true vectorised (ufunc) closed-form expressions instead of the previous
np.vectorize wrapper -- against the engine's own scalar per-element output.

The rewrite reproduces the exact published Wagner (liquid) / Goff-Gratch (ice)
formulas the engine already computes scalar-wise; this test proves that claim
rather than assuming it, by comparing element-by-element against a Python-loop
call into the engine's SaturationProperties directly (not against the removed
np.vectorize wrapper, which no longer exists in the module).
"""
import numpy as np
import pytest

from met_water_nucleation import SaturationProperties
from meteorological_flow import thermodynamics as th


def _T_grid(n=200):
    # spans the LookupConfig default T_range (230..305 K) plus a bit of
    # margin on both sides, staying within the engine's extended Wagner range.
    return np.linspace(200.0, 320.0, n)


def _reference_psat_water(T_array):
    return np.array([SaturationProperties.Psat_water(float(t), extended=True)
                      for t in T_array])


def _reference_psat_ice(T_array):
    return np.array([SaturationProperties.Psat_ice(float(t)) for t in T_array])


def test_psat_water_matches_engine_scalar_over_grid():
    T = _T_grid()
    got = th.psat_water(T)
    want = _reference_psat_water(T)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_psat_ice_matches_engine_scalar_over_grid():
    # Goff-Gratch is documented valid ~173..273.16K (extended beyond, flagged);
    # stay inside the well-behaved range here.
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


def test_saturation_ratios_still_vectorised_and_finite():
    T = _T_grid()
    p_v = 500.0 * np.ones_like(T)
    S_w, S_i, RH_w, RH_i = th.saturation_ratios(T, p_v)
    assert np.all(np.isfinite(S_w)) and np.all(np.isfinite(S_i))
    assert np.allclose(RH_w, 100.0 * S_w)
    assert np.allclose(RH_i, 100.0 * S_i)
