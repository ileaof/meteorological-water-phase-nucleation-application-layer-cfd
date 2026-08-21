#!/usr/bin/env python3
"""test_lookup_accuracy.py -- lookup interpolation vs the direct kernel (test 13).

The lookup is precomputed by calling the validated kernel at the table nodes
(scan_resolution=N); we compare the interpolated lookup to a *direct* kernel
call built with the SAME scan_resolution, so the only discrepancy is the
trilinear interpolation error (not a scan-resolution mismatch).  The validated
equations stay read-only.

Two complementary checks:
  * a SYNTHETIC linear table -> the interpolator reproduces a linear function
    to ~machine precision (tight; verifies the trilinear wiring / assembly /
    save-load independently of the expensive, fast-varying kernel).
  * a REAL kernel build (small, scan_res=20) -> off-node log10I within ~1
    (the rate varies over orders of magnitude) and rC_2nd median rel-error
    within ~0.5.  These tolerances reflect the deliberately coarse TEST table;
    the production 28x20x9 table (exercised by the reference demo) is finer and
    more accurate.  NaN structural fields are nearest-filled at the phase
    boundary so interpolation is defined everywhere (documented parameterization).
"""
import math
import os
import tempfile

import numpy as np

import met_water_nucleation as M
from meteorological_flow.config import LookupConfig, NucleationConfig
from meteorological_flow.nucleation_lookup import _TABLE_FIELDS, NucleationLookup

SCAN = 20   # scan_resolution shared by the lookup build and the direct sim


def _direct_sim(ncfg, scan=SCAN):
    """A direct kernel sim with the SAME scan_resolution as the lookup."""
    atm = M.un.AtmosphericInput(theta=ncfg.theta, mode=ncfg.mode,
                                phase_mode=ncfg.phase_mode,
                                scenario="single_state", scan_resolution=scan)
    return M.un.UnifiedNucleationSimulator(atm)


def _build_lookup():
    ncfg = NucleationConfig()
    lc = LookupConfig(n_T=8, n_pv=5, n_grad=5, scan_resolution=SCAN,
                      T_range=[230.0, 305.0], pv_range=[40.0, 3500.0],
                      grad_range=[1e-3, 20.0])
    lk = NucleationLookup(ncfg, lc)
    lk.build(threads=4)           # ~7s; deterministic, matches threads=1 exactly
    return ncfg, lk


def test_13a_interpolator_exact_on_linear_synthetic():
    """[math] The trilinear RegularGridInterpolator reproduces a linear function
    a*T + b*pv + c*log(grad) + d exactly (to ~1e-9) at arbitrary points.  This
    verifies the lookup's coordinate wiring (log-spaced grad axis) and assembly
    without the expensive kernel."""
    ncfg = NucleationConfig()
    lc = LookupConfig(n_T=8, n_pv=6, n_grad=5, scan_resolution=SCAN,
                      T_range=[230.0, 305.0], pv_range=[40.0, 3500.0],
                      grad_range=[1e-3, 20.0])
    lk = NucleationLookup(ncfg, lc)
    # plant a known linear field directly into the table (table dicts are empty
    # until build/_assemble, so allocate the arrays here)
    a, b, c, d = 0.7, 0.002, 1.3, 5.0
    shape = (lc.n_T, lc.n_pv, lc.n_grad)
    # fill every stored field with the same linear function so the interpolator
    # build (which iterates all _TABLE_FIELDS) succeeds; we assert only log10I.
    lin = np.empty(shape)
    for it, T in enumerate(lk.T_axis):
        for ip, pv in enumerate(lk.pv_axis):
            for ig, lg in enumerate(lk.log_grad_axis):
                lin[it, ip, ig] = a * T + b * pv + c * lg + d
    for ph in (0, 1):
        for f in _TABLE_FIELDS:
            lk.table[ph][f] = lin.copy()
    lk._build_interpolators()
    rng = np.random.default_rng(11)
    for _ in range(20):
        T = float(rng.uniform(235.0, 300.0))
        pv = float(rng.uniform(200.0, 3000.0))
        g = float(10 ** rng.uniform(-2.5, 1.0))
        interp = float(lk._interp[0]["log10I"]([[T, pv, math.log10(g)]])[0])
        exact = a * T + b * pv + c * math.log10(g) + d
        assert abs(interp - exact) < 1e-9, f"linear not exact: {interp} vs {exact}"


def test_13b_lookup_exact_at_nodes():
    """[num] The interpolator returns the stored node value exactly at a node
    coordinate (identity at nodes)."""
    ncfg, lk = _build_lookup()
    # find a finite liquid node
    vals = lk.table[0]["log10I"]
    for it in range(lk.T_axis.size):
        for ip in range(lk.pv_axis.size):
            for ig in range(lk.grad_axis.size):
                if math.isfinite(vals[it, ip, ig]):
                    T = float(lk.T_axis[it]); pv = float(lk.pv_axis[ip])
                    g = float(lk.grad_axis[ig])
                    node_val = vals[it, ip, ig]
                    interp = float(lk._interp[0]["log10I"]([[T, pv, math.log10(g)]])[0])
                    assert abs(interp - node_val) < 1e-9, \
                        f"lookup not exact at node: {interp} vs {node_val}"
                    return
    raise AssertionError("no finite node found in lookup table")


def test_13c_lookup_matches_direct_within_tolerance():
    """[ref] At off-node sample points (interior, well-resolved region) the
    interpolated lookup agrees with the direct kernel (same scan_resolution):
    log10I within ~1.0 (fast-varying, log-scaled), rC_2nd MEDIAN rel-error
    within ~0.5.  Tolerances reflect the coarse TEST table; production is
    finer (see the reference demo)."""
    ncfg, lk = _build_lookup()
    sim = _direct_sim(ncfg)
    rng = np.random.default_rng(123)
    errs_logI = []
    errs_rC = []
    for _ in range(25):
        T = float(rng.uniform(252.0, 292.0))
        pv = float(rng.uniform(400.0, 2600.0))
        g = float(10 ** rng.uniform(-2.0, 0.6))
        res = sim.evaluate_point(T, 70000.0, pv, r_ref=ncfg.r_ref or M.un.R_REF_DEFAULT,
                                 grad_T_req=g)
        r = res.get("liquid")
        if r is None:
            continue
        dL = getattr(r, "log10I", None)
        dR = getattr(r, "rC_2nd", None)
        if dL is None or not math.isfinite(dL):
            continue
        iL = float(lk._interp[0]["log10I"]([[T, pv, math.log10(g)]])[0])
        iR = float(lk._interp[0]["rC_2nd"]([[T, pv, math.log10(g)]])[0])
        errs_logI.append(abs(iL - dL))
        if dR and math.isfinite(dR) and math.isfinite(iR):
            errs_rC.append(abs(iR - dR) / abs(dR))
    assert errs_logI, "no comparable samples"
    assert max(errs_logI) < 1.0, f"log10I interp error too large: {max(errs_logI)}"
    assert errs_rC, "no finite rC samples"
    assert float(np.median(errs_rC)) < 0.5, \
        f"rC_2nd median rel error too large: {np.median(errs_rC)}"
    # the structural field is finite everywhere in range after NaN-fill
    assert math.isfinite(iR)


def test_13d_lookup_save_load_roundtrip():
    """[num] Saving and reloading the lookup reproduces the table exactly (the
    cache is the reproducibility mechanism across runs)."""
    ncfg, lk = _build_lookup()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lk.npz")
        lk.save(path)
        lk2 = NucleationLookup.load(path, ncfg, lk.cfg)
        assert np.allclose(lk.T_axis, lk2.T_axis)
        assert np.allclose(lk.table[0]["log10I"], lk2.table[0]["log10I"], equal_nan=True)
        assert np.allclose(lk.table[1]["rC_2nd"], lk2.table[1]["rC_2nd"], equal_nan=True)
        # the reloaded interpolator is also exact at a finite node
        v = lk2.table[0]["log10I"][3, 2, 2]
        if math.isfinite(v):
            iv = float(lk2._interp[0]["log10I"]([[lk2.T_axis[3], lk2.pv_axis[2],
                                                  lk2.log_grad_axis[2]]])[0])
            assert abs(iv - v) < 1e-9