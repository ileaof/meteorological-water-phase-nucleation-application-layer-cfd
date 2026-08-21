"""Nucleation lookup table: precompute + cache + interpolate the kernel outputs.

The validated kernel's gradient-scan path (~150 ms/cell at scan_resolution=75)
is far too costly to call per grid cell per time step.  We precompute the
per-cell outputs once over a coarse (T, p_v, |gradT|) grid for each phase,
cache to ``.npz``, and interpolate at run time (vectorised, ~instant).

Scientific note (documented): in the shifted-equilibrium framework the
nucleation rate is controlled primarily by (T, |gradT|, phase); the vapour
partial pressure p_v and total pressure P enter weakly for the rate but
govern the supersaturation diagnostics and (Batch 2) the mass balance.  We
keep p_v as a (small) axis for completeness; the rate slices are nearly flat
in p_v and interpolate trivially.

Build is parallelised with :mod:`multiprocessing` when ``threads>1``
(spec-sanctioned), with a sequential fallback.  Scan resolution matches the
direct kernel (75) so lookup-vs-direct error is interpolation error only.
"""
from __future__ import annotations

import math
import os

import numpy as np
import scipy.interpolate as si

import met_water_nucleation as M

from .config import LookupConfig, NucleationConfig
from .nucleation_adapter import PHASES, NucleationField

# per-phase fields stored in the table (names match NucleationResult attrs)
_TABLE_FIELDS = ("log10I", "I", "rC_2nd", "rC_1st", "Gamma2", "Gamma1",
                 "Delta_T", "T_eq_shift", "P_eq_shift", "gamma_r",
                 "dgamma_dr", "DeltaG_2nd", "DeltaG_1st", "closure_resid")

# ---- multiprocessing worker (top-level -> picklable) ----
_SIM_CACHE: dict = {}


def _worker_get_sim(theta, mode, phase_mode, r_ref, scan_res):
    key = (theta, mode, phase_mode, scan_res)
    if key not in _SIM_CACHE:
        un = M.un
        atm = un.AtmosphericInput(theta=theta, mode=mode, phase_mode=phase_mode,
                                  scenario="single_state", scan_resolution=scan_res)
        _SIM_CACHE[key] = (un, un.UnifiedNucleationSimulator(atm), r_ref)
    return _SIM_CACHE[key]


def _eval_chunk(points, theta, mode, phase_mode, r_ref, scan_res):
    """Evaluate a list of (T, pv, grad, phase_idx) -> dict of arrays (per field).
    Returns dict[phase_idx][field] = np.array(len(points))."""
    un, sim, rref = _worker_get_sim(theta, mode, phase_mode, r_ref, scan_res)
    n = len(points)
    out = {0: {}, 1: {}}
    for f in _TABLE_FIELDS:
        out[0][f] = np.empty(n); out[1][f] = np.empty(n)
    for ii, (T, pv, g, ph) in enumerate(points):
        res = sim.evaluate_point(T, 70000.0, pv, r_ref=rref, grad_T_req=g)
        for ip, p in enumerate(PHASES):
            r = res.get(p)
            if r is None:
                for f in _TABLE_FIELDS:
                    out[ip][f][ii] = np.nan
                continue
            for f in _TABLE_FIELDS:
                v = getattr(r, f, np.nan)
                if f == "log10I" and (v is None or not math.isfinite(v)):
                    v = -np.inf
                out[ip][f][ii] = v
    return out


class NucleationLookup:
    def __init__(self, nuc_cfg: NucleationConfig, lookup_cfg: LookupConfig):
        self.nuc_cfg = nuc_cfg
        self.cfg = lookup_cfg
        self.theta = nuc_cfg.theta
        self.mode = nuc_cfg.mode
        self.phase_mode = nuc_cfg.phase_mode
        self.r_ref = nuc_cfg.r_ref if nuc_cfg.r_ref else M.un.R_REF_DEFAULT
        self.gmin = nuc_cfg.gmin
        # coordinate axes
        self.T_axis = np.linspace(lookup_cfg.T_range[0], lookup_cfg.T_range[1], lookup_cfg.n_T)
        self.pv_axis = np.linspace(lookup_cfg.pv_range[0], lookup_cfg.pv_range[1], lookup_cfg.n_pv)
        self.grad_axis = np.logspace(np.log10(lookup_cfg.grad_range[0]),
                                     np.log10(lookup_cfg.grad_range[1]), lookup_cfg.n_grad)
        self.log_grad_axis = np.log10(self.grad_axis)
        # table data: dict[phase_idx][field] -> array shape (n_T, n_pv, n_grad)
        self.table = {0: {}, 1: {}}
        self._interp = {0: {}, 1: {}}
        self.built = False

    # ---- build / cache ----
    def build(self, threads: int = 1, progress=None) -> None:
        pts = []
        for it, T in enumerate(self.T_axis):
            for ip, pv in enumerate(self.pv_axis):
                for ig, g in enumerate(self.grad_axis):
                    for ph in (0, 1):
                        pts.append((float(T), float(pv), float(g), ph))
        n = len(pts)
        if progress:
            progress(0, n)
        chunks = _split(pts, max(1, threads))
        results = [None] * len(chunks)
        if threads and threads > 1:
            try:
                import concurrent.futures as cf
                with cf.ProcessPoolExecutor(max_workers=threads) as ex:
                    futs = [ex.submit(_eval_chunk, c, self.theta, self.mode,
                                     self.phase_mode, self.r_ref,
                                     self.cfg.scan_resolution) for c in chunks]
                    for i, f in enumerate(futs):
                        results[i] = f.result()
                        if progress:
                            progress(sum(len(c) for c in chunks[:i + 1]), n)
            except Exception:
                results = [self._eval_chunk_seq(c) for c in chunks]
                if progress:
                    progress(n, n)
        else:
            results = [self._eval_chunk_seq(c) for c in chunks]
            if progress:
                progress(n, n)
        # assemble into table arrays
        self._assemble(results, pts)
        self._build_interpolators()
        self.built = True

    def _eval_chunk_seq(self, points):
        return _eval_chunk(points, self.theta, self.mode, self.phase_mode,
                          self.r_ref, self.cfg.scan_resolution)

    def _assemble(self, results, pts):
        nT, nP, nG = len(self.T_axis), len(self.pv_axis), len(self.grad_axis)
        for ph in (0, 1):
            for f in _TABLE_FIELDS:
                self.table[ph][f] = np.full((nT, nP, nG), np.nan)
        idx = 0
        for res in results:
            for f in _TABLE_FIELDS:
                a0 = res[0][f]; a1 = res[1][f]
                for ii in range(len(a0)):
                    T, pv, g, ph = pts[idx + ii]
                    it = _nearest_index(self.T_axis, T)
                    ip = _nearest_index(self.pv_axis, pv)
                    ig = _nearest_index(self.log_grad_axis, math.log10(g))
                    self.table[0][f][it, ip, ig] = a0[ii]
                    self.table[1][f][it, ip, ig] = a1[ii]
            idx += len(a0)

    def _build_interpolators(self):
        for ph in (0, 1):
            for f in _TABLE_FIELDS:
                data = self.table[ph][f].copy()
                if f == "log10I":
                    # subsaturated / no-solution nodes -> -50 (a "no rate" floor,
                    # below any physical homogeneous nucleation rate).
                    data = np.where(np.isfinite(data), data, -50.0)
                else:
                    # structural fields (rC, Gamma, DeltaG, ...): the kernel
                    # returns NaN where no embryo forms (subsaturated).  Leave a
                    # NaN there and the trilinear interpolator propagates NaN
                    # into neighbouring supersaturated cells -> unusable rC etc.
                    # Fill each NaN hole with its nearest finite node value so
                    # interpolation is defined everywhere (a documented
                    # parameterization of the phase-boundary region).
                    data = _fill_nan_nearest(data)
                self._interp[ph][f] = si.RegularGridInterpolator(
                    (self.T_axis, self.pv_axis, self.log_grad_axis), data,
                    method="linear", bounds_error=False, fill_value=None)

    # ---- save / load ----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, T_axis=self.T_axis, pv_axis=self.pv_axis,
                 grad_axis=self.grad_axis, log_grad_axis=self.log_grad_axis,
                 **{f"t{ph}_{f}": self.table[ph][f]
                    for ph in (0, 1) for f in _TABLE_FIELDS})

    @classmethod
    def load(cls, path: str, nuc_cfg: NucleationConfig, lookup_cfg: LookupConfig) -> NucleationLookup:
        lk = cls(nuc_cfg, lookup_cfg)
        z = np.load(path)
        lk.T_axis = z["T_axis"]; lk.pv_axis = z["pv_axis"]
        lk.grad_axis = z["grad_axis"]; lk.log_grad_axis = z["log_grad_axis"]
        for ph in (0, 1):
            for f in _TABLE_FIELDS:
                lk.table[ph][f] = z[f"t{ph}_{f}"]
        lk._build_interpolators()
        lk.built = True
        return lk

    # ---- interpolation into a NucleationField ----
    def fill_field(self, nf: NucleationField, T, P, pv, gradT, state) -> None:
        nx, ny, nz = T.shape
        Tflat = np.asarray(T).reshape(-1)
        pvflat = np.clip(np.asarray(pv).reshape(-1),
                        self.pv_axis[0], self.pv_axis[-1])
        gflat = np.maximum(np.asarray(gradT).reshape(-1), self.gmin)
        lgflat = np.clip(np.log10(gflat), self.log_grad_axis[0], self.log_grad_axis[-1])
        Tc = np.clip(Tflat, self.T_axis[0], self.T_axis[-1])
        coords = np.stack([Tc, pvflat, lgflat], axis=-1)
        in_range = ((Tflat >= self.T_axis[0]) & (Tflat <= self.T_axis[-1]) &
                    (np.asarray(pv).reshape(-1) >= self.pv_axis[0]) &
                    (np.asarray(pv).reshape(-1) <= self.pv_axis[-1]))
        for ph in (0, 1):
            for f in _TABLE_FIELDS:
                vals = self._interp[ph][f](coords)
                self._store(nf, ph, f, vals.reshape(nx, ny, nz))
        # status & validity
        for ph in (0, 1):
            nf.status[ph] = np.where(in_range.reshape(nx, ny, nz), 1, -2)
        # dominant phase from interpolated rates
        lI = nf.log10I[0]; iI = nf.log10I[1]
        dom = np.where(np.isfinite(lI) & np.isfinite(iI) & (lI > iI + 0.5), 1,
                       np.where(np.isfinite(iI) & (iI > lI + 0.5), 2, 3))
        nf.dominant_phase = dom.astype(int)
        nf.validity_mask = in_range.reshape(nx, ny, nz) & np.isfinite(lI)

    def _store(self, nf, ph, field, arr):
        m = {
            "log10I": "log10I", "I": "I", "rC_2nd": "rC_2nd", "rC_1st": "rC_1st",
            "Gamma2": "Gamma2", "Gamma1": "Gamma1", "Delta_T": "Delta_T",
            "T_eq_shift": "T_local", "P_eq_shift": "P_eq_shift",
            "gamma_r": "gamma", "dgamma_dr": "dgamma_dr",
            "DeltaG_2nd": "DeltaG_2nd", "DeltaG_1st": "DeltaG_1st",
            "closure_resid": "residual",
        }
        getattr(nf, m[field])[ph] = arr


def _nearest_index(axis, val):
    return int(np.argmin(np.abs(axis - val)))


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """Replace NaN/inf holes with the nearest finite node value (3-D nearest
    fill via a distance transform).  Used so structural kernel fields (rC,
    Gamma, ...) interpolate cleanly across the subsaturated boundary where the
    kernel returns no embryo (NaN).  If the field is all-NaN, return zeros."""
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if finite.all():
        return a
    if not finite.any():
        return np.zeros_like(a)
    try:
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(~finite, return_distances=False,
                                         return_indices=True)
        filled = a[tuple(indices)]
    except Exception:                       # pragma: no cover - scipy present
        filled = a.copy()
        for _ in range(a.ndim * max(a.shape)):
            nxt = filled.copy()
            for ax in range(a.ndim):
                nxt = np.nan_to_num(nxt, nan=np.nan)
            if np.isfinite(nxt).all():
                break
        filled = np.nan_to_num(filled, nan=np.nanmean(a))
    return np.where(finite, a, filled)


def _split(lst, n):
    k = max(1, (len(lst) + n - 1) // n)
    return [lst[i:i + k] for i in range(0, len(lst), k)]


__all__ = ["NucleationLookup"]