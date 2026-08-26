"""NucleationAdapter: faithful wrapper around the validated kernel.

Treats ``met_water_nucleation.un`` as a read-only thermodynamic microphysics
subroutine.  For each cell it calls ``UnifiedNucleationSimulator.evaluate_point``
with (T, P, p_v, r_ref, grad_T_req=|gradT|) and collects the kernel's outputs
into grid-shaped arrays.  Nothing in the validated equations is modified; the
adapter is a pure pass-through.

Two evaluation methods:
  * ``direct``  -- calls the kernel per cell (accurate, ~150 ms/cell with the
    gradient scan; only for validation / small samples).
  * ``lookup``  -- interpolates a precomputed table (see :mod:`nucleation_lookup`);
    the only viable method for field-scale runs.

The macroscopic cell |gradT| is floored at ``gmin`` (the kernel's
well-behaved near-equilibrium lower bound) -- documented parameterization that
keeps the shifted-equilibrium scan in its valid range; the |gradT|->0 limit is
therefore the kernel's own near-equilibrium result, NOT the CNT limit.
"""
from __future__ import annotations

import math

import numpy as np

import met_water_nucleation as M

from .config import NucleationConfig
from .state import FlowState

PHASES = ("liquid", "ice")
STATUS_CODE = {"ok": 1, "subsaturated": 0, "no_solution": -1, "out_of_range": -2}
DOMINANT_CODE = {"none": 0, "liquid": 1, "ice": 2, "competition": 3}

# NucleationResult attributes extracted per phase (names from the kernel).
_RESULT_FIELDS = (
    "log10I", "I", "rC_2nd", "rC_1st", "Gamma2", "Gamma1", "Delta_T",
    "T_eq_shift", "P_eq_shift", "gamma_r", "dgamma_dr", "DeltaG_2nd",
    "DeltaG_1st", "closure_resid", "parabolic_resid", "gibbs_thomson_resid",
)


class NucleationField:
    """Container for grid-shaped nucleation outputs (all per-phase stacked on
    axis 0: 0=liquid, 1=ice unless noted)."""

    def __init__(self, shape):
        s = (2,) + shape
        self.log10I = np.full(s, -np.inf)
        self.I = np.zeros(s)
        self.rC_2nd = np.full(s, np.nan)
        self.rC_1st = np.full(s, np.nan)
        self.Gamma2 = np.full(s, np.nan)
        self.Gamma1 = np.full(s, np.nan)
        self.Delta_T = np.full(s, np.nan)
        self.T_local = np.full(s, np.nan)
        self.P_eq_shift = np.full(s, np.nan)
        self.gamma = np.full(s, np.nan)
        self.dgamma_dr = np.full(s, np.nan)
        self.DeltaG_2nd = np.full(s, np.nan)
        self.DeltaG_1st = np.full(s, np.nan)
        self.status = np.zeros(s, dtype=int)     # STATUS_CODE
        self.residual = np.full(s, np.nan)
        self.expected_events = np.zeros(s)
        self.dominant_phase = np.zeros(shape, dtype=int)   # DOMINANT_CODE
        self.validity_mask = np.zeros(shape, dtype=bool)

    def field_dict(self):
        return {k: getattr(self, k) for k in [
            "log10I", "I", "rC_2nd", "rC_1st", "Gamma2", "Gamma1", "Delta_T",
            "T_local", "P_eq_shift", "gamma", "dgamma_dr", "DeltaG_2nd",
            "DeltaG_1st", "status", "residual", "expected_events",
            "dominant_phase", "validity_mask"]}


class NucleationAdapter:
    def __init__(self, nuc_cfg: NucleationConfig):
        self.un = M.un
        self.mode = nuc_cfg.mode
        self.phase_mode = nuc_cfg.phase_mode
        self.theta = nuc_cfg.theta
        self.r_ref = nuc_cfg.r_ref if nuc_cfg.r_ref else self.un.R_REF_DEFAULT
        self.gmin = nuc_cfg.gmin
        self.method = nuc_cfg.method
        self.lookup = None
        self._build_sim()

    def _build_sim(self):
        atm = self.un.AtmosphericInput(
            theta=self.theta, mode=self.mode, phase_mode=self.phase_mode,
            scenario="single_state")
        self.sim = self.un.UnifiedNucleationSimulator(atm)

    def set_lookup(self, lookup) -> None:
        self.lookup = lookup
        self.method = "lookup"

    def evaluate_cell(self, T, P, p_v, grad_T):
        """Direct kernel call for a single cell -> dict[phase -> NucleationResult]."""
        g = grad_T if grad_T and grad_T > self.gmin else self.gmin
        return self.sim.evaluate_point(T, P, p_v, r_ref=self.r_ref, grad_T_req=g)

    def _fill_from_result(self, nf, ip, idx, r):
        """Fill NucleationField arrays at flat index idx for phase index ip."""
        for f in _RESULT_FIELDS:
            if f == "log10I":
                val = getattr(r, "log10I", None)
                nf.log10I[ip].flat[idx] = val if val is not None and math.isfinite(val) else -np.inf
            elif f == "T_local":
                nf.T_local[ip].flat[idx] = getattr(r, "T_eq_shift", np.nan)
            elif f == "gamma":
                nf.gamma[ip].flat[idx] = getattr(r, "gamma_r", np.nan)
            elif f == "dgamma_dr":
                nf.dgamma_dr[ip].flat[idx] = getattr(r, "dgamma_dr", np.nan)
            elif f == "DeltaG_2nd":
                nf.DeltaG_2nd[ip].flat[idx] = getattr(r, "DeltaG_2nd", np.nan)
            elif f == "DeltaG_1st":
                nf.DeltaG_1st[ip].flat[idx] = getattr(r, "DeltaG_1st", np.nan)
            elif f == "rC_2nd":
                nf.rC_2nd[ip].flat[idx] = getattr(r, "rC_2nd", np.nan)
            elif f == "rC_1st":
                nf.rC_1st[ip].flat[idx] = getattr(r, "rC_1st", np.nan)
            elif f == "Gamma2":
                nf.Gamma2[ip].flat[idx] = getattr(r, "Gamma2", np.nan)
            elif f == "Gamma1":
                nf.Gamma1[ip].flat[idx] = getattr(r, "Gamma1", np.nan)
            elif f == "Delta_T":
                nf.Delta_T[ip].flat[idx] = getattr(r, "Delta_T", np.nan)
            elif f == "P_eq_shift":
                nf.P_eq_shift[ip].flat[idx] = getattr(r, "P_eq_shift", np.nan)
            elif f == "I":
                v = getattr(r, "I", 0.0); nf.I[ip].flat[idx] = v if v is not None and math.isfinite(v) else 0.0
            elif f == "closure_resid":
                nf.residual[ip].flat[idx] = getattr(r, "closure_resid", np.nan)
            elif f == "parabolic_resid":
                pass  # stored within residual (closure_resid used as primary)
            elif f == "gibbs_thomson_resid":
                pass
        nf.status[ip].flat[idx] = STATUS_CODE.get(getattr(r, "status", "ok"), 1)

    def evaluate_field(self, state: FlowState, dt: float = 0.0,
                       cell_volume: float | None = None) -> NucleationField:
        """Evaluate nucleation over the whole grid.  Returns NucleationField."""
        shape = state.grid.center_shape
        nf = NucleationField(shape)
        gradT = np.maximum(state.gradT_mag, self.gmin)
        T, P, pv = state.T, state.P_total, state.pv

        if self.method == "lookup" and self.lookup is not None:
            # The lookup/interpolation layer (and the immutable engine it
            # ultimately wraps) is CPU-only by design -- see backend.py's
            # module docstring. T/P/pv/gradT may be GPU-resident; convert
            # once here, at this call boundary, rather than inside the
            # lookup table code.
            to_cpu = state.grid.backend.to_cpu
            self.lookup.fill_field(nf, to_cpu(T), to_cpu(P), to_cpu(pv), to_cpu(gradT), state)
        else:
            self._evaluate_direct(nf, T, P, pv, gradT, state)

        # expected events (mean-field subgrid): N = J * dV * dt
        if cell_volume is None:
            cell_volume = state.grid.cell_vol
        nf.expected_events = nf.I * cell_volume * max(dt, 0.0)
        # validity: finite rate & status ok
        nf.validity_mask = np.isfinite(nf.log10I[0]) | np.isfinite(nf.log10I[1])
        return nf

    def _evaluate_direct(self, nf, T, P, pv, gradT, state):
        nx, ny, nz = state.grid.center_shape
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = (i * ny + j) * nz + k
                    res = self.sim.evaluate_point(
                        float(T[i, j, k]), float(P[i, j, k]), float(pv[i, j, k]),
                        r_ref=self.r_ref, grad_T_req=float(gradT[i, j, k]))
                    for ip, ph in enumerate(PHASES):
                        r = res.get(ph)
                        if r is None:
                            continue
                        self._fill_from_result(nf, ip, idx, r)
                    # dominant from the liquid result (both carry it)
                    r0 = res.get("liquid")
                    if r0 is not None:
                        nf.dominant_phase[i, j, k] = DOMINANT_CODE.get(
                            getattr(r0, "dominant", "none"), 0)


__all__ = [
    "DOMINANT_CODE",
    "PHASES",
    "STATUS_CODE",
    "NucleationAdapter",
    "NucleationField",
]