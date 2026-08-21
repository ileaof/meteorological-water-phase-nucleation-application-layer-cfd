"""FlowState: prognostic fields + diagnosed thermodynamics for the 3D solver.

Staggered storage mirrors :class:`grid.Grid`.  Prognostic variables are the
Boussinesq set ``u, v, w, p'`` (perturbation pressure), and the transported
scalars ``theta`` (potential temperature), ``q_v, q_l, q_i``.  All remaining
fields (T, rho, p_v, RH, S, |gradT|, P_total) are *diagnosed* from these via
:mod:`thermodynamics` and cached on the state for the nucleation adapter and
output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import thermodynamics as th
from .config import SimulationConfig
from .grid import Grid


@dataclass
class FlowState:
    grid: Grid
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray            # perturbation pressure at centres
    theta: np.ndarray        # potential temperature at centres
    qv: np.ndarray
    ql: np.ndarray
    qi: np.ndarray
    t: float = 0.0           # simulation time [s]
    # diagnosed (filled by .diagnose())
    T: np.ndarray = None
    P_total: np.ndarray = None
    rho: np.ndarray = None
    pv: np.ndarray = None
    S_w: np.ndarray = None
    S_i: np.ndarray = None
    RH_w: np.ndarray = None
    RH_i: np.ndarray = None
    gradT_mag: np.ndarray = None

    @classmethod
    def zeros(cls, grid: Grid) -> FlowState:
        return cls(
            grid=grid,
            u=np.zeros(grid.u_shape), v=np.zeros(grid.v_shape), w=np.zeros(grid.w_shape),
            p=grid.zeros_c(), theta=grid.zeros_c(),
            qv=grid.zeros_c(), ql=grid.zeros_c(), qi=grid.zeros_c(),
        )

    def diagnose(self, cfg: SimulationConfig) -> None:
        """Fill the diagnosed thermodynamic fields from the prognostic state.

        Boussinesq: total pressure = P0 + p' (p' is tiny; used consistently).
        T = T(theta, P_total); p_v from q_v; S/RH from saturation (engine).
        """
        g = self.grid
        P0 = cfg.physics.P0
        P_total = np.full(g.center_shape, P0, dtype=float) + self.p
        # defensive positivity guard: the Boussinesq perturbation p' is O(Pa) and
        # should never drive P_total <= 0; if a transient overshoot does, floor it
        # so the theta->T power stays real (and flag it via the clip).
        P_total = np.where(P_total > 0.0, P_total, P0)
        self.P_total = P_total
        if cfg.physics.theta_transport:
            # theta is defined with P0_REF (100000 Pa) as the reference pressure,
            # NOT the scenario background P0; recover T with the same reference.
            self.T = th.T_from_theta(self.theta, P_total, th.P0_REF)
        else:
            self.T = self.theta.copy()  # T transported directly
        self.pv = th.p_v_from_q_v(self.qv, P_total)
        S_w, S_i, RH_w, RH_i = th.saturation_ratios(self.T, self.pv)
        self.S_w, self.S_i, self.RH_w, self.RH_i = S_w, S_i, RH_w, RH_i
        self.rho = th.density_moist(P_total, self.T, self.qv)
        self.gradT_mag = g.grad_magnitude(self.T)

    def copy(self) -> FlowState:
        return FlowState(
            grid=self.grid,
            u=self.u.copy(), v=self.v.copy(), w=self.w.copy(), p=self.p.copy(),
            theta=self.theta.copy(), qv=self.qv.copy(), ql=self.ql.copy(),
            qi=self.qi.copy(), t=self.t,
            T=None if self.T is None else self.T.copy(),
            P_total=None if self.P_total is None else self.P_total.copy(),
            rho=None if self.rho is None else self.rho.copy(),
            pv=None if self.pv is None else self.pv.copy(),
            S_w=None if self.S_w is None else self.S_w.copy(),
            S_i=None if self.S_i is None else self.S_i.copy(),
            RH_w=None if self.RH_w is None else self.RH_w.copy(),
            RH_i=None if self.RH_i is None else self.RH_i.copy(),
            gradT_mag=None if self.gradT_mag is None else self.gradT_mag.copy(),
        )

    def velocity_magnitude_center(self) -> np.ndarray:
        uc = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        vc = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return np.sqrt(uc ** 2 + vc ** 2 + wc ** 2)

    def total_water(self) -> float:
        return float((self.qv + self.ql + self.qi).sum() * self.grid.cell_vol)


__all__ = ["FlowState"]