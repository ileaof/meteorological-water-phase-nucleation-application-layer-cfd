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
    ql: np.ndarray           # cloud liquid (q_c)
    qi: np.ndarray           # cloud ice
    # precipitating hydrometeors (Increment 2 two-way microphysics; default zeros)
    qr: np.ndarray = None    # rain
    qs: np.ndarray = None    # snow
    qg: np.ndarray = None    # graupel
    qh: np.ndarray = None    # hail
    # accumulated surface precipitation [kg/m^2 == mm], per category (2-D over x,y)
    surface_precip: dict = None
    # hydrostatic base pressure [Pa] (deep_convection scenario; None -> uniform P0)
    p0_field: np.ndarray = None
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
        xp = grid.xp
        return cls(
            grid=grid,
            u=xp.zeros(grid.u_shape), v=xp.zeros(grid.v_shape), w=xp.zeros(grid.w_shape),
            p=grid.zeros_c(), theta=grid.zeros_c(),
            qv=grid.zeros_c(), ql=grid.zeros_c(), qi=grid.zeros_c(),
            qr=grid.zeros_c(), qs=grid.zeros_c(), qg=grid.zeros_c(), qh=grid.zeros_c(),
            surface_precip={c: xp.zeros((grid.nx, grid.ny)) for c in ("rain", "snow", "graupel", "hail")},
        )

    def ensure_hydrometeors(self) -> None:
        """Guarantee the precipitating fields exist (zeros) — e.g. after a
        restart written before Increment 2."""
        g = self.grid
        for name in ("qr", "qs", "qg", "qh"):
            if getattr(self, name) is None:
                setattr(self, name, g.zeros_c())
        if self.surface_precip is None:
            self.surface_precip = {c: g.xp.zeros((g.nx, g.ny))
                                   for c in ("rain", "snow", "graupel", "hail")}

    def diagnose(self, cfg: SimulationConfig) -> None:
        """Fill the diagnosed thermodynamic fields from the prognostic state.

        Boussinesq: total pressure = P0 + p' (p' is tiny; used consistently).
        T = T(theta, P_total); p_v from q_v; S/RH from saturation (engine).
        """
        g = self.grid
        xp = g.xp
        if self.p0_field is not None:
            P_base = self.p0_field                 # stratified base (deep convection)
        else:
            # dtype matches the (possibly float32, under --float32) state array,
            # not a hardcoded float64 -- otherwise P_base + self.p below would
            # silently upcast P_total back to float64 even in --float32 mode.
            P_base = xp.full(g.center_shape, cfg.physics.P0, dtype=self.p.dtype)
        P_total = P_base + self.p
        # defensive positivity guard: the Boussinesq perturbation p' is O(Pa) and
        # should never drive P_total <= 0; if a transient overshoot does, floor it
        # so the theta->T power stays real (and flag it via the clip).
        P_total = xp.where(P_total > 0.0, P_total, P_base)
        self.P_total = P_total
        if cfg.physics.theta_transport:
            # theta is defined with P0_REF (100000 Pa) as the reference pressure,
            # NOT the scenario background P0; recover T with the same reference.
            self.T = th.T_from_theta(self.theta, P_total, th.P0_REF, xp=xp)
        else:
            self.T = self.theta.copy()  # T transported directly
        self.pv = th.p_v_from_q_v(self.qv, P_total, xp=xp)
        S_w, S_i, RH_w, RH_i = th.saturation_ratios(self.T, self.pv, xp=xp)
        self.S_w, self.S_i, self.RH_w, self.RH_i = S_w, S_i, RH_w, RH_i
        self.rho = th.density_moist(P_total, self.T, self.qv, xp=xp)
        self.gradT_mag = g.grad_magnitude(self.T)

    def copy(self) -> FlowState:
        _c = lambda a: None if a is None else a.copy()
        return FlowState(
            grid=self.grid,
            u=self.u.copy(), v=self.v.copy(), w=self.w.copy(), p=self.p.copy(),
            theta=self.theta.copy(), qv=self.qv.copy(), ql=self.ql.copy(),
            qi=self.qi.copy(),
            qr=_c(self.qr), qs=_c(self.qs), qg=_c(self.qg), qh=_c(self.qh),
            surface_precip=None if self.surface_precip is None
            else {k: v.copy() for k, v in self.surface_precip.items()},
            p0_field=_c(self.p0_field),
            t=self.t,
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
        return self.grid.xp.sqrt(uc ** 2 + vc ** 2 + wc ** 2)

    def total_water(self) -> float:
        tot = self.qv + self.ql + self.qi
        for name in ("qr", "qs", "qg", "qh"):
            a = getattr(self, name)
            if a is not None:
                tot = tot + a
        return float(tot.sum() * self.grid.cell_vol)


__all__ = ["FlowState"]