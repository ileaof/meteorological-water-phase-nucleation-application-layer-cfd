"""Moist Boussinesq buoyancy for the vertical (w) momentum equation.

    B = g [ (T - T_ref)/T_ref + 0.61 (q_v - q_v,ref) - (q_l + q_i) ]

with T_ref, q_v,ref the fixed initial references (Boussinesq convention).  The
buoyancy is a cell-centre scalar; it is averaged onto the z-faces and added as
a dw/dt tendency.  Moisture buoyancy (water-vapour virtual effect and the
negative loading of condensate) is included when ``moisture_buoyancy`` is set.
"""
from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .grid import Grid
from .state import FlowState


def _condensate_loading(state: FlowState) -> np.ndarray:
    loading = state.ql + state.qi
    for name in ("qr", "qs", "qg", "qh"):
        a = getattr(state, name, None)
        if a is not None:
            loading = loading + a
    return loading


def buoyancy_w_tendency(state: FlowState, grid: Grid, cfg: SimulationConfig,
                        T_ref: float, qv_ref: float,
                        theta0=None, qv0=None) -> np.ndarray:
    """Return the w-face tendency [m/s^2] from buoyancy.

    With a stratified base state (``theta0``/``qv0`` given, deep-convection
    scenario) the buoyancy is the **perturbation** form ``b = g[theta'/theta0 +
    0.61 q_v' - condensate loading]`` referenced to the local base state;
    otherwise the mixing-chamber form referenced to the scalar ``T_ref``.
    """
    g = cfg.flow.gravity
    if theta0 is not None:
        thp = state.theta - theta0
        if cfg.physics.moisture_buoyancy:
            qvp = state.qv - qv0
            B = g * (thp / theta0 + 0.61 * qvp - _condensate_loading(state))
        else:
            B = g * (thp / theta0)
    else:
        T = state.T if state.T is not None else state.theta
        dT = (T - T_ref) / T_ref
        if cfg.physics.moisture_buoyancy:
            B = g * (dT + 0.61 * (state.qv - qv_ref) - _condensate_loading(state))
        else:
            B = g * dT
    # average cell-centre buoyancy onto z-faces
    Bf = grid.xp.zeros(grid.w_shape)
    Bf[:, :, 1:-1] = 0.5 * (B[:, :, :-1] + B[:, :, 1:])
    Bf[:, :, 0] = B[:, :, 0]
    Bf[:, :, -1] = B[:, :, -1]
    return Bf


__all__ = ["buoyancy_w_tendency"]