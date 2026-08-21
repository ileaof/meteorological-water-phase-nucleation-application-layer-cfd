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


def buoyancy_w_tendency(state: FlowState, grid: Grid, cfg: SimulationConfig,
                        T_ref: float, qv_ref: float) -> np.ndarray:
    """Return the w-face tendency [m/s^2] from buoyancy."""
    T = state.T if state.T is not None else state.theta
    dT = (T - T_ref) / T_ref
    if cfg.physics.moisture_buoyancy:
        B = cfg.flow.gravity * (dT + 0.61 * (state.qv - qv_ref) - (state.ql + state.qi))
    else:
        B = cfg.flow.gravity * dT
    # average cell-centre buoyancy onto z-faces
    Bf = np.zeros(grid.w_shape)
    Bf[:, :, 1:-1] = 0.5 * (B[:, :, :-1] + B[:, :, 1:])
    Bf[:, :, 0] = B[:, :, 0]
    Bf[:, :, -1] = B[:, :, -1]
    return Bf


__all__ = ["buoyancy_w_tendency"]