"""Meteorological diagnostics and conservation budgets.

Qualifiers (masks over the grid) follow the spec's list: vapor subsaturation,
liquid/ice supersaturation, droplet / ice-crystal nucleation, supercooled
liquid, mixed phase, cloud, and rain/snow/graupel/hail *favorability*.

IMPORTANT (scientific integrity): rain/snow/graupel/hail are reported as
**thermodynamic / microphysical favorability** diagnostics, NOT predictions of
precipitation.  Hydrometeor growth (collision-coalescence, riming, accretion,
melting) is not modelled in Batch 1 (and is an optional Batch-2 extension); a
high nucleation rate never by itself implies rain or hail.
"""
from __future__ import annotations

import numpy as np

from .nucleation_adapter import NucleationField
from .state import FlowState


def qualifiers(state: FlowState, nf: NucleationField) -> dict:
    """Return a dict of boolean masks qualifying the local state."""
    T = state.T
    Sw, Si = state.S_w, state.S_i
    ql, qi = state.ql, state.qi
    wc = vertical_velocity_center(state)
    # nucleation "active" where the respective phase is supersaturated AND a
    # finite kernel rate is reported (supersaturation is the physical gate; the
    # kernel rate is the shifted-equilibrium tendency).
    liq_active = (Sw > 1.0) & np.isfinite(nf.log10I[0])
    ice_active = (Si > 1.0) & np.isfinite(nf.log10I[1])
    out = {
        "vapor_subsaturation": (Sw < 1.0) & (Si < 1.0),
        "liquid_supersaturation": Sw > 1.0,
        "ice_supersaturation": Si > 1.0,
        "droplet_nucleation": liq_active,
        "ice_crystal_nucleation": ice_active,
        "supercooled_liquid": (T < 273.16) & (ql > 1e-9),
        "mixed_phase": (ql > 1e-9) & (qi > 1e-9),
        "cloud": (ql + qi) > 1e-8,
        # favorability (NOT prediction): warm-rain favors warm, LWC, updraft;
        # snow favors ice supersaturation; hail favors strong updraft + LWC + cold.
        "rain_favorable": (T > 273.16) & (Sw > 1.0) & (ql > 1e-9),
        "snow_favorable": (T < 273.16) & (Si > 1.0) & (ice_active),
        "graupel_favorable": (T < 273.16) & (ql > 1e-9) & (qi > 1e-9),
        "hail_favorable": (T < 273.16) & (wc > 5.0) & (ql > 1e-9),
    }
    return out


def vertical_velocity_center(state: FlowState) -> np.ndarray:
    return 0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:])


def summary_stats(state: FlowState, nf: NucleationField) -> dict:
    """Extrema and aggregate metrics for the JSON report."""
    umag = state.velocity_magnitude_center()
    wc = vertical_velocity_center(state)
    liq = nf.log10I[0]; ice = nf.log10I[1]
    liq_f = liq[np.isfinite(liq)] if np.any(np.isfinite(liq)) else np.array([0.0])
    ice_f = ice[np.isfinite(ice)] if np.any(np.isfinite(ice)) else np.array([0.0])
    return {
        "T_min": float(np.nanmin(state.T)), "T_max": float(np.nanmax(state.T)),
        "qv_min": float(np.nanmin(state.qv)), "qv_max": float(np.nanmax(state.qv)),
        "S_w_max": float(np.nanmax(state.S_w)), "S_i_max": float(np.nanmax(state.S_i)),
        "S_w_min": float(np.nanmin(state.S_w)),
        "umax": float(np.nanmax(umag)), "wmax": float(np.nanmax(np.abs(wc))),
        "gradT_max": float(np.nanmax(state.gradT_mag)),
        "log10I_liq_max": float(np.nanmax(liq_f)),
        "log10I_ice_max": float(np.nanmax(ice_f)),
        "n_liq_nucleation_cells": int(np.sum((state.S_w > 1.0) & np.isfinite(liq))),
        "n_ice_nucleation_cells": int(np.sum((state.S_i > 1.0) & np.isfinite(ice))),
        "total_water_kg": float(state.total_water()),
    }


def conservation_budgets(state: FlowState, initial: dict, rho0: float) -> dict:
    """Conservation diagnostics vs the initial state (absolute + relative).

    NOTE: with open inflow/outflow boundaries these are NOT expected to be
    conserved (mass/energy flux through boundaries).  In a closed/periodic
    domain they should hold to discretization error.
    """
    g = state.grid
    dv = g.cell_vol
    tw = float((state.qv + state.ql + state.qi).sum() * dv)
    ke = 0.5 * rho0 * float((state.u ** 2).sum() * (g.dx * g.dy * g.dz)
                           + (state.v ** 2).sum() * (g.dx * g.dy * g.dz)
                           + (state.w ** 2).sum() * (g.dx * g.dy * g.dz))
    zc = g.zc.reshape(1, 1, -1)
    pe = rho0 * 9.81 * float((zc * np.ones(g.center_shape)).sum() * dv)
    th = rho0 * 1005.0 * float(state.T.sum() * dv)
    return {
        "total_water_kg": tw,
        "total_water_rel_err": (tw - initial["total_water"]) / max(abs(initial["total_water"]), 1e-12),
        "kinetic_energy_J": ke, "potential_energy_J": pe, "thermal_energy_J": th,
        "total_energy_J": ke + pe + th,
        "total_energy_rel_err": ((ke + pe + th) - initial["total_energy"]) / max(abs(initial["total_energy"]), 1e-12),
    }


def initial_budgets(state: FlowState, rho0: float) -> dict:
    g = state.grid
    dv = g.cell_vol
    tw = float((state.qv + state.ql + state.qi).sum() * dv)
    ke = 0.5 * rho0 * float((state.u ** 2).sum() + (state.v ** 2).sum() + (state.w ** 2).sum()) * dv
    zc = g.zc.reshape(1, 1, -1)
    pe = rho0 * 9.81 * float((zc * np.ones(g.center_shape)).sum() * dv)
    th = rho0 * 1005.0 * float(state.T.sum() * dv)
    return {"total_water": tw, "total_energy": ke + pe + th,
            "kinetic_energy": ke, "potential_energy": pe, "thermal_energy": th}


__all__ = [
    "conservation_budgets",
    "initial_budgets",
    "qualifiers",
    "summary_stats",
    "vertical_velocity_center",
]