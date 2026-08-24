"""Meteorological diagnostics and conservation budgets.

Qualifiers (masks over the grid) follow the spec's list: vapor subsaturation,
liquid/ice supersaturation, droplet / ice-crystal nucleation, supercooled
liquid, mixed phase, cloud, and rain/snow/graupel/hail *favorability*.

IMPORTANT (scientific integrity): in the one-way stage rain/snow/graupel/hail are
reported as **thermodynamic / microphysical favorability** diagnostics, NOT
predictions of precipitation -- a high nucleation rate never by itself implies
rain or hail.  The two-way stage (`precip_microphysics`) does model hydrometeor
growth (condensation/deposition, autoconversion, accretion, riming, melting) and
sedimentation, so precipitation forms -- qualitatively at demonstration resolution.
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


def surface_water_kg(state: FlowState) -> float:
    """Accumulated surface precipitation as a water mass [kg].

    ``surface_precip[c]`` is [kg/m^2] over (x,y); times the cell footprint dx*dy
    and summed gives the mass that has left the airborne field via sedimentation.
    """
    sp = getattr(state, "surface_precip", None)
    if not sp:
        return 0.0
    g = state.grid
    area = g.dx * g.dy
    return float(sum(np.asarray(v).sum() for v in sp.values()) * area)


def _rho_weight(rho, grid):
    """Density weighting: None/scalar -> as-is; (nz,) profile -> (1,1,nz)."""
    if rho is None:
        return 1.0
    if np.ndim(rho) == 0:
        return float(rho)
    return np.asarray(rho, dtype=float).reshape(1, 1, -1)


def total_water_kg(state: FlowState, rho=None) -> float:
    """Complete water inventory [kg]: airborne (vapour + all hydrometeors) plus
    the water accumulated at the surface.  Sedimentation moves water from airborne
    to surface, so the SUM is what a closed domain should conserve.

    ``rho`` is the density weighting used for the airborne term.  For the anelastic
    system pass the reference-density PROFILE rho0(z) -- the transport conserves
    ``int rho0 q``, so weighting by rho0(z) is what makes the budget consistent
    (an unweighted sum spuriously drifts as water is redistributed vertically
    through the rho0 gradient by a strong updraft; M6).
    """
    r = _rho_weight(rho, state.grid)
    q = state.qv + state.ql + state.qi
    for nm in ("qr", "qs", "qg", "qh"):
        a = getattr(state, nm, None)
        if a is not None:
            q = q + a
    return float((r * q).sum() * state.grid.cell_vol) + surface_water_kg(state)


def mass_continuity_residual(state: FlowState, rho0_c=None, rho0_wface=None) -> dict:
    """Interior residual of the (an)elastic mass constraint.

    Anelastic (rho0_c, rho0_wface given): max |div(rho0 u)| over interior cells,
    normalised by a characteristic mass-flux-divergence scale.  Boussinesq
    (rho0_* None): max |div u|.  Small values confirm the projection enforces the
    core's continuity constraint (mass conservation), not the limiters.
    """
    g = state.grid
    if rho0_c is not None and rho0_wface is not None:
        rc = np.asarray(rho0_c).reshape(1, 1, -1)
        rwf = np.asarray(rho0_wface).reshape(1, 1, -1)
        dudx = (state.u[1:] - state.u[:-1]) / g.dx
        dvdy = (state.v[:, 1:] - state.v[:, :-1]) / g.dy
        wflux = rwf * state.w
        dwdz = (wflux[:, :, 1:] - wflux[:, :, :-1]) / g.dz
        div = rc * (dudx + dvdy) + dwdz
        wmax = float(np.max(np.abs(state.w))) if state.w.size else 0.0
        scale = float(np.max(np.abs(rho0_c))) * wmax / g.dz + 1e-12
    else:
        div = g.divergence(state.u, state.v, state.w)
        umax = float(np.max(np.abs(state.velocity_magnitude_center())))
        scale = umax / min(g.dx, g.dy, g.dz) + 1e-12
    interior = div[1:-1, 1:-1, 1:-1] if min(div.shape) > 2 else div
    absmax = float(np.max(np.abs(interior))) if interior.size else 0.0
    return {"abs_max": absmax, "normalised": absmax / scale}


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


def _energy_budget(state: FlowState, rho) -> tuple:
    """Density-weighted kinetic / potential / thermal energy [J].  ``rho`` is a
    scalar (Boussinesq) or the reference-density PROFILE rho0(z) (anelastic), so
    the budget is consistent with the mass the transport actually conserves."""
    g = state.grid
    dv = g.cell_vol
    r = _rho_weight(rho, g)
    umag2 = state.velocity_magnitude_center() ** 2          # (nx,ny,nz)
    ke = 0.5 * float((r * umag2).sum() * dv)
    zc = g.zc.reshape(1, 1, -1) * np.ones(g.center_shape)
    pe = 9.81 * float((r * zc).sum() * dv)
    th = 1005.0 * float((r * state.T).sum() * dv)
    return ke, pe, th


def conservation_budgets(state: FlowState, initial: dict, rho=None) -> dict:
    """Conservation diagnostics vs the initial state (absolute + relative),
    density-weighted by ``rho`` (scalar Boussinesq, or rho0(z) profile anelastic).

    NOTE: with open inflow/outflow boundaries these are NOT expected to be
    conserved (mass/energy flux through boundaries).  In a closed/periodic
    domain they should hold to discretization error.
    """
    tw = total_water_kg(state, rho)     # rho0-weighted airborne + surface accumulation
    ke, pe, th = _energy_budget(state, rho)
    return {
        "total_water_kg": tw,
        "total_water_rel_err": (tw - initial["total_water"]) / max(abs(initial["total_water"]), 1e-12),
        "kinetic_energy_J": ke, "potential_energy_J": pe, "thermal_energy_J": th,
        "total_energy_J": ke + pe + th,
        "total_energy_rel_err": ((ke + pe + th) - initial["total_energy"]) / max(abs(initial["total_energy"]), 1e-12),
    }


def initial_budgets(state: FlowState, rho=None) -> dict:
    tw = total_water_kg(state, rho)
    ke, pe, th = _energy_budget(state, rho)
    return {"total_water": tw, "total_energy": ke + pe + th,
            "kinetic_energy": ke, "potential_energy": pe, "thermal_energy": th}


__all__ = [
    "conservation_budgets",
    "initial_budgets",
    "mass_continuity_residual",
    "qualifiers",
    "summary_stats",
    "surface_water_kg",
    "total_water_kg",
    "vertical_velocity_center",
]