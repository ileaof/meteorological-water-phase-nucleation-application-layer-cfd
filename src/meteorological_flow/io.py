"""I/O for the meteorological_flow solver: time-dependent NetCDF (xarray,
scipy engine -> NetCDF3, consistent with the rest of the repo), JSON summary,
CSV histories, and binary restart/checkpoint files.
"""
from __future__ import annotations

import json
import os

import numpy as np
import xarray as xr

from .grid import Grid
from .nucleation_adapter import NucleationField
from .state import FlowState


def _centers(state: FlowState) -> dict:
    """Interpolate staggered velocities to cell centres for output."""
    return {
        "u": 0.5 * (state.u[:-1, :, :] + state.u[1:, :, :]),
        "v": 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :]),
        "w": 0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:]),
    }


def snapshot(state: FlowState, nf: NucleationField, t: float, rho0: float) -> dict:
    """Collect a time-slice of all output fields (cell-centred) as a dict."""
    c = _centers(state)
    d = {
        "time": float(t),
        "u": c["u"], "v": c["v"], "w": c["w"],
        "T": state.T, "T_local_liquid": nf.T_local[0], "T_local_ice": nf.T_local[1],
        "P": state.P_total, "p_v": state.pv,
        "RH_water": state.RH_w, "RH_ice": state.RH_i,
        "q_v": state.qv, "q_l": state.ql, "q_i": state.qi,
        "S_w": state.S_w, "S_i": state.S_i,
        "gradT_mag": state.gradT_mag,
        "DeltaT_liquid": nf.Delta_T[0], "DeltaT_ice": nf.Delta_T[1],
        "P_eq_shift_liquid": nf.P_eq_shift[0], "P_eq_shift_ice": nf.P_eq_shift[1],
        "Gamma2_liquid": nf.Gamma2[0], "Gamma2_ice": nf.Gamma2[1],
        "rC_2nd_liquid": nf.rC_2nd[0], "rC_2nd_ice": nf.rC_2nd[1],
        "log10I_liquid": nf.log10I[0], "log10I_ice": nf.log10I[1],
        "dominant_phase": nf.dominant_phase.astype(np.float64),
        "buoyancy": np.zeros_like(state.T),  # filled by simulation if available
        "latent_heat_rate": np.zeros_like(state.T),
        "solver_residual": np.full(state.grid.center_shape, nf.residual[0].mean() if np.isfinite(nf.residual[0]).any() else 0.0),
        "validity_mask": nf.validity_mask.astype(np.float64),
        "rho": state.rho,
    }
    return d


def write_netcdf(snapshots: list, path: str, grid: Grid, attrs: dict) -> str:
    """Write a time-dependent NetCDF (scipy engine / NetCDF3) from snapshots."""
    if not snapshots:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    times = np.array([s["time"] for s in snapshots], dtype=float)
    fields = [k for k in snapshots[0] if k != "time"]
    data = {}
    for f in fields:
        # cell-centred fields are stored (nx, ny, nz); stack over time then
        # transpose to the documented CF-style (time, z, y, x) ordering.  The
        # transpose is essential for non-cubic grids (nx != nz), e.g. the
        # storm-scale run -- a cubic grid only masked the axis mislabelling.
        arr = np.stack([s[f] for s in snapshots])          # (time, nx, ny, nz)
        arr = np.transpose(arr, (0, 3, 2, 1))              # (time, nz, ny, nx)
        data[f] = (["time", "z", "y", "x"], arr)
    ds = xr.Dataset(
        data_vars=data,
        coords={"time": times, "z": grid.zc, "y": grid.yc, "x": grid.xc},
        attrs=attrs,
    )
    ds.to_netcdf(path, engine="scipy", format="NETCDF3_CLASSIC")
    return path


def write_json(obj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_csv(rows: list, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        return path
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        fh.writelines(",".join(_csv_cell(r[k]) for k in keys) + "\n" for r in rows)
    return path


def _csv_cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else repr(v)
    return str(v)


def write_restart(state: FlowState, path: str, t: float) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, t=t, u=state.u, v=state.v, w=state.w, p=state.p,
             theta=state.theta, qv=state.qv, ql=state.ql, qi=state.qi)
    return path


def load_restart(path: str, grid: Grid) -> FlowState:
    z = np.load(path)
    st = FlowState(grid=grid, u=z["u"], v=z["v"], w=z["w"], p=z["p"],
                   theta=z["theta"], qv=z["qv"], ql=z["ql"], qi=z["qi"], t=float(z["t"]))
    return st


__all__ = [
    "load_restart",
    "snapshot",
    "write_csv",
    "write_json",
    "write_netcdf",
    "write_restart",
]