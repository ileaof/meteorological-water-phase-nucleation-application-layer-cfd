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


def _centers(state: FlowState, to_cpu) -> dict:
    """Interpolate staggered velocities to cell centres for output."""
    return {
        "u": to_cpu(0.5 * (state.u[:-1, :, :] + state.u[1:, :, :])),
        "v": to_cpu(0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])),
        "w": to_cpu(0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:])),
    }


def _hydro(state: FlowState, name: str, to_cpu) -> np.ndarray:
    """Precipitating-species field, or zeros if the run does not carry it."""
    a = getattr(state, name, None)
    return to_cpu(a) if a is not None else np.zeros(state.grid.center_shape)


def snapshot(state: FlowState, nf: NucleationField, t: float, rho0: float) -> dict:
    """Collect a time-slice of all output fields (cell-centred) as a dict.

    ``nf`` (the nucleation field) is always host/NumPy already -- it is
    produced by the CPU-side lookup/engine layer. ``state``'s fields may be
    GPU-resident (CuPy); every one is pulled to the host here, at the output
    boundary, via ``to_cpu`` -- NetCDF/JSON/CSV writers only ever see NumPy.
    """
    to_cpu = state.grid.backend.to_cpu
    c = _centers(state, to_cpu)
    d = {
        "time": float(t),
        "u": c["u"], "v": c["v"], "w": c["w"],
        "T": to_cpu(state.T), "T_local_liquid": nf.T_local[0], "T_local_ice": nf.T_local[1],
        "P": to_cpu(state.P_total), "p_v": to_cpu(state.pv),
        "RH_water": to_cpu(state.RH_w), "RH_ice": to_cpu(state.RH_i),
        "q_v": to_cpu(state.qv), "q_l": to_cpu(state.ql), "q_i": to_cpu(state.qi),
        "q_r": _hydro(state, "qr", to_cpu), "q_s": _hydro(state, "qs", to_cpu),
        "q_g": _hydro(state, "qg", to_cpu), "q_h": _hydro(state, "qh", to_cpu),
        "S_w": to_cpu(state.S_w), "S_i": to_cpu(state.S_i),
        "gradT_mag": to_cpu(state.gradT_mag),
        "DeltaT_liquid": nf.Delta_T[0], "DeltaT_ice": nf.Delta_T[1],
        "P_eq_shift_liquid": nf.P_eq_shift[0], "P_eq_shift_ice": nf.P_eq_shift[1],
        "Gamma2_liquid": nf.Gamma2[0], "Gamma2_ice": nf.Gamma2[1],
        "rC_2nd_liquid": nf.rC_2nd[0], "rC_2nd_ice": nf.rC_2nd[1],
        "log10I_liquid": nf.log10I[0], "log10I_ice": nf.log10I[1],
        "dominant_phase": nf.dominant_phase.astype(np.float64),
        "buoyancy": np.zeros(state.grid.center_shape),  # filled by simulation if available
        "latent_heat_rate": np.zeros(state.grid.center_shape),
        "solver_residual": np.full(state.grid.center_shape, nf.residual[0].mean() if np.isfinite(nf.residual[0]).any() else 0.0),
        "validity_mask": nf.validity_mask.astype(np.float64),
        "rho": to_cpu(state.rho),
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
    # 64-bit offset format lifts the 2 GB NetCDF3-CLASSIC limit (int32 offsets),
    # which a large grid x many snapshots easily exceeds (e.g. 64^3 x ~50 slices).
    # Guard the write so a NetCDF failure never discards the completed run's
    # summary.json / history.csv (the physics is already done at this point).
    n_bytes = sum(v[1].nbytes for v in data.values())
    try:
        ds.to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")
    except Exception as exc:                                  # noqa: BLE001
        import warnings
        warnings.warn(
            "flow.nc not written (%s). Estimated size ~%.2f GB. For large grids "
            "increase --output-interval (fewer time slices) or reduce N."
            % (exc, n_bytes / 1e9), RuntimeWarning)
    return path


# Tecplot 360 ASCII export -------------------------------------------------
# Matches the CFDPYGPU dialect (visualization/tecplot_writer.py): a structured
# ORDERED zone with DATAPACKING=POINT, one node record per line, STRANDID +
# SOLUTIONTIME for time animation, and Fortran-order flattening so the I (x)
# index varies fastest, then J (y), then K (z).  Variable names carry their SI
# unit in brackets (py2tec / Tecplot 360 accept quoted names with brackets).
#
# The meteorological variable set (adapted from the CFDPYGPU X,Y,Z,U,V,W,
# Pressure,Temperature,Alpha core): the cloud condensate q_cloud (=q_l+q_i)
# plays the role of the VOF "Alpha" phase fraction, and the precipitating
# species q_rain/q_snow/q_graupel/q_hail are exported as their own variables.
_TECPLOT_VARS = [
    "X [m]", "Y [m]", "Z [m]",
    "U [m/s]", "V [m/s]", "W [m/s]",
    "Pressure [Pa]", "Temperature [K]",
    "q_v [kg/kg]", "q_cloud [kg/kg]",
    "q_rain [kg/kg]", "q_snow [kg/kg]", "q_graupel [kg/kg]", "q_hail [kg/kg]",
    "S_w [-]",
]


def _tecplot_columns(grid: Grid, snap: dict, Xf, Yf, Zf):
    """Fortran-flattened node columns for one snapshot (order matches _TECPLOT_VARS).

    Missing precipitating-species keys default to zeros so the exporter also
    works for one-way / no-microphysics snapshots.
    """
    shp = grid.center_shape

    def r(key):
        a = snap.get(key)
        return np.zeros(shp).ravel(order="F") if a is None else np.asarray(a).ravel(order="F")

    qcloud = (np.asarray(snap["q_l"]) + np.asarray(snap["q_i"])).ravel(order="F")
    return np.column_stack([
        Xf, Yf, Zf,
        r("u"), r("v"), r("w"), r("P"), r("T"),
        r("q_v"), qcloud, r("q_r"), r("q_s"), r("q_g"), r("q_h"), r("S_w"),
    ])


def write_tecplot(snapshots: list, path: str, grid: Grid, attrs: dict | None = None,
                  title: str = "meteorological_flow") -> str:
    """Write a Tecplot 360 ASCII ``.dat`` (one ORDERED/POINT zone per snapshot).

    Zones share ``STRANDID=1`` and carry ``SOLUTIONTIME`` so Tecplot 360 treats
    the file as a time sequence (Play animates the storm).  Matches the CFDPYGPU
    ``TecplotExporter`` dialect; readable by py2tec / ParaView (Tecplot reader).
    """
    if not snapshots:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # cell-centre mesh, Fortran-flattened (I=x fastest, then J=y, then K=z).
    # grid.xc/yc/zc may be GPU-resident; np.meshgrid/np.column_stack (used by
    # _tecplot_columns below) are host-only, so convert once here -- the
    # snapshot dicts themselves are already host (see io.snapshot's to_cpu
    # boundary), this is the one remaining GPU-resident input to this writer.
    to_cpu = grid.backend.to_cpu
    Xg, Yg, Zg = np.meshgrid(to_cpu(grid.xc), to_cpu(grid.yc), to_cpu(grid.zc), indexing="ij")
    Xf, Yf, Zf = (a.ravel(order="F") for a in (Xg, Yg, Zg))
    dims = "I=%d J=%d K=%d" % (grid.nx, grid.ny, grid.nz)
    # guard the write so a failure never discards the run's summary/history
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('TITLE = "%s"\n' % title)
            fh.write("VARIABLES = " + ", ".join('"%s"' % v for v in _TECPLOT_VARS) + "\n")
            for s in snapshots:
                t = float(s["time"])
                fh.write('ZONE T="t=%.6f" ZONETYPE=ORDERED %s '
                         'DATAPACKING=POINT STRANDID=1 SOLUTIONTIME=%.6f\n'
                         % (t, dims, t))
                np.savetxt(fh, _tecplot_columns(grid, s, Xf, Yf, Zf), fmt="%.6e")
                fh.write("\n")
    except Exception as exc:                                  # noqa: BLE001
        import warnings
        warnings.warn("flow.dat (Tecplot) not written (%s)." % exc, RuntimeWarning)
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
    to_cpu = state.grid.backend.to_cpu
    np.savez(path, t=t, u=to_cpu(state.u), v=to_cpu(state.v), w=to_cpu(state.w),
             p=to_cpu(state.p), theta=to_cpu(state.theta), qv=to_cpu(state.qv),
             ql=to_cpu(state.ql), qi=to_cpu(state.qi))
    return path


def load_restart(path: str, grid: Grid) -> FlowState:
    z = np.load(path)
    xp = grid.xp
    st = FlowState(grid=grid, u=xp.asarray(z["u"]), v=xp.asarray(z["v"]), w=xp.asarray(z["w"]),
                   p=xp.asarray(z["p"]), theta=xp.asarray(z["theta"]), qv=xp.asarray(z["qv"]),
                   ql=xp.asarray(z["ql"]), qi=xp.asarray(z["qi"]), t=float(z["t"]))
    return st


__all__ = [
    "load_restart",
    "snapshot",
    "write_csv",
    "write_json",
    "write_netcdf",
    "write_restart",
    "write_tecplot",
]