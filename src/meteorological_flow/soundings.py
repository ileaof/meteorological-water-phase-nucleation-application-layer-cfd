"""Reference-sounding I/O: build a :class:`base_state.BaseState` from data.

Supports idealised analytic profiles (see :mod:`base_state`), CSV / radiosonde
tables, in-memory arrays, and NetCDF/xarray.  Data are interpolated onto the
model levels ``grid.zc``; potential temperature, density and (if absent) the
hydrostatic pressure are derived consistently with the engine thermodynamics.

Accepted humidity forms: vapour mixing ratio ``qv`` [kg/kg], relative humidity
``RH`` [%], or dew-point ``Td`` [K].  Wind may be given as components (u, v) or
as speed/direction (radiosonde convention: direction FROM, degrees).
"""
from __future__ import annotations

import numpy as np

from . import thermodynamics as th
from .base_state import BaseState


def _interp(zc, z, f):
    z = np.asarray(z, dtype=float)
    order = np.argsort(z)
    return np.interp(np.asarray(zc, dtype=float), z[order], np.asarray(f, dtype=float)[order])


def from_arrays(grid, z, T, *, p=None, qv=None, RH=None, Td=None,
                u=None, v=None, p_sfc=100000.0) -> BaseState:
    """Build a BaseState from sounding arrays (interpolated to grid.zc)."""
    # deliberately CPU-only (see base_state.py); grid.zc may be GPU-resident.
    zc = np.asarray(grid.backend.to_cpu(grid.zc), dtype=float)
    nz = zc.size
    T0 = _interp(zc, z, T)
    u0 = _interp(zc, z, u) if u is not None else np.zeros(nz)
    v0 = _interp(zc, z, v) if v is not None else np.zeros(nz)

    p0 = _interp(zc, z, p) if p is not None else None

    def _qv_from(RHi, Tdi, Ti, pi):
        if qv is not None:
            return _interp(zc, z, qv)
        if RHi is not None:
            return (RHi / 100.0) * th.q_v_from_p_v(th.psat_water(Ti), pi)
        if Tdi is not None:
            return th.q_v_from_p_v(th.psat_water(Tdi), pi)   # e = e_sat(Td)
        return np.zeros(nz)

    RH0 = _interp(zc, z, RH) if RH is not None else None
    Td0 = _interp(zc, z, Td) if Td is not None else None

    if p0 is None:
        # hydrostatic integration from p_sfc using T0 and the moisture form
        qv_col = _interp(zc, z, qv) if qv is not None else None
        p0 = np.empty(nz)
        p_prev, z_prev = float(p_sfc), 0.0
        for k in range(nz):
            dz = zc[k] - z_prev
            p_new = p_prev
            for _ in range(3):                            # fixed-point (qv needs p)
                if qv_col is not None:
                    qk = float(qv_col[k])
                elif RH0 is not None:
                    qk = float(RH0[k] / 100.0
                               * th.q_v_from_p_v(th.psat_water(float(T0[k])), p_new))
                elif Td0 is not None:
                    qk = float(th.q_v_from_p_v(th.psat_water(float(Td0[k])), p_new))
                else:
                    qk = 0.0
                Tv = float(T0[k]) * (1.0 + 0.61 * qk)
                p_new = p_prev * float(np.exp(-th.g0 * dz / (th.R_d * Tv)))
            p0[k] = p_new
            p_prev, z_prev = p_new, zc[k]

    qv0 = _qv_from(RH0, Td0, T0, p0)
    qv0 = np.maximum(np.asarray(qv0, dtype=float), 0.0)
    theta0 = th.theta_from_T(T0, p0, th.P0_REF)
    rho0 = p0 / (th.R_d * T0 * (1.0 + 0.61 * qv0))
    return BaseState(zc=zc, theta0=np.asarray(theta0), qv0=qv0, p0=np.asarray(p0),
                     T0=T0, rho0=rho0, u0=u0, v0=v0)


# canonical column aliases (lower-cased, stripped)
_ALIASES = {
    "z": ("z", "height", "height_m", "altitude", "gph", "z_m"),
    "T": ("t", "temp", "temperature", "t_k", "temperature_k"),
    "p": ("p", "pres", "pressure", "p_pa", "pressure_pa", "pres_pa"),
    "qv": ("qv", "q", "mixing_ratio", "qv_kgkg", "q_v"),
    "RH": ("rh", "relhum", "relative_humidity", "rh_percent"),
    "Td": ("td", "dewpoint", "dew_point", "td_k"),
    "u": ("u", "u_ms", "uwind"),
    "v": ("v", "v_ms", "vwind"),
    "speed": ("speed", "wind_speed", "ff", "spd"),
    "direction": ("dir", "direction", "wind_dir", "dd", "wdir"),
}


def _map_columns(header):
    hl = [h.strip().lower() for h in header]
    out = {}
    for canon, al in _ALIASES.items():
        for i, h in enumerate(hl):
            if h in al:
                out[canon] = i
                break
    return out


def _wind_components(speed, direction):
    """Meteorological wind (speed, direction-FROM in degrees) -> (u, v)."""
    rad = np.deg2rad(np.asarray(direction, dtype=float))
    spd = np.asarray(speed, dtype=float)
    u = -spd * np.sin(rad)
    v = -spd * np.cos(rad)
    return u, v


def from_csv(grid, path, *, p_sfc=100000.0) -> BaseState:
    """Read a CSV/radiosonde table (named columns; see ``_ALIASES``)."""
    import csv
    with open(path, "r", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith("#")]
    header, data = rows[0], rows[1:]
    cols = _map_columns(header)
    if "z" not in cols or "T" not in cols:
        raise ValueError("sounding CSV needs at least height and temperature columns")
    arr = np.array([[float(x) for x in r] for r in data], dtype=float)

    def col(name):
        return arr[:, cols[name]] if name in cols else None

    u = col("u")
    v = col("v")
    if u is None and "speed" in cols and "direction" in cols:
        u, v = _wind_components(col("speed"), col("direction"))
    return from_arrays(grid, col("z"), col("T"), p=col("p"), qv=col("qv"),
                       RH=col("RH"), Td=col("Td"), u=u, v=v, p_sfc=p_sfc)


def from_netcdf(grid, path, *, p_sfc=100000.0) -> BaseState:
    """Read a sounding from a NetCDF file via xarray (standard variable names)."""
    import xarray as xr
    ds = xr.open_dataset(path)
    try:
        def g(*names):
            for n in names:
                if n in ds:
                    return np.asarray(ds[n].values, dtype=float).ravel()
            return None
        z = g("z", "height", "altitude")
        T = g("T", "temperature", "t")
        return from_arrays(grid, z, T, p=g("p", "pressure"), qv=g("qv", "q"),
                           RH=g("RH", "rh"), Td=g("Td", "dewpoint"),
                           u=g("u"), v=g("v"), p_sfc=p_sfc)
    finally:
        ds.close()


def to_csv(base: BaseState, path: str) -> str:
    """Write a BaseState as a sounding CSV (z, p, T, qv, u, v)."""
    import csv
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["z_m", "p_Pa", "T_K", "qv_kgkg", "u_ms", "v_ms"])
        for k in range(base.zc.size):
            w.writerow(["%.2f" % base.zc[k], "%.2f" % base.p0[k], "%.3f" % base.T0[k],
                        "%.6e" % base.qv0[k], "%.3f" % base.u0[k], "%.3f" % base.v0[k]])
    return path


__all__ = ["from_arrays", "from_csv", "from_netcdf", "to_csv"]
