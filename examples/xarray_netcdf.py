#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_met_xarray_netcdf.py
=============================
Build a small xarray.Dataset (a 1-D vertical field of T, P, p_v, dynamics),
write it to NetCDF3 (scipy engine -- no extra backend needed), read it back
via `from_netcdf`, run the meteorological nucleation layer on each level, and
write the per-phase report as JSON, CSV and NetCDF.

Demonstrates:
  * xarray field ingestion (from_xarray);
  * NetCDF3 round-trip via the scipy engine (no netCDF4/h5netcdf required);
  * the `evaluate_profile` driver producing per-level reports;
  * structured xarray/NetCDF output of the mandatory schema.

Note: GRIB and NetCDF4/HDF5 ingestion require backends (cfgrib, netCDF4 or
h5netcdf) not installed in this environment; those paths degrade gracefully to
'undetermined' naming the missing dependency (see from_grib / from_netcdf).
"""
import io
import os
import sys

import numpy as np
import xarray as xr

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

# Bootstrap the package import when running from a source checkout without
# `pip install -e .` (path resolved relative to this file, not to CWD).
try:
    import met_water_nucleation as M
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    import met_water_nucleation as M

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "xarray_netcdf")
os.makedirs(OUTDIR, exist_ok=True)


def build_field():
    z = np.array([800.0, 1500.0, 2500.0, 3500.0, 4500.0, 5500.0])
    T = np.array([279.0, 273.5, 267.0, 260.0, 253.0, 246.0])
    P = np.array([92000.0, 85000.0, 75000.0, 66000.0, 57000.0, 49000.0])
    # vapour partial pressure: keep near-ice-saturation aloft, subsaturated low
    p_v = np.array([800.0, 600.0, 380.0, 220.0, 120.0, 60.0])
    w = np.array([0.5, 2.0, 3.0, 3.0, 1.5, 0.5])
    LWC = np.array([0.0, 3e-4, 6e-4, 6e-4, 2e-4, 0.0])
    IWC = np.array([0.0, 0.0, 1e-5, 1e-4, 3e-4, 2e-4])
    ds = xr.Dataset(
        {"T": ("z", T), "P": ("z", P), "p_v": ("z", p_v),
         "w": ("z", w), "LWC": ("z", LWC), "IWC": ("z", IWC)},
        coords={"z": z})
    return ds


def main():
    ds = build_field()
    print("Input xarray.Dataset:")
    print(ds)

    # write input field to NetCDF3 (scipy engine) and read back
    infield = os.path.join(OUTDIR, "met_input_field.nc")
    for eng in ("netcdf4", "h5netcdf", "scipy"):
        try:
            ds.to_netcdf(infield, engine=eng)
            print(f"\nWrote input field -> {infield} (engine={eng})")
            break
        except Exception:
            continue

    met = M.from_netcdf(infield)
    print("\nReconstructed MetInput from NetCDF:")
    print(f"  T = {np.asarray(met.T)}")
    print(f"  P = {np.asarray(met.P)}")
    print(f"  p_v = {np.asarray(met.p_v)}")
    print(f"  w = {np.asarray(met.w)}")

    # run per-level
    met.phase_mode = "both"
    met.dt_micro = 60.0
    met.cell_volume = 1.0e6
    runner = M.MetNucleationRunner(met)
    T = np.asarray(met.T).reshape(-1)
    P = np.asarray(met.P).reshape(-1)
    pv = np.asarray(met.p_v).reshape(-1)
    w = np.asarray(met.w).reshape(-1)
    LWC = np.asarray(met.LWC).reshape(-1)
    IWC = np.asarray(met.IWC).reshape(-1)

    all_reps = []
    for i in range(len(T)):
        reps = runner.evaluate_point(float(T[i]), float(P[i]), float(pv[i]),
                                      dynamics={"w": float(w[i]),
                                                "LWC": float(LWC[i]),
                                                "IWC": float(IWC[i]),
                                                "z": float(np.asarray(met.z)[i])})
        all_reps.append(reps)

    # outputs: JSON, CSV, NetCDF
    M.to_json(all_reps, os.path.join(OUTDIR, "xarray_report.json"))
    M.to_csv(all_reps, os.path.join(OUTDIR, "xarray_report.csv"))
    M.to_netcdf(all_reps, os.path.join(OUTDIR, "xarray_report.nc"))
    print(f"\nSaved reports -> {OUTDIR}/xarray_report.{{json,csv,nc}}")

    # show the output xarray Dataset
    outds = M.to_xarray(all_reps)
    print("\nOutput xarray.Dataset (per-phase mandatory fields):")
    print(outds)
    print("\nphase_names attr:", outds.attrs.get("phase_names"))
    return 0


if __name__ == "__main__":
    sys.exit(main())