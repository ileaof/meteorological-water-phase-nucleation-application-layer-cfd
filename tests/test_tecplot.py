"""Tests for the Tecplot 360 ASCII exporter (io.write_tecplot).

Verifies the CFDPYGPU-dialect structure: ORDERED/POINT zones, one zone per
snapshot with STRANDID + SOLUTIONTIME, I*J*K node records per zone in Fortran
(I-fastest) order, and the CLI/config --tecplot wiring.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from meteorological_flow import io as fio
from meteorological_flow.config import SimulationConfig, apply_overrides
from meteorological_flow.grid import Grid


def _fake_snapshot(grid, t):
    shp = grid.center_shape
    ones = np.ones(shp)
    return {
        "time": t,
        "u": 0.1 * ones, "v": 0.2 * ones, "w": 0.3 * ones,
        "P": 1.0e5 * ones, "T": 290.0 * ones,
        "q_v": 1e-2 * ones, "q_l": 1e-4 * ones, "q_i": 2e-4 * ones,
        "q_r": 5e-4 * ones, "q_s": 6e-4 * ones, "q_g": 7e-4 * ones, "q_h": 8e-4 * ones,
        "S_w": 1.05 * ones,
    }


def _parse(path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    header = [ln for ln in lines if ln.startswith(("TITLE", "VARIABLES"))]
    zones = [ln for ln in lines if ln.startswith("ZONE")]
    data = [ln for ln in lines if ln and ln[0].lstrip()[:1].isdigit() or ln.startswith("-")]
    return header, zones, lines


def test_tecplot_structure_and_zone_count():
    g = Grid(nx=4, ny=5, nz=6, Lx=4000, Ly=5000, Lz=6000)
    snaps = [_fake_snapshot(g, t) for t in (0.0, 10.0, 20.0)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "flow.dat")
        fio.write_tecplot(snaps, p, g)
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        lines = text.splitlines()
    assert lines[0].startswith('TITLE =')
    assert lines[1].startswith('VARIABLES =') and lines[1].count('"') == 2 * 15
    zone_hdrs = [ln for ln in lines if ln.startswith("ZONE")]
    assert len(zone_hdrs) == 3                                  # one zone per snapshot
    for z, t in zip(zone_hdrs, (0.0, 10.0, 20.0)):
        assert "ZONETYPE=ORDERED" in z and "DATAPACKING=POINT" in z
        assert "I=4 J=5 K=6" in z and "STRANDID=1" in z
        assert ("SOLUTIONTIME=%.6f" % t) in z
    # each zone carries I*J*K = 120 node records
    data_lines = [ln for ln in lines if ln[:1].isdigit() or ln.startswith("-")]
    assert len(data_lines) == 3 * (4 * 5 * 6)


def test_tecplot_fortran_order_and_values():
    g = Grid(nx=3, ny=2, nz=2, Lx=3000, Ly=2000, Lz=2000)
    snap = _fake_snapshot(g, 5.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.dat")
        fio.write_tecplot([snap], p, g)
        rows = np.loadtxt(p, skiprows=3)          # TITLE, VARIABLES, ZONE
    # first two nodes differ in X only (I varies fastest): xc=[500,1500,2500]
    assert np.isclose(rows[0, 0], 500.0) and np.isclose(rows[1, 0], 1500.0)
    assert np.isclose(rows[0, 1], rows[1, 1])     # same Y
    # column order: X0 Y1 Z2 U3 V4 W5 P6 T7 qv8 qcloud9 qr10 qs11 qg12 qh13 Sw14
    assert np.allclose(rows[:, 3], 0.1) and np.allclose(rows[:, 5], 0.3)
    assert np.allclose(rows[:, 9], 3e-4)          # q_cloud = q_l+q_i
    assert np.allclose(rows[:, 10], 5e-4)         # q_rain
    assert np.allclose(rows[:, 13], 8e-4)         # q_hail
    assert np.allclose(rows[:, 14], 1.05)         # S_w
    assert rows.shape == (3 * 2 * 2, 15)


def test_cli_tecplot_flag_adds_format():
    cfg = apply_overrides(SimulationConfig(), storm_scale=True, tecplot=True)
    assert "tecplot" in cfg.output.format
    cfg2 = apply_overrides(SimulationConfig(), storm_scale=True)
    assert "tecplot" not in cfg2.output.format


def _gpu_available() -> bool:
    try:
        import cupy
        _ = cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(),
                    reason="no working CUDA/CuPy GPU in this environment")
def test_tecplot_writes_on_gpu_backed_grid():
    # regression test for a real bug: write_tecplot built its coordinate mesh
    # from grid.xc/yc/zc directly (np.meshgrid) without converting a
    # GPU-resident grid to host first -- np.column_stack (inside
    # _tecplot_columns) then raised "Only cupy arrays can be column stacked",
    # silently truncating flow.dat to just its header (caught via a real
    # --storm-scale --z-stretch ... --tecplot --device auto run).
    from meteorological_flow.backend import get_backend
    backend = get_backend("gpu")
    g = Grid(nx=4, ny=5, nz=6, Lx=4000, Ly=5000, Lz=6000, backend=backend)
    snaps = [_fake_snapshot(g, t) for t in (0.0, 10.0)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "flow.dat")
        fio.write_tecplot(snaps, p, g)
        size = os.path.getsize(p)
        assert size > 1000, "flow.dat was truncated (GPU coordinate conversion regressed)"
        with open(p, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    zone_hdrs = [ln for ln in lines if ln.startswith("ZONE")]
    assert len(zone_hdrs) == 2
