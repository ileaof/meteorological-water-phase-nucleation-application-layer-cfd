"""CPU-friendly matplotlib visualisation (Agg backend): central horizontal and
vertical slices of the key fields, velocity vectors, and budget histories.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .grid import Grid


def _slice(arr, axis, mid):
    if axis == "h":       # horizontal slice at z=mid
        return arr[:, :, mid]
    if axis == "v":       # vertical slice at y=mid
        return arr[:, mid, :]
    raise ValueError(axis)


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _field_slice(ax, grid: Grid, arr, axis, mid, title, cmap="viridis"):
    to_cpu = grid.backend.to_cpu
    a = _slice(arr, axis, mid)
    if axis == "h":
        X, Y = np.meshgrid(to_cpu(grid.xc), to_cpu(grid.yc), indexing="ij")
        im = ax.pcolormesh(X, Y, a, shading="auto", cmap=cmap)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    else:
        X, Z = np.meshgrid(to_cpu(grid.xc), to_cpu(grid.zc), indexing="ij")
        im = ax.pcolormesh(X, Z, a, shading="auto", cmap=cmap)
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.set_title(title)
    fig = ax.figure
    fig.colorbar(im, ax=ax)


def plot_snapshot(state, nf, grid: Grid, outdir: str, t: float, tag: str = "") -> list:
    """Plot the standard slice suite for one time slice; returns file list.

    matplotlib has no CuPy support, so every ``state``-derived array is
    pulled to the host here, at the plotting boundary; ``nf`` (the
    nucleation field) is always host/NumPy already."""
    files = []
    to_cpu = grid.backend.to_cpu
    nz, ny = grid.nz, grid.ny
    midz, midy = nz // 2, ny // 2
    uc = to_cpu(0.5 * (state.u[:-1, :, :] + state.u[1:, :, :]))
    vc = to_cpu(0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :]))
    wc_full = to_cpu(0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:]))
    umag = np.sqrt(uc ** 2 + vc ** 2 + wc_full ** 2)
    panels = [
        ("T", to_cpu(state.T), "viridis", "Temperature [K]"),
        ("S_w", to_cpu(state.S_w), "BrBG", "S_w (liquid)"),
        ("S_i", to_cpu(state.S_i), "BrBG", "S_i (ice)"),
        ("p_prime", to_cpu(state.p), "RdBu_r", "p' [Pa]"),
        ("gradT", to_cpu(state.gradT_mag), "magma", "|gradT| [K/m]"),
        ("log10I_liq", np.where(np.isfinite(nf.log10I[0]), nf.log10I[0], np.nan), "plasma", "log10 I (liquid)"),
        ("log10I_ice", np.where(np.isfinite(nf.log10I[1]), nf.log10I[1], np.nan), "plasma", "log10 I (ice)"),
        ("q_v", to_cpu(state.qv) * 1000.0, "BuGn", "q_v [g/kg]"),
    ]
    for name, arr, cmap, title in panels:
        fig, (axh, axv) = plt.subplots(1, 2, figsize=(11, 4.5))
        _field_slice(axh, grid, arr, "h", midz, f"{title} (z=mid) {tag}")
        _field_slice(axv, grid, arr, "v", midy, f"{title} (y=mid) {tag}")
        fig.tight_layout()
        p = os.path.join(outdir, f"{name}_{tag}.png")
        _save(fig, p); files.append(p)

    # velocity magnitude + vectors (horizontal slice)
    fig, ax = plt.subplots(figsize=(6, 5))
    X, Y = np.meshgrid(to_cpu(grid.xc), to_cpu(grid.yc), indexing="ij")
    ax.pcolormesh(X, Y, _slice(umag, "h", midz), shading="auto", cmap="cividis")
    step = max(1, grid.nx // 12)
    uq = np.nan_to_num(_slice(uc, "h", midz)[::step, ::step])
    vq = np.nan_to_num(_slice(vc, "h", midz)[::step, ::step])
    # skip the quiver when the flow is essentially at rest: matplotlib's autoscale
    # divides by the vector magnitude, so all-zero vectors raise a divide-by-zero
    # warning (harmless, but noisy at the first steps before convection develops).
    vmax = float(np.max(np.hypot(uq, vq)))
    if vmax > 1e-9:
        ax.quiver(X[::step, ::step], Y[::step, ::step], uq, vq,
                  color="white", scale=max(vmax * 20.0, 1e-9))
    ax.set_title(f"|u| + vectors (z=mid) {tag}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout()
    p = os.path.join(outdir, f"velocity_vectors_{tag}.png")
    _save(fig, p); files.append(p)

    # vertical velocity (vertical slice)
    fig, ax = plt.subplots(figsize=(6, 5))
    wc = to_cpu(0.5 * (state.w[:, :, :-1] + state.w[:, :, 1:]))
    _field_slice(ax, grid, wc, "v", midy, f"w [m/s] (y=mid) {tag}", cmap="RdBu_r")
    fig.tight_layout()
    p = os.path.join(outdir, f"w_{tag}.png")
    _save(fig, p); files.append(p)
    return files


def plot_budgets(history: list, outdir: str) -> list:
    files = []
    if not history:
        return files
    t = [r["time"] for r in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, [r.get("total_water_kg", np.nan) for r in history], label="total water [kg]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("total water [kg]")
    ax.legend(); ax.set_title("Integrated water budget")
    fig.tight_layout()
    p = os.path.join(outdir, "budget_water.png"); _save(fig, p); files.append(p)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, [r.get("mean_S_w", np.nan) for r in history], label="mean S_w")
    ax.plot(t, [r.get("mean_S_i", np.nan) for r in history], label="mean S_i")
    ax.set_xlabel("time [s]"); ax.set_ylabel("S"); ax.legend(); ax.set_title("Mean supersaturation")
    fig.tight_layout()
    p = os.path.join(outdir, "budget_supersat.png"); _save(fig, p); files.append(p)
    return files


__all__ = ["plot_budgets", "plot_snapshot"]