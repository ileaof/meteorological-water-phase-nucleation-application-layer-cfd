"""Configuration loading and validation for meteorological_flow.

Reads a YAML scenario file into nested dataclasses, validates ranges, and
applies CLI overrides (grid resolution, duration, output interval, coupling
stage, ...).  Mirrors the repo's existing YAML style (configs/*.yaml).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class DomainConfig:
    Lx: float = 100.0
    Ly: float = 100.0
    Lz: float = 100.0


@dataclass
class GridConfig:
    nx: int = 20
    ny: int = 20
    nz: int = 20
    z_stretch: float = 1.0     # >1 clusters vertical levels near the surface (1=uniform)


@dataclass
class TimeConfig:
    duration: float = 120.0
    cfl: float = 0.5
    dt_max: float = 0.25


@dataclass
class FlowConfig:
    formulation: str = "boussinesq"
    nu: float = 2.0             # momentum eddy/SGS viscosity [m^2/s] (molecular
                                # 1.5e-5 is negligible at dx~5 m; an eddy value
                                # gives real subgrid dissipation)
    kappa: float = 2.0          # scalar eddy diffusivity [m^2/s]
    advection_order: int = 1   # 1=upwind (monotone), 2=MUSCL(minmod)
    p_drop: float = 30.0       # Pa, applied as a uniform x body force (20-100)
    gravity: float = 9.81
    gamma_damp: float = 0.2    # 1/s, linear (Rayleigh) momentum drag -- a
                                # documented bulk subgrid dissipation that bounds
                                # the otherwise-unbounded Boussinesq buoyant
                                # convection (warm parcels do not cool on ascent).
                                # 0 disables it.
    smagorinsky: bool = False


@dataclass
class PhysicsConfig:
    P0: float = 70000.0
    theta_transport: bool = True
    moisture_buoyancy: bool = True
    T_ref: float | None = None   # reference T for Boussinesq buoyancy; None=mean
    scenario: str = "mixing_chamber"   # mixing_chamber | deep_convection (storm-scale)
    bubble_dtheta: float = 3.0         # warm-bubble amplitude [K] (deep_convection)
    precision: str = "float64"         # float64 (scientific) | float32 (performance)
    pressure_gradient: float | None = None   # if set, p_drop = gradient * Lx [Pa/m]
    dynamics: str = "boussinesq"       # boussinesq (test mode) | anelastic (deep convection)


@dataclass
class InflowConfig:
    side: str = "west"
    T: float = 293.0
    RH_water: float = 90.0
    u: float = 2.0


@dataclass
class BoundaryConfig:
    x_west: str = "inflow"        # inflow | outflow | periodic | wall
    x_east: str = "inflow"        # inflow (cold, -x) | outflow | periodic | wall
    y: str = "free_slip"          # free_slip | periodic | wall
    z_bottom: str = "free_slip"   # free_slip | no_slip
    z_top: str = "open"           # open (mass-balanced outflow) | damping_layer | rigid_lid
    warm_inflow: InflowConfig = field(default_factory=InflowConfig)
    cold_inflow: InflowConfig = field(default_factory=lambda: InflowConfig(side="east", T=258.0, RH_water=30.0, u=2.0))


@dataclass
class LookupConfig:
    enabled: bool = True
    n_T: int = 28
    n_pv: int = 20
    n_grad: int = 9
    T_range: list[float] = field(default_factory=lambda: [230.0, 305.0])
    pv_range: list[float] = field(default_factory=lambda: [40.0, 3500.0])
    grad_range: list[float] = field(default_factory=lambda: [1e-3, 20.0])  # log-spaced
    scan_resolution: int = 30                 # kernel radius scan (30 fast/accy; 75 matches direct)
    cache_path: str | None = None   # None -> <outdir>/nucleation_lookup.npz
    rebuild: bool = False


@dataclass
class NucleationConfig:
    mode: str = "homogeneous"      # homogeneous | heterogeneous
    phase_mode: str = "both"        # auto | liquid | ice | both
    method: str = "lookup"          # direct | lookup
    stage: str = "one_way"          # one_way | vapor_depletion | thermal_feedback | hydrometeor
    stochastic: bool = False
    seed: int = 20260820
    theta: float = 3.141592653589793   # radians; pi = homogeneous limit
    r_ref: float = 1.0e-7
    gmin: float = 1.0e-3            # floor for |gradT| (K/m), framework well-behaved limit
    dt_diagnostic: float = 60.0     # recompute nucleation diagnostics every this many s
    couple_kernel: bool = False     # two-way stage: feed the kernel rate J as the
                                    # microphysics embryo source (eq39 pathway) (M7)
    lookup: LookupConfig = field(default_factory=LookupConfig)


@dataclass
class PerformanceConfig:
    device: str = "auto"                 # auto | cpu | gpu (GPU is optional; see backend.py)
    compute_threads: int | None = None   # BLAS/OpenMP thread cap for the solver; None=library default


@dataclass
class OutputConfig:
    outdir: str = "outputs/flow_reference"
    interval_steps: int = 20
    format: list[str] = field(default_factory=lambda: ["netcdf", "json", "csv"])
    figures: list[str] = field(default_factory=lambda: ["slices", "vectors", "budgets"])
    restart: bool = True
    animate: bool = False   # build per-field + combined-panel MP4/GIF after the run (see animate.py)


@dataclass
class SimulationConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    boundaries: BoundaryConfig = field(default_factory=BoundaryConfig)
    nucleation: NucleationConfig = field(default_factory=NucleationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    random_seed: int = 20260820


# ---------------------------------------------------------------------------
def _get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if d is not None else default


def from_dict(d: dict[str, Any]) -> SimulationConfig:
    d = d or {}
    cfg = SimulationConfig()
    dom = _get(d, "domain", {})
    cfg.domain = DomainConfig(
        Lx=float(_get(dom, "Lx", 100.0)), Ly=float(_get(dom, "Ly", 100.0)),
        Lz=float(_get(dom, "Lz", 100.0)))
    gr = _get(d, "grid", {})
    cfg.grid = GridConfig(nx=int(_get(gr, "nx", 20)), ny=int(_get(gr, "ny", 20)),
                          nz=int(_get(gr, "nz", 20)),
                          z_stretch=float(_get(gr, "z_stretch", 1.0)))
    tm = _get(d, "time", {})
    cfg.time = TimeConfig(duration=float(_get(tm, "duration", 120.0)),
                          cfl=float(_get(tm, "cfl", 0.5)), dt_max=float(_get(tm, "dt_max", 0.25)))
    fl = _get(d, "flow", {})
    cfg.flow = FlowConfig(formulation=str(_get(fl, "formulation", "boussinesq")),
                          nu=float(_get(fl, "nu", 2.0)), kappa=float(_get(fl, "kappa", 2.0)),
                          advection_order=int(_get(fl, "advection_order", 1)),
                          p_drop=float(_get(fl, "p_drop", 30.0)),
                          gravity=float(_get(fl, "gravity", 9.81)),
                          gamma_damp=float(_get(fl, "gamma_damp", 0.2)),
                          smagorinsky=bool(_get(fl, "smagorinsky", False)))
    ph = _get(d, "physics", {})
    cfg.physics = PhysicsConfig(P0=float(_get(ph, "P0", 70000.0)),
                               theta_transport=bool(_get(ph, "theta_transport", True)),
                               moisture_buoyancy=bool(_get(ph, "moisture_buoyancy", True)),
                               T_ref=_get(ph, "T_ref", None),
                               scenario=str(_get(ph, "scenario", "mixing_chamber")),
                               bubble_dtheta=float(_get(ph, "bubble_dtheta", 3.0)),
                               precision=str(_get(ph, "precision", "float64")),
                               dynamics=str(_get(ph, "dynamics", "boussinesq")))
    bd = _get(d, "boundaries", {})
    warm = _get(bd, "warm_inflow", {})
    cold = _get(bd, "cold_inflow", {})
    cfg.boundaries = BoundaryConfig(
        x_west=str(_get(bd.get("x", {}), "west", _get(bd.get("x", {}), "inflow", "inflow"))),
        x_east=str(_get(bd.get("x", {}), "east", _get(bd.get("x", {}), "outflow", "inflow"))),
        y=str(_get(bd, "y", "free_slip")),
        z_bottom=str(_get(bd.get("z", {}), "bottom", "free_slip")),
        z_top=str(_get(bd.get("z", {}), "top", "open")),
        warm_inflow=InflowConfig(side=str(_get(warm, "side", "west")),
                                 T=float(_get(warm, "T", 293.0)),
                                 RH_water=float(_get(warm, "RH_water", 90.0)),
                                 u=float(_get(warm, "u", 2.0))),
        cold_inflow=InflowConfig(side=str(_get(cold, "side", "east")),
                                 T=float(_get(cold, "T", 258.0)),
                                 RH_water=float(_get(cold, "RH_water", 30.0)),
                                 u=float(_get(cold, "u", 2.0))))
    nu = _get(d, "nucleation", {})
    lk = _get(nu, "lookup", {})
    cfg.nucleation = NucleationConfig(
        mode=str(_get(nu, "mode", "homogeneous")),
        phase_mode=str(_get(nu, "phase_mode", "both")),
        method=str(_get(nu, "method", "lookup")),
        stage=str(_get(nu, "stage", "one_way")),
        stochastic=bool(_get(nu, "stochastic", False)),
        seed=int(_get(nu, "seed", 20260820)),
        theta=float(_get(nu, "theta", 3.141592653589793)),
        r_ref=float(_get(nu, "r_ref", 1.0e-7)),
        gmin=float(_get(nu, "gmin", 1.0e-3)),
        dt_diagnostic=float(_get(nu, "dt_diagnostic", 60.0)),
        couple_kernel=bool(_get(nu, "couple_kernel", False)),
        lookup=LookupConfig(enabled=bool(_get(lk, "enabled", True)),
                            n_T=int(_get(lk, "n_T", 28)), n_pv=int(_get(lk, "n_pv", 20)),
                            n_grad=int(_get(lk, "n_grad", 9)),
                            T_range=_get(lk, "T_range", [230.0, 305.0]),
                            pv_range=_get(lk, "pv_range", [40.0, 3500.0]),
                            grad_range=_get(lk, "grad_range", [1e-3, 20.0]),
                            scan_resolution=int(_get(lk, "scan_resolution", 30)),
                            cache_path=_get(lk, "cache_path", None),
                            rebuild=bool(_get(lk, "rebuild", False))))
    ou = _get(d, "output", {})
    cfg.output = OutputConfig(outdir=str(_get(ou, "outdir", "outputs/flow_reference")),
                              interval_steps=int(_get(ou, "interval_steps", 20)),
                              format=list(_get(ou, "format", ["netcdf", "json", "csv"])),
                              figures=list(_get(ou, "figures", ["slices", "vectors", "budgets"])),
                              restart=bool(_get(ou, "restart", True)),
                              animate=bool(_get(ou, "animate", False)))
    pf = _get(d, "performance", {})
    _ct = _get(pf, "compute_threads", None)
    cfg.performance = PerformanceConfig(
        device=str(_get(pf, "device", "auto")),
        compute_threads=(int(_ct) if _ct is not None else None))
    cfg.random_seed = int(_get(d, "random_seed", 20260820))
    validate(cfg)
    return cfg


def from_yaml(path: str) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return from_dict(raw if isinstance(raw, dict) else {})


def validate(cfg: SimulationConfig) -> None:
    import math as _m
    g, d = cfg.grid, cfg.domain
    _MIN = 3   # central-difference / Laplacian stencils need >= 3 cells per axis
    assert g.nx >= _MIN and g.ny >= _MIN and g.nz >= _MIN, \
        "grid must have >= %d cells per axis (stencil minimum); got %dx%dx%d" % (
            _MIN, g.nx, g.ny, g.nz)
    assert all(_m.isfinite(v) and v > 0 for v in (d.Lx, d.Ly, d.Lz)), \
        "domain lengths must be positive and finite"
    ncells = g.nx * g.ny * g.nz
    assert 0 < ncells < 2_000_000_000, "number of cells out of range: %d" % ncells
    assert cfg.physics.precision in ("float32", "float64"), \
        "precision must be float32 or float64"
    assert cfg.physics.dynamics in ("boussinesq", "anelastic"), \
        "dynamics must be boussinesq or anelastic"
    assert cfg.performance.device in ("auto", "cpu", "gpu"), \
        "performance.device must be auto, cpu or gpu"
    assert cfg.performance.compute_threads is None or cfg.performance.compute_threads > 0, \
        "performance.compute_threads must be a positive integer or unset"
    assert 0 < cfg.time.cfl <= 1.0, "CFL must be in (0, 1]"
    assert cfg.time.dt_max > 0 and cfg.time.duration >= 0
    assert cfg.flow.advection_order in (1, 2)
    assert cfg.nucleation.mode in ("homogeneous", "heterogeneous")
    assert cfg.nucleation.phase_mode in ("auto", "liquid", "ice", "both")
    assert cfg.nucleation.method in ("direct", "lookup")
    assert cfg.nucleation.stage in ("none", "one_way", "vapor_depletion", "thermal_feedback", "hydrometeor")
    assert cfg.boundaries.warm_inflow.T > 100, "warm inflow T unphysical"
    assert cfg.boundaries.cold_inflow.T > 100, "cold inflow T unphysical"


# ---------------------------------------------------------------------------
# geometry / memory helpers (derived quantities; single source of truth)
# ---------------------------------------------------------------------------
def geometry(cfg: SimulationConfig) -> dict:
    """Derived grid geometry.  Cell-centred convention: dx = Lx / Nx (Nx CELLS,
    not nodes), cell_volume = dx*dy*dz, domain_volume = Lx*Ly*Lz."""
    d, g = cfg.domain, cfg.grid
    dx, dy, dz = d.Lx / g.nx, d.Ly / g.ny, d.Lz / g.nz
    return {
        "Nx": g.nx, "Ny": g.ny, "Nz": g.nz,
        "Lx_m": d.Lx, "Ly_m": d.Ly, "Lz_m": d.Lz,
        "dx_m": dx, "dy_m": dy, "dz_m": dz,
        "cell_volume_m3": dx * dy * dz,
        "domain_volume_m3": d.Lx * d.Ly * d.Lz,
        "n_cells": g.nx * g.ny * g.nz,
        "cubic": abs(dx - dy) < 1e-9 and abs(dy - dz) < 1e-9,
    }


# ~ number of persistent + temporary 3-D fields carried by the solver
_N_FIELDS_ESTIMATE = 45


def estimate_memory_gb(cfg: SimulationConfig, n_fields: int = _N_FIELDS_ESTIMATE,
                       safety: float = 2.0, device: str = "cpu") -> float:
    """Rough field-memory estimate [GB] = N_cells * n_fields * bytes * safety.
    Note: the direct pressure factorisation (splu, used only for grids <= ~40^3)
    can add substantial fill-in beyond this; large grids use iterative CG.

    ``device`` is accepted for API symmetry with the GPU-VRAM preflight check
    (:func:`meteorological_flow.backend.check_gpu_memory`) -- the field-count/
    byte/safety formula itself is backend-agnostic (same fields, same dtype,
    whether they live in host or device memory), so it is currently unused."""
    n = cfg.grid.nx * cfg.grid.ny * cfg.grid.nz
    bytes_per = 4 if cfg.physics.precision == "float32" else 8
    return n * n_fields * bytes_per * safety / 1e9


def format_geometry(cfg: SimulationConfig) -> str:
    gm = geometry(cfg)
    mem = estimate_memory_gb(cfg)
    lines = [
        "grid       : %d x %d x %d = %d cells (%s spacing)" % (
            gm["Nx"], gm["Ny"], gm["Nz"], gm["n_cells"],
            "isotropic" if gm["cubic"] else "anisotropic"),
        "domain     : %.1f x %.1f x %.1f m   V_domain = %.3e m^3" % (
            gm["Lx_m"], gm["Ly_m"], gm["Lz_m"], gm["domain_volume_m3"]),
        "spacing    : dx=%.3f dy=%.3f dz=%.3f m   V_cell = %.4g m^3" % (
            gm["dx_m"], gm["dy_m"], gm["dz_m"], gm["cell_volume_m3"]),
        "memory est.: ~%.2f GB (%s, ~%d fields, safety x%.1f)" % (
            mem, cfg.physics.precision, _N_FIELDS_ESTIMATE, 2.0),
        "time       : cfl=%.2f dt_max=%.3g s duration=%.1f s" % (
            cfg.time.cfl, cfg.time.dt_max, cfg.time.duration),
        "device     : %s%s" % (cfg.performance.device,
            "" if cfg.performance.compute_threads is None
            else " (compute_threads=%d)" % cfg.performance.compute_threads),
    ]
    return "\n".join("  " + ln for ln in lines)


# ---------------------------------------------------------------------------
# named CPU mesh presets (see the manual)
# ---------------------------------------------------------------------------
PRESETS = {
    # shallow mixing-chamber meshes (cubic ~1 km box)
    "fast":     dict(Lx=1000.0, Ly=1000.0, Lz=1000.0, Nx=25, Ny=25, Nz=25),
    "light":    dict(Lx=1000.0, Ly=1000.0, Lz=1000.0, Nx=40, Ny=40, Nz=40),
    "recommended": dict(Lx=1000.0, Ly=1000.0, Lz=1000.0, Nx=50, Ny=50, Nz=50),
    "advanced": dict(Lx=1000.0, Ly=1000.0, Lz=1000.0, Nx=100, Ny=100, Nz=100),
    "convective-column": dict(Lx=2000.0, Ly=2000.0, Lz=5000.0, Nx=50, Ny=50, Nz=125),
    # deep-convection storm meshes (imply the storm setup + anelastic core).
    # `storm=True` -> apply_overrides configures the stratified deep-convection
    # scenario (walls, damping top, two-way microphysics) around this grid.
    # dx = Lx/Nx, dz = Lz/Nz; the direct pressure solver is used up to 64k cells
    # (storm-quick/storm/storm-refined), CG above (storm-fine/storm-hires).
    "storm-quick":   dict(Lx=16000.0, Ly=16000.0, Lz=16000.0, Nx=16, Ny=16, Nz=40, storm=True),
    "storm":         dict(Lx=20000.0, Ly=20000.0, Lz=18000.0, Nx=24, Ny=24, Nz=45, storm=True),
    "storm-refined": dict(Lx=24000.0, Ly=24000.0, Lz=18000.0, Nx=32, Ny=32, Nz=50, storm=True),
    "storm-fine":    dict(Lx=24000.0, Ly=24000.0, Lz=18000.0, Nx=40, Ny=40, Nz=60, storm=True),
    "storm-hires":   dict(Lx=24000.0, Ly=24000.0, Lz=18000.0, Nx=48, Ny=48, Nz=64, storm=True),
}


def _apply_storm_physics(cfg: SimulationConfig) -> None:
    """Configure the km-scale deep-convection storm scenario (physics only, not
    the grid/domain): stratified base state + warm-bubble trigger + closed walls
    + damping top + two-way microphysics.  Shared by ``--storm-scale`` and the
    ``storm-*`` presets.  Demonstration-scale (see base_state.py)."""
    cfg.physics.scenario = "deep_convection"
    cfg.physics.P0 = 100000.0
    cfg.nucleation.stage = "hydrometeor"
    cfg.boundaries.x_west = cfg.boundaries.x_east = "wall"
    cfg.boundaries.y = "free_slip"
    cfg.boundaries.z_top = "damping_layer"
    cfg.boundaries.z_bottom = "free_slip"
    cfg.flow.p_drop = 0.0
    # With the M5 conservative transport (no spurious low-level concentration of
    # theta'/moisture) the storm no longer runs away numerically, so the SGS
    # dissipation is much lighter than before -- letting the CAPE drive a physical
    # updraft (~30% of the parcel ceiling on a coarse grid, stronger when refined).
    cfg.flow.gamma_damp = 0.005            # light Rayleigh drag (bounds extremes)
    cfg.flow.nu = cfg.flow.kappa = 20.0    # eddy viscosity/diffusivity at ~km res
    cfg.time.cfl = 0.3
    cfg.time.dt_max = 3.0                    # deep column: keep the fall/advective CFL < 1


def apply_overrides(cfg: SimulationConfig, *,
                   grid_resolution: int | None = None,
                   Lx: float | None = None, Ly: float | None = None, Lz: float | None = None,
                   Nx: int | None = None, Ny: int | None = None, Nz: int | None = None,
                   z_stretch: float | None = None,
                   cfl: float | None = None, dt_max: float | None = None,
                   sgs: float | None = None,
                   pressure_drop: float | None = None,
                   pressure_gradient: float | None = None,
                   float32: bool = False, preset: str | None = None,
                   duration: float | None = None,
                   output_interval: int | None = None,
                   output: str | None = None,
                   no_microphysics: bool = False,
                   one_way: bool = False,
                   diagnostic_only: bool = False,
                   two_way: bool = False,
                   storm_scale: bool = False,
                   dynamics: str | None = None,
                   tecplot: bool = False,
                   periodic: bool = False,
                   kernel_nucleation: bool = False,
                   method: str | None = None,
                   threads: int | None = None,
                   device: str | None = None,
                   compute_threads: int | None = None,
                   precision: str | None = None,
                   animate: bool = False) -> SimulationConfig:
    """Return a copy of cfg with CLI overrides applied.  Precedence (low->high):
    preset -> storm_scale -> grid_resolution -> explicit Lx/Ly/Lz/Nx/Ny/Nz."""
    cfg = copy.deepcopy(cfg)
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError("unknown preset '%s'; choices: %s"
                             % (preset, ", ".join(sorted(PRESETS))))
        pv = PRESETS[preset]
        if pv.get("storm"):
            # a storm-* preset carries the whole deep-convection setup and
            # defaults to the anelastic core (explicit --dynamics still wins).
            _apply_storm_physics(cfg)
            cfg.physics.dynamics = "anelastic"
        cfg.domain.Lx, cfg.domain.Ly, cfg.domain.Lz = pv["Lx"], pv["Ly"], pv["Lz"]
        cfg.grid.nx, cfg.grid.ny, cfg.grid.nz = pv["Nx"], pv["Ny"], pv["Nz"]
    if storm_scale:
        # km-scale deep-convection storm (physics), with the legacy default grid/
        # domain.  For deeper, EL-containing meshes prefer the storm-* presets.
        _apply_storm_physics(cfg)
        cfg.domain.Lx = cfg.domain.Ly = 16000.0
        cfg.domain.Lz = 10000.0
        cfg.grid.nx = cfg.grid.ny = 24
        cfg.grid.nz = 40
    if grid_resolution is not None:
        n = int(grid_resolution)
        cfg.grid.nx = cfg.grid.ny = cfg.grid.nz = n
    # explicit domain/grid dimensions -- highest precedence (CLI over YAML/preset)
    if Lx is not None:
        cfg.domain.Lx = float(Lx)
    if Ly is not None:
        cfg.domain.Ly = float(Ly)
    if Lz is not None:
        cfg.domain.Lz = float(Lz)
    if Nx is not None:
        cfg.grid.nx = int(Nx)
    if Ny is not None:
        cfg.grid.ny = int(Ny)
    if Nz is not None:
        cfg.grid.nz = int(Nz)
    if z_stretch is not None:
        cfg.grid.z_stretch = float(z_stretch)
    if cfl is not None:
        cfg.time.cfl = float(cfl)
    if dt_max is not None:
        cfg.time.dt_max = float(dt_max)
    if sgs is not None:
        cfg.flow.nu = cfg.flow.kappa = float(sgs)   # subgrid eddy viscosity/diffusivity
    # precision: --precision supersedes the older --float32 flag; both may be
    # given together only if they agree (mirrors the pressure_drop/gradient
    # mutual-exclusion pattern below).
    if precision is not None and float32 and precision != "float32":
        raise ValueError("--float32 conflicts with --precision %s" % precision)
    if precision is not None:
        cfg.physics.precision = str(precision)
    elif float32:
        cfg.physics.precision = "float32"
    if device is not None:
        cfg.performance.device = str(device)
    if compute_threads is not None:
        cfg.performance.compute_threads = int(compute_threads)
    # pressure forcing: total drop [Pa] vs gradient [Pa/m] (mutually exclusive).
    # Increasing Lx at fixed drop lowers the gradient -- surfaced here explicitly.
    if pressure_drop is not None and pressure_gradient is not None:
        raise ValueError("provide pressure_drop OR pressure_gradient, not both")
    if pressure_drop is not None:
        cfg.flow.p_drop = float(pressure_drop)
        cfg.physics.pressure_gradient = None
    if pressure_gradient is not None:
        cfg.physics.pressure_gradient = float(pressure_gradient)
        cfg.flow.p_drop = float(pressure_gradient) * cfg.domain.Lx
    if duration is not None:
        cfg.time.duration = float(duration)
    if output_interval is not None:
        cfg.output.interval_steps = int(output_interval)
    if output is not None:
        cfg.output.outdir = output
    if no_microphysics:
        cfg.nucleation.stage = "none"
    if one_way or diagnostic_only:
        cfg.nucleation.stage = "one_way"
    if two_way:
        cfg.nucleation.stage = "hydrometeor"   # full two-way microphysics coupling
    if dynamics is not None:
        cfg.physics.dynamics = str(dynamics)
    if tecplot and "tecplot" not in cfg.output.format:
        cfg.output.format = list(cfg.output.format) + ["tecplot"]
    if animate:
        cfg.output.animate = True
    if periodic:
        # periodic lateral boundaries (mean-wind / shear storm): the environmental
        # wind is ingested and the projection/advection wrap in x and y.
        cfg.boundaries.x_west = cfg.boundaries.x_east = "periodic"
        cfg.boundaries.y = "periodic"
    if kernel_nucleation:
        cfg.nucleation.couple_kernel = True
    if method is not None:
        cfg.nucleation.method = method
    # threads stored on the lookup config for the table build
    if threads is not None:
        cfg.nucleation.lookup.threads = int(threads)  # type: ignore[attr-defined]
    else:
        cfg.nucleation.lookup.threads = None  # type: ignore[attr-defined]
    validate(cfg)
    return cfg


__all__ = [
    "BoundaryConfig",
    "DomainConfig",
    "FlowConfig",
    "GridConfig",
    "InflowConfig",
    "LookupConfig",
    "NucleationConfig",
    "OutputConfig",
    "PerformanceConfig",
    "PhysicsConfig",
    "SimulationConfig",
    "TimeConfig",
    "PRESETS",
    "apply_overrides",
    "estimate_memory_gb",
    "format_geometry",
    "from_dict",
    "from_yaml",
    "geometry",
    "validate",
]