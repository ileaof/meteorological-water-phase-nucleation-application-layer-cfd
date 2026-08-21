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
    lookup: LookupConfig = field(default_factory=LookupConfig)


@dataclass
class OutputConfig:
    outdir: str = "outputs/flow_reference"
    interval_steps: int = 20
    format: list[str] = field(default_factory=lambda: ["netcdf", "json", "csv"])
    figures: list[str] = field(default_factory=lambda: ["slices", "vectors", "budgets"])
    restart: bool = True


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
                          nz=int(_get(gr, "nz", 20)))
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
                               T_ref=_get(ph, "T_ref", None))
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
                              restart=bool(_get(ou, "restart", True)))
    cfg.random_seed = int(_get(d, "random_seed", 20260820))
    validate(cfg)
    return cfg


def from_yaml(path: str) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return from_dict(raw if isinstance(raw, dict) else {})


def validate(cfg: SimulationConfig) -> None:
    g = cfg.grid
    assert g.nx > 0 and g.ny > 0 and g.nz > 0, "grid cells must be positive"
    assert cfg.domain.Lx > 0 and cfg.domain.Ly > 0 and cfg.domain.Lz > 0
    assert 0 < cfg.time.cfl <= 1.0, "CFL must be in (0, 1]"
    assert cfg.time.dt_max > 0 and cfg.time.duration >= 0
    assert cfg.flow.advection_order in (1, 2)
    assert cfg.nucleation.mode in ("homogeneous", "heterogeneous")
    assert cfg.nucleation.phase_mode in ("auto", "liquid", "ice", "both")
    assert cfg.nucleation.method in ("direct", "lookup")
    assert cfg.nucleation.stage in ("none", "one_way", "vapor_depletion", "thermal_feedback", "hydrometeor")
    assert cfg.boundaries.warm_inflow.T > 100, "warm inflow T unphysical"
    assert cfg.boundaries.cold_inflow.T > 100, "cold inflow T unphysical"


def apply_overrides(cfg: SimulationConfig, *,
                   grid_resolution: int | None = None,
                   duration: float | None = None,
                   output_interval: int | None = None,
                   output: str | None = None,
                   no_microphysics: bool = False,
                   one_way: bool = False,
                   diagnostic_only: bool = False,
                   method: str | None = None,
                   threads: int | None = None) -> SimulationConfig:
    """Return a copy of cfg with CLI overrides applied."""
    cfg = copy.deepcopy(cfg)
    if grid_resolution is not None:
        n = int(grid_resolution)
        cfg.grid.nx = cfg.grid.ny = cfg.grid.nz = n
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
    "PhysicsConfig",
    "SimulationConfig",
    "TimeConfig",
    "apply_overrides",
    "from_dict",
    "from_yaml",
    "validate",
]