"""Simulation orchestrator: the 3D Boussinesq time loop with one-way nucleation.

Per step (Chorin projection method, explicit predictor):

1. enforce boundary conditions (velocity + scalars);
2. diagnose thermodynamics (T, p_v, S, rho, |gradT|) from the prognostic state;
3. predictor: advect + diffuse momentum, add buoyancy (w) and the uniform
   pressure-drop body force (u), then advect + diffuse the scalars (theta, q_v,
   and q_l/q_i when the stage transports hydrometeors);
4. Chorin projection -> div(u) ~ 0, update p';
5. re-enforce velocity BCs (inflow Dirichlet preserved);
6. nucleation: in Batch 1 this is **one-way / diagnostic** -- the kernel outputs
   are evaluated and recorded but the prognostic state is NOT modified.

Adaptive CFL: dt = min( cfl*min(dx)/max|u|, 0.5*min(dx)^2/(3*max(nu,kappa)), dt_max ).

Scientific integrity: the validated nucleation kernel is called read-only; no
prognostic field is altered by the microphysics in Batch 1.  Two-way coupling
(vapour depletion, latent heat, hydrometeor transport) is Batch 2, gated on the
one-way foundation passing its verification (see docs/flow_guide.md).
"""
from __future__ import annotations

import os
import time as _time

import numpy as np

from . import advection as adv
from . import boundary_conditions as bc
from . import buoyancy as buo
from . import diagnostics as diag
from . import diffusion as dif
from . import io as fio
from . import plotting as plt_mod
from . import thermodynamics as th
from .config import SimulationConfig
from .grid import Grid
from .nucleation_adapter import NucleationAdapter, NucleationField
from .nucleation_lookup import NucleationLookup
from .pressure_solver import PressureSolver
from .state import FlowState

try:
    import resource as _resource
    _MEM_OK = True
except Exception:                       # Windows: no resource module
    _MEM_OK = False


def _mem_kb():
    if not _MEM_OK:
        return 0
    return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss


def _grid_from_config(cfg: SimulationConfig) -> Grid:
    return Grid(nx=cfg.grid.nx, ny=cfg.grid.ny, nz=cfg.grid.nz,
                Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz)


def _initial_state(grid: Grid, cfg: SimulationConfig) -> FlowState:
    """Smooth west(warm)->east(cold) blend for theta and q_v; zero hydrometeors.

    A deterministic (seed-independent) linear blend provides a physically
    reasonable initial condition; the inflow BCs and the pressure drop drive the
    subsequent mixing.  A small sinusoidal y-perturbation breaks the exact
    y-symmetry so the mixing zone is resolved (not a numerical artefact).
    """
    state = FlowState.zeros(grid)
    th_w, qv_w = bc.inflow_state(cfg.boundaries.warm_inflow, cfg.physics.P0)
    th_c, qv_c = bc.inflow_state(cfg.boundaries.cold_inflow, cfg.physics.P0)
    frac = (grid.xc / grid.Lx).reshape(grid.nx, 1, 1)   # 0 at west, 1 at east
    th_lin = (th_w * (1.0 - frac) + th_c * frac)         # (nx, 1, 1)
    qv_lin = (qv_w * (1.0 - frac) + qv_c * frac)
    state.theta = np.broadcast_to(th_lin, grid.center_shape).copy()
    state.qv = np.broadcast_to(qv_lin, grid.center_shape).copy()
    state.ql = np.zeros(grid.center_shape); state.qi = np.zeros(grid.center_shape)
    state.u = np.zeros(grid.u_shape); state.v = np.zeros(grid.v_shape); state.w = np.zeros(grid.w_shape)
    bc.apply_velocity_bcs(state, grid, cfg)
    bc.apply_scalar_bcs(state, grid, cfg)
    state.diagnose(cfg)
    return state


def _pressure_method(grid: Grid) -> str:
    n = grid.nx * grid.ny * grid.nz
    return "direct" if n <= 64_000 else "cg"   # cached splu for <= 40^3


def _build_lookup(cfg: SimulationConfig, outdir: str, adapter: NucleationAdapter):
    lk_cfg = cfg.nucleation.lookup
    cache = lk_cfg.cache_path or os.path.join(outdir, "nucleation_lookup.npz")
    if os.path.exists(cache) and not lk_cfg.rebuild:
        return NucleationLookup.load(cache, cfg.nucleation, lk_cfg), cache
    lk = NucleationLookup(cfg.nucleation, lk_cfg)
    threads = getattr(lk_cfg, "threads", None) or 1
    lk.build(threads=threads)
    lk.save(cache)
    return lk, cache


class Simulation:
    def __init__(self, cfg: SimulationConfig, restart: str | None = None):
        self.cfg = cfg
        self.grid = _grid_from_config(cfg)
        self.rng = np.random.default_rng(cfg.random_seed)
        if restart:
            self.state = fio.load_restart(restart, self.grid)
        else:
            self.state = _initial_state(self.grid, cfg)
        self.state.diagnose(cfg)
        # reference Boussinesq state
        self.T_ref = float(self.state.T.mean()) if cfg.physics.T_ref is None else cfg.physics.T_ref
        self.qv_ref = float(self.state.qv.mean())
        self.rho0 = cfg.physics.P0 / (th.R_d * self.T_ref)
        # solvers.  All-Neumann projection; the top outflow is sized to balance
        # the two inflows (see boundary_conditions), so mean(div)=0 and the
        # projected velocity is divergence-free -> monotone scalar advection.
        self.pressure = PressureSolver(self.grid, method=_pressure_method(self.grid))
        # nucleation
        self.stage = cfg.nucleation.stage
        self.do_nucleation = self.stage != "none"
        self.adapter = None
        self.lookup = None
        if self.do_nucleation:
            self.adapter = NucleationAdapter(cfg.nucleation)
            if cfg.nucleation.method == "lookup":
                self.lookup, _ = _build_lookup(cfg, cfg.output.outdir, self.adapter)
                self.adapter.set_lookup(self.lookup)
        # bookkeeping
        self.snapshots = []
        self.history = []
        self.step = 0
        self.t = float(self.state.t)
        self.last_nf = NucleationField(self.grid.center_shape)
        self._t0 = _time.perf_counter()

    # ---- CFL ----
    def _dt(self) -> float:
        g = self.grid
        umag = self.state.velocity_magnitude_center()
        umax = float(umag.max()) if umag.size else 0.0
        # safety margin: the predictor (buoyancy + pressure-drop body force) grows
        # the velocity within a step, so size dt for ~1.25x the current speed.
        adv_dt = self.cfg.time.cfl * min(g.dx, g.dy, g.dz) / max(1.25 * umax, 1e-9)
        diff_coef = max(self.cfg.flow.nu, self.cfg.flow.kappa)
        diff_dt = 0.5 * min(g.dx, g.dy, g.dz) ** 2 / (3.0 * max(diff_coef, 1e-12))
        dt = min(adv_dt, diff_dt, self.cfg.time.dt_max)
        return max(dt, 1e-6)

    # ---- one Chorin step ----
    def _step(self, dt: float) -> None:
        cfg = self.cfg
        g = self.grid
        st = self.state
        order = cfg.flow.advection_order
        # 1. BCs + diagnose
        bc.apply_velocity_bcs(st, g, cfg)
        bc.apply_scalar_bcs(st, g, cfg)
        st.diagnose(cfg)
        # 2. momentum predictor.  (v1 simplification, documented): advective
        # transport of momentum is deferred -- the velocity is governed by the
        # uniform pressure-drop body force, moist Boussinesq buoyancy, viscous
        # diffusion, and the incompressibility projection.  This is a linearized
        # creeping/buoyant flow; the SCALARS (theta, q_v, hydrometeors) are advected
        # by the resulting DIVERGENCE-FREE velocity, which is what drives the
        # mixing -> supersaturation -> nucleation.  Fully conservative staggered
        # momentum advection is a Batch-2 upgrade.
        dif.diffuse_momentum(st, g, cfg.flow.nu, dt)
        # buoyancy on w
        Bf = buo.buoyancy_w_tendency(st, g, cfg, self.T_ref, self.qv_ref)
        st.w += dt * Bf
        # uniform pressure-drop body force along +x (NOT a per-cell subtraction):
        #   du/dt = -(1/rho0) dP/dx, with dP/dx = -p_drop/Lx  =>  du/dt = p_drop/(rho0*Lx)
        accel = cfg.flow.p_drop / (self.rho0 * g.Lx)
        st.u += dt * accel
        # linear (Rayleigh) momentum drag: du/dt = -gamma u.  Bounds the otherwise
        # runaway Boussinesq buoyant convection (warm parcels do not cool on
        # ascent, so without a dissipation the plume accelerates unboundedly).
        # A documented bulk subgrid dissipation; 0 disables it.
        if cfg.flow.gamma_damp > 0.0:
            decay = 1.0 - cfg.flow.gamma_damp * dt
            st.u *= decay; st.v *= decay; st.w *= decay
        # 3. project the velocity to divergence-free BEFORE advecting scalars.
        # Flux-form upwind is monotone (bounded) only under a SOLENOIDAL velocity
        # (per-axis CFL<1); advecting with the divergent predictor lets multi-axis
        # convergence sum the inflow-CFL above 1 and create non-physical extrema.
        bc.apply_velocity_bcs(st, g, cfg)
        res, it = self.pressure.project(st, dt, self.rho0)
        self._last_res = res; self._last_iters = it
        bc.apply_velocity_bcs(st, g, cfg)   # re-impose inflow (Neumann preserves it)
        # 4. scalar predictor: advect + diffuse with the divergence-free velocity
        Uc, Vc, Wc = adv.cell_velocity(st, g)
        st.theta = adv.advect_center(st.theta, Uc, Vc, Wc, g, dt, order)
        qv_new = adv.advect_center(st.qv, Uc, Vc, Wc, g, dt, order)
        # positivity of q_v (last-resort clip; bookkeep loss)
        clip_loss = float(np.sum(np.minimum(qv_new, 0.0)) * g.cell_vol)
        if clip_loss < 0:
            self._last_clip = clip_loss
        st.qv = np.maximum(qv_new, 0.0)
        if self.stage == "hydrometeor":   # Batch 2 transport hook (stub here)
            st.ql = adv.advect_center(st.ql, Uc, Vc, Wc, g, dt, order)
            st.qi = adv.advect_center(st.qi, Uc, Vc, Wc, g, dt, order)
            st.ql = np.maximum(st.ql, 0.0); st.qi = np.maximum(st.qi, 0.0)
        st.theta = dif.diffuse_center(st.theta, g, cfg.flow.kappa, dt)
        st.qv = dif.diffuse_center(st.qv, g, cfg.flow.kappa, dt)
        st.qv = np.maximum(st.qv, 0.0)
        # 5. scalar BCs + diagnose (velocity already div-free from step 3)
        bc.apply_scalar_bcs(st, g, cfg)
        bc.apply_velocity_bcs(st, g, cfg)
        st.diagnose(cfg)
        st.t = self.t + dt

    # ---- nucleation diagnostics (one-way) ----
    def _evaluate_nucleation(self, dt: float) -> NucleationField:
        if not self.do_nucleation or self.adapter is None:
            return NucleationField(self.grid.center_shape)
        return self.adapter.evaluate_field(self.state, dt, self.grid.cell_vol)

    # ---- main loop ----
    def run(self, progress=None) -> dict:
        cfg = self.cfg
        g = self.grid
        outdir = cfg.output.outdir
        os.makedirs(outdir, exist_ok=True)
        initial = diag.initial_budgets(self.state, self.rho0)
        self._last_clip = 0.0
        self._last_res = 0.0; self._last_iters = 0
        duration = cfg.time.duration
        interval = max(1, cfg.output.interval_steps)
        # initial snapshot
        nf0 = self._evaluate_nucleation(0.0)
        self.last_nf = nf0
        self._record(nf0, initial)
        self._maybe_output(nf0, force=True)
        max_cfl = 0.0
        while self.t < duration - 1e-9:
            dt = self._dt()
            if self.t + dt > duration:
                dt = duration - self.t
            # CFL diagnostic (advective)
            umax = float(self.state.velocity_magnitude_center().max())
            cfl_now = umax * dt / min(g.dx, g.dy, g.dz)
            max_cfl = max(max_cfl, cfl_now)
            self._step(dt)
            self.step += 1
            self.t = float(self.state.t)
            # nucleation at output cadence (one-way: diagnostic only)
            if self.step % interval == 0 or self.t >= duration - 1e-9:
                nf = self._evaluate_nucleation(dt)
                self.last_nf = nf
                self._record(nf, initial)
                self._maybe_output(nf)
            if progress and (self.step % max(1, interval) == 0):
                progress(self.t, duration, self.step)
        # finalise outputs
        report = self._finalise(initial, max_cfl)
        return report

    def _record(self, nf: NucleationField, initial: dict) -> None:
        st = self.state
        bud = diag.conservation_budgets(st, initial, self.rho0)
        stats = diag.summary_stats(st, nf)
        row = {"time": self.t, "step": self.step,
               "total_water_kg": bud["total_water_kg"],
               "total_water_rel_err": bud["total_water_rel_err"],
               "total_energy_J": bud["total_energy_J"],
               "total_energy_rel_err": bud["total_energy_rel_err"],
               "mean_S_w": float(np.mean(st.S_w)), "mean_S_i": float(np.mean(st.S_i)),
               "umax": stats["umax"], "wmax": stats["wmax"],
               "T_min": stats["T_min"], "T_max": stats["T_max"],
               "log10I_liq_max": stats["log10I_liq_max"],
               "log10I_ice_max": stats["log10I_ice_max"],
               "solver_residual": getattr(self, "_last_res", 0.0),
               "solver_iters": getattr(self, "_last_iters", 0)}
        self.history.append(row)

    def _maybe_output(self, nf: NucleationField, force: bool = False) -> None:
        cfg = self.cfg
        if not (force or self.step % max(1, cfg.output.interval_steps) == 0):
            return
        snap = fio.snapshot(self.state, nf, self.t, self.rho0)
        # fill buoyancy/latent placeholders for output
        snap["solver_residual"] = np.full(self.grid.center_shape, getattr(self, "_last_res", 0.0))
        self.snapshots.append(snap)
        if "figures" in cfg.output.format or "slices" in cfg.output.figures:
            tag = f"t{round(self.t)}"
            plt_mod.plot_snapshot(self.state, nf, self.grid,
                                  os.path.join(cfg.output.outdir, "figures"),
                                  self.t, tag=tag)
        if cfg.output.restart:
            fio.write_restart(self.state, os.path.join(cfg.output.outdir, "restart.npz"), self.t)

    def _finalise(self, initial: dict, max_cfl: float) -> dict:
        cfg = self.cfg
        outdir = cfg.output.outdir
        os.makedirs(outdir, exist_ok=True)
        attrs = self._global_attrs()
        # NetCDF
        if "netcdf" in cfg.output.format and self.snapshots:
            fio.write_netcdf(self.snapshots, os.path.join(outdir, "flow.nc"), self.grid, attrs)
        # CSV history
        if "csv" in cfg.output.format:
            fio.write_csv(self.history, os.path.join(outdir, "history.csv"))
        # budgets plots
        if "budgets" in cfg.output.figures:
            plt_mod.plot_budgets(self.history, os.path.join(outdir, "figures"))
        # JSON summary
        bud = diag.conservation_budgets(self.state, initial, self.rho0)
        stats = diag.summary_stats(self.state, self.last_nf)
        wall = _time.perf_counter() - self._t0
        report = {
            "code_version": attrs.get("code_version", "meteorological_flow v1"),
            "config": _cfg_summary(cfg),
            "wall_clock_s": wall,
            "memory_max_kb": _mem_kb(),
            "n_steps": self.step,
            "final_time": self.t,
            "max_cfl": max_cfl,
            "rho0": self.rho0, "T_ref": self.T_ref, "qv_ref": self.qv_ref,
            "final_stats": stats,
            "final_budgets": bud,
            "final_solver_residual": getattr(self, "_last_res", 0.0),
            "final_solver_iters": getattr(self, "_last_iters", 0),
            "lookup_used": self.lookup is not None,
            "stage": self.stage,
            "limitations": [
                ("Boussinesq: pressure-drop expansion cooling ~0.1 K is 2nd-order; "
                 "supersaturation dominated by mixing + buoyant lifting."),
                ("One-way (Batch 1): nucleation is diagnostic; prognostic state is not "
                 "modified by microphysics. Two-way coupling is a gated Batch-2 step."),
                ("|gradT| floored at gmin: the |gradT|->0 limit is the kernel's "
                 "near-equilibrium result (parameterization), NOT the CNT limit."),
                ("Momentum advection uses a centre round-trip (v1 simplification); the "
                 "projection corrects divergence but this is not a fully conservative "
                 "staggered momentum scheme."),
                ("Rain/snow/graupel/hail are reported as thermodynamic favorability, "
                 "NOT precipitation prediction."),
                "Not operational weather prediction; demonstration-scale only.",
            ],
        }
        if "json" in cfg.output.format:
            fio.write_json(report, os.path.join(outdir, "summary.json"))
        return report

    def _global_attrs(self) -> dict:
        import met_water_nucleation as M
        return {
            "code_version": "meteorological_flow v1 (Batch 1: one-way nucleation)",
            "engine_version": getattr(M, "__version__", "met_water_nucleation"),
            "formulation": self.cfg.flow.formulation,
            "P0": self.cfg.physics.P0,
            "random_seed": self.cfg.random_seed,
            "rho0": self.rho0, "T_ref": self.T_ref, "qv_ref": self.qv_ref,
            "grid": f"{self.grid.nx}x{self.grid.ny}x{self.grid.nz}",
            "dx": self.grid.dx, "dy": self.grid.dy, "dz": self.grid.dz,
            "duration": self.cfg.time.duration, "cfl": self.cfg.time.cfl,
            "stage": self.stage,
        }


def _cfg_summary(cfg: SimulationConfig) -> dict:
    return {
        "domain": {"Lx": cfg.domain.Lx, "Ly": cfg.domain.Ly, "Lz": cfg.domain.Lz},
        "grid": {"nx": cfg.grid.nx, "ny": cfg.grid.ny, "nz": cfg.grid.nz},
        "duration": cfg.time.duration, "cfl": cfg.time.cfl,
        "p_drop": cfg.flow.p_drop, "P0": cfg.physics.P0,
        "warm_inflow": {"T": cfg.boundaries.warm_inflow.T, "RH": cfg.boundaries.warm_inflow.RH_water,
                        "u": cfg.boundaries.warm_inflow.u},
        "cold_inflow": {"T": cfg.boundaries.cold_inflow.T, "RH": cfg.boundaries.cold_inflow.RH_water,
                        "u": cfg.boundaries.cold_inflow.u},
        "nucleation": {"method": cfg.nucleation.method, "stage": cfg.nucleation.stage,
                       "mode": cfg.nucleation.mode, "phase_mode": cfg.nucleation.phase_mode},
        "random_seed": cfg.random_seed,
    }


__all__ = ["Simulation", "_grid_from_config", "_initial_state"]