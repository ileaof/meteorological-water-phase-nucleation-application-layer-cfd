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
from .backend import Backend, get_backend
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


def _grid_from_config(cfg: SimulationConfig, backend: Backend | None = None) -> Grid:
    periodic = getattr(cfg.boundaries, "x_west", None) == "periodic"
    return Grid(nx=cfg.grid.nx, ny=cfg.grid.ny, nz=cfg.grid.nz,
                Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz,
                z_stretch=getattr(cfg.grid, "z_stretch", 1.0), periodic=periodic,
                backend=backend)


def _deep_convection_initial(grid: Grid, cfg: SimulationConfig, base=None) -> FlowState:
    """Stratified conditionally-unstable base state + a warm-bubble trigger.

    The hydrostatic base state (:mod:`base_state`) sets ``theta0(z)``, ``qv0(z)``
    and the base pressure ``p0(z)``; a Gaussian warm bubble at low levels seeds
    the updraft.  The velocities start at rest; buoyancy (perturbation form) and
    the two-way microphysics latent heat drive the deep, precipitating plume.
    """
    from .base_state import build_base_state, warm_bubble
    if base is None:
        base = build_base_state(grid)
    state = FlowState.zeros(grid)
    theta0 = base.field(base.theta0, grid.center_shape, xp=grid.xp)
    qv0 = base.field(base.qv0, grid.center_shape, xp=grid.xp)
    dth, dqv = warm_bubble(grid, dtheta=cfg.physics.bubble_dtheta)
    state.theta = theta0 + dth
    state.qv = grid.xp.maximum(qv0 + dqv, 0.0)
    state.p0_field = base.field(base.p0, grid.center_shape, xp=grid.xp)
    # environmental wind: periodic lateral BCs ingest the mean wind u0(z)/v0(z)
    # (from the sounding shear) so the flow is sheared and the updraft can tilt /
    # organise; the closed-wall storm starts from rest (u0 not advectable there).
    if getattr(grid, "periodic", False):
        state.u[:] = grid.xp.asarray(base.u0, dtype=float)[None, None, :]
        state.v[:] = grid.xp.asarray(base.v0, dtype=float)[None, None, :]
    bc.apply_velocity_bcs(state, grid, cfg)
    bc.apply_scalar_bcs(state, grid, cfg, theta0=theta0, qv0=qv0)
    state.diagnose(cfg)
    # Moisten the warm-bubble core to near-saturation: a warm-but-dry thermal has
    # LOWER RH (warming raises q_sat) and would not condense, so it never taps the
    # CAPE.  A saturated thermal condenses on the first ascent -> latent heat ->
    # a physically-triggered updraft.  (With the M5 conservative transport, this
    # honest trigger is what drives the storm; the old dry bubble relied on the
    # non-conservative scheme spuriously concentrating moisture.)
    if cfg.physics.bubble_dtheta > 0.0:
        xp = grid.xp
        qsat = th.q_v_from_p_v(th.psat_water(state.T, xp=xp), state.P_total, xp=xp)
        core = dth > 0.5 * cfg.physics.bubble_dtheta
        state.qv = xp.where(core, xp.maximum(state.qv, 0.97 * qsat), state.qv)
        state.diagnose(cfg)
    return state


def _initial_state(grid: Grid, cfg: SimulationConfig, base=None) -> FlowState:
    """Smooth west(warm)->east(cold) blend for theta and q_v; zero hydrometeors.

    A deterministic (seed-independent) linear blend provides a physically
    reasonable initial condition; the inflow BCs and the pressure drop drive the
    subsequent mixing.  A small sinusoidal y-perturbation breaks the exact
    y-symmetry so the mixing zone is resolved (not a numerical artefact).

    For the ``deep_convection`` scenario the stratified base state is used
    instead (see :func:`_deep_convection_initial`).
    """
    if cfg.physics.scenario == "deep_convection":
        return _deep_convection_initial(grid, cfg, base)
    xp = grid.xp
    state = FlowState.zeros(grid)
    th_w, qv_w = bc.inflow_state(cfg.boundaries.warm_inflow, cfg.physics.P0)
    th_c, qv_c = bc.inflow_state(cfg.boundaries.cold_inflow, cfg.physics.P0)
    frac = (grid.xc / grid.Lx).reshape(grid.nx, 1, 1)   # 0 at west, 1 at east
    th_lin = (th_w * (1.0 - frac) + th_c * frac)         # (nx, 1, 1)
    qv_lin = (qv_w * (1.0 - frac) + qv_c * frac)
    state.theta = xp.broadcast_to(th_lin, grid.center_shape).copy()
    state.qv = xp.broadcast_to(qv_lin, grid.center_shape).copy()
    state.ql = xp.zeros(grid.center_shape); state.qi = xp.zeros(grid.center_shape)
    state.u = xp.zeros(grid.u_shape); state.v = xp.zeros(grid.v_shape); state.w = xp.zeros(grid.w_shape)
    bc.apply_velocity_bcs(state, grid, cfg)
    bc.apply_scalar_bcs(state, grid, cfg)
    state.diagnose(cfg)
    return state


def _pressure_method(grid: Grid) -> str:
    n = grid.nx * grid.ny * grid.nz
    if getattr(grid, "stretched", False):
        return "direct"   # variable-dz operator is asymmetric -> splu (not CG)
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
    def __init__(self, cfg: SimulationConfig, restart: str | None = None, base=None,
                 backend: Backend | None = None):
        self.cfg = cfg
        # compute backend (see backend.py). Defaulting to CPU here means every
        # existing direct `Simulation(cfg, ...)` call site (tests, scripts)
        # that doesn't pass backend= is unaffected.
        self.backend = backend if backend is not None else get_backend("cpu")
        self.grid = _grid_from_config(cfg, backend=self.backend)
        self.rng = np.random.default_rng(cfg.random_seed)
        # deep-convection base state (stratified sounding) for perturbation buoyancy;
        # an explicit `base` (e.g. Weisman-Klemp or a radiosonde) overrides the default.
        self._base_override = base
        self.base = None
        self.theta0_field = None
        self.qv0_field = None
        if cfg.physics.scenario == "deep_convection":
            from .base_state import build_base_state
            self.base = base if base is not None else build_base_state(self.grid)
            self.theta0_field = self.base.field(self.base.theta0, self.grid.center_shape, xp=self.grid.xp)
            self.qv0_field = self.base.field(self.base.qv0, self.grid.center_shape, xp=self.grid.xp)
        if restart:
            self.state = fio.load_restart(restart, self.grid)
            if self.base is not None and self.state.p0_field is None:
                self.state.p0_field = self.base.field(self.base.p0, self.grid.center_shape, xp=self.grid.xp)
        else:
            self.state = _initial_state(self.grid, cfg, base=self.base)
        if cfg.physics.precision == "float32":
            # performance mode: store the prognostic state in float32 (halves the
            # persistent-state / restart memory).  Intermediate numpy operations
            # still upcast to float64, so this is a memory mode, not a full
            # float32 kernel -- documented; float64 is the scientific default.
            for _nm in ("u", "v", "w", "p", "theta", "qv", "ql", "qi",
                        "qr", "qs", "qg", "qh", "p0_field"):
                _a = getattr(self.state, _nm, None)
                if _a is not None:
                    setattr(self.state, _nm, _a.astype(np.float32))
        self.state.diagnose(cfg)
        # reference Boussinesq state
        self.T_ref = float(self.state.T.mean()) if cfg.physics.T_ref is None else cfg.physics.T_ref
        self.qv_ref = float(self.state.qv.mean())
        self.rho0 = cfg.physics.P0 / (th.R_d * self.T_ref)
        # solvers.  All-Neumann projection; the top outflow is sized to balance
        # the two inflows (see boundary_conditions), so mean(div)=0 and the
        # projected velocity is divergence-free -> monotone scalar advection.
        self.pressure = PressureSolver(self.grid, method=_pressure_method(self.grid))
        # dynamical core: Boussinesq (constant rho0, test mode) vs anelastic
        # (rho0(z) reference density, div(rho0 u)=0 -> deep-convection mass
        # expansion).  Anelastic needs a stratified reference density profile;
        # use the base state's rho0(z), or build a default hydrostatic one.
        self.dynamics = getattr(cfg.physics, "dynamics", "boussinesq")
        xp = self.grid.xp
        self.rho0_c = None
        self.rho0_wface = None
        if self.dynamics == "anelastic":
            if self.base is not None:
                rho0_prof = np.asarray(self.base.rho0, dtype=float)
            else:
                from .base_state import build_base_state
                rho0_prof = np.asarray(build_base_state(self.grid).rho0, dtype=float)
            # rho0 on the z-faces (nz+1); np.interp clamps to the edge values
            # beyond the cell-centre range (constant extrapolation at ground/top).
            # These are tiny (nz,)/(nz+1,) profiles built once at init -- np.interp
            # has no direct GPU equivalent, so this one-time interpolation is done
            # on the host, then both profiles are moved to the compute backend.
            rho0_wface_h = np.interp(self.grid.backend.to_cpu(self.grid.zf),
                                     self.grid.backend.to_cpu(self.grid.zc), rho0_prof)
            self.rho0_c = xp.asarray(rho0_prof)                       # (nz,) cell centres
            self.rho0_wface = xp.asarray(rho0_wface_h)
        # conservative scalar transport (M5): the deep_convection scenario advects
        # with the projected divergence-free staggered velocity + reference density
        # (int rho0 q conserved, no wall leak).  Anelastic uses rho0(z); the
        # Boussinesq storm uses constant 1 (still exact flux form).
        self._transport_rho_c = None
        self._transport_rho_wf = None
        if cfg.physics.scenario == "deep_convection":
            if self.rho0_c is not None:
                self._transport_rho_c = self.rho0_c
                self._transport_rho_wf = self.rho0_wface
            else:
                self._transport_rho_c = xp.ones(self.grid.nz)
                self._transport_rho_wf = xp.ones(self.grid.nz + 1)
        # reference density for the CONSERVATION diagnostics: the anelastic
        # transport conserves int rho0(z) q, so the budgets must weight by rho0(z)
        # too (an unweighted sum drifts as a strong updraft redistributes water
        # vertically through the rho0 gradient -- M6).  Boussinesq: scalar rho0.
        self.rho_ref = (self.rho0_c if (self.dynamics == "anelastic" and self.rho0_c is not None)
                        else self.rho0)
        # periodic storm: face-broadcast environmental wind so the Rayleigh drag
        # relaxes the PERTURBATION toward it (the mean shear persists rather than
        # being damped away, which would defeat the point of ingesting it).
        self._u0_face = self._v0_face = None
        if getattr(self.grid, "periodic", False) and self.base is not None:
            self._u0_face = xp.broadcast_to(xp.asarray(self.base.u0, dtype=float)[None, None, :],
                                            self.grid.u_shape).copy()
            self._v0_face = xp.broadcast_to(xp.asarray(self.base.v0, dtype=float)[None, None, :],
                                            self.grid.v_shape).copy()
        # nucleation (diagnostic, one-way) vs microphysics (two-way coupling)
        self.stage = cfg.nucleation.stage
        self.do_nucleation = self.stage == "one_way"
        self.do_microphysics = self.stage in ("vapor_depletion", "thermal_feedback", "hydrometeor")
        # M7: couple the validated kernel rate J as the microphysics embryo source
        # (eq39 pathway) in the two-way stage.  The kernel supplies the SOURCE; the
        # microphysics still grows/converts/precipitates -- nucleation alone never
        # confirms precipitation.
        self.couple_nucleation = self.do_microphysics and getattr(
            cfg.nucleation, "couple_kernel", False)
        self.adapter = None
        self.lookup = None
        if self.do_nucleation or self.couple_nucleation:
            self.adapter = NucleationAdapter(cfg.nucleation)
            if cfg.nucleation.method == "lookup":
                self.lookup, _ = _build_lookup(cfg, cfg.output.outdir, self.adapter)
                self.adapter.set_lookup(self.lookup)
        self.coupler = None
        if self.do_microphysics:
            from .microphysics_coupling import MicrophysicsCoupler
            self.coupler = MicrophysicsCoupler()
        # bookkeeping
        self.snapshots = []
        self.history = []
        self.step = 0
        self.t = float(self.state.t)
        self.last_nf = NucleationField(self.grid.center_shape)
        self._t0 = _time.perf_counter()

    # ---- CFL (anisotropic, per-axis) ----
    def _dt(self) -> float:
        g = self.grid
        xp = g.xp
        st = self.state
        # per-component cell-centre velocity maxima -> anisotropic advective CFL
        #   dt_adv = cfl / (|u|/dx + |v|/dy + |w|/dz)
        uc = 0.5 * (st.u[:-1, :, :] + st.u[1:, :, :])
        vc = 0.5 * (st.v[:, :-1, :] + st.v[:, 1:, :])
        wc = 0.5 * (st.w[:, :, :-1] + st.w[:, :, 1:])
        umax = float(xp.abs(uc).max()) if uc.size else 0.0
        vmax = float(xp.abs(vc).max()) if vc.size else 0.0
        wmax = float(xp.abs(wc).max()) if wc.size else 0.0
        dzmin = g.dz if not getattr(g, "stretched", False) else float(g.dz_c.min())
        inv_adv = umax / g.dx + vmax / g.dy + wmax / dzmin
        self._inv_adv = inv_adv
        # 1.25 margin: the predictor (buoyancy + body force) grows |u| within a step.
        adv_dt = self.cfg.time.cfl / max(1.25 * inv_adv, 1e-12)
        # anisotropic diffusive limit: dt_diff = 0.5 / (K (1/dx^2 + 1/dy^2 + 1/dz^2))
        diff_coef = max(self.cfg.flow.nu, self.cfg.flow.kappa)
        diff_dt = 0.5 / (max(diff_coef, 1e-12)
                         * (1.0 / g.dx ** 2 + 1.0 / g.dy ** 2 + 1.0 / dzmin ** 2))
        dt_max = self.cfg.time.dt_max
        candidates = {"advective": adv_dt, "diffusive": diff_dt, "dt_max": dt_max}
        self._dt_limiter = min(candidates, key=candidates.get)
        dt = candidates[self._dt_limiter]
        if self._dt_limiter != "dt_max":
            self._n_dt_reductions = getattr(self, "_n_dt_reductions", 0) + 1
        return max(dt, 1e-6)

    # ---- one Chorin step ----
    def _step(self, dt: float) -> None:
        cfg = self.cfg
        g = self.grid
        xp = g.xp
        st = self.state
        order = cfg.flow.advection_order
        # 1. BCs + diagnose.  For the stratified base state the scalar z-BCs
        # preserve theta0(z)/qv0(z) (zero-gradient on the perturbation).
        bc.apply_velocity_bcs(st, g, cfg)
        bc.apply_scalar_bcs(st, g, cfg, theta0=self.theta0_field, qv0=self.qv0_field)
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
        # buoyancy on w (perturbation form vs base state for deep convection)
        Bf = buo.buoyancy_w_tendency(st, g, cfg, self.T_ref, self.qv_ref,
                                     theta0=self.theta0_field, qv0=self.qv0_field)
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
            if self._u0_face is not None:      # relax perturbation -> mean wind persists
                st.u = self._u0_face + (st.u - self._u0_face) * decay
                st.v = self._v0_face + (st.v - self._v0_face) * decay
            else:
                st.u *= decay; st.v *= decay
            st.w *= decay
        # safety limiter: bound velocities so the explicit upwind scheme stays
        # stable under strong buoyant acceleration (documented; only bites at
        # extreme speeds, never in the shallow mixing-chamber reference).
        _VCAP = 120.0
        xp.clip(st.u, -_VCAP, _VCAP, out=st.u)
        xp.clip(st.v, -_VCAP, _VCAP, out=st.v)
        xp.clip(st.w, -_VCAP, _VCAP, out=st.w)
        # 3. project the velocity to divergence-free BEFORE advecting scalars.
        # Flux-form upwind is monotone (bounded) only under a SOLENOIDAL velocity
        # (per-axis CFL<1); advecting with the divergent predictor lets multi-axis
        # convergence sum the inflow-CFL above 1 and create non-physical extrema.
        bc.apply_velocity_bcs(st, g, cfg)
        if self.dynamics == "anelastic":
            res, it = self.pressure.project_anelastic(st, dt, self.rho0_c, self.rho0_wface)
        else:
            res, it = self.pressure.project(st, dt, self.rho0)
        self._last_res = res; self._last_iters = it
        bc.apply_velocity_bcs(st, g, cfg)   # re-impose inflow (Neumann preserves it)
        # 4. scalar predictor: advect with the divergence-free velocity.  The
        # deep_convection storm uses the CONSERVATIVE mass-flux transport (M5):
        # the projected staggered velocity + reference density, in flux form, so
        # int rho0 q telescopes exactly to the boundary flux (conservative at any
        # intensity) and the closed walls carry zero flux (no spurious leak).  The
        # 2nd-order MUSCL/minmod reconstruction keeps it from diffusing away the
        # steep base theta0(z)/qv0(z) (which 1st-order upwind would erode, killing
        # the stratification / CAPE).  The mixing chamber keeps the cell-centre
        # scheme (its inflow/outflow BCs).
        if cfg.physics.scenario == "deep_convection":
            trc, trwf = self._transport_rho_c, self._transport_rho_wf
            _adv = lambda f: adv.advect_center_massflux(f, st.u, st.v, st.w, g, dt, trc, trwf, order=2)
        else:
            Uc, Vc, Wc = adv.cell_velocity(st, g)
            _adv = lambda f: adv.advect_center(f, Uc, Vc, Wc, g, dt, order)
        st.theta = _adv(st.theta)
        qv_new = _adv(st.qv)
        # positivity of q_v (last-resort clip; bookkeep loss).  Monotone upwind
        # under CFL<=1 should keep q_v>=0, so this rarely bites.
        clip_loss = float(xp.sum(xp.minimum(qv_new, 0.0)) * g.cell_vol)
        if clip_loss < 0:
            self._last_clip = clip_loss
        st.qv = xp.maximum(qv_new, 0.0)
        if self.do_microphysics:   # transport cloud + precipitating hydrometeors
            st.ensure_hydrometeors()
            st.ql = xp.maximum(_adv(st.ql), 0.0)
            st.qi = xp.maximum(_adv(st.qi), 0.0)
            st.qr = xp.maximum(_adv(st.qr), 0.0)
            st.qs = xp.maximum(_adv(st.qs), 0.0)
            st.qg = xp.maximum(_adv(st.qg), 0.0)
            st.qh = xp.maximum(_adv(st.qh), 0.0)
        # perturbation-only diffusion vs the stratified reference (deep_convection):
        # diffusing the full curved theta0(z)/qv0(z) would inject spurious buoyancy.
        st.theta = dif.diffuse_center(st.theta, g, cfg.flow.kappa, dt, base=self.theta0_field)
        st.qv = dif.diffuse_center(st.qv, g, cfg.flow.kappa, dt, base=self.qv0_field)
        st.qv = xp.maximum(st.qv, 0.0)
        # 5. scalar BCs + diagnose (velocity already div-free from step 3)
        bc.apply_scalar_bcs(st, g, cfg, theta0=self.theta0_field, qv0=self.qv0_field)
        bc.apply_velocity_bcs(st, g, cfg)
        st.diagnose(cfg)
        # 6. two-way microphysics: growth/conversion + embryo source + latent-heat
        #    feedback on theta, then gravitational sedimentation to the surface.
        if self.do_microphysics and self.coupler is not None:
            # M7: when kernel coupling is on, evaluate the validated 2nd-order rate
            # on the CURRENT (post-transport) state each step and pass it as the
            # embryo source; else the microphysics uses its own CCN/IN activation.
            if self.couple_nucleation and self.adapter is not None:
                nf = self.adapter.evaluate_field(st, dt, g.cell_vol)
                self.last_nf = nf
            else:
                nf = None
            self.coupler.apply(st, g, dt, nf=nf)
            # anelastic: sediment with the reference density so the airborne->
            # surface transfer is rho0-consistent with the transport/budget (M7).
            self.coupler.sediment(st, g, dt, rho_ref=self._transport_rho_c)
            self.coupler.zero_inflow_hydrometeors(st)
            bc.apply_scalar_bcs(st, g, cfg, theta0=self.theta0_field, qv0=self.qv0_field)
            st.diagnose(cfg)
            # deep-column safety: keep T inside the physical/correlation range so
            # a transient numerical overshoot cannot corrupt the saturation
            # curves (documented stability guard; only bites at extreme cells).
            if cfg.physics.scenario == "deep_convection":
                Tc = xp.clip(st.T, 180.0, 335.0)
                if not bool(xp.array_equal(Tc, st.T)):
                    st.theta = th.theta_from_T(Tc, st.P_total, th.P0_REF, xp=xp)
                    st.diagnose(cfg)
        st.t = self.t + dt

    # ---- nucleation diagnostics (one-way or kernel-coupled two-way) ----
    def _evaluate_nucleation(self, dt: float) -> NucleationField:
        if self.adapter is None:
            return NucleationField(self.grid.center_shape)
        return self.adapter.evaluate_field(self.state, dt, self.grid.cell_vol)

    # ---- main loop ----
    def run(self, progress=None) -> dict:
        cfg = self.cfg
        g = self.grid
        outdir = cfg.output.outdir
        os.makedirs(outdir, exist_ok=True)
        initial = diag.initial_budgets(self.state, self.rho_ref)
        self._last_clip = 0.0
        self._last_res = 0.0; self._last_iters = 0
        duration = cfg.time.duration
        interval = max(1, cfg.output.interval_steps)
        # progress cadence is DECOUPLED from the output cadence: a large
        # --output-interval (to keep flow.nc small) must not hide the progress.
        prog_every = max(1, min(interval, 10))
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
            # anisotropic advective CFL diagnostic: (|u|/dx+|v|/dy+|w|/dz) * dt
            cfl_now = getattr(self, "_inv_adv", 0.0) * dt
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
            if progress and (self.step % prog_every == 0):
                progress(self.t, duration, self.step)
        # finalise outputs
        report = self._finalise(initial, max_cfl)
        return report

    def _record(self, nf: NucleationField, initial: dict) -> None:
        st = self.state
        bud = diag.conservation_budgets(st, initial, self.rho_ref)
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
        # CSV history
        if "csv" in cfg.output.format:
            fio.write_csv(self.history, os.path.join(outdir, "history.csv"))
        # budgets plots
        if "budgets" in cfg.output.figures:
            plt_mod.plot_budgets(self.history, os.path.join(outdir, "figures"))
        # JSON summary
        bud = diag.conservation_budgets(self.state, initial, self.rho_ref)
        stats = diag.summary_stats(self.state, self.last_nf)
        wall = _time.perf_counter() - self._t0
        if self.dynamics == "anelastic":
            core_note = ("Anelastic core: rho0(z) reference density with div(rho0 u)=0, "
                         "so deep-column mass expansion (updrafts amplifying with height) "
                         "is represented -- a strict improvement over Boussinesq for deep "
                         "convection.  Scalar transport is conservative flux-form (M5): "
                         "the projected staggered velocity + rho0 weighting + 2nd-order "
                         "MUSCL, so int rho0 q is conserved.  The conservation budget is "
                         "rho0(z)-weighted (M6), consistent with what the transport "
                         "conserves -- the water error stays ~1e-3 at any updraft strength "
                         "(an unweighted budget spuriously drifted ~2% at ~24 m/s as the "
                         "updraft redistributed water through the rho0 gradient).")
        else:
            core_note = ("Boussinesq: density variations enter only through buoyancy; over a "
                         "deep storm column (10-12 km) this is stretched beyond strict "
                         "validity.  Use --dynamics anelastic for the deep-convection core.")
        limitations = [
            core_note,
            ("|gradT| floored at gmin: the |gradT|->0 limit is the kernel's "
             "near-equilibrium result (parameterization), NOT the CNT limit."),
            ("Momentum advection uses a centre round-trip (v1 simplification); the "
             "projection corrects divergence but this is not a fully conservative "
             "staggered momentum scheme."),
            "Not operational weather prediction; demonstration-scale only.",
        ]
        if self.do_microphysics:
            limitations.insert(1, ("Two-way microphysics active: hydrometeor growth + "
                "latent-heat feedback + sedimentation; per-step latent heating, velocity "
                "and temperature are bounded as documented stability safeguards."))
            if cfg.physics.scenario == "deep_convection":
                _core = "anelastic" if self.dynamics == "anelastic" else "Boussinesq-stretched"
                limitations.insert(2, ("Storm scale: stratified base state + warm-bubble "
                    "trigger; %s demonstration on a coarse grid -- updraft speeds, "
                    "condensate loading and surface totals are indicative, not "
                    "quantitative until the M4-M8 conservation/convergence criteria are met."
                    % _core))
        else:
            limitations.insert(1, ("One-way: nucleation is diagnostic; the prognostic "
                "state is NOT modified by microphysics (Batch 1)."))
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
            "limitations": limitations,
        }
        report["stage_microphysics"] = bool(self.do_microphysics)
        report["kernel_coupled"] = bool(self.couple_nucleation)
        from .config import estimate_memory_gb as _mem
        from .config import geometry as _geom
        report["geometry"] = _geom(cfg)
        report["memory_estimate_gb"] = _mem(cfg)
        report["precision"] = cfg.physics.precision
        report["dynamics"] = self.dynamics
        report["backend"] = {
            "name": self.backend.name,
            "device_label": self.backend.device_info().get("label"),
            "fallback_reason": self.backend.fallback_reason,
        }
        # M4 conservation: the (an)elastic mass-continuity residual (should be
        # ~0 -> the projection, not the limiters, enforces the constraint) and the
        # complete water budget (all species + surface accumulation).
        mres = diag.mass_continuity_residual(
            self.state,
            rho0_c=self.rho0_c if self.dynamics == "anelastic" else None,
            rho0_wface=self.rho0_wface if self.dynamics == "anelastic" else None)
        report["conservation"] = {
            "mass_continuity_residual_abs": mres["abs_max"],
            "mass_continuity_residual_norm": mres["normalised"],
            "total_water_rel_err": bud["total_water_rel_err"],
            "total_energy_rel_err": bud["total_energy_rel_err"],
            "water_measure": "airborne(all species)+surface accumulation",
            "note": ("water/energy are conserved to discretization error only in a "
                     "closed domain; boundary flux (top outflow / damping) is the "
                     "expected non-conservation channel."),
        }
        report["cfl_limiter_last"] = getattr(self, "_dt_limiter", None)
        report["n_dt_reductions"] = getattr(self, "_n_dt_reductions", 0)
        if getattr(self.state, "surface_precip", None) is not None:
            prec = {c: float(np.mean(v)) for c, v in self.state.surface_precip.items()}
            prec["total_mm"] = float(sum(prec.values()))
            report["surface_precip_mm"] = prec
        if "json" in cfg.output.format:
            fio.write_json(report, os.path.join(outdir, "summary.json"))
        # NetCDF last: it can be large/slow at high resolution, and its write is
        # guarded (io.write_netcdf), so the summary/history above always survive.
        if "netcdf" in cfg.output.format and self.snapshots:
            fio.write_netcdf(self.snapshots, os.path.join(outdir, "flow.nc"), self.grid, attrs)
        # Tecplot 360 ASCII (.dat): one ORDERED/POINT zone per snapshot, STRANDID
        # time animation.  Guarded like the NetCDF write.
        if "tecplot" in cfg.output.format and self.snapshots:
            fio.write_tecplot(self.snapshots, os.path.join(outdir, "flow.dat"), self.grid,
                              attrs, title="meteorological_flow storm t<=%.0fs" % self.t)
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
            "dynamics": self.dynamics,
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
        "performance": {"device": cfg.performance.device,
                        "compute_threads": cfg.performance.compute_threads},
        "random_seed": cfg.random_seed,
    }


__all__ = ["Simulation", "_grid_from_config", "_initial_state"]