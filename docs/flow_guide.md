# `meteorological_flow` — 3D Boussinesq flow with one-way water-phase nucleation

This is a **CPU 3D fluid-flow application** built *around* the repository's
validated, immutable water-phase nucleation engine (`met_water_nucleation`).
It simulates a 100 m mixing chamber: warm/humid air inflows from the west,
cold/dry air from the east, a uniform pressure drop drives a gentle
through-flow, gravity acts along −z, and the **second-order (shifted-equilibrium)
nucleation kernel** is evaluated one-way (diagnostic) over the resulting mixing
zone. The engine is treated **read-only** — no validated equation is modified;
the flow layer only feeds the kernel `(T, P, p_v, |∇T|)` and reads its outputs.

This guide covers the formulation, its documented consequences and
limitations, how to run it, and what the outputs mean. The two-way
microphysics (vapour depletion, latent heat, hydrometeor transport) is a
**gated Batch-2 extension**; Batch 1 stops at the one-way verification gate.

---

## 1. Formulation

### 1.1 Boussinesq, staggered Arakawa C-grid

Density variations enter **only through buoyancy**; the prognostic velocity is
incompressible (projected each step). Scalars — potential temperature θ,
vapour `q_v`, liquid `q_l`, ice `q_i`, perturbation pressure `p'` — live at
cell centres; `u, v, w` on the east/north/top faces. z is vertical (gravity
−z). Grid metrics and operators (divergence, face gradients, centred gradients,
Laplacian, centre→face interpolation) are plain numpy finite differences
(`grid.py`).

**Potential temperature θ is the transported conserved scalar**, with

```
T = θ · (P / P0_REF)^(R_d / c_p) ,   P0_REF = 100000 Pa
```

so adiabatic lifting cools the parcel (this is how supersaturation by ascent
enters). `P0_REF` is the *θ reference* pressure — **not** the scenario
background `P0 = 70000 Pa` (the two are deliberately distinct; conflating them
makes the initial `T` come back ~30 K too warm).

Moist Boussinesq buoyancy on the w-equation:

```
B = g · [ (T − T_ref)/T_ref + 0.61 (q_v − q_v,ref) − (q_l + q_i) ]
```

with `T_ref, q_v,ref` the fixed initial references.

### 1.2 Time stepping (Chorin projection)

Each step (`simulation.py`):

1. enforce velocity + scalar BCs; diagnose `T, p_v, S, ρ, |∇T|`;
2. **momentum predictor**: viscous diffusion, add buoyancy to `w`, add the
   uniform pressure-drop body force `du/dt = p_drop/(ρ0·Lx)` to `u`, apply
   linear Rayleigh momentum drag `u ← u·(1 − γ·dt)`. *(v1 simplification,
   documented: advective transport of momentum is deferred — see §3.)*
3. **project the velocity to divergence-free BEFORE advecting scalars**
   (Chorin: `∇²p' = (ρ0/Δt)∇·u*`, `u ← u* − (Δt/ρ0)∇p'`);
4. **scalar predictor**: advect + diffuse θ, `q_v` (and `q_l, q_i` at the
   hydrometeor stage) with the now-solenoidal velocity; clip `q_v ≥ 0` as a
   last-resort safety (bookkept as a conservation loss);
5. re-enforce BCs; diagnose.

**Adaptive CFL** each step:
`dt = min( cfl·min(dx)/max(1.25·|u|), 0.5·min(dx)²/(3·max(ν,κ)), dt_max )`.
The 1.25 margin sizes `dt` for the predictor growing the velocity within a
step.

### 1.3 Pressure solver

The 7-point cell-centre Laplacian is assembled **once** as a `scipy.sparse`
matrix (`A = −∇²`, positive semi-definite + a 1e-12 diagonal pin) with
all-Neumann BCs. Small grids (≤ ~40³) use a cached `splu` factorisation; larger
grids use CG + Jacobi. All-Neumann makes the system singular (constant null
space); the RHS is mean-subtracted for compatibility and the solution is
mean-zeroed, so `p'` has zero mean. The residual and iteration count are
reported each step.

**Sign convention** (a subtle point that bit us during development): with
`A = −∇²`, Chorin's `∇²p' = +(ρ0/Δt)∇·u` requires `rhs = −(ρ0/Δt)·div`.
Boundary faces use `dp'/dn = 0`, so the projection does **not** alter the
boundary (inflow Dirichlet) velocity.

### 1.4 Boundary conditions & the mass-balanced top outflow

- **West / east**: Dirichlet inflow (warm/moist west, cold/dry east, both at
  2 m/s into the domain). Inflow scalars are fixed from the configured T/RH
  via the engine's saturation curves; `q_l = q_i = 0` at inflows.
- **y**: free-slip (zero normal `v`), zero-gradient scalars.
- **z bottom**: free-slip (`w = 0`).
- **z top**: a **mass-balanced outflow** — `w_top = (u_warm + u_cold)·Lz/Lx`,
  sized so the net boundary flux is exactly zero. This is what makes the
  all-Neumann projection yield a genuinely divergence-free (not merely
  constant-divergence) field, which is the precondition for monotone scalar
  advection (see §3).

### 1.5 Nucleation coupling (one-way / diagnostic, Batch 1)

The adapter (`nucleation_adapter.py`) builds the kernel simulator **once** and
calls, per cell,
`sim.evaluate_point(T, P, p_v, r_ref, grad_T_req=|∇T|)`, collecting every
kernel output (`ΔT, T_eq_shift, P_eq_shift, γ, ∂γ/∂r, Γ1, Γ2, rC_1st, rC_2nd,
ΔG_1st, ΔG_2nd, I, log10I, dominant, closure_resid`) into grid arrays — the
1st-order / CNT / 2nd-order results are kept **distinct** (the kernel exposes
`rC_1st`/`rC_2nd`, `Gamma1`/`Gamma2`, `DeltaG_1st`/`DeltaG_2nd`).

**One-way means the prognostic state is NOT modified by the microphysics.**
The kernel outputs are recorded (NetCDF, plots, JSON) but `q_v, θ, …` are
untouched. Two-way coupling is Batch 2, gated on this foundation passing its
verification.

**Lookup table** (`nucleation_lookup.py`): the gradient-scan kernel path is
too costly to call per cell per step, so the outputs are precomputed once over
`(T, p_v, |∇T|, phase)` — `|∇T|` log-spaced — cached to `.npz`, and interpolated
per cell with `scipy.RegularGridInterpolator` (trilinear). Build is
parallelised with `multiprocessing`. Subgrid structural fields (`rC`, `Gamma`,
…) that the kernel returns as NaN where no embryo forms are **nearest-filled**
at the phase boundary so interpolation is defined everywhere (a documented
parameterization).

---

## 2. Scientific integrity

- The validated core (`src/met_water_nucleation/_engine/**`) is **never
  modified** — SHA-256-guarded, ruff-excluded. All physics is reused via
  `import met_water_nucleation as M`.
- Existing reference results are **not overwritten**; flow outputs go to a
  separate gitignored `outputs/flow_reference/` (and `out_met_nucleation/`).
- 1st-order / CNT / 2nd-order results are reported distinctly.
- Parameterizations and extrapolations are **labelled** (see §3).
- **Reproducibility**: config + code version + random seed are written into
  every output (NetCDF global attrs, `summary.json`).

---

## 3. Documented consequences & limitations

These are not bugs; they are the honest scope of a Batch-1 demonstration-scale
solver. The `summary.json` report carries this list verbatim.

1. **Boussinesq**: the imposed `ΔP = 30 Pa` over 70000 Pa gives adiabatic
   expansion cooling ~0.1 K — second-order. Supersaturation is dominated by
   **mixing** and **buoyant lifting**, not by the pressure drop itself. A
   low-Mach/compressible solver is future work.
2. **One-way (Batch 1)**: nucleation is diagnostic; the prognostic state is not
   modified by microphysics. Two-way coupling is a gated Batch-2 step.
3. **`|∇T|` floored at `gmin`**: the `|∇T| → 0` limit is the kernel's
   near-equilibrium result (parameterization), **not** the CNT limit. The CFD
   Cartesian `|∇T|` is identified with the kernel's radial `|dT/dr|` as a
   first-order subgrid closure (validation test 14 checks the limit).
4. **Momentum advection deferred** (v1 simplification): the velocity is
   governed by the body force + buoyancy + diffusion + projection; the
   **scalars** are advected by the resulting divergence-free velocity (which
   drives the mixing → supersaturation → nucleation). Fully conservative
   staggered momentum advection is a Batch-2 upgrade. The projection still
   enforces `∇·u ≈ 0`.
5. **Rayleigh momentum drag** `γ = 0.2 /s` is a documented bulk subgrid
   dissipation that bounds the otherwise-unbounded Boussinesq buoyant
   convection (warm parcels do not cool on ascent, so without a sink the plume
   accelerates unboundedly). `γ = 0` disables it.
6. **Rain/snow/graupel/hail** are reported as thermodynamic / microphysical
   **favorability** diagnostics, **not** precipitation prediction.
   Hydrometeor growth (collision-coalescence, riming, melting) is not modelled
   in Batch 1; a high nucleation rate never by itself implies rain or hail.
7. **Not operational weather prediction**; demonstration-scale only.

A consequence worth stating plainly: the projected velocity is
divergence-free **only because the top outflow is mass-balanced**. With two
unbalanced inflows the all-Neumann projection leaves a constant residual
divergence (`div(u_new) = mean(div) ≠ 0`), and flux-form upwind — monotone
*only under a solenoidal velocity* — would then create non-physical scalar
extrema. The mass-balanced top outflow makes `mean(div) = 0`, so the projected
field is divergence-free everywhere (the reference demo reaches `divmax ≈
1e-11`).

---

## 4. Running it

### 4.1 As a module / console script

```bash
# full one-way reference demo (20^3, builds the lookup once, ~6 min first time;
# the table is cached and reused on subsequent runs):
python -m meteorological_flow.cli --config configs/cold_dry_vs_warm_moist.yaml --grid-resolution 20 --duration 60 --one-way-coupling --output outputs/flow_reference --threads 8

# pure flow (no microphysics), fast sanity check:
python -m meteorological_flow.cli --config configs/cold_dry_vs_warm_moist.yaml --grid-resolution 20 --duration 60 --no-microphysics --output outputs/flow_pure

# after `pip install -e .` the console script is available:
meteorological-flow --config configs/cold_dry_vs_warm_moist.yaml --grid-resolution 40 --duration 120 --one-way-coupling
```

### 4.2 Flags

| flag | meaning |
|---|---|
| `--config PATH` | YAML scenario (default `configs/cold_dry_vs_warm_moist.yaml`) |
| `--grid-resolution {20,40,50}` | override `nx=ny=nz` (20³ dev default, dx=5 m; **not** 100³) |
| `--duration S` | override simulated duration |
| `--output DIR` | output directory |
| `--output-interval N` | snapshot + nucleation cadence in steps |
| `--threads N` | multiprocessing threads for the lookup build |
| `--one-way-coupling` | stage = one_way (diagnostic nucleation) |
| `--no-microphysics` | stage = none (pure flow) |
| `--diagnostic-only` | alias for one-way |
| `--method direct\|lookup` | kernel evaluation method (lookup required at scale) |
| `--restart PATH` | restart from a `.npz` checkpoint |
| `--dry-run` | print the plan (grid, dt estimate, table size) and exit |
| `--validate` | run the flow validation suite, exit 0/1 |

### 4.3 Python API

```python
from meteorological_flow import SimulationConfig, from_yaml, apply_overrides, Simulation
cfg = apply_overrides(from_yaml("configs/cold_dry_vs_warm_moist.yaml"),
                      grid_resolution=20, duration=60, one_way=True)
report = Simulation(cfg).run()      # -> summary dict (also written to summary.json)
```

A thin runner is `examples/run_reference_demo.py`.

---

## 5. Outputs

All go to the output directory (default `outputs/flow_reference/`):

- `flow.nc` — time-dependent NetCDF3 (scipy engine), dims `(time, z, y, x)`:
  `u,v,w,T,T_local_*,P,p_v,RH_*,q_*,S_*,gradT,ΔT,P_eq_shift_*,Γ2_*,rC_2nd_*,
  log10I_*,dominant_phase,solver_residual,validity_mask,rho`. Global attrs carry
  code version, formulation, P0, seed, ρ0, T_ref, grid, dx/dy/dz, stage.
- `history.csv` — domain-integral budgets (water/energy, mean S, extrema,
  solver residual) per output cadence.
- `summary.json` — wall-clock, memory, n_steps, max CFL, ρ0, T_ref, final
  stats (extrema, max S, max log10I, nucleation cell counts), final budgets,
  solver residual, the **limitations list**, and the full config+seed.
- `restart.npz` — checkpoint at the output cadence.
- `nucleation_lookup.npz` — the cached lookup table (reused across runs).
- `figures/` — horizontal/vertical slices of T, S_w/S_i, p', |u|+vectors, w,
  |∇T|, log10I(liquid/ice), q_v; budget plots.

### Reference demo numbers (20³, 60 s, one-way)

```
wall clock   : 35.9 s  (excludes the one-time lookup build)
steps        : 240     final t = 60.00 s
max CFL      : 0.271
T range      : 255.45 .. 293.67 K
max |u|      : 5.43 m/s   max |w| : 5.05 m/s
max S_w/S_i  : 1.707 / 1.779
max log10I   : liq = 57.88   ice = 54.23
liq nuc cells: 2220   ice nuc cells : 1820
solver resid : 9.5e-14
```

The high `log10I` is the kernel's honest homogeneous-limit output at the
mixing-zone supersaturation (`S_w ≈ 1.7`); it is **not** adjusted for visual
plausibility (per the integrity constraint). Water/energy "errors" are nonzero
because the boundaries are open (mass/energy flux through them) — expected and
documented, not a conservation failure.

---

## 6. Verification (the Batch-1 gate)

The spec mandates a verification gate before any two-way coupling:

1. `python -m pytest tests/` — the original 24 nucleation tests **plus** the
   flow suite (`tests/test_grid.py`, `test_advection.py`,
   `test_pressure_projection.py`, `test_scalar_conservation.py`,
   `test_boundary_conditions.py`, `test_nucleation_adapter.py`,
   `test_lookup_accuracy.py`, `test_reference_scenario.py`) all pass.
2. `met_h2o_nucleation.py --validate` still PASS (the guarded core is
   untouched).
3. `python -m meteorological_flow.cli --validate` — the flow suite green.
4. The 20³ one-way reference demo produces NetCDF + JSON + CSV + PNG and a
   physically sane report.

**Batch 2 (gated, next)** — `phase_change.py`: vapour depletion (mass-conserving,
`q_v ≥ 0`) + latent heat (condensation/deposition/freezing/melting signs) +
buoyancy feedback; then hydrometeor transport + sedimentation; precip
favorability diagnostics (labelled, not prediction); optional LES/Smagorinsky
toward operational microphysics.