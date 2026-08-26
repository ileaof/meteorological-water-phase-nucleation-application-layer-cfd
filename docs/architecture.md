# Architecture

The `met_water_nucleation` package is a thin, importable facade over an
**immutable validated physics engine**. The engine is loaded **read-only** via
`importlib` and is never modified or refactored; its integrity is SHA-256
guarded (`--validate`).

## Two-layer model

```
                 +------------------------------------------+
   met input --> |  application / diagnosis layer            |  met_h2o_nucleation.py
                 |  MetInput / Runner / Diagnosis           |  + het_contact_angle.py
                 |  free-energy decomposition / I/O / viz    |
                 +---------------------+--------------------+
                                       |  imports READ-ONLY (importlib, by path)
                                       v
                 +------------------------------------------+
                 |  validated core                          |  unified_h2o_nucleation_climate.py
                 |  closure, r_C, Gamma, rate, tests [1]-[21]|  (SHA-256 guarded)
                 +------------------------------------------+
```

The core owns the closure `F(g;r)=Γ²/(4πr²)−g=0`, the 1st/2nd-order critical
radius, the surface-stress law, the nucleation rate and the validation suite.
The application layer adds only what the core deliberately does not own:
free-energy decomposition, precipitation diagnosis, I/O adapters, the full
report schema and visualisation.

## Scientific data flow

```
meteorological input  (T, P, humidity, optional w/LWC/IWC/N_ccn/N_inp/...)
        |
        v
atmospheric thermodynamics   resolve_humidity -> p_v; S_w, S_i (IAPWS Wagner / Goff-Gratch)
        |
        v
nonequilibrium nucleation    thermal closure (Brent) -> T_local; P_eq,shift = P_sat(T_local)
        |                    2nd-order Gibbs-Thomson; critical radius r_C,2nd (Eq.39b)
        v
phase competition            liquid vs ice nucleation rate I; kinetic dominance
        |
        v
optional fluid-flow coupling (NONE in this repo: no CFD solver; `--w`/cooling are inputs)
        |
        v
meteorological diagnostics   rain/snow/graupel/hail favourability + diagnostic_class
        |
        v
outputs & visualisation      JSON / CSV / NetCDF (xarray) / PNG
```

Notes:
- "optional fluid-flow coupling" is a **placeholder within this package
  only**: `met_water_nucleation` itself has no grid, advection, pressure
  solver or turbulence module; updraft `w` and `cooling_rate` are scalar
  inputs to it, not a solved flow. The repository DOES have a real,
  separate CFD solver -- the `meteorological_flow` package (staggered
  Arakawa C-grid, advection, diffusion, buoyancy, a pressure-projection
  Poisson solve, optional two-way microphysics) -- which calls this
  package's engine as its nucleation kernel; see "The meteorological_flow
  CFD solver" and "CPU/GPU execution backends" below. This note describes
  `met_water_nucleation` in isolation, not the whole repository.
- The contact angle θ is solved self-consistently from Ferreira Eq.17
  (`r_C,Het/r_C,Hom`) and reported as `contact_angle_deg`; with no substrate
  surface energies modelled, the solver returns the homogeneous limit.

## The meteorological_flow CFD solver

`src/meteorological_flow/` is the repository's actual CFD layer: a 3-D
staggered Arakawa C-grid (`grid.py`), finite-volume advection (`advection.py`,
1st-order upwind or 2nd-order MUSCL/minmod), explicit diffusion
(`diffusion.py`), moist Boussinesq/anelastic buoyancy (`buoyancy.py`), a
Chorin pressure-projection Poisson solve (`pressure_solver.py`, direct `splu`
or iterative CG), and boundary conditions (`boundary_conditions.py`). It calls
`met_water_nucleation`'s validated engine (via `nucleation_adapter.py` /
`nucleation_lookup.py`) as its nucleation kernel, evaluated at output cadence
against a precomputed interpolation table -- the engine itself is never
modified, consistent with its read-only/SHA-256-guarded status above.
`simulation.py` is the time-stepping orchestrator (`Simulation._step`/`.run`);
`cli.py` is its command-line entry point (`meteorological-flow` console
script / `python -m meteorological_flow.cli`).

## CPU/GPU execution backends

The solver runs on CPU (NumPy/SciPy) by default and optionally on GPU (CuPy),
selected via `--device auto|cpu|gpu` (default `auto`) or
`SimulationConfig.performance.device`. See `backend.py` for the full design;
summary:

- **`get_backend(device)`** (`backend.py`) resolves a `Backend` object
  (`.xp` = `numpy` or `cupy`, `.sparse`/`.sparse_linalg`, `.to_cpu()`,
  `.synchronize()`, `.device_info()`) via a real detection state machine:
  library import -> CUDA device count -> a trivial kernel launch -> a VRAM
  preflight check against `config.estimate_memory_gb()`. `--device gpu` fails
  loudly on any failure (a categorized `BackendError`: missing library,
  incompatible driver, CUDA unavailable, no GPU detected, insufficient
  memory, or -- from a specific call site rather than this up-front probe --
  an unsupported kernel) -- it never silently runs CPU math while claiming
  GPU. `--device auto` catches the same errors and falls back to CPU with a
  logged reason.
- **Threading, not per-array dispatch**: `Grid` and `FlowState` each carry
  `self.xp`/`self.backend` (set once at construction); every operator
  (`advection.py`, `diffusion.py`, `buoyancy.py`, `boundary_conditions.py`,
  `pressure_solver.py`, `thermodynamics.py`, the `Simulation._step`/`._dt`
  hot path, `base_state.py`'s field-broadcasting) reads `grid.xp` rather than
  hardcoding `numpy`, so a GPU-resolved backend keeps the large 3-D fields
  resident on the device across the whole time loop -- no per-step transfer.
  `thermodynamics.psat_water`/`psat_ice` were rewritten from a hidden
  Python-loop `np.vectorize` wrapper into true vectorised `xp.exp`/`xp.log10`
  closed-form expressions (same published Wagner/Goff-Gratch coefficients,
  pinned equal to the engine's per-element output by
  `tests/test_thermodynamics_vectorization.py`) -- this was both a real CPU
  speedup and a hard prerequisite for GPU residency (`np.vectorize` cannot
  accept a CuPy array).
- **Explicit host/device boundaries**: everything that leaves the solver --
  NetCDF/JSON/CSV/restart I/O (`io.py`), matplotlib plotting (`plotting.py`),
  and the nucleation-lookup interpolation layer (`nucleation_lookup.py`,
  `nucleation_adapter.py`) -- pulls GPU-resident arrays to the host once, at
  that boundary, via `backend.to_cpu()`, rather than relying on any implicit
  conversion (CuPy deliberately raises `TypeError` on an implicit
  GPU-\>NumPy conversion, which is what makes these boundaries easy to find
  and keep correct rather than a source of silent wrong results).
- **Pressure solver, GPU-vs-CPU difference (documented, not silent)**: CuPy
  has no GPU-native direct sparse solver equivalent to SciPy's `splu` (only
  iterative solvers). For a **non-stretched** grid, the GPU backend therefore
  always uses iterative CG, overriding whatever `_pressure_method()` would
  pick for CPU on that grid size (`direct` for small grids, `cg` for large
  ones) -- logged once per run
  (`[pressure_solver] GPU backend: using iterative CG ...`). For a
  **stretched** grid (`--z-stretch`), the vertical operator is asymmetric
  (see `pressure_solver.py`'s `_build()` comment), and CG is not guaranteed
  to converge on it -- confirmed the hard way: an early build forced CG there
  too and it diverged (max CFL ~1e12, every field NaN) on a real
  `--storm-scale --z-stretch ... --device auto` run. Stretched grids
  therefore **always** use the direct solve, regardless of backend, exactly
  like the CPU heuristic already does; when the rest of the solver is on
  GPU, only the small `(n,)`-sized RHS/solution vectors cross the host/device
  boundary each step for that solve, never the operator or any 3-D field
  (`[pressure_solver] GPU backend + stretched grid: forcing the direct CPU
  solve ...`). The Laplacian matrix itself is always assembled on the host (a
  one-time, not per-step, index-bookkeeping loop -- no vectorisation win from
  moving it), then transferred to the GPU once at `PressureSolver`
  construction (only for the CG/non-stretched case).
- **Precision**: `float64` is the default and the scientific baseline on
  both backends; `--precision float32` (or the older `--float32` flag) is an
  explicit opt-in memory-savings mode, documented and unchanged in behaviour
  by the GPU work.
- **Two-way microphysics coupling is GPU-accelerated too**: the
  `precip_microphysics` package (hydrometeor growth, latent-heat feedback,
  sedimentation -- `--two-way-coupling`/`--storm-scale`'s default
  `hydrometeor` stage) is backend-aware, following the same pattern as the
  core solver. Its `MicrophysicsState` has no `Grid` of its own (it is a
  framework-agnostic dataclass, reused identically for a 0-D parcel and for
  full 3-D flow-coupled fields, and has no dependency on this module to avoid
  inverting the existing package layering), so it carries a plain `xp` array-
  module reference instead of a `Backend` object; the flow-coupled path
  (`meteorological_flow/microphysics_coupling.py`) passes `xp=grid.xp`
  explicitly. `precip_microphysics/thermo.py` had the same hidden
  `np.vectorize`-into-the-engine problem as `thermodynamics.py`'s
  `psat_water`/`psat_ice` and got the same fix, duplicated independently
  (not shared, again to avoid the layering inversion) and pinned by
  `tests/test_precip_microphysics_thermo_vectorization.py`. See
  `tests/test_flow_microphysics_coupling.py`'s `TestCpuVsGpuMicrophysics` and
  `tests/test_backend_equivalence.py::test_cpu_vs_gpu_storm_two_way` for the
  equivalence coverage. The package's genuinely 0-D-only standalone surface
  (`column.py`, `scenarios.py` -- confirmed by survey to have zero 1-D/3-D
  call sites anywhere in the repo) was left untouched; it stays numpy-only
  with no behavioural change. The `met_water_nucleation` CLI/report tool is a
  thin wrapper around the immutable engine's scalar `evaluate_point` API and
  has no GPU-relevant hot path.

### Numeric equivalence (CPU vs GPU)

`tests/test_backend_equivalence.py` runs the same case once per backend and
compares `report["final_stats"]`/`["conservation"]`/`["final_budgets"]`
within explicitly justified tolerances -- **not** bit-for-bit equality, since
the GPU path's CG pressure solve and the CPU path's direct `splu` solve (for
small grids) are different, individually-correct discretisations of the same
Poisson problem. Field extrema/thermodynamic stats: `rtol=1e-4`;
conservation/budget relative errors: `atol=1e-3`. Both backends are also
checked for NaN/Inf and same-backend run-to-run reproducibility.

### Dependencies

`threadpoolctl` (required) gives runtime BLAS/OpenMP thread control
(`--compute-threads N`) without relying on environment variables being set
before NumPy is imported (which this CLI's own import order does not
guarantee). `cupy-cuda12x` (optional, `pip install "met_water_nucleation[gpu]"`)
is never required for CPU-only use; see the README's GPU install section for
platform-specific notes and `scripts/benchmark_backends.py` for measured
CPU-vs-GPU timings on real hardware.

## Import mechanics (why the bundle is co-located)

The core's SHA-256 guard computes the reference-script location as
`dirname(dirname(__file__))` and the loaders find the core by a
`__file__`-relative search. The whole bundle (core, two reference scripts,
`met`, `hca`) therefore moved **as a unit** into
`src/met_water_nucleation/_engine/` so that the relative arrangement — and
hence the guard and the loaders — keeps resolving identically. The move was
byte-identical (`git mv`); all five engine files retain their pre-reorg
SHA-256 checksums.