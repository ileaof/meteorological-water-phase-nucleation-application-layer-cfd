# met_water_nucleation — meteorological water-phase nucleation application layer

A Python package for **non-equilibrium (shifted-equilibrium) water-phase
nucleation** under a thermal gradient, plus meteorological diagnostics, built
on the Ferreira Eq.39a/39b framework (Physica B **695** (2024) 416494; MRS
Meeting 2026). It computes vapour→liquid (condensation) and vapour→ice
(deposition) nucleation — homogeneous or heterogeneous — the shifted
equilibrium, 1st/2nd-order critical radii, the free-energy decomposition, the
nucleation rate, and transparent rain/snow/graupel/hail *favourability*
diagnostics for one or more atmospheric states.

> **Scope vs. precipitation.** This is a **nucleation** diagnosis layer. A high
> nucleation rate never by itself implies rain or hail — hydrometeor growth
> (condensation/deposition, collision–coalescence, accretion, riming,
> melting/refreezing) is **not modelled**. See `docs/MET_NUCLEATION_HYPOTHESES.md`.

---

## Scientific scope

For one or more atmospheric states (T, P, humidity + optional dynamics) the
package reports, for each admissible phase:

- vapour→liquid and vapour→ice nucleation, homogeneous or heterogeneous;
- the non-equilibrium thermal closure (radius as continuation variable, the
  thermal gradient as the Brent-solved unknown);
- the 2nd-order Gibbs–Thomson coefficient and 1st/2nd-order critical radii
  (Ferreira Eq.39b parabola);
- the free-energy decomposition ΔG_V / ΔG_bulk / ΔG_surface / ΔG_config / ΔG_total;
- the shifted equilibrium pressure P_eq,shift = P_sat,phase(T_local);
- the nucleation rate I, log10 I, and expected event count I·dt·V_cell;
- transparent 0..1 rain/snow/graupel/hail favourability indices with
  contributing/missing variables, confidence and a caveat;
- ingestion of scalars, profiles, time series, xarray/NetCDF/GRIB fields;
- structured JSON/CSV/NetCDF output and optional PNG figures.

All internal quantities are SI. Undetermined quantities are reported as
`"undetermined"` with the missing information named.

---

## Installation

Requires **Python ≥ 3.9**. `numpy`, `scipy` and `matplotlib` are required (the
bundled core imports all three at load time); the I/O backends are optional.

```bash
git clone <this repo>.git
cd met_h2o_nucleation_cfd
python -m pip install -e .                 # core deps; installs the package
# optional backends:
python -m pip install -e ".[netcdf]"       # xarray + netCDF4
python -m pip install -e ".[grib]"         # cfgrib + eccodes (needs system eccodes)
python -m pip install -e ".[io]"           # all I/O backends + pandas
python -m pip install -e ".[dev]"          # pytest + ruff
```

The validated physics core, the two SHA-256-guarded reference models and the
application modules are all bundled under
`src/met_water_nucleation/_engine/`, so **no `PYTHONPATH` or external checkout
is needed**.

---

## Quick start

```bash
# prove the bundled core is intact and the layer self-checks pass
met-water-nucleation --validate

# one state, both phases, supersaturated, with dynamics
met-water-nucleation --T 260 --P 70000 --RH 110 --phase-mode both \
    --w 2.0 --LWC 5e-4 --IWC 1e-4 --dt 60 --Vcell 1e6 --summary \
    --json outputs/cli_report.json
```

Python API:

```python
import met_water_nucleation as M

met = M.MetInput(T=260.0, P=70000.0, RH=110.0, rh_reference="water",
                 phase_mode="both", mode="homogeneous",
                 w=2.0, LWC=5e-4, IWC=1e-4, cooling_rate=-2.0e-4,
                 N_ccn=3.0e8, N_inp=1.0e4, dt_micro=60.0, cell_volume=1.0e6)
runner = M.MetNucleationRunner(met)
p_v, src, warns = M.resolve_humidity(met, 260.0, 70000.0)
reps = runner.evaluate_point(260.0, 70000.0, p_v,
                             dynamics={"w": 2.0, "LWC": 5e-4, "IWC": 1e-4})
for phase, report in reps.items():
    print(phase, report.status, report.log10_nucleation_rate,
          report.r_critical_2nd_m, report.diagnostic_class)
M.to_json(reps, "outputs/my_report.json")
```

---

## Package structure

```
met_h2o_nucleation_cfd/
├── pyproject.toml            metadata, deps, CLI entry point, optional extras
├── README.md                 this file
├── met_h2o_nucleation.py     BACKWARD-COMPAT SHIM (delegates to the package CLI)
├── src/met_water_nucleation/
│   ├── __init__.py           package facade; re-exports the engine API
│   ├── cli.py / __main__.py  console entry point + `python -m …`
│   └── _engine/              IMMUTABLE bundle (loaded read-only, never edited)
│       ├── met_h2o_nucleation.py            application / diagnosis layer
│       ├── het_contact_angle.py             heterogeneous contact-angle models
│       ├── unified_h2o_nucleation_climate/
│       │   └── unified_h2o_nucleation_climate.py   validated core (SHA-256 guarded)
│       └── Nucleation_model_H2O_vapour_{solid,liquid}_Sim_2026*.py  reference models
├── tests/                    test_met_nucleation.py + conftest.py
├── examples/                 single_state, vertical_profile, xarray_netcdf,
│                             figures, frontal_collision
├── configs/                  declarative scenario YAMLs
├── scripts/                  run_validation, regenerate_outputs
├── docs/                      MANUAL, HYPOTHESES, architecture, migration guide
├── references/               reserved for papers/presentations (see README)
├── legacy/                   review area for ambiguous / historical files
├── data/                     reserved for input/reference datasets
└── outputs/                  generated outputs (outputs/<scenario>/<run-id>/)
```

The engine bundle under `_engine/` is **immutable**: the core's SHA-256
integrity guard (`--validate`) verifies the two reference models byte-for-byte.
Production code accesses the engine only through the package facade; it is
never refactored together with the engine.

---

## Command line

```
met-water-nucleation [--validate]
    [--T K] [--P Pa] [--RH %] [--p-v Pa]
    [--phase-mode auto|liquid|ice|both] [--mode homogeneous|heterogeneous]
    [--theta DEG] [--r-ref m] [--gradT K/m]
    [--w m/s] [--LWC kg/m3] [--IWC kg/m3] [--dt s] [--Vcell m3]
    [--outdir DIR] [--json PATH] [--summary]
```

`--validate` runs the core validation suite [1]–[21] (proving the guarded core
is untouched) plus the met-layer self-checks; exits 0/1. `--summary` prints the
compact one-row-per-phase table; without it the full 48-field report is printed.
See `docs/MANUAL_met_h2o_nucleation.md` §14 for the full flag reference and 10
verified CLI cases.

---

## Testing

```bash
python -m pytest tests/        # 24 nucleation tests + 36 flow-suite tests
# or, without pytest:
python tests/test_met_nucleation.py
```

Tests are CWD-independent (`tests/conftest.py` bootstraps the `src/` path). The
flow suite (`tests/test_{grid,advection,pressure_projection,
scalar_conservation,boundary_conditions,nucleation_adapter,lookup_accuracy,
reference_scenario}.py`) covers the staggered operators, projection, scalar
conservation/positivity, BCs, the one-way adapter fidelity, lookup accuracy,
and the reference-scenario smoke/budget report.

---

## 3D flow solver (`meteorological_flow`)

Alongside the single-state nucleation tool, the repository now ships a
**CPU 3D Boussinesq flow application** that drives the validated kernel as its
microphysics subroutine: a 100 m mixing chamber with warm/moist vs cold/dry
opposing inflows, a pressure drop, gravity, and **one-way (diagnostic)**
water-phase nucleation over the mixing zone. The engine stays read-only; the
flow layer only feeds the kernel `(T, P, p_v, |∇T|)` and records its outputs.

```bash
# 20^3 one-way reference demo (builds the lookup once, caches it):
python -m meteorological_flow.cli --config configs/cold_dry_vs_warm_moist.yaml \
    --grid-resolution 20 --duration 60 --one-way-coupling --threads 8
# or, after `pip install -e .`:
meteorological-flow --config configs/cold_dry_vs_warm_moist.yaml --one-way-coupling
```

The formulation (staggered C-grid, Chorin projection, mass-balanced top
outflow, lookup-table nucleation), its documented consequences and
limitations, run instructions, and the Batch-2 (two-way) roadmap are in
**`docs/flow_guide.md`**. Two-way microphysics (vapour depletion, latent heat,
hydrometeor transport) is gated on the one-way foundation passing its
verification.

---

## CPU expectations

A single-state report is sub-second. The `--validate` suite runs in ~1 s. The
`examples/figures.py` figure suite (scanning ∇T and T over 12–18 points for both
phases, plus the P_eq,shift surface) takes ~10–30 s on a laptop. The validated
core is a single-threaded microphysics tool, not a global model. The 3D
`meteorological_flow` solver (20³, 60 s, one-way) runs in ~35 s after a one-time
lookup build (~6 min with 8 threads, cached and reused); see
`docs/flow_guide.md`.

---

## Documentation

- `docs/MANUAL_met_h2o_nucleation.md` — full reference manual (also `.html`).
- `docs/MET_NUCLEATION_HYPOTHESES.md` — hypotheses, validity ranges, validation report.
- `docs/architecture.md` — the data-flow pipeline.
- `docs/flow_guide.md` — the 3D Boussinesq flow solver (formulation, limitations, running).
- `docs/migration-guide.md` — migrating from the pre-reorganization flat layout.
- `MIGRATION_MANIFEST.md` — old path → new path for every moved file.

---

## Citation & license

If you use this tool in academic work, cite the shifted-equilibrium framework:

> Ferreira, *Physica B: Condensed Matter* **695** (2024) 416494; and the MRS
> Meeting 2026 contribution. (See `CITATION.cff` — complete the author/title/doi.)

**License:** MIT — see `LICENSE`. The bundled core and its reference models are
integrity-guarded and read-only; the MIT licence permits modification, but
editing those guarded files invalidates the `--validate` integrity check (see
the note in `LICENSE`).