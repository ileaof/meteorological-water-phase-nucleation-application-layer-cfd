# Migration Manifest — old path → new path

Every file moved during the reorganization is listed below (Git rename
preserved; SHA-256 of engine files unchanged). No file was deleted.

Legend: **engine** = immutable validated bundle (loaded read-only);
**shim** = new backward-compatibility wrapper; **new** = newly created.

## Engine bundle (moved as a unit into `src/met_water_nucleation/_engine/`)

| Old path | New path | Kind | SHA-256 unchanged? |
|---|---|---|---|
| `met_h2o_nucleation.py` | `src/met_water_nucleation/_engine/met_h2o_nucleation.py` | engine | ✅ `d717c5ff…` |
| `het_contact_angle.py` | `src/met_water_nucleation/_engine/het_contact_angle.py` | engine | ✅ `91b74e60…` |
| `unified_h2o_nucleation_climate/unified_h2o_nucleation_climate.py` | `src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/unified_h2o_nucleation_climate.py` | engine (core) | ✅ `5d3aec3f…` |
| `Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py` | `src/met_water_nucleation/_engine/Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py` | engine (ref) | ✅ `c9fa9c01…` |
| `Nucleation_model_H2O_vapour_liquid_Sim_2026.py` | `src/met_water_nucleation/_engine/Nucleation_model_H2O_vapour_liquid_Sim_2026.py` | engine (ref) | ✅ `06d55a8c…` |

## Tests, examples, docs, outputs

| Old path | New path | Kind | Note |
|---|---|---|---|
| `test_met_nucleation.py` | `tests/test_met_nucleation.py` | verification | import → package; round-trip path → temp dir |
| `example_met_single_state.py` | `examples/single_state.py` | example | import → package; OUTDIR → `outputs/single_state/` |
| `example_met_vertical_profile.py` | `examples/vertical_profile.py` | example | as above |
| `example_met_xarray_netcdf.py` | `examples/xarray_netcdf.py` | example | as above |
| `example_met_figures.py` | `examples/figures.py` | example | as above |
| `example_met_frontal_collision.py` | `examples/frontal_collision.py` | example | as above |
| `README.md` | `docs/index.md` | doc | (original moved; new root `README.md` created) |
| `MANUAL_met_h2o_nucleation.md` | `docs/MANUAL_met_h2o_nucleation.md` | doc | unchanged + reorg note |
| `MANUAL_met_h2o_nucleation.html` | `docs/MANUAL_met_h2o_nucleation.html` | doc | unchanged |
| `MET_NUCLEATION_HYPOTHESES.md` | `docs/MET_NUCLEATION_HYPOTHESES.md` | doc | unchanged + reorg note |
| `out_met_nucleation/*` (19 files) | `outputs/*` | generated ref | kept tracked as historical reference; new runs → `outputs/<scenario>/` |

## Unchanged / kept in place

| Path | Note |
|---|---|
| `requirements.txt` | kept for backward compatibility (superseded by `pyproject.toml`) |
| `.gitignore` | updated in place (outputs subdirs, orphaned core dir) |

## Newly created

| Path | Purpose |
|---|---|
| `pyproject.toml` | package metadata, deps, optional extras, CLI entry point |
| `met_h2o_nucleation.py` (root) | **shim** — `python met_h2o_nucleation.py …` still works (deprecation warning) |
| `src/met_water_nucleation/__init__.py` | package facade; re-exports engine API |
| `src/met_water_nucleation/cli.py`, `__main__.py` | console entry + `python -m …` |
| `src/met_water_nucleation/_engine/__init__.py`, `…/unified_h2o_nucleation_climate/__init__.py` | subpackage markers (engine ships with installs) |
| `tests/conftest.py` | CWD-independent `src/` path bootstrap |
| `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff` | repo meta |
| `MIGRATION_MANIFEST.md`, `docs/architecture.md`, `docs/migration-guide.md` | this + architecture + migration |
| `configs/*.yaml` | declarative scenario configs |
| `scripts/run_validation.py`, `scripts/regenerate_outputs.py` | maintenance utilities |
| `outputs/README.md`, `outputs/.gitkeep` | output naming convention |
| `legacy/README.md`, `references/README.md`, `data/README.md` | section READMEs |

## Core's own ecosystem (relocated — decision 2026-08-20)

| Old path | New path | Kind | Note |
|---|---|---|---|
| `unified_h2o_nucleation_climate/README.md` | `src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/README.md` | untracked collateral | core's own docs; restored alongside the core; still gitignored |
| `unified_h2o_nucleation_climate/MANUAL_unified_h2o_nucleation_climate.html` | `…/unified_h2o_nucleation_climate/MANUAL_unified_h2o_nucleation_climate.html` | untracked collateral | as above |
| `unified_h2o_nucleation_climate/out_*`, `unified_nucleation_out/` | `…/unified_h2o_nucleation_climate/out_*`, `…/unified_nucleation_out/` | untracked collateral | core's own historical outputs; restored alongside the core; still gitignored |
| `unified_h2o_nucleation_climate/__pycache__/` | (discarded) | regenerable cache | not moved; regenerated on next run |

These were relocated (option 2 from `legacy/README.md`) to restore the core's
self-contained bundle — core `.py` + its own docs + its own historical outputs
in one folder — matching the original layout. They were deliberately kept out
of version control by the original author and remain untracked/gitignored.
The empty old shell at repo root was removed.

## Dangling references (not present in repo)

`het_contact_angle.py` and `met_h2o_nucleation.py` mention
`theta_liquid_nucleation_assessment.md` and `test_gt_2nd_het_parabola.py` in
comments — these files do not exist in the repository (historical, from a
prior environment). They are not created or fabricated.