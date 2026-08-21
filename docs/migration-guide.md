# Migration guide

The repository was reorganized from a flat layout into an installable
`src/`-layout package. Scientific results are unchanged (the engine moved
byte-identically; SHA-256 checksums preserved).

## What changed

- The application layer is now the installable package `met_water_nucleation`.
- The validated engine (core + 2 reference scripts + `met` + `hca`) moved as a
  unit to `src/met_water_nucleation/_engine/` (read-only, never edited).
- Examples → `examples/`, tests → `tests/`, docs → `docs/`,
  `out_met_nucleation/` → `outputs/`.

## Old → new commands

| Before | After |
|---|---|
| `python met_h2o_nucleation.py --validate` | `met-water-nucleation --validate` (or `python -m met_water_nucleation --validate`) |
| `python met_h2o_nucleation.py --T 260 ...` | `met-water-nucleation --T 260 ...` |
| `python test_met_nucleation.py` | `python -m pytest tests/` (or `python tests/test_met_nucleation.py`) |
| `python example_met_single_state.py` | `python examples/single_state.py` |

`python met_h2o_nucleation.py …` **still works** — a backward-compatibility
shim remains at the repo root and emits a `DeprecationWarning` pointing here.

## Old → new imports

| Before | After |
|---|---|
| `import met_h2o_nucleation as M` | `import met_water_nucleation as M` |
| `import het_contact_angle as hca` | `import met_water_nucleation as M; hca = M._engine.het_contact_angle` (advanced; prefer the re-exported API) |

The public API names (`MetInput`, `MetNucleationRunner`, `resolve_humidity`,
`to_json`, `to_csv`, `to_netcdf`, `MetNucleationPlotter`, `SaturationProperties`,
`LiquidNucleationModel`, `IceNucleationModel`, `AtmosphericInput`, `ftheta`,
`PHASE_LIQUID`, `PHASE_ICE`, `R_REF_DEFAULT`, `MANDATORY_FIELDS`, `un`, …) are
unchanged — only the import root changes.

## One-time setup

```bash
python -m pip install -e .
```

This installs the `met-water-nucleation` console script and makes
`import met_water_nucleation` work from any directory (no CWD or PYTHONPATH
hacks). Tests (`tests/conftest.py`) also bootstrap the `src/` path, so they
run without installation too.

## Output location

New run outputs should follow `outputs/<scenario>/<run-id>/`. The examples now
write to `outputs/<scenario>/` (e.g. `outputs/single_state/`). The original
committed flat reference outputs (`outputs/*.{json,csv,nc,png}`) are kept
tracked as historical regression references.