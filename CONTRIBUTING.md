# Contributing

## The immutable engine

The physics engine under `src/met_water_nucleation/_engine/` is **validated and
integrity-guarded**:

- `unified_h2o_nucleation_climate/unified_h2o_nucleation_climate.py` — the core.
- `Nucleation_model_H2O_vapour_{solid,liquid}_Sim_2026*.py` — SHA-256-guarded.
- `met_h2o_nucleation.py`, `het_contact_angle.py` — the application layer.

**Do not edit these files to simplify organization.** The `--validate` command
checks the two reference models byte-for-byte. If a physics change is needed,
open an issue first, change the smallest possible unit, update the checksum
constant in the core, and re-run the full `tests/` suite.

Production code accesses the engine only through the `met_water_nucleation`
package facade (`src/met_water_nucleation/__init__.py`), which loads the engine
read-only via `importlib`.

## Layout

| Where | What |
|---|---|
| `src/met_water_nucleation/_engine/` | immutable engine — do not refactor |
| `src/met_water_nucleation/` (non-`_engine`) | package facade / CLI (edits OK) |
| `tests/` | automated verification |
| `examples/` | user-facing demonstrations |
| `configs/` | declarative scenario YAMLs |
| `scripts/` | maintenance utilities |
| `docs/` | manual, hypotheses, architecture |
| `outputs/` | generated outputs (gitignored subdirs) |

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/
ruff check .
met-water-nucleation --validate
```

Keep heavy dependencies optional (xarray/netCDF4/cfgrib/pandas) — the package
must import with only numpy/scipy/matplotlib.