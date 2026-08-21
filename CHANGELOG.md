# Changelog

All notable changes to this repository's organization are documented here.
Scientific behaviour is unchanged across the 1.0.0 reorganization (the engine
moved byte-identically; SHA-256 checksums preserved).

## [1.0.0] — 2026-08-20 — package reorganization

### Added
- Installable `src/`-layout package `met_water_nucleation` with a facade that
  re-exports the validated engine API.
- `pyproject.toml` with metadata, required deps (numpy/scipy/matplotlib),
  optional extras (`netcdf`, `grib`, `pandas`, `io`, `dev`) and the
  `met-water-nucleation` console entry point.
- `python -m met_water_nucleation` module entry point.
- Root `met_h2o_nucleation.py` backward-compatibility shim (DeprecationWarning).
- `tests/conftest.py` (CWD-independent import bootstrap).
- `docs/architecture.md`, `docs/migration-guide.md`, `MIGRATION_MANIFEST.md`.
- `configs/` (declarative scenario YAMLs), `scripts/` (run_validation,
  regenerate_outputs), `legacy/`, `references/`, `data/` section READMEs.
- `outputs/README.md` naming convention (`outputs/<scenario>/<run-id>/`).

### Changed
- Engine bundle (core + 2 reference scripts + `met` + `hca`) relocated as a
  unit to `src/met_water_nucleation/_engine/` (byte-identical `git mv`).
- Core's own ecosystem (its `README.md`, `MANUAL_*.html` and historical `out_*`
  outputs) relocated into the core's new folder alongside the core `.py`,
  restoring the original self-contained bundle (still untracked/gitignored).
- Examples → `examples/` (imports → package; write to `outputs/<scenario>/`).
- Tests → `tests/` (import → package; round-trip artifact → system temp dir).
- Docs → `docs/` (`README.md` → `docs/index.md`; new root `README.md`).
- `out_met_nucleation/` → `outputs/`.
- `.gitignore` updated (per-scenario output subdirs, orphaned core dir).

### Preserved
- All five engine files retain their pre-reorg SHA-256 checksums; `--validate`
  still passes (core [1]–[21], ice SHA-256 unchanged).
- 24/24 tests pass before and after.
- Committed flat reference outputs kept tracked and unchanged on disk.

### Resolved decisions (2026-08-20)
- `LICENSE`: **MIT** added at the repo root (`pyproject.toml` license → MIT +
  SPDX classifier). The integrity-guarded core/reference models remain
  read-only; the MIT licence permits modification but editing guarded files
  invalidates `--validate` (noted in `LICENSE`).
- Orphaned `unified_h2o_nucleation_climate/` directory (the core's own untracked
  docs + historical `out_*` outputs): **relocated into the core's new folder**
  `src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/` (option 2
  from `legacy/README.md`), restoring the core's self-contained ecosystem
  alongside the core `.py`. Still untracked/gitignored as before; the empty old
  shell at repo root was removed.