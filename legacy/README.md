# legacy/

Review area for ambiguous / historical files. Nothing here is imported by
production code.

## Resolved decisions (2026-08-20)

### `unified_h2o_nucleation_climate/` (orphaned, formerly at repo root)

When the validated core was moved (as a unit) into
`src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/`, this *old*
directory was left in place holding only the core's **own** collateral — not
the core itself:

- `README.md`, `MANUAL_unified_h2o_nucleation_climate.html` — the core's own
  documentation.
- `out_alt/`, `out_climate/`, `out_gradsweep/`, `out_het/`, `out_ice_only/`,
  `out_phasemap/`, `out_psatgrad/`, `out_Psweep/`, `out_RHsweep/`,
  `out_single_both/`, `out_subsat/`, `out_ts/`, `out_Tsweep/`,
  `unified_nucleation_out/` — the core's own historical generated outputs.
- `__pycache__/` — regenerable (discarded on the move).

**Decision: option 2** — relocate the whole ecosystem into the core's new folder
`src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/`, so the
core's self-contained bundle (core `.py` + its own docs + its own historical
outputs) is restored as it was in the original layout. Everything remains
**untracked/gitignored** (the original author deliberately kept this collateral
out of version control; the `.gitignore` now excludes the core's `README.md`,
`*.html`, `unified_nucleation_out` and `out_*/` in the new location). The empty
old shell at the repo root was removed.

Nothing in this collateral is required for tests or the CLI; it is kept purely
for provenance alongside the core.

## Dangling references (not present in the repo)

The engine source comments reference two files that do **not** exist in this
repository (historical, from a prior environment):

- `theta_liquid_nucleation_assessment.md`
- `test_gt_2nd_het_parabola.py`

They are documented here for provenance; they were **not** fabricated.