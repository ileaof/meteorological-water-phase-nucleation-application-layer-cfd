# Handoff — precipitation microphysics extension

**Date:** 2026-08-21 · **Branch:** `main` · **Status:** Increment 1 complete & tested.

## Objective

Extend the meteorological water-phase nucleation application layer so precipitation
diagnostics rest on hydrometeor **formation → growth → transport → phase evolution**,
not on the nucleation rate alone — resolving the limitation *"a high nucleation rate
never by itself implies rain or hail"* by implementing the physics and the evidence,
**not** by lowering confidence thresholds. Validated 2nd-order nucleation equations
must be preserved.

## Decisions locked (user-approved)

1. **Architecture:** new framework-agnostic package `src/precip_microphysics/`, outside
   the immutable `_engine/`; consumes the nucleation kernel read-only. `_engine/` untouched.
2. **Scope:** Increment 1 = 0-D/1-D core + diagnostics + standalone driver + tests +
   scenarios. Increment 2 = wire into the 3D `meteorological_flow` solver. **← remaining**
3. **Fidelity:** single-moment bulk, all categories (`q_c,q_r,q_i,q_s,q_g,q_h`), diagnostic
   number concentrations via Marshall-Palmer closure.
4. **API compat:** old `Favorability`/`PrecipitationDiagnosis`/49-field report kept intact;
   new evidence-based diagnostics added in parallel.

## What was delivered (Increment 1)

Full single-moment scheme (Kessler warm rain + Lin/Rutledge-Hobbs ice/snow/graupel +
hail wet/dry growth), sedimentation, nucleation→embryo coupling, an **evidence-based
confidence model** (`data_completeness × model_validity × process_evidence ×
numerical_quality`), **diagnostic levels 0–4**, **reason codes**, the caveat logic, a
standalone parcel/column driver, 4 reference scenarios, 26 new tests, config + YAML, docs.

**Success criterion met:** a category is `confirmed` only at Level 4 with the *complete*
growth chain (`process_evidence = 1`), confidence ≥ threshold, and no hard block. A high
nucleation rate alone reaches **Level 1** and confirms nothing (scenario 1: log10 I ≈ 55,
confirmed = none).

## File map (new)

```
src/precip_microphysics/
  constants.py          physical constants + category params (EMPIRICAL flagged, cited)
  state.py              MicrophysicsState (7 mixing ratios, 0-D or column, conserving)
  config.py             MicrophysicsConfig + ProcessSwitches + from_dict/from_yaml
  thermo.py             engine SaturationProperties (read-only), latent heat, ventilation
  size_distributions.py MP slope, number, radius, mass-weighted terminal velocity
  processes.py          all process rates as mass Transfers (Kessler/Lin/RH; PROCESS_ORDER)
  nucleation_source.py  J -> embryo mass (N=J·dV·dt, CCN/IN-capped, vapour-limited)
  scheme.py             BulkMicrophysics.step: applies transfers, conserves water + latent heat
  sedimentation.py      0-D box + 1-D column upwind fall -> surface flux + accumulation
  evidence.py           levels 0-4, 4-part confidence, reason codes, caveat (exact string)
  diagnostics.py        build evidence context -> per-category output schema + hail extras
  column.py             ColumnModel: standalone parcel/column driver (adiabatic ascent)
  scenarios.py          4 reference scenarios
tests/test_microphysics.py            20 tests
tests/test_microphysics_scenarios.py  6 tests (scenarios + config loader)
examples/microphysics_scenarios.py    runnable 4-scenario comparison
examples/heavy_rain_hail_scenario.py  ~100 mm rain+hail severe storm (two cores)
tests/test_heavy_scenario.py          regression: ~100 mm rain + confirmed hail
configs/microphysics_reference.yaml   reference config
docs/microphysics_guide.md            when precip CAN/CANNOT be confirmed; evidence table; limits
pyproject.toml                        packages.find include += "precip_microphysics*"
```

## Test status

`PYTHONPATH=src python -m pytest tests/ -q` → **87 passed** (60 original + 20 microphysics
+ 6 scenario/config + 1 heavy-storm). Original 60 unchanged; `_engine/` untouched so
`--validate` (tests [1]–[21] + SHA-256) stays green. Warnings are pre-existing
`PytestReturnNotNone`.

Severe-storm example (`examples/heavy_rain_hail_scenario.py`): two coupled cores —
a warm moisture-convergence-fed rain core (steady-state isothermal) + a supercell
hail core (sustained supercooled reservoir, short growth, then descent/melt) —
produce ~100 mm rain (warm + melted graupel/hail) plus ~2 mm confirmed surface hail
(realistic ~1:40 hail:rain). Moisture supply is bounded by the vertical flux
S_max ≈ w·q_v/H (~3e-5 kg/kg/s for a mean updraft). Both rain & hail Level 4 confirmed;
water conserved to ~1e-16.

## How to run

```bash
PYTHONPATH=src python examples/microphysics_scenarios.py --kernel   # 4-stage comparison
PYTHONPATH=src python -m pytest tests/test_microphysics.py -q       # 20 unit tests (fast, ~3s)
```

Scenarios (kernel-free is fast; `--kernel` adds one kernel call each):
1 nucleation-only → Level 1, no confirmation · 2 warm rain → Level 4 rain (~5 mm) ·
3 mixed-phase → Level 4 snow · 4 deep hail → Level 4 hail (wet growth, ~35% melt, ~65% survival).

## Remaining work (Increment 2)

- Wire `BulkMicrophysics` into `meteorological_flow.simulation` at `stage="hydrometeor"`:
  add `q_r,q_s,q_g,q_h` to `FlowState`, advect them, apply the scheme per cell, feed the
  nucleation adapter's `I` into `nucleation_source`, apply latent-heat feedback to `theta`,
  and column sedimentation. Config already reserves the stage names.
- 3D natural hail (a parcel can't sustain supercooled LWC vs Bergeron; the 3D updraft can).
- Optional double-moment upgrade; observational validation vs WSM6/Thompson/Morrison.

## Key implementation notes / gotchas

- Water conservation is structural: processes return **mass transfers** (src→dst), the
  scheme re-caps at apply time (safe operator split) and derives latent-heat sign from the
  phase-rank change (vapour<liquid<ice). Sedimentation is the only water sink (booked).
- Nucleation is **decoupled from confirmation**: kernel `J` sets favourability/rate; CCN/IN
  set droplet number; vapour limits mass. `activation_pathway` (eq39/ccn/homogeneous) picks
  ONE source → no double counting.
- Confidence gate: `confirmed` requires `process_evidence == 1` (every required process
  contributed) — this is why removing any one growth process blocks confirmation even if the
  numeric confidence would otherwise clear the threshold.
- Parcel driver uses dry-adiabatic ascent cooling; condensation latent heat then yields the
  moist adiabat automatically. Hail scenario imposes a supercell-core SLW supply (documented
  idealisation).
- Exact caveat string and 0.50/0.75 thresholds are preserved (evidence.py `CAVEAT`,
  config `threshold_*`).
