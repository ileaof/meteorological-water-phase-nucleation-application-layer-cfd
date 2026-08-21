# Precipitation microphysics & evidence-based diagnostics

`precip_microphysics` extends the meteorological water-phase nucleation
application layer so that precipitation diagnostics rest on the **formation,
growth, transport and phase evolution of hydrometeors** — not on the nucleation
rate alone. It resolves the standing limitation:

> A high nucleation rate never by itself implies rain or hail.

It does so by *implementing the missing physics and the evidence needed to
support meaningful diagnostics* — never by relaxing the confidence thresholds.

The validated second-order nucleation core (`met_water_nucleation._engine`,
SHA-256 guarded) is imported **read-only** and never modified; this package is a
separate, framework-agnostic layer consumed by both a standalone parcel/column
driver and (Increment 2) the 3D flow solver.

---

## 1. The five-stage principle

The code keeps these stages strictly distinct; only stages 4–5 are
"precipitation", and a high nucleation rate is only ever evidence for stage 1.

| Stage | Meaning | Diagnostic level |
|---|---|---|
| 1 · Nucleation | initial droplet/ice-embryo formation | Level 1 |
| 2 · Cloud formation | sustained liquid/ice content in a cell | Level 2 (cloud) |
| 3 · Hydrometeor growth | cloud → precipitating particles | Level 2 |
| 4 · Precipitation production | rain/snow/graupel/hail with mass & fall speed | Level 3 |
| 5 · Surface precipitation | particles survive transport/melting to the ground | Level 4 |

## 2. Diagnostic levels (0–4)

| Level | Name | Criterion |
|---|---|---|
| 0 | `insufficient_information` | no reliable assessment (e.g. state outside validity, nothing supersaturated) |
| 1 | `thermodynamic_favourability` | supersaturation / nucleation favourable, growth unresolved |
| 2 | `hydrometeor_production` | category mass being generated aloft (`production_rate > 0` or `q > 1e-9`) |
| 3 | `precipitation_development` | `q > 1e-5 kg/kg` **and** `V_t > V_t,min` (particles can precipitate) |
| 4 | `surface_precipitation` | positive category surface flux (`> 1e-8 kg m⁻² s⁻¹`) |

A category is **`confirmed`** only at **Level 4**, with the **complete growth
chain present** (`process_evidence = 1`), confidence **at/above its threshold**,
and no hard blocking reason. Hail is reported with high confidence only at
Level 4; hail aloft is labelled *hail development aloft* (Level 2–3).

## 3. Confidence model

An auditable product of four components, each in [0, 1]:

```
confidence = data_completeness × model_validity × process_evidence × numerical_quality
```

| Component | What it measures |
|---|---|
| `data_completeness` | fraction of the dynamic/microphysical variables the phenomenon needs that were supplied |
| `model_validity` | 1 inside the correlation envelope (`T ∈ [233, 320] K`), reduced when extrapolated |
| `process_evidence` | fraction of the required growth processes that were enabled **and contributed mass** |
| `numerical_quality` | 1 when water is conserved and no numerical failure occurred, sharply reduced otherwise |

`process_evidence` is what makes nucleation-only evidence insufficient: with no
growth processes contributing, it is 0 and confidence is 0 regardless of how
high the nucleation rate is.

## 4. When precipitation CAN and CANNOT be confirmed

**CAN be confirmed** (`confirmed = true`, no caveat) when *all* hold:

1. the category reached **Level 4** (positive surface flux);
2. **every** required growth/transport process contributed mass
   (`process_evidence = 1`);
3. `confidence ≥ threshold` (0.50 rain/snow/graupel, 0.75 hail);
4. the state is inside the validity envelope and water is conserved.

**CANNOT be confirmed** (caveat attached) when *any* hold — these are the caveat
triggers:

- confidence `< 0.50` (rain/snow/graupel) or `< 0.75` (hail);
- only nucleation thermodynamics is available (`microphysics_enabled = false`);
- a required growth process was disabled or never contributed;
- required dynamic fields are missing (e.g. no updraft for hail);
- total-water or numerical validity checks fail;
- the state lies outside a correlation's validity range;
- the category never reached the surface (production aloft only).

The exact caveat string is preserved verbatim:

> *"Thermodynamically favourable to nucleation, but the dynamic and
> microphysical data are insufficient to confirm precipitation or hail."*

## 5. Reason codes

Machine-readable, returned in `reason_codes`, alongside a human `caveat` and
`missing_variables`:

`THERMODYNAMICS_ONLY`, `MISSING_VERTICAL_VELOCITY`,
`MISSING_SUPERCOOLED_LIQUID_WATER`, `NO_COLLISION_COALESCENCE`,
`NO_DEPOSITION_GROWTH`, `NO_AGGREGATION`, `NO_RIMING_MODEL`,
`NO_SEDIMENTATION`, `NO_SURFACE_FLUX`, `INSUFFICIENT_RESIDENCE_TIME`,
`HAIL_SURVIVAL_NOT_EVALUATED`, `OUTSIDE_MODEL_VALIDITY`,
`NUMERICAL_CONSERVATION_FAILURE`.

## 6. Category → required-evidence table

| Category | Required evidence (variables) | Required processes | Reason code if missing |
|---|---|---|---|
| **Rain** | cloud liquid water; updraft; layer depth; timestep | condensation growth; collision-coalescence (autoconv/accretion); sedimentation | `MISSING_SUPERCOOLED_LIQUID_WATER`, `MISSING_VERTICAL_VELOCITY`, `NO_COLLISION_COALESCENCE`, `NO_SEDIMENTATION`, `NO_SURFACE_FLUX` |
| **Snow** | ice water; sub-freezing layer; layer depth; timestep | ice nucleation; deposition growth; aggregation; sedimentation | `NO_DEPOSITION_GROWTH`, `NO_AGGREGATION`, `NO_SEDIMENTATION`, `NO_SURFACE_FLUX` |
| **Graupel** | ice/snow embryo; supercooled liquid water; updraft; layer depth | ice nucleation; riming; graupel conversion; sedimentation | `MISSING_SUPERCOOLED_LIQUID_WATER`, `NO_RIMING_MODEL`, `NO_SEDIMENTATION` |
| **Hail** | graupel embryo; substantial supercooled LWC; strong & deep updraft; residence time; freezing level; layer depth | hail embryo; wet/dry growth (riming); sedimentation; melting/survival | `MISSING_SUPERCOOLED_LIQUID_WATER`, `MISSING_VERTICAL_VELOCITY`, `INSUFFICIENT_RESIDENCE_TIME`, `NO_RIMING_MODEL`, `HAIL_SURVIVAL_NOT_EVALUATED`, `NO_SURFACE_FLUX` |
| **Any** | validity range; conservation; numerics | — | `THERMODYNAMICS_ONLY`, `OUTSIDE_MODEL_VALIDITY`, `NUMERICAL_CONSERVATION_FAILURE` |

## 7. Microphysical processes and their sources

Single-moment bulk scheme (prognostic mass mixing ratios `q_c,q_r,q_i,q_s,q_g,q_h`;
diagnostic number concentrations from Marshall-Palmer closure). Empirical
coefficients are flagged `EMPIRICAL` in [constants.py](../src/precip_microphysics/constants.py).

| Process | Parameterization / source |
|---|---|
| Condensation / evaporation | saturation adjustment with latent feedback (Soong & Ogura 1973) |
| Autoconversion, accretion | Kessler (1969) |
| Rain evaporation | ventilated diffusion (Rutledge & Hobbs 1983) |
| Ice nucleation (embryo source) | second-order kernel rate `J`, `N = J·dV·dt`, capped by CCN/IN, vapour-limited; Fletcher (1962) IN / Bigg (1953) immersion |
| Deposition / sublimation | diffusional growth over ice, `A_K + A_D` denominator (Byers 1965) |
| Aggregation (`q_i → q_s`) | autoconversion threshold (Lin et al. 1983) |
| Riming (cloud → snow/graupel/hail) | continuous bulk collection of cloud water (Lin83; Rutledge & Hobbs 1983) |
| Graupel conversion | rimed-snow threshold (Lin83) |
| Melting (`q_s,q_g,q_h → q_r`) | ventilated bulk melting (Rutledge & Hobbs 1984) |
| Hail embryo + wet/dry growth | graupel gate (supercooled LWC + updraft) → wet-growth collection |
| Sedimentation | mass-weighted terminal velocity, exponential distribution, density correction (Foote & du Toit 1969); upwind column flux |

**Latent heat** is derived from the phase-rank change of each mass transfer:
condensation/deposition/freezing warm (+`L_v`/`L_s`/`L_f` over `c_p`), and
evaporation/sublimation/melting cool.

**No double counting**: `activation_pathway` selects exactly one embryo source
(`eq39` kernel-rate, `ccn` Twomey activation, or `homogeneous`); the kernel gives
the *rate/favourability* while CCN/IN set the *number*, and the mass is limited by
available supersaturated vapour.

## 8. Output schema (per category)

```json
{
  "category": "rain|snow|graupel|hail",
  "diagnostic_level": 0, "diagnostic_level_name": "...",
  "thermodynamic_favourability": 0.0,
  "production_rate_kg_m3_s": 0.0, "mixing_ratio_kg_kg": 0.0,
  "number_concentration_m3": 0.0, "characteristic_radius_m": 0.0,
  "terminal_velocity_m_s": 0.0, "surface_flux_kg_m2_s": 0.0, "accumulation_mm": 0.0,
  "confidence": 0.0, "threshold": 0.5, "confirmed": false,
  "caveat_required": true, "caveat": "...", "reason_codes": [],
  "supporting_evidence": {}, "missing_variables": [],
  "model_validity": {}, "numerical_quality": {}, "confidence_components": {}
}
```

Hail additionally returns `embryo_source`, `supercooled_liquid_water_kg_m3`,
`max_updraft_m_s`, `growth_region_depth_m`, `residence_time_s`, `growth_regime`
(wet/dry), `max_diameter_m`, `melting_fraction`, `surface_survival_probability`.

## 9. Reference scenarios

Run `python examples/microphysics_scenarios.py [--kernel]`:

| Scenario | log10 I | Max level | Confirmed | Note |
|---|---|---|---|---|
| 1 · high nucleation, no microphysics | ~55 | **1** | **none** | caveat holds despite enormous rate |
| 2 · warm rain | ~55 | 4 | rain | condensation → coalescence → surface rain |
| 3 · mixed phase | ~55 | 4 | snow | ice deposition → aggregation → surface snow |
| 4 · deep convective hail | ~55 | 4 | hail | supercell core → hail growth → descent → melt/survival |

Scenario 1 vs 2–4 is the proof that nucleation and precipitation are distinct:
the **same** favourable nucleation rate confirms nothing without the growth
evidence.

## 10. Limitations & recommendations for future validation

This is a physically structured, conservation-respecting **demonstration-scale**
scheme, not an observationally validated operational model.

- **Single-moment**: number concentrations are diagnostic (fixed `N0`); radii and
  fall speeds inherit that assumption. A double-moment upgrade (prognostic `N`)
  would improve size/fall-speed fidelity and CCN/INP sensitivity.
- **0-D/1-D parcel**: a parcel cannot self-sustain supercooled liquid water
  against the Bergeron process, so the hail scenario imposes the supercell-core
  condition; full hail growth belongs in the 3D flow coupling (Increment 2).
- **Empirical coefficients** (autoconversion thresholds, `N0`, fall-speed `a,b`,
  Bigg/Fletcher constants) are standard literature values, **not tuned to
  observations** here.
- **Idealised hail wet/dry growth**: the Schumann-Ludlam limit is applied as a
  regime gate, not a full surface heat-balance integration.
- **Recommended validation**: compare against a reference bulk scheme (WSM6 /
  Thompson / Morrison) in 1-D and 3-D; check surface precipitation and hydrometeor
  profiles against observations (radar, disdrometer, hail pads); verify
  conservation and reproducibility at scale; and add convergence tests for the
  sedimentation and saturation-adjustment operators.

## 11. Scientific integrity

- The validated nucleation core and its reference models (`_engine/**`) are
  **never modified** — imported read-only via `import met_water_nucleation as M`;
  `--validate` (tests [1]–[21] + SHA-256) stays green.
- Every mass transfer conserves total water; sedimentation surface flux is the
  only sink, and it is booked. Latent heating is derived from the transfers, so
  its sign is always physically consistent.
- Parameterizations and extrapolations are labelled; reason codes and confidence
  components make every diagnosis auditable.
- 1st-order / CNT / 2nd-order nucleation results remain distinct and untouched.

## 12. Validation status

- **86 tests pass** (60 original nucleation/flow + 20 microphysics + 6 scenario/config).
- The 20 microphysics tests cover: nucleation-only → Level ≤ 1; each category's
  required evidence; exact 0.50/0.75 thresholds and caveat text; reason codes;
  water conservation; latent-heat signs; non-negativity; sedimentation mass
  balance; melting/evaporation; disable-microphysics → thermodynamic-only;
  numerical/validity confidence reduction; and reproducibility.
