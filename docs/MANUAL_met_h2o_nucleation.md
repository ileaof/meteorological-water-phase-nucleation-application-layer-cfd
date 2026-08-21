# Manual — `met_h2o_nucleation.py`

> **Reorganization note (2026-08-20).** This manual documents the application
> layer's API, which is unchanged. The module has been relocated (byte-identical)
> into the installable package `met_water_nucleation`; import it as
> `import met_water_nucleation as M`, or run `met-water-nucleation …` /
> `python -m met_water_nucleation …`. The legacy `python met_h2o_nucleation.py …`
> command still works via a root shim. The validated core now lives at
> `src/met_water_nucleation/_engine/unified_h2o_nucleation_climate/` (read-only,
> SHA-256 guarded). See `docs/migration-guide.md` and `MIGRATION_MANIFEST.md`.

**Meteorological water-phase nucleation application layer** for the Ferreira
Eq.39a/39b shifted-equilibrium framework
(Physica B **695** (2024) 416494; MRS Meeting 2026).

This is the reference manual for `met_h2o_nucleation.py` (the application /
diagnosis layer). The physics core it wraps —
`unified_h2o_nucleation_climate/unified_h2o_nucleation_climate.py` — is
**bundled in this repository** and **imported read-only, never modified** (its
integrity is SHA-256 guarded). For hypotheses, validity ranges and the
validation report see
`MET_NUCLEATION_HYPOTHESES.md`; for the test suite see
`test_met_nucleation.py`.

---

## Contents

1. [What this module is](#1-what-this-module-is) — incl. [architecture](#11-architecture) & [the physics in brief](#12-the-physics-in-brief)
2. [Dependencies](#2-dependencies)
3. [Installation & quick start](#3-installation--quick-start)
4. [Constants & re-exported core symbols](#4-constants--re-exported-core-symbols)
5. [Input — `MetInput`](#5-input--metinput)
6. [Humidity helpers](#6-humidity-helpers)
7. [Free-energy decomposition](#7-free-energy-decomposition)
8. [Precipitation diagnosis](#8-precipitation-diagnosis)
9. [Output — `MetNucleationReport`](#9-output--metnucleationreport)
10. [Runner — `MetNucleationRunner`](#10-runner--metnucleationrunner)
11. [I/O adapters](#11-io-adapters)
12. [Visualisation — `MetNucleationPlotter`](#12-visualisation--metnucleationplotter)
13. [Self-checks & validation](#13-self-checks--validation)
14. [Command-line reference](#14-command-line-reference) — incl. the 10 verified [CLI cases](#141-cli-cases-with-verified-output)
15. [Examples](#15-examples)
16. [Conventions](#16-conventions)
17. [Validity ranges & what remains hypothesis](#17-validity-ranges--what-remains-hypothesis)
18. [Troubleshooting](#18-troubleshooting)
19. [Citation & license](#19-citation--license)
20. [File map](#20-file-map)

---

## 1. What this module is

`met_h2o_nucleation.py` computes, for one or more atmospheric states:

* **vapour → liquid** (condensation) and **vapour → ice** (deposition)
  nucleation, homogeneous or heterogeneous;
* the **non-equilibrium thermal closure** (radius as continuation variable,
  gradient as the Brent-solved unknown);
* the **2nd-order Gibbs–Thomson coefficient** and **1st/2nd-order critical
  radii** (Ferreira Eq.39b parabola, heterogeneous with ∂f/∂r);
* the **free-energy decomposition** ΔG_V / ΔG_bulk / ΔG_surface / ΔG_config /
  ΔG_total at the evaluated radius;
* the **shifted equilibrium pressure** P_eq,shift = P_sat,phase(T_local);
* the **nucleation rate** I and log10 I (overflow-safe) and the expected event
  count in a cell/timestep;
* transparent **rain / snow / graupel / hail favourability indices** (0..1)
  with contributing / missing variables, confidence and a caveat — *a high
  rate never by itself implies precipitation*;
* ingestion of scalars, profiles, time series, **xarray / NetCDF / GRIB**
  fields and structured **JSON / CSV / NetCDF** output;
* optional **PNG figures**.

All internal quantities are **SI**. When a quantity cannot be determined from
the inputs it is reported as `"undetermined"` (the constant `NA`) with the
missing information named.

### 1.1 Architecture

```
                 +-------------------------------+
   met input --> |  met_h2o_nucleation.py         |  <-- this layer (application/diagnosis)
                 |  MetInput / Runner / Diagnosis |
                 |  free-energy decomp / IO / viz |
                 +---------------+---------------+
                                 |  imports READ-ONLY (importlib)
                                 v
                 +-------------------------------+
                 |  unified_h2o_nucleation_      |  <-- validated core (DO NOT MODIFY)
                 |  climate.py                   |      closure, r_C, Gamma, rate, tests [1]-[21]
                 +-------------------------------+
```

The core closure (F(g;r)=Γ²/(4πr²)−g=0), the critical-radius parabola, the
surface-stress law, the nucleation rate and the validation suite (incl. the
ice-reference SHA-256 guard) are **delegated** to the core. This layer adds only
what the core deliberately does not own: free-energy decomposition,
precipitation diagnosis, I/O adapters, the full report schema, visualisation.

### 1.2 The physics in brief

Classical nucleation theory fixes the critical radius from a balance of bulk and
surface free energy at a *single* equilibrium. This framework (Ferreira,
Eq. 39a/39b) treats nucleation under a **thermal gradient** instead: a non-zero
∇T across the embryo *shifts* the local equilibrium, so the saturation pressure
the germ actually sees is `P_eq,shift = P_sat,phase(T_local)` rather than
`P_sat,phase(T_ambient)`. What the tool reports follows from that shift:

* the **closure** `F(g;r) = Γ²/(4πr²) − g = 0` ties the gradient `g = ∇T` to the
  continuation radius `r`; with `r` pinned at `r_ref`, the gradient is the
  Brent-solved unknown (or you prescribe it with `--gradT`);
* the **critical radius** is the root of a **2nd-order (parabolic) stationarity**
  condition (Eq. 39b), reported as `r_critical_2nd_m` — the principal result —
  next to the classical 1st-order value for comparison;
* the **nucleation barrier and rate** follow from the shifted state and are
  decomposed into bulk / surface / configurational parts;
* everything downstream (rate → favourability → diagnostic class) is a
  *diagnosis* of this shifted-equilibrium state. The tool never invents
  hydrometeor growth, and never turns a high rate into a precipitation forecast.

Read `∇T` here as the **local** temperature gradient at the embryo interface
(validated 1–10⁴ K/m), not a synoptic front gradient (~10⁻³ K/m).

---

## 2. Dependencies

| Package | Required? | Used for |
|---|---|---|
| `numpy` | yes | arrays, numerics |
| `scipy` | yes | `brentq` (thermal closure); imported by the bundled core at load |
| `matplotlib` | yes | imported by the bundled core at load (headless `Agg` backend); also drives `MetNucleationPlotter` figures |
| `xarray` | optional | `from_xarray`, `to_xarray`, NetCDF I/O |
| `netCDF4` / `h5netcdf` | optional | NetCDF4/HDF5 read/write |
| `cfgrib` + `eccodes` | optional | GRIB ingestion |
| `pandas` | optional | convenience |

`numpy`, `scipy` and `matplotlib` are **required** just to import the package,
because the bundled core imports all three at load time. The remaining backends
are optional: if one is absent the relevant path **degrades gracefully** to
`"undetermined"` naming the missing dependency rather than crashing
(`from_grib` raises a clear `RuntimeError` telling you to install `cfgrib`;
`from_netcdf` tries `netcdf4 → h5netcdf → scipy` and falls back to NetCDF3 via
the scipy engine when only scipy is present). Install everything with
`pip install -r requirements.txt`.

The validated core is **bundled in this repository** under
`unified_h2o_nucleation_climate/` and loaded by `importlib` via a
`__file__`-relative search: the module (which sits at the repo root) looks for
`unified_h2o_nucleation_climate/` in its own folder, then one and two levels up,
so it runs from any working directory with **no `PYTHONPATH`**. The core and its
two SHA-256-guarded reference models (`Nucleation_model_H2O_vapour_*_Sim_2026*.py`,
which the guard checks byte-for-byte) are **never modified** by this layer.

---

## 3. Installation & quick start

### 3.1 Install

```bash
git clone https://github.com/ileaof/meteorological-water-phase-nucleation-application-layer.git
cd meteorological-water-phase-nucleation-application-layer
python -m pip install -r requirements.txt      # numpy, scipy, matplotlib (required)
python met_h2o_nucleation.py --validate         # prove the bundled core is intact -> SELF-CHECKS PASS
```

Requires **Python ≥ 3.9**. The repository is **self-contained**: the validated
core (`unified_h2o_nucleation_climate/`), the `het_contact_angle` module and the
two SHA-256-guarded reference models are all bundled, so no `PYTHONPATH` or
external checkout is needed. A successful `--validate` run ends with
`SELF-CHECKS PASS` (see Case 9).

### 3.2 Command line

```bash
# one state, both phases, supersaturated, with dynamics + a JSON dump
# (run from the repo root)
python met_h2o_nucleation.py --T 260 --P 70000 --RH 110 --phase-mode both --w 2.0 --LWC 5e-4 --IWC 1e-4 --dt 60 --Vcell 1e6 --json out_met_nucleation/cli_report.json

# prove the core is untouched and the met-layer self-checks pass
python met_h2o_nucleation.py --validate
```

The CLI prints the full 48-field report for each admissible phase. See §14 for
all flags.

### 3.3 Python API (minimal)

```python
import met_h2o_nucleation as M

met = M.MetInput(T=260.0, P=70000.0, RH=110.0, rh_reference="water",
                 phase_mode="both", mode="homogeneous",
                 w=2.0, LWC=5e-4, IWC=1e-4, cooling_rate=-2.0e-4,
                 N_ccn=3.0e8, N_inp=1.0e4,
                 dt_micro=60.0, cell_volume=1.0e6)

runner = M.MetNucleationRunner(met)
p_v, src, warns = M.resolve_humidity(met, 260.0, 70000.0)   # -> Pa
reps = runner.evaluate_point(260.0, 70000.0, p_v,
                             dynamics={"w": 2.0, "LWC": 5e-4, "IWC": 1e-4})

for phase, report in reps.items():          # dict[phase] -> MetNucleationReport
    print(phase, report.status, report.log10_nucleation_rate,
          report.r_critical_2nd_m, report.diagnostic_class)

M.to_json(reps, "out.json")                  # full 48-field schema
```

---

## 4. Constants & re-exported core symbols

| Name | Value / source | Meaning |
|---|---|---|
| `PHASE_LIQUID` | `"liquid"` | phase tag |
| `PHASE_ICE` | `"ice"` | phase tag |
| `Tt` | `273.16` K | triple-point temperature |
| `Pt` | `611.657` Pa | triple-point pressure |
| `THETA0` | `radians(45)` | default contact angle — **brentq fallback only**; θ is solved by Eq. 17 and reported as `contact_angle_deg` |
| `R_REF_DEFAULT` | `1e-7` m | default continuation radius |
| `T_MIN_LOCAL` | `233` K | deep-supercooling lower bound (extrapolation flag) |
| `EPS_MW` | `0.622` | M_H2O / M_dry_air (mixing-ratio epsilon) |
| `NA` | `"undetermined"` | "could not be determined" sentinel |
| `MANDATORY_FIELDS` | list[str] | the 48-field output schema (order) |
| `UNITS` | dict[str,str] | SI unit of each output field |
| `FIELD_ALIASES` | dict[str,list] | accepted input variable names per canonical field |

Re-exported from the core (so you need not import the core directly):
`SaturationProperties`, `UnifiedNucleationSimulator`, `AtmosphericInput`,
`LiquidNucleationModel`, `IceNucleationModel`, `ftheta`.

`ftheta(theta) = 2 − 3 cos θ + cos³ θ` (un-normalised, 0..4); the heterogeneous
factor used internally is `ftheta(theta)/4` (normalised, 0..1).

---

## 5. Input — `MetInput`

A dataclass holding the thermo fields shared with the core **plus** the
dynamic/microphysical/coordinate fields the core does not carry. Scalars, 1-D
arrays (profiles / time series) or callables are accepted; the array drivers
iterate elementwise. Dynamic/microphysical fields default to `None` →
`"undetermined"` in the report.

### 5.1 Thermodynamic fields (shared with core)

| Field | Type | Default | Unit | Notes |
|---|---|---|---|---|
| `T` | float/array/callable | `258.15` | K | ambient temperature |
| `P` | float/array/callable | `70000.0` | Pa | **total** atmospheric pressure |
| `RH` | optional | `None` | % | relative humidity |
| `rh_reference` | str | `"water"` | — | `"water"` or `"ice"` (reference for RH) |
| `y_v` | optional | `None` | 0..1 | vapour mole fraction |
| `p_v` | optional | `None` | Pa | vapour **partial** pressure |
| `q_v` | optional | `None` | kg/kg | specific humidity |
| `r_mix` | optional | `None` | kg/kg | mass mixing ratio |
| `grad_T` | optional | `None` | K/m | requested \|dT/dr\| (else solved) |

> At least one of `p_v, RH, y_v, r_mix, q_v` must be provided (see
> `resolve_humidity`).

### 5.2 Continuation / heterogeneous

| Field | Default | Unit | Notes |
|---|---|---|---|
| `r_ref` | `R_REF_DEFAULT` (1e-7) | m | continuation radius |
| `theta` | `THETA0` (45°) | rad | contact angle — **solver fallback only**. The nucleation angle θ is *calculated* by the solution via Ferreira Eq. 17 (`r_C,Het/r_C,Hom` self-consistency) and reported as `contact_angle_deg`; `theta` is used only when no self-consistent angle is found (brentq fallback). |
| `mode` | `"homogeneous"` | — | `"homogeneous"` or `"heterogeneous"` |
| `phase_mode` | `"auto"` | — | `auto` / `liquid` / `ice` / `both` |

`phase_mode` semantics:
* `auto` — compute the thermodynamically admissible phase(s) and report the
  kinetically dominant one;
* `both` — compute liquid and ice side by side;
* `liquid` / `ice` — single phase.

### 5.3 Dynamic / microphysical (new, not in core)

| Field | Unit | Meaning |
|---|---|---|
| `w` | m/s | vertical velocity (updraft) |
| `LWC` | kg/m³ | liquid water content |
| `IWC` | kg/m³ | ice water content |
| `N_ccn` | 1/m³ | cloud condensation nuclei number concentration |
| `N_inp` | 1/m³ | ice nucleating particle number concentration |
| `cooling_rate` | K/s | dT/dt (**<0 means cooling**) |
| `dt_micro` | s | microphysics timestep (enables `expected_events`) |
| `cell_volume` | m³ | grid-cell volume (enables `expected_events`) |
| `freezing_level` | m | altitude of the 0 °C isotherm |

### 5.4 Coordinates / metadata

`z` (geopotential altitude, m), `lat`, `lon`, `time` (s since reference) — all
optional, carried through to output.

`__post_init__` validates `phase_mode`, `mode`, `rh_reference` and raises
`ValueError` on a bad value.

---

## 6. Humidity helpers

```python
p_v, source, warnings = M.resolve_humidity(met, T, P)   # -> (Pa, str, list[str])
```

Resolves `p_v` [Pa] from whichever humidity input is given, **cross-checking
consistency** when more than one is provided (1 % relative tolerance; on
mismatch it keeps the first and appends a warning). `source` is one of
`"p_v"`, `"RH"`, `"y_v"`, `"r_mix"`, `"q_v"`. Uses the core's
`SaturationProperties` correlations (IAPWS Wagner liquid, Goff-Gratch ice).

Moist-air relations (`eps = 0.622`):
```
p_v = r * P / (r + eps)        r = eps * p_v / (P - p_v)
q   = r / (1 + r)               r = q / (1 - q)
```
Helpers: `mixing_ratio_from_p_v(p_v, P)`, `specific_humidity_from_p_v(p_v, P)`.

---

## 7. Free-energy decomposition

```python
fe = M.free_energy_decomposition(model, st, theta)   # -> dict
```

Decomposes the nucleation free energy at the evaluated radius `st['r']`, using
the core model's own hooks (`st['dGv']`, `st['gam']`) — **no re-derivation of
the physics**. Returns:

| Key | Unit | Definition |
|---|---|---|
| `DeltaG_V_J_m3` | J/m³ | ΔS_V · ΔT |
| `DeltaS_bulk` | J/(m³·K) | volumetric entropy change (st['dsv']) |
| `DeltaG_bulk_J` | J | (4π/3) r³ ΔG_V |
| `DeltaG_surface_J` | J | 4π r² γ(r, T_local) |
| `DeltaG_config_J` | J | (f/4 − 1)·(ΔG_bulk + ΔG_surface)  (hetero correction) |
| `DeltaG_total_J` | J | (f/4)·(ΔG_bulk + ΔG_surface) |
| `f_theta` | — | `2 − 3cos θ + cos³ θ` |
| `f_theta_normalised` | 0..1 | `f/4` |

Homogeneous limit θ = π → f/4 = 1 → `DeltaG_config_J = 0`.

> This decomposition is a **diagnostic at the representative continuation
> radius `r`**. The *critical* barriers ΔG_C come from the validated core
> (`r_C_1st` / `r_C_2nd` parabolic stationarity) and are reported separately as
> `DeltaG_critical_1st_J` / `DeltaG_critical_2nd_J`.

---

## 8. Precipitation diagnosis

### 8.1 `Favorability` dataclass

| Field | Meaning |
|---|---|
| `value` | 0..1 favourability |
| `contributing_vars` | factors that were present and contributed |
| `missing_vars` | factors that were absent |
| `confidence` | 0..1 = (#present ideal factors)/(#ideal) |
| `explanation` | short physical explanation |
| `caveat` | standard caveat when confidence is low (see below) |

### 8.2 `PrecipitationDiagnosis`

```python
diag = M.PrecipitationDiagnosis(T, S_w, S_i, log10I, phase,
                                 w=None, LWC=None, IWC=None, cooling_rate=None,
                                 freezing_level=None, N_ccn=None, N_inp=None, z=None)
rain  = diag.rain()      # -> Favorability
snow  = diag.snow()
graup = diag.graupel()
hail  = diag.hail()
klass = diag.diagnostic_class()
```

**Honesty guard.** A high nucleation rate **never by itself** implies rain or
hail — hydrometeor growth (condensation/deposition, collision-coalescence,
accretion, riming, melting/refreezing) is **not modelled**. When the
dynamic/microphysical data are absent, the index reflects thermodynamic
favourability only, confidence is low, and the caveat is attached:

> *"Thermodynamically favourable to nucleation, but the dynamic and
> microphysical data are insufficient to confirm precipitation or hail."*

Caveat triggers: confidence < 0.5 for rain/snow/graupel; < 0.75 for hail (the
highest data bar).

#### Elementary normalised factors (transparent, no hidden tuning)

| Factor | Formula | Saturation point |
|---|---|---|
| `thermo_supw` / `sup_ice` | (S − 1)/0.20 | 20 % supersaturation |
| `thermo_nuc` / `nuc` | sigmoid(log10I; x0=6, width=1.5) | I ≈ 1e6 /m³/s |
| `cold` | (273.15 − T)/40 | 0 at 0 °C, 1 at −40 °C |
| `warm` | (T − 273.15)/20 | 0 at 0 °C, 1 at 20 °C |
| `updraft` | w/5 | 5 m/s |
| `hail_updraft` | (w − 5)/15 | 20 m/s |
| `LWC` | LWC/1e-3 | 1 g/m³ |
| `IWC` | IWC/1e-3 | 1 g/m³ |
| `cool` | \|cooling_rate\|/5e-4 | ~1.8 K/h |
| `CCN` | N_ccn/1e9 | 1000 /cm³ |
| `INP` | N_inp/1e6 | 1 /cm³ |

The `_sigmoid` mapping is a **documented diagnostic mapping, not a physical
rate law** (threshold log10I = 6).

#### The four indices

| Index | Factors | Ideal set |
|---|---|---|
| `rain` | thermo_supw, thermo_nuc, updraft, LWC, CCN (+ cold_rain_melt when **both** IWC and freezing_level present) | 5–6 |
| `snow` | sup_ice, cold, nuc, IWC, INP | 5 |
| `graupel` | cold, sup_ice, nuc, LWC, IWC, updraft | 6 |
| `hail` | cold, nuc, LWC, hail_updraft, IWC | 5 (ideal also wants supercooled_depth + melt_below) |

`_combine` is a **weighted mean over present factors (equal weights,
renormalised)** — absent factors do not penalise the value but lower the
confidence.

#### `diagnostic_class()` — actually returned labels

`subsaturated`, `saturated_water`, `saturated_ice`, `condensation_favorable`,
`warm_rain`, `mixed_phase`, `supercooled_liquid`, `deposition_favorable`,
`insufficient_data`.

(Gradients like `cold_rain` / `snow` / `graupel` / `hail` / `freezing_rain` are
expressed through the **favourability indices**, not through this class.)

---

## 9. Output — `MetNucleationReport`

One record per phase per ambient point. Carries the 48 mandatory fields plus
`favorability_detail`, `metadata`, and the assumptions/warnings/validity_flags
lists. Use `report.to_dict()` for a plain dict (NaN → `None`).

### 9.1 The 48-field schema (with units)

| # | Field | Unit |
|---|---|---|
| 1 | `status` | — (`ok` / `subsaturated` / `no_solution`) |
| 2 | `phase` | — (`liquid` / `ice`) |
| 3 | `nucleation_mode` | — (`homogeneous` / `heterogeneous`) |
| 4 | `contact_angle_deg` | deg — nucleation angle θ **solved** by Ferreira Eq. 17 (`r_C,Het/r_C,Hom` self-consistency); `≈180` for the homogeneous / no-substrate limit. `--theta`/THETA0 is only the brentq fallback, not this value. |
| 5 | `T_ambient_K` | K |
| 6 | `T_local_K` | K (shifted local temperature) |
| 7 | `P_total_Pa` | Pa |
| 8 | `p_v_Pa` | Pa |
| 9 | `RH_water_percent` | % |
| 10 | `RH_ice_percent` | % |
| 11 | `S_water` | 1 (saturation ratio wrt water) |
| 12 | `S_ice` | 1 (saturation ratio wrt ice) |
| 13 | `gradT_K_m` | K/m |
| 14 | `DeltaT_K` | K |
| 15 | `P_eq_classical_Pa` | Pa |
| 16 | `P_eq_shift_Pa` | Pa (shifted equilibrium pressure) |
| 17 | `DeltaP_eq_Pa` | Pa (P_eq_classical − P_eq_shift) |
| 18 | `gamma_J_m2` | J/m² (surface energy at r) |
| 19 | `dgamma_dr_J_m3` | J/m³ (∂γ/∂r) |
| 20 | `surface_stress_N_m` | N/m (τ; Shuttleworth/Gurtin-Murdoch for ice) |
| 21 | `DeltaS_bulk` | J/(m³·K) |
| 22 | `DeltaG_V_J_m3` | J/m³ |
| 23 | `DeltaG_bulk_J` | J |
| 24 | `DeltaG_surface_J` | J |
| 25 | `DeltaG_config_J` | J |
| 26 | `DeltaG_total_J` | J |
| 27 | `Gamma_1st` | K·m (1st-order Gibbs-Thomson) |
| 28 | `Gamma_2nd` | K·m (2nd-order Gibbs-Thomson) |
| 29 | `r_critical_1st_m` | m |
| 30 | `r_critical_2nd_m` | m (principal result) |
| 31 | `DeltaG_critical_1st_J` | J |
| 32 | `DeltaG_critical_2nd_J` | J |
| 33 | `nucleation_rate_m3_s` | 1/(m³·s) |
| 34 | `log10_nucleation_rate` | log10(1/(m³·s)) |
| 35 | `expected_events` | 1 (I·dt_micro·cell_volume; `None` if undetermined) |
| 36 | `dominant_phase` | — |
| 37 | `rain_favorability` | 0..1 |
| 38 | `snow_favorability` | 0..1 |
| 39 | `graupel_favorability` | 0..1 |
| 40 | `hail_favorability` | 0..1 |
| 41 | `diagnostic_class` | — (see §8.2) |
| 42 | `confidence` | 0..1 (mean of per-index confidences) |
| 43 | `assumptions` | list[str] |
| 44 | `warnings` | list[str] |
| 45 | `validity_flags` | list[str] |
| 46 | `solver_iterations` | 1 (Brent iterations; `None` if not captured) |
| 47 | `closure_residual` | K/m (residual of F(g;r)=0) |
| 48 | `critical_radius_residual` | J (parabolic stationarity residual) |

### 9.2 Validity flags

`in_valid_range` / `out_of_range`, `supercooled_liquid_meta`,
`T_local_near_lower_bound_extrapolated`, `above_triple_point_liquid_stable`,
`subsaturated`, `no_solution`.

### 9.3 Metadata block (`report.metadata`)

`units` (= `UNITS`), `sign_conventions`, `sources` (Psat water/ice, surface
liquid/ice, rate, framework), `validity_ranges`, `f_theta_convention`,
`GT_convention`, and a `note` that hydrometeor growth is not modelled.

---

## 10. Runner — `MetNucleationRunner`

```python
runner = M.MetNucleationRunner(met)
reps = runner.evaluate_point(T, P, p_v, grad_T=None, dynamics=None)
```

| Driver | Signature | Returns |
|---|---|---|
| `evaluate_point` | `(T, P, p_v, grad_T=None, dynamics=None)` | `dict[phase] → MetNucleationReport` |
| `evaluate_profile` | `(T_arr, P_arr, p_v_arr, z_arr, dyn_arrs=None)` | `list[dict]` (elementwise over arrays) |
| `evaluate_series` | `(T_arr, P_arr, p_v_arr, t_arr, dyn_arrs=None)` | `list[dict]` (elementwise over time) |

`dynamics` / `dyn_arrs` keys: `w, LWC, IWC, cooling_rate, freezing_level,
N_ccn, N_inp, z` (any subset; absent → `"undetermined"`).

Internals worth knowing:
* `_atm` builds a core `AtmosphericInput` (no modification of the core class).
* `_solver_iterations` re-runs the **same** bracket + `brentq` the core uses,
  with `full_output=True`, to capture the iteration count deterministically
  (guarded; returns `None` on any failure).
* `_build_report` assembles the core result + state dict + free-energy
  decomposition + expected_events + diagnosis into a `MetNucleationReport`,
  and sets assumptions/warnings/validity_flags.
* `brentq_local(func, a, b, args=(), xtol, rtol, maxiter)` — `scipy.optimize.brentq` with `full_output=True`.

---

## 11. I/O adapters

### 11.1 Ingestion

| Function | Notes |
|---|---|
| `from_xarray(ds)` | name-tolerant field mapping (see `FIELD_ALIASES`); missing → `None` |
| `from_netcdf(path)` | tries `netcdf4 → h5netcdf → scipy`; clear error naming the missing backend |
| `from_grib(path)` | requires `cfgrib`; raises a `RuntimeError` telling you to `pip install cfgrib eccodes` if absent |

`FIELD_ALIASES` maps canonical names to accepted input names, e.g. `T` ←
`T, tair, air_temperature, temp, temperature, T_ambient`; `P` ←
`P, ps, surface_pressure, pressure, air_pressure, sp`; `q_v` ←
`q_v, q, specific_humidity, hus`; `z` ← `z, altitude, height, h,
geopotential_height, gph`; etc. (see source for the full table).

### 11.2 Output

| Function | Notes |
|---|---|
| `reports_to_records(reports)` | flatten (dict-per-point, or list of those) → list of JSON records |
| `to_json(reports, path)` | full schema; NaN → `null`; lists/dicts JSON-encoded |
| `to_csv(reports, path)` | 48 columns in `MANDATORY_FIELDS` order; NaN/None → `NA` |
| `to_xarray(reports, path=None)` | numeric fields over an **unnamed `phase` dimension**; string fields (`phase, status, nucleation_mode, dominant_phase, diagnostic_class`) are kept in JSON/CSV only; phase-name mapping stored in `ds.attrs['phase_names']` (comma-separated, sorted) |
| `to_netcdf(reports, path)` | `to_xarray` + write (scipy engine → NetCDF3 unless netCDF4/h5netcdf present) |

> The unnamed-dimension + `phase_names` attribute design is deliberate: string
> phase coordinates do not round-trip cleanly through NetCDF3/scipy, so the
> phase identity is preserved as an attribute rather than a coordinate
> variable. Read it back with `ds.attrs['phase_names'].split(',')`.

---

## 12. Visualisation — `MetNucleationPlotter`

```python
plot = M.MetNucleationPlotter("out_met_nucleation")   # uses the Agg backend
```

| Method | Output file | Requires |
|---|---|---|
| `plot_peq_shift_surface(phase, T_range, nT, g_range, ng)` | `peq_shift_surface_{phase}.png` | core models (direct) |
| `plot_gibbs_thomson_and_radii(reports_by_gradT, phase)` | `gt_and_radii_{phase}.png` | `[(g, reps), ...]` |
| `plot_free_energy(model, T, P, p_v, theta=π, n=40)` | `free_energy_vs_r.png` | a core model |
| `plot_rates(reports_liquid, reports_ice)` | `rates_vs_T.png` | lists of reports (per phase) |
| `plot_vertical_profile(profile_reports, z_arr)` | `vertical_profile.png` | list of per-level report dicts |
| `plot_favorability_bars(report)` | `favorability_bars.png` | one `MetNucleationReport` |

`plot_peq_shift_surface` brute-force scans `r` to find the radius whose closure
yields each requested `g`; reduce `nT`/`ng` if runtime matters.

The complete figure suite is generated by **`example_met_figures.py`**.

---

## 13. Self-checks & validation

```python
M.run_self_checks(verbose=True)   # -> bool
# 1) runs the CORE validation suite [1]-[21] (proves the core is untouched);
# 2) met-layer free-energy identity (DeltaG_total == bulk + surface + config);
# 3) runner end-to-end at one point (favourability in [0,1], confidence in [0,1]).
```

Or from the CLI: `python met_h2o_nucleation.py --validate`.

The full 24-test suite (20 mandatory + 4 bonus, each labelled
math / num / ref / reg) lives in **`test_met_nucleation.py`**:

```bash
python test_met_nucleation.py
```

---

## 14. Command-line reference

```
python met_h2o_nucleation.py [--validate]
        [--T K] [--P Pa] [--RH %] [--p-v Pa]
        [--phase-mode auto|liquid|ice|both]
        [--mode homogeneous|heterogeneous]
        [--theta DEG] [--r-ref m] [--gradT K/m]
        [--w m/s] [--LWC kg/m3] [--IWC kg/m3]
        [--dt s] [--Vcell m3]
        [--outdir DIR] [--json PATH] [--summary]
```

| Flag | Default | Meaning |
|---|---|---|
| `--validate` | off | run the core validation suite [1]-[21] + met-layer self-checks; exit 0/1 |
| `--T` | 260.0 | ambient temperature [K] |
| `--P` | 70000.0 | **total** pressure [Pa] |
| `--RH` | — | relative humidity [%] (mode A; cross-checked if others given) |
| `--p-v` | — | vapour **partial** pressure [Pa] (direct; alternative to `--RH`) |
| `--phase-mode` | `auto` | `auto` / `liquid` / `ice` / `both` |
| `--mode` | `homogeneous` | `homogeneous` / `heterogeneous` |
| `--theta` | 45 (THETA0) | heterogeneous contact angle in **degrees** — **brentq fallback only**; θ is *solved* by Eq. 17 and reported as `contact_angle_deg`. With no substrate modelled, the solver returns the homogeneous limit (θ≈180°) and `--theta` is unused. |
| `--r-ref` | `R_REF_DEFAULT` (1e-7) | continuation radius [m] |
| `--gradT` | — | requested thermal gradient [K/m] (else Brent-solved by the closure) |
| `--w` | — | vertical velocity [m/s] |
| `--LWC` | — | liquid water content [kg/m³] |
| `--IWC` | — | ice water content [kg/m³] |
| `--dt` | — | microphysics timestep [s] (enables `expected_events`) |
| `--Vcell` | — | grid-cell volume [m³] (enables `expected_events`) |
| `--outdir` | `out_met_nucleation` | output directory |
| `--json` | — | write the full JSON report to this path |
| `--summary` | off | print the compact one-row-per-phase table shown below **instead of** the default full 48-field vertical report. The full report is still written to `--json`. Without this flag the CLI prints every field, one per line. |

Humidity is accepted as `--RH`, `--p-v`, (or `y_v`/`r_mix`/`q_v` via the API); if
several are given they are cross-checked (≤1 % relative). At least one is
required. Phase admissibility: liquid iff `S_w > 1`, ice iff `S_i > 1`;
`both` computes regardless; `auto` reports only the admissible phase(s) and the
kinetically dominant one.

### 14.1 CLI cases with verified output

The runs below were executed from the repo root with the `--summary` flag and
their terminal output captured verbatim. `--summary` prints the compact
one-row-per-phase table (`status`, saturation ratios, solved gradient, 2nd-order
critical radius, `log10 I`, dominant phase, the four favourability indices, the
diagnostic class, and `expected_events`) — this is exactly what is shown in each
block below. Without `--summary` the CLI instead prints the **full 48-field
vertical report** (one field per line, under a per-phase header), which is what
is written verbatim to `--json`. The summary table and the full report carry the
same values; only the layout differs.

**Variable key** — every column in the example terminal blocks below:

| Column | Unit | Meaning |
|---|---|---|
| `phase` | — | which phase this row reports: `liquid` (condensation) or `ice` (deposition) |
| `status` | — | admissibility flag: `ok` if the phase is supersaturated (S>1) and the closure solved; `subsaturated` if S<1 (no nucleation; all physics fields `undetermined`) |
| `S_w` | — | saturation ratio wrt **water**: `S_w = p_v / P_sat,w(T)`; >1 ⇒ liquid supersaturated, =1 at equilibrium, <1 subsaturated |
| `S_i` | — | saturation ratio wrt **ice**: `S_i = p_v / P_sub,i(T)`; >1 ⇒ ice supersaturated. At T<0 °C, `S_i > S_w` because the sublimation curve lies below the vaporisation curve |
| `gradT` | K/m | thermal gradient: the Brent-solved closure value (or the `--gradT` you prescribed). Drives the shifted-equilibrium local temperature `T_local = T − 8πr·g/…` via the core closure |
| `rC2nd` | m | **2nd-order critical radius** (Eq.39b parabola root) — the continuation radius at which the nucleation barrier peaks. Micrometre order at the default `r_ref=1e-7 m`; a sub-micron value flags a bad `r_ref`, not a closure bug |
| `log10I` | log₁₀(m⁻³s⁻¹) | base-10 nucleation rate. Higher = faster germ formation. Compare across phases to read the kinetic competition |
| `dominant` | — | kinetically dominant phase — the phase with the larger `I` at this state. `none` when both are subsaturated |
| `rain` | 0–1 | rain (warm condensation/coalescence) favourability index; combines `S_w` supersaturation with the cold factor `(273.15−T)/40`. Saturates at 1.0 for ≥20 % water supersaturation |
| `snow` | 0–1 | snow (vapour deposition) favourability index; driven by `S_i` and the cold factor |
| `graup` | 0–1 | graupel (riming/accretion) favourability index; needs supercooled liquid + ice together |
| `hail` | 0–1 | hail (deep wet growth) favourability index; the most demanding — needs strong updraft, large LWC, cold cloud depth. **Never** set high solely because `I` is high |
| `class` | — | diagnostic class: `mixed_phase` (both phases supersaturated, T<0 °C), `condensation_favorable` (warm, only water supersaturated), `subsaturated` (neither), and others |
| `exp_events` | count | `expected_events = I · dt · V_cell` — expected new germs in the cell over the timestep. `undetermined` when `--dt` or `--Vcell` is not supplied |

> The four favourability indices are **diagnostic flags, not forecasts**: they
> combine nucleation tendency with whatever dynamics/microphysics were supplied
> and carry a confidence + caveat. When dynamics are absent they reflect
> thermodynamic favourability only (LOW confidence). Hydrometeor growth is not
> modelled.

**Case 1 — both phases, supersaturated, with dynamics + expected events.**
The canonical reference point: 260 K, 700 hPa, 110 % RH over water, a modest
updraft, supercooled LWC/IWC, and a timestep × cell volume so `expected_events`
is determined.

```bash
python met_h2o_nucleation.py --T 260 --P 70000 --RH 110 --phase-mode both --w 2.0 --LWC 5e-4 --IWC 1e-4 --dt 60 --Vcell 1e6 --summary
```

```text
  phase  | status | S_w  | S_i  | gradT | rC2nd    | log10I | dominant | rain  | snow  | graup | hail  | class       | exp_events
  liquid | ok     | 1.10 | 1.25 | 74.33 | 5.11e-06 | 53.10  | liquid   | 0.600 | 0.607 | 0.555 | 0.386 | mixed_phase | 7.61e+60
  ice    | ok     | 1.10 | 1.25 | 148.1 | 4.20e-06 | 49.08  | liquid   | 0.600 | 0.607 | 0.555 | 0.386 | mixed_phase | 7.20e+56
```
> **Read.** Both phases supersaturated (S_w=1.10, S_i=1.25). Liquid wins
> kinetically (log₁₀I 53.1 vs 49.1); r_C,2nd ≈ 5.1e-6 m (liquid), 4.2e-6 m
> (ice). With dt=60 s and V_cell=1e6 m³, `expected_events = I·dt·V_cell` is
> enormous (≈7.6e60 liquid) — nucleation is not the bottleneck here; growth is
> (unmodelled). The dynamics raise the favourability confidence; class is
> `mixed_phase` (supersaturated wrt both phases, T<0 °C).

**Case 2 — auto phase mode (dominant phase reported).**
Same state, but `--phase-mode auto`: both admissible phases are computed and the
kinetically dominant one is reported. Without `--dt/--Vcell`, `expected_events`
is `"undetermined"`.

```bash
python met_h2o_nucleation.py --T 260 --P 70000 --RH 110 --phase-mode auto --summary
```

```text
  phase  | status | S_w  | S_i  | gradT | rC2nd    | log10I | dominant | rain  | snow  | graup | hail  | class       | exp_events
  liquid | ok     | 1.10 | 1.25 | 74.33 | 5.11e-06 | 53.10  | liquid   | 0.750 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
  ice    | ok     | 1.10 | 1.25 | 148.1 | 4.20e-06 | 49.08  | liquid   | 0.750 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
```
> **Read.** `dominant_phase = liquid` (Δlog₁₀I ≈ 4 decades). Favourability
> confidence is higher than Case 1 only in appearance — the indices themselves
> are unchanged; the difference is that the diagnostic weights report the
> thermodynamic-only confidence when dynamics are absent. `expected_events` is
> `"undetermined"` because no timestep/cell volume was supplied.

**Case 3 — ice-only, RH = 130 %.**
Force the ice phase and drive it hard against the sublimation curve (S_i = 1.51).
Useful for cirrus / ice-cloud regimes.

```bash
python met_h2o_nucleation.py --T 258.15 --P 70000 --RH 130 --phase-mode ice --summary
```

```text
  phase | status | S_w  | S_i  | gradT | rC2nd    | log10I | dominant | rain  | snow  | graup | hail  | class       | exp_events
  ice   | ok     | 1.30 | 1.51 | 147   | 4.20e-06 | 49.08  | ice      | 1.000 | 0.792 | 0.792 | 0.687 | mixed_phase | undetermined
```
> **Read.** Only ice is computed; `dominant = ice` (no competitor). RH_i is
> 1.51 by construction (RH = 130 % wrt water ⇒ S_i = 1.51 at 258.15 K); RH_w
> (=1.30) is still reported for cross-reference. rain favourability saturates
> at 1.0 because the warm-rain supersaturation factor `(S_w−1)/0.20` clips at
> 1 — note this is a *thermodynamic* rain favourability, not a rain forecast.

**Case 4 — heterogeneous nucleation, θ solved by Ferreira Eq. 17.**
Switch to heterogeneous mode. The nucleation angle θ is no longer fixed: the core
solves it self-consistently from Eq. 17 (`r_C,Het/r_C,Hom`) and reports it as
`contact_angle_deg`. Compare log₁₀I against Case 2 (homogeneous).

```bash
python met_h2o_nucleation.py --T 260 --P 70000 --RH 110 --phase-mode both --mode heterogeneous --theta 60 --summary
```

```text
  phase  | status | S_w  | S_i  | gradT | rC2nd    | log10I | dominant | rain  | snow  | graup | hail  | class       | exp_events
  liquid | ok     | 1.10 | 1.25 | 74.33 | 5.11e-06 | 50.93  | liquid   | 0.750 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
  ice    | ok     | 1.10 | 1.25 | 148.1 | 4.20e-06 | 46.91  | liquid   | 0.750 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
```
> **Read.** The solved `contact_angle_deg ≈ 180°` (reported for both phases):
> the core/model carries **no substrate (aerosol) surface energies** (γ_AC,
> γ_BC), so the Eq. 17 self-consistency `r_C,Het(θ) = r_C,Hom·ratio(θ)` has only
> the trivial homogeneous-limit root θ = π. The supplied `--theta 60` is the
> brentq fallback and is **not** used here — a non-trivial θ < 180° would
> require substrate surface energies the core does not currently model.
> log₁₀I drops ~2.2 decades vs Case 2 (53.10→50.93 liquid) because the
> heterogeneous rate uses the solved-θ barrier/prefactor (full-sphere geometry
> at θ = π), not a fixed 60° cap. r_C,2nd is unchanged — it comes from the
> validated core closure, independent of the contact angle.

**Case 5 — prescribed thermal gradient (∇T = 1e3 K/m).**
Override the closure: prescribe the gradient instead of Brent-solving it. A
steeper gradient shrinks the continuation radius and shifts the local state.

```bash
python met_h2o_nucleation.py --T 260 --P 70000 --RH 110 --phase-mode both --gradT 1e3 --summary
```

```text
  phase  | status | S_w  | S_i  | gradT | rC2nd    | log10I | dominant | rain  | snow  | graup | hail  | class       | exp_events
  liquid | ok     | 1.10 | 1.25 | 1101  | 1.34e-06 | 51.94  | liquid   | 0.750 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
  ice    | ok     | 1.10 | 1.25 | 1142  | 1.52e-06 | 48.20  | liquid   | 0.750 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
```
> **Read.** Forcing ∇T = 1e3 K/m (≈14× the solved value of Case 2) collapses
> r_C,2nd from ~5.1e-6 m to ~1.3e-6 m and lowers log₁₀I by ~1.2 decades. The
> saturation ratios are unchanged (they depend on T and p_v, not the gradient).
> This is the sensitivity lever: the gradient is an *independent* control, not
> a consequence of supersaturation.

**Case 6 — subsaturated state (no nucleation).**
At 80 % RH both phases are subsaturated. In `auto` mode the tool must not
fabricate a rate: it returns `status = subsaturated`, NaN nucleation fields, and
`dominant = none`.

```bash
python met_h2o_nucleation.py --T 258.15 --P 70000 --RH 80 --phase-mode auto --summary
```

```text
  phase  | status       | S_w  | S_i  | gradT  | rC2nd  | log10I | dominant | rain  | snow  | graup | hail  | class        | exp_events
  liquid | subsaturated | 0.80 | 0.93 | undet. | undet. | undet. | none     | 0.000 | 0.125 | 0.125 | 0.188 | subsaturated | undetermined
  ice    | subsaturated | 0.80 | 0.93 | undet. | undet. | undet. | none     | 0.000 | 0.125 | 0.125 | 0.188 | subsaturated | undetermined
```
> **Behaviour.** S_w = 0.80, S_i = 0.93 — both below 1. `status = "subsaturated"`,
> all nucleation fields `"undetermined"`, `dominant = "none"`. The
> favourability indices fall to their thermodynamic floor (no supersaturation,
> no nucleation tendency) and the caveat is attached. No silent caps, no forced
> convergence. (In `both` mode the thermal closure still solves — it is a
> thermal-field closure independent of p_v — so use `auto` to surface the
> subsaturated status.)

**Case 7 — vapour partial pressure directly (p_v = 500 Pa).**
Skip RH and supply the vapour partial pressure directly. At 260 K this is
strongly supersaturated wrt both phases (S_w = 2.25).

```bash
python met_h2o_nucleation.py --T 260 --P 70000 --p-v 500 --phase-mode both --summary
```

```text
  phase  | status | S_w  | S_i  | gradT | rC2nd    | log10I | dominant | rain  | snow  | graup | hail  | class       | exp_events
  liquid | ok     | 2.25 | 2.55 | 74.33 | 5.11e-06 | 53.10  | liquid   | 1.000 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
  ice    | ok     | 2.25 | 2.55 | 148.1 | 4.20e-06 | 49.08  | liquid   | 1.000 | 0.776 | 0.776 | 0.664 | mixed_phase | undetermined
```
> **Read.** `--p-v 500` Pa ⇒ S_w = 500/P_sat,w(260 K) = 2.25, S_i = 2.55. The
> rain warm-supersaturation factor saturates at 1.0 (>20 % SS). The solved
> gradient and r_C,2nd are identical to Case 2 — the closure depends on the
> phase constitutive laws at T_local, not on how humidity was specified. This
> is the mode to use when p_v comes from a forecast field rather than RH.

**Case 8 — warm regime (T = 285 K, RH = 102 %).**
Above freezing: liquid is slightly supersaturated, ice is *sub* saturated
(S_i = 0.91). The diagnostic class flips to `condensation_favorable` and the
cold-type indices collapse.

```bash
python met_h2o_nucleation.py --T 285 --P 90000 --RH 102 --phase-mode both --w 1.0 --LWC 3e-4 --dt 60 --Vcell 1e6 --summary
```

```text
  phase  | status | S_w  | S_i  | gradT | rC2nd    | log10I | dominant | rain  | snow  | graup | hail  | class                  | exp_events
  liquid | ok     | 1.02 | 0.91 | 79.62 | 5.11e-06 | 53.06  | liquid   | 0.400 | 0.333 | 0.300 | 0.325 | condensation_favorable | 6.96e+60
  ice    | ok     | 1.02 | 0.91 | 163.5 | 4.20e-06 | 49.11  | liquid   | 0.400 | 0.333 | 0.300 | 0.325 | condensation_favorable | 7.71e+56
```
> **Read.** T > 273.15 K, S_w = 1.02 (slight supersaturation), S_i = 0.91 (ice
> subsaturated). Class = `condensation_favorable` (warm, supersaturated wrt
> water only). The `cold` factor `(273.15−T)/40` is 0, so snow/graupel/hail
> fall to their warm-floor. `expected_events` is determined (dt + Vcell given).
> Ice is still computed in `both` mode even though it is not admissible — useful
> for comparison; `auto` would drop it.

**Case 9 — self-validation (`--validate`).**
Run the core validation suite [1]-[21] (proving the guarded core is untouched)
plus the met-layer self-checks (free-energy identity, runner end-to-end). Exits
0 on success, 1 on any failure.

```bash
python met_h2o_nucleation.py --validate
```

```text
==============================================================================
SELF-CHECKS  met_h2o_nucleation.py
==============================================================================
[core validation] -> PASS                      (tests [1]-[21], ice SHA-256 unchanged)
[decomposition identity] dG_total==sum: PASS   (free-energy identity)
[runner end-to-end] 2 phases: PASS             (favourability in [0,1], confidence in [0,1])
------------------------------------------------------------------------------
SELF-CHECKS PASS
```
> **Read.** The full 24-test suite (`test_met_nucleation.py`) is the
> comprehensive check; `--validate` is the lightweight in-process gate. Both
> pass in the current build. The core SHA-256 guard (test 18) is the proof that
> this application layer has not modified the validated core.

**Case 10 — warm-moist × cold-dry air-mass collision (frontal mixing cloud).**
A warm, moist air mass (T = 293.15 K, RH = 95 %) collides with a cold, dry one
(T = 268.15 K, RH = 40 %) at a near-surface front (P = 900 hPa). Neither parent
is saturated, yet **isobaric mixing** yields a supersaturated parcel: mixing
temperature and vapour pressure linearly (the Rogers–Yau mixing-cloud
construction) and scanning the warm-air mass fraction *f*, the supersaturation
peaks at *f* = 0.50 → T = 280.75 K, p_v = 1203.69 Pa, **S_water = 1.153**. That
mixed state *is* the frontal cloud; `example_met_frontal_collision.py` builds it
(re-using the core `SaturationProperties` correlations) and diagnoses it. Feeding
the mixed state to the CLI with a modest updraft over the cold wedge:

```bash
python met_h2o_nucleation.py --T 280.75 --P 90000 --p-v 1203.69 --phase-mode both --w 1.5 --LWC 5e-4 --dt 60 --Vcell 1e6 --summary
```

```text
  phase  | status | S_w  | S_i  | gradT | rC2nd    | log10I | theta_deg | dominant | rain  | snow  | graup | hail  | class     | theta_model   | exp_events
  liquid | ok     | 1.15 | 1.07 | 78.71 | 5.11e-06 | 53.05  | 90.04     | liquid   | 0.641 | 0.452 | 0.431 | 0.375 | warm_rain | ferreira_eq17 | 6.79e+60
  ice    | ok     | 1.15 | 1.07 | 160.8 | 4.20e-06 | 49.10  | 90.03     | liquid   | 0.641 | 0.452 | 0.431 | 0.375 | warm_rain | ferreira_eq17 | 7.63e+56
```
> **Read.** The collision is not itself a tool input — it is encoded as the
> *mixed parcel* it produces. Both parents were subsaturated (RH 95 % and 40 %),
> yet the mixture reaches S_water = 1.15 because e_sat(T) is convex and the
> straight mixing line bulges above it: this is mixing fog / frontal cloud.
> T = 280.75 K > 273.15 K, so the class is `warm_rain` and liquid dominates
> (log₁₀I 53.05 vs 49.10 for ice); ∇T is solved by the thermal closure
> (78.7 K/m liquid). Note the peak-supersaturation mixture is *warm* — for a
> supercooled / mixed-phase ice case pick a colder mixture (larger cold-air
> fraction, T < 273 K) from the same `T_mix`/`S_mix` scan the script computes.
> Do **not** map the synoptic front's ∇T onto `--gradT`: that field is the local
> interface gradient (validated 1–10⁴ K/m), not the ~10⁻³ K/m synoptic gradient.

---

## 15. Examples

| Script | Demonstrates |
|---|---|
| `example_met_single_state.py` | one state (T=260 K, P=700 hPa, RH=110 %, both phases) → full 48-field report + JSON/CSV/NetCDF + `favorability_bars.png` |
| `example_met_vertical_profile.py` | 20-level hydrostatic profile → per-level reports, CSV + `vertical_profile.png` + JSON; subsaturated levels handled |
| `example_met_xarray_netcdf.py` | build an `xarray.Dataset`, NetCDF3 round-trip (scipy engine), per-level reports, structured xarray/NetCDF output |
| `example_met_figures.py` | the full figure suite (P_eq,shift surface, Γ & r_C vs ∇T, ΔG vs r, rates vs T, vertical profile, favourability bars) |
| `example_met_frontal_collision.py` | warm-moist × cold-dry air-mass collision → isobaric mixing to the supersaturated frontal-cloud state → both-phase nucleation report + JSON/CSV (see Case 10) |

```bash
# run from the repo root (examples auto-write into out_met_nucleation/)
python example_met_single_state.py
python example_met_vertical_profile.py
python example_met_xarray_netcdf.py
python example_met_figures.py
python example_met_frontal_collision.py
```

---

## 16. Conventions

* **Units:** all internal quantities are SI.
* **Pressures:** `P`/`P_total_Pa` = total atmospheric pressure; `p_v` = water
  vapour partial pressure; `P_eq_*` = phase equilibrium (saturation) pressure;
  `P_eq_shift` = shifted equilibrium pressure P_sat,phase(T_local).
* **Sign conventions** (in `metadata.sign_conventions`):
  `DeltaS_bulk < 0` (volumetric entropy, condensible minus vapour);
  `DeltaG_V < 0` drives nucleation; `DeltaP_eq = P_eq_classical − P_eq_shift > 0`
  under cooling; `cooling_rate < 0` means cooling.
* **Heterogeneous geometry:** `f(θ) = 2 − 3 cos θ + cos³ θ` (un-normalised
  0..4); factor `f/4` (0..1); homogeneous limit θ = π → f/4 = 1.
* **Gibbs-Thomson:** `GT = r_C · ΔT / 2` (core convention; **not** 4πr²g).
* **Liquid surface:** Tolman curvature `γ(r) = γ∞/(1 + 2δ_T/r)`.
* **Ice surface stress:** Shuttleworth / Gurtin-Murdoch `τ = γ + r·∂γ/∂r`
  (distinct from the liquid `τ = γ`).

---

## 17. Validity ranges & what remains hypothesis

| Quantity | Validity |
|---|---|
| T ambient | 233..373 K (ice 233..273; liquid 233..647) |
| gradT | 1..1e4 K/m validated; beyond = extrapolation |
| r continuation | 1e-9..1e-2 m |
| Psat water | IAPWS Wagner, **extended below triple point** (extrapolated, stated) |
| Psat ice | Goff-Gratch, anchored at the triple point |

**Remains hypothesis (not validated against observations):** the favourability
indices (transparent documented weights, not tuned), `expected_events`
(well-mixed cell + single-step Poisson), the sigmoid nucleation-tendency
mapping, the self-consistent θ at r_C, and IAPWS below the triple point.

**Out of scope (explicitly stated):** hydrometeor growth
(condensation/deposition, collision-coalescence, accretion, riming,
melting/refreezing). A high nucleation rate never by itself implies rain or
hail.

For the full hypotheses table (H1–H17) and the CNT-vs-1st-vs-2nd comparison see
`MET_NUCLEATION_HYPOTHESES.md`. For an operational rain/hail forecast the module
additionally requires: vertical-velocity / CAPE / residence-time profiles,
LWC/IWC + size distributions + growth rates + melting/refreezing,
CCN/INP/INAS spectra, the full T/q profile + freezing level + cloud depth, and
the microphysical timestep + cell volume.

---

## 18. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `ModuleNotFoundError: het_contact_angle` (or the core) | Run from the cloned repo root. `het_contact_angle.py` and the `unified_h2o_nucleation_climate/` core are bundled there; if you copied only `met_h2o_nucleation.py` elsewhere, either copy the siblings too or put the repo root on `PYTHONPATH`. |
| `--validate` → `ice reference script not found` / `VALIDATION FAILED` | The SHA-256 guard needs the two reference models (`Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py`, `…_liquid_Sim_2026.py`) at the repo root, one level above `unified_h2o_nucleation_climate/`. They are bundled; if you copied only the module, copy those two files too. |
| `ImportError` for `scipy` or `matplotlib` | Both are **required** — the bundled core imports them at load. `pip install -r requirements.txt`. matplotlib uses the headless `Agg` backend, so no display is needed. |
| NetCDF read/write warns about the engine or fails | `to_netcdf`/`from_netcdf` fall back `netcdf4 → h5netcdf → scipy`; with only scipy present you get NetCDF3. `pip install netCDF4` for NetCDF4/HDF5. |
| `from_grib` raises `RuntimeError` | GRIB ingestion needs `cfgrib` + `eccodes` (`pip install cfgrib eccodes`). |
| Every physics field is `undetermined`, `status = subsaturated` | The state is below saturation (S<1) in `auto`/single-phase mode — correct behaviour, not an error. Raise `--RH`/`--p-v`, or use `--phase-mode both` to force the closure regardless. |
| `r_critical_2nd_m` comes out sub-micron | Usually a bad `--r-ref`; the default `1e-7 m` yields micrometre-order critical radii (see §17). |
| Unicode / `cp1252` errors when printing on Windows | The examples wrap stdout as UTF-8. If you print reports yourself, set `PYTHONUTF8=1` or reconfigure stdout to UTF-8. |

---

## 19. Citation & license

If you use this tool in academic work, please cite the underlying
shifted-equilibrium framework:

> Ferreira, *Physica B: Condensed Matter* **695** (2024) 416494; and the
> MRS Meeting 2026 contribution on the meteorological water-phase nucleation
> application layer.

```bibtex
@article{ferreira2024shifted,
  author  = {Ferreira},          % TODO: complete the author list
  title   = {},                   % TODO: article title
  journal = {Physica B: Condensed Matter},
  volume  = {695},
  pages   = {416494},
  year    = {2024},
  doi     = {}                     % TODO
}
```

Complete the `author` / `title` / `doi` fields from the published reference.

**License.** No `LICENSE` file is currently included, which by default means **all
rights reserved**. To permit reuse, add a licence (e.g. MIT, BSD-3-Clause,
Apache-2.0); until then, contact the author for permission. Note the bundled core
and its reference models are integrity-guarded and marked read-only.

---

## 20. File map

> Updated for the 2026-08-20 reorganization. API names are unchanged; only
> paths moved. See `MIGRATION_MANIFEST.md` for the full old→new mapping.

```
src/met_water_nucleation/
    __init__.py                    package facade (import met_water_nucleation as M)
    cli.py / __main__.py           console entry point + `python -m …`
    _engine/                       IMMUTABLE bundle (read-only, SHA-256 guarded)
        met_h2o_nucleation.py        the application/diagnosis module
        het_contact_angle.py         heterogeneous contact-angle models
        Nucleation_model_H2O_vapour_solid_Sim_2026_paper.py   ice reference model
        Nucleation_model_H2O_vapour_liquid_Sim_2026.py        liquid reference model
        unified_h2o_nucleation_climate/
            unified_h2o_nucleation_climate.py   the validated core (DO NOT MODIFY)

tests/test_met_nucleation.py       24-test validation suite
examples/                          single_state, vertical_profile, xarray_netcdf,
                                   figures, frontal_collision
configs/                           declarative scenario YAMLs
scripts/                           run_validation, regenerate_outputs
docs/                              this manual (+.html), hypotheses, architecture, migration guide
outputs/                           generated outputs (outputs/<scenario>/<run-id>/)

met_h2o_nucleation.py              BACKWARD-COMPAT SHIM (repo root) — delegates to the package CLI
```