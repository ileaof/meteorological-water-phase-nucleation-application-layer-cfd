# `met_h2o_nucleation` — Hypotheses, Limitations, Validity & Validation Report

> **Reorganization note (2026-08-20).** The module and core moved (byte-identical)
> into the installable package `met_water_nucleation` under
> `src/met_water_nucleation/_engine/`. The science, equations and validation
> status below are unchanged. Import as `import met_water_nucleation as M`;
> the validated core is `M.un`. See `docs/migration-guide.md`.

This document records the scientific hypotheses, extrapolations, validity
ranges and the validation status of the meteorological water-phase nucleation
module `met_h2o_nucleation.py` (the application/diagnosis layer on top of the
validated core `unified_h2o_nucleation_climate/unified_h2o_nucleation_climate.py`).

## 1. Architecture

`met_h2o_nucleation.py` imports the validated core **read-only** and adds:

* free-energy decomposition (`free_energy_decomposition`);
* precipitation diagnosis (`PrecipitationDiagnosis`);
* xarray / NetCDF / GRIB I/O adapters;
* the full mandatory 47-field report + metadata;
* visualisation;
* the 20-test suite (`test_met_nucleation.py`).

The core closure, the 1st/2nd-order critical radius, the 2nd-order
heterogeneous parabola (Ferreira Eq.39b), the surface-stress law and the
nucleation rate are **delegated** to the core; this layer does not re-derive
them.  The ice reference script remains SHA-256-guarded (test 18).

**Core correction (Ferreira Eq. 17).**  The core function
`het_hom_radius_ratio(theta)` — the antiderivative of the Eq. 17 integrand
`sinϑ(1+cosϑ)/(2−cosϑ−cos²ϑ)` giving `r_C,Het/r_C,Hom` — was corrected to the
closed form `(2+cosθ)^{1/3}(1−cosθ)^{2/3}/2^{2/3}` (ratio(π)=1, ratio(0)=0).
The previous form (`exp` of a single log of `2−cos−cos²`) was wrong and
exceeded 1 for θ>π/2, which manufactured a spurious contact angle.  This is
the only core edit; the ice reference script SHA-256 is unchanged (test [1]/18)
and all core tests [1]–[21] still PASS.  The solved θ is surfaced in the met
report as `contact_angle_deg` (field #4).  Because the model carries no
substrate (aerosol) surface energies, the Eq. 17 self-consistency
`r_C,Het(θ)=r_C,Hom·ratio(θ)` has only the trivial root θ=π; the solver reports
the homogeneous limit `contact_angle_deg ≈ 180°`, and `--theta`/THETA0 is
only the brentq fallback (unused when no substrate is modelled).

## 2. Equations effectively implemented

| Quantity | Source / form |
|---|---|
| Closure `F(g;r)=Γ²/(4πr²)−g=0` | core (Brent, bracket) |
| `ΔT=8πrg`, `T_local=T−ΔT`, `∂ΔT/∂r=8πg` | core |
| `P_eq,shift=P_sat,phase(T_local)` | core |
| `Γ²` (2nd-order homogeneous) | core `−(3/4)(ΔS_VΔT+∂γ/∂r)/(∂ΔS_V/∂r+(ΔS_V/ΔT)∂ΔT/∂r)` |
| 2nd-order heterogeneous parabola `Ar²+Br+C=0` (with `∂f/∂r`) | core `_rC_2nd_het` |
| `r_C,1st`, `r_C,2nd` (parabolic root, physical branch) | core |
| Nucleation rate `I` and `log10 I` (D·A·N_v/λ⁴·exp(dGc/dGc_eq)) | core |
| `expected_events = I · dt_micro · V_cell` | **new** |
| `ΔG_V=ΔS_V·ΔT`, `ΔG_bulk=(4π/3)r³ΔG_V`, `ΔG_surface=4πr²γ(r,T_local)` | **new** (core hooks) |
| `ΔG_config=(f(θ)/4−1)(ΔG_bulk+ΔG_surface)`, `ΔG_total=(f/4)(ΔG_bulk+ΔG_surface)` | **new** |
| `f(θ)=2−3cosθ+cos³θ` (unnormalised 0..4), factor `f/4` (0..1) | core convention |
| **Eq. 17** `r_C,Het/r_C,Hom = exp[−∫_θ^π sinϑ(1+cosϑ)/(2−cosϑ−cos²ϑ) dϑ]` = `(2+cosθ)^{1/3}(1−cosθ)^{2/3}/2^{2/3}`; θ solved self-consistently, reported as `contact_angle_deg` | core (`het_hom_radius_ratio`, `_solve_theta`) — **corrected** |
| Tolman curvature `γ_VL(r)=γ∞/(1+2δ/r)` (liquid) | core |
| Shuttleworth / Gurtin–Murdoch `τ=γ+r∂γ/∂r` (ice) | core |
| Favourability indices (0..1) | **new** (transparent weighted diagnostic, see §4) |

Sign conventions: `ΔS_V<0` (volumetric entropy, condensible minus vapour);
`ΔG_V<0` drives nucleation; `ΔP_eq=P_eq_classical−P_eq_shift>0` under cooling;
`cooling_rate=dT/dt<0` means cooling.

## 3. Hypotheses, extrapolations and validity ranges

| # | Hypothesis / assumption | Validity | Status |
|---|---|---|---|
| H1 | Radius `r` is the continuation variable; the thermal gradient is the Brent-solved unknown (Ferreira closure). | `r ∈ [1e-9,1e-2] m` | Validated (core tests [8]–[12]) |
| H2 | All gradient-dependent quantities recomputed every residual (not frozen). | — | Validated (core) |
| H3 | Liquid: IAPWS Wagner saturation, extended below the triple point for supercooled states. | `T ∈ [233, 647] K` | Validated above Tt; **extrapolated** below Tt (stated) |
| H4 | Ice: Goff–Gratch sublimation, anchored at the triple point. | `T ∈ [233, 273] K` | Validated |
| H5 | Liquid surface energy: Tolman curvature with `δ_T=0.2 nm` (configurable). | r not too small | Validated; becomes stiff as `r→2δ_T` |
| H6 | Ice surface energy: literature value `γ₀≈0.105 J/m²` (NOT the 1.3 of the ice reference script). | — | Validated (core test [3]) |
| H7 | Ice surface STRESS via Gurtin–Murdoch (`τ=γ+r dγ/dr`), distinct from the liquid `τ=γ`. | — | Validated |
| H8 | Heterogeneous geometry: spherical-cap factor `f(θ)/4`; `∂f/∂r` included when θ depends on r; `θ=π` ⇒ homogeneous. The nucleation angle θ is **solved** self-consistently from Ferreira Eq. 17 (`r_C,Het/r_C,Hom`) by the core and reported as `contact_angle_deg`; `--theta`/THETA0 is the brentq fallback only. | — | Validated (tests 9, 10, b1) |
| H9 | The 2nd-order heterogeneous parabola uses `dθ/dr` evaluated self-consistently at `r_C`; at `r_C` it is small and the het correction is negligible (`rC_het≈rC_hom`). | — | **Hypothesis** (documented core refinement) |
| H10 | Gibbs–Thomson coefficient `GT = r_C·ΔT/2` (NOT `4πr²g`). | — | Validated (core test [17]) |
| H11 | Free-energy decomposition is a **diagnostic at the evaluated continuation radius `r`**; the *critical* barriers `ΔG_C` come from the validated core. | — | Stated convention |
| H12 | `expected_events = I·dt·V_cell` assumes a well-mixed cell and a single-step Poisson count. | — | **Hypothesis** |
| H13 | Favourability indices are **transparent weighted diagnostics**, NOT a physical growth model. Weights are equal over present factors; absent factors do not penalise the value but lower the confidence. | — | **Hypothesis / not validated against observations** |
| H14 | Hydrometeor growth (condensation/deposition, collision–coalescence, accretion, riming, melting/refreezing) is **NOT modelled**. A high `I` never by itself implies rain or hail. | — | Stated limitation |
| H15 | Nucleation tendency mapped to [0,1] via `sigmoid((log10I−6)/1.5)` — a documented diagnostic mapping, not a physical rate law. | — | **Hypothesis** |
| H16 | GRIB read requires `cfgrib`; NetCDF4/HDF5 read requires `netCDF4` or `h5netcdf`. Absent backends degrade to "undetermined" naming the dependency. | — | Validated (test b4) |
| H17 | xarray/NetCDF output uses the scipy engine (NetCDF3); string phase coordinates are stored as an attribute mapping over an unnamed integer dimension. | — | Validated (test b3) |

## 4. Favourability indices — definition (transparent, no hidden tuning)

Each index is a weighted mean over **present** normalised factors (equal
weights; renormalised), with `confidence = (#present ideal factors)/(#ideal)`.
Absent dynamic/microphysical factors lower the confidence but do not bias the
value.  When `confidence < 0.5` (rain/snow/graupel) or `< 0.75` (hail) the
standard caveat is attached.

| Index | Factors (ideal set) |
|---|---|
| rain | warm: `{S_w, log10I, w, LWC, N_ccn}`; cold: `{log10I, IWC, freezing_level}`; else thermodynamic-only |
| snow | `{S_i, (273−T)/40, log10I, IWC, N_inp}` |
| graupel | `{(273−T)/40, S_i, log10I, LWC, IWC, w/5}` |
| hail | `{(273−T)/40, log10I, LWC/1e-3, (w−5)/15, IWC}` — **highest data bar** |

Normalised factors saturate at documented values (e.g. `LWC/1e-3` saturates at
1 g/m³; `w/5` at 5 m/s; hail updraft `(w−5)/15` saturates at 20 m/s).  These
choices are documented in the code, not tuned to observations.

## 5. Validation categories (per the specification)

The 20 mandatory tests are labelled:
* **math** (mathematical/analytical identity): 1, 4, 5, 6, 8, 9, 10, 11, 12, 16, b1;
* **num** (numerical verification): 3, 7, 14, 15, b3, b4;
* **ref** (reference comparison): 2, 17, 18, 19;
* **reg** (meteorological-output regression): 13, 20, b2.

No test is labelled **experimental** (no in-situ observation is compared) or
**extrapolation** in the sense of asserting an unvalidated regime — the only
extrapolation flagged is H3 (IAPWS below Tt), which is inherited from the core
and stated as such.

## 6. CNT vs 1st-order vs 2nd-order

* **CNT** is implemented only as a reference / limiting case (core
  `_cnt_reference`, `r_CNT`, `ΔG_CNT`).  Test 19 reproduces the classical
  limit `r_C,1st → r_CNT` (rel 4e-7) at a prescribed large undercooling.
* **1st-order** (`r_C,1st`, `Γ¹`) is reported for comparison; for the liquid
  it is **negative near equilibrium** (no positive 1st-order root there) — a
  known property of this framework, not a defect.  The physically selected
  radius is the **2nd-order** one.
* **2nd-order** (`r_C,2nd`, `Γ²`, the Eq.39b parabola) is the principal
  result; test 8 verifies it is the stationary point of `ΔG_total`, and
  test 7 that the parabolic residual is ~1e-17.
* The full Ferreira 2nd-order formulation is **not** replaced by CNT; CNT
  is exposed only via `r_CNT`/`ΔG_CNT` and the near-equilibrium limit.

## 7. What is needed for an operational rain / hail forecast

This module supplies the **nucleation** leg.  A real precipitation / hail
forecast additionally requires data and models the diagnosis layer explicitly
flags as **missing**:

* **Dynamics**: vertical velocity `w` and its profile, updraft width/depth,
  convective intensity (CAPE/CIN), residence time in the supercooled layer.
* **Microphysics**: LWC/IWC profiles, drop/crystal size distributions,
  collision–coalescence / accretion / riming rates, melting and refreezing
  below the freezing level, hydrometeor trajectories.
* **Aerosol**: CCN/INP spectra (composition + size), ice-active surface-site
  density (e.g. INAS), their vertical distribution.
* **Thermodynamic context**: the full T/q profile, freezing level, cloud
  depth, environmental lapse rate, cold-trap / warm-rain layers.
* **Timescale**: the microphysical timestep and grid-cell volume for
  `expected_events` (without them `expected_events` is "undetermined").

Without these, the module correctly returns
**"Thermodynamically favourable to nucleation, but the dynamic and
microphysical data are insufficient to confirm precipitation or hail"** with a low
confidence, rather than an over-claim.

## 8. Final validation report (filled after the test run)

* `python met_h2o_nucleation.py --validate` → core [1]–[21] PASS (ice
  SHA-256 unchanged) + decomposition identity + runner end-to-end: **PASS**.
* `python test_met_nucleation.py` → **20/20 mandatory + 4/4 bonus PASS**
  (see the run log; categories per §5).
* `python example_met_single_state.py` → full 47-field report (both phases),
  JSON/CSV/NetCDF + `favorability_bars.png`.
* `python example_met_vertical_profile.py` → 20-level profile, CSV +
  `vertical_profile.png`, subsaturated levels handled.
* `python example_met_xarray_netcdf.py` → NetCDF3 round-trip, per-level
  reports, xarray/NetCDF output.
* `python example_met_figures.py` → full visualization suite:
  P_eq,shift(T,∇T) surface (liquid+ice), Γ¹/Γ² & r_C1/r_C2 vs ∇T (liquid+ice),
  ΔG decomposition vs r, log10 I vs T, vertical profile, favourability bars.

**Conclusion:** the mathematical identities, numerical limits and repository
references are validated; the meteorological outputs regress cleanly; the
core is provably untouched.  The favourability indices and `expected_events`
remain **hypotheses / diagnostics** (not validated against observations), and
hydrometeor growth is explicitly out of scope (§7).