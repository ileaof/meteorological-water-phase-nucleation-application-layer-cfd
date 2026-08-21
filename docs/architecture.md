# Architecture

The `met_water_nucleation` package is a thin, importable facade over an
**immutable validated physics engine**. The engine is loaded **read-only** via
`importlib` and is never modified or refactored; its integrity is SHA-256
guarded (`--validate`).

## Two-layer model

```
                 +------------------------------------------+
   met input --> |  application / diagnosis layer            |  met_h2o_nucleation.py
                 |  MetInput / Runner / Diagnosis           |  + het_contact_angle.py
                 |  free-energy decomposition / I/O / viz    |
                 +---------------------+--------------------+
                                       |  imports READ-ONLY (importlib, by path)
                                       v
                 +------------------------------------------+
                 |  validated core                          |  unified_h2o_nucleation_climate.py
                 |  closure, r_C, Gamma, rate, tests [1]-[21]|  (SHA-256 guarded)
                 +------------------------------------------+
```

The core owns the closure `F(g;r)=Γ²/(4πr²)−g=0`, the 1st/2nd-order critical
radius, the surface-stress law, the nucleation rate and the validation suite.
The application layer adds only what the core deliberately does not own:
free-energy decomposition, precipitation diagnosis, I/O adapters, the full
report schema and visualisation.

## Scientific data flow

```
meteorological input  (T, P, humidity, optional w/LWC/IWC/N_ccn/N_inp/...)
        |
        v
atmospheric thermodynamics   resolve_humidity -> p_v; S_w, S_i (IAPWS Wagner / Goff-Gratch)
        |
        v
nonequilibrium nucleation    thermal closure (Brent) -> T_local; P_eq,shift = P_sat(T_local)
        |                    2nd-order Gibbs-Thomson; critical radius r_C,2nd (Eq.39b)
        v
phase competition            liquid vs ice nucleation rate I; kinetic dominance
        |
        v
optional fluid-flow coupling (NONE in this repo: no CFD solver; `--w`/cooling are inputs)
        |
        v
meteorological diagnostics   rain/snow/graupel/hail favourability + diagnostic_class
        |
        v
outputs & visualisation      JSON / CSV / NetCDF (xarray) / PNG
```

Notes:
- "optional fluid-flow coupling" is a **placeholder**: despite the repository
  name `..._cfd`, there is no grid, advection, pressure solver or turbulence
  module. Updraft `w` and `cooling_rate` are scalar inputs, not a solved flow.
- The contact angle θ is solved self-consistently from Ferreira Eq.17
  (`r_C,Het/r_C,Hom`) and reported as `contact_angle_deg`; with no substrate
  surface energies modelled, the solver returns the homogeneous limit.

## Import mechanics (why the bundle is co-located)

The core's SHA-256 guard computes the reference-script location as
`dirname(dirname(__file__))` and the loaders find the core by a
`__file__`-relative search. The whole bundle (core, two reference scripts,
`met`, `hca`) therefore moved **as a unit** into
`src/met_water_nucleation/_engine/` so that the relative arrangement — and
hence the guard and the loaders — keeps resolving identically. The move was
byte-identical (`git mv`); all five engine files retain their pre-reorg
SHA-256 checksums.