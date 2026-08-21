# configs/

Declarative scenario configurations (YAML). Each mirrors an `examples/`
script's parameters so a scenario can be read, version-controlled and diffed
separately from code.

These are **declarative references**. A YAML→`MetInput` loader is a documented
future task (it would be a *new* package module — `met_water_nucleation.config`
— with `PyYAML` as an optional dependency; it would not touch the immutable
engine). Until then, treat these as the canonical parameter sets the examples
reproduce.

| File | Scenario |
|---|---|
| `default.yaml` | package defaults (MetInput defaults) |
| `single_state.yaml` | one supercooled supersaturated parcel, both phases |
| `vertical_profile.yaml` | 20-level hydrostatic mid-latitude ascent |
| `xarray_netcdf.yaml` | 6-level field, NetCDF3 round-trip |
| `frontal_collision.yaml` | warm-moist × cold-dry isobaric mixing (peak-S parcel) |