# data/

Reserved for input and immutable reference datasets.

- `data/input/` — scenario input files (CSV/NetCDF/GRIB) fed to the package.
- `data/reference/` — small, immutable validation datasets required by tests.

Nothing is stored here yet. The tests generate their own small fixtures
on the fly (e.g. the xarray round-trip uses the system temp dir), so no
committed dataset is currently required. Add small, test-required datasets
here and keep them tracked; keep large/regenerable data out of the repo.