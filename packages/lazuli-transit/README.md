# lazuli-transit

Instrument-agnostic exoplanet transit models and NASA Exoplanet Archive
helpers for the Lazuli mission. Depends only on numpy/astropy/pandas (plus
optional `jaxoplanet` and `astroquery`); it has no dependency on any specific
instrument package, so WCC, the IFS, and others can all build light-curve
simulators on top of it.
