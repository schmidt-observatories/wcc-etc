# NASA Exoplanet Archive → Transit Params

**Date:** 2026-06-19
**Status:** Design approved

## Goal

Make it easy to download a local CSV of NASA Exoplanet Archive planet
parameters and build a `TransitModel` for any confirmed planet, so users can
estimate transit light curves for real planets with a single call instead of
hand-entering batman parameters.

## Scope

- Python API only (no CLI, no notebook entry point in this change).
- Bulk download of the **PSCompPars** table (one complete, self-consistent row
  per confirmed planet) to a local CSV cache.
- A `TransitModel.from_planet(name)` classmethod that maps archive columns to
  batman parameters with sensible fallbacks.

Out of scope: per-planet TAP queries, the full PS table (multiple parameter
sets per planet), limb-darkening lookup, multi-planet systems beyond name
selection.

## Dependency

`astroquery` becomes an optional dependency:

```toml
[project.optional-dependencies]
lightcurve = ["batman-package"]
exoarchive = ["astroquery"]
```

Import is lazy inside the functions that need it (mirroring how `batman` is
handled in `lightcurve.py`), raising `ImportError` with the hint:

> `download_exoplanet_archive requires 'astroquery'. Install it with: pip install wcc-etc[exoarchive]`

The package must still import fine without `astroquery` installed.

## Components

### New module: `src/wcc_etc/exoarchive.py`

#### `download_exoplanet_archive(path=None, refresh=False) -> pandas.DataFrame`

- Queries the PSCompPars table via
  `astroquery.ipac.nexsci.nasa_exoplanet_archive.NasaExoplanetArchive`,
  selecting the identifier + transit columns listed below.
- `path` defaults to `~/.wcc_etc/exoplanet_archive_pscomppars.csv`. The
  parent directory is created if missing.
- If the cache file exists and `refresh=False`, load it from disk instead of
  hitting the network. If `refresh=True` (or the file is absent), download and
  write the CSV.
- Returns a `pandas.DataFrame` in both paths.

#### `load_exoplanet_archive(path=None) -> pandas.DataFrame`

- Reads the cached CSV (default path as above) and returns a DataFrame.
- If the file does not exist, raise `FileNotFoundError` with a clear message
  telling the user to run `download_exoplanet_archive()` first. Does **not**
  hit the network.

Columns selected/persisted: `pl_name`, `hostname`, `pl_orbper`, `pl_ratror`,
`pl_ratdor`, `pl_orbincl`, `pl_tranmid`, `pl_orbeccen`, `pl_orblper`,
`pl_radj`, `pl_orbsmax`, `st_rad`.

### Extend `src/wcc_etc/lightcurve.py`

#### `TransitModel.from_planet(name, df=None, limb_dark="quadratic", u=(0.1, 0.3)) -> TransitModel`

- If `df is None`, call `load_exoplanet_archive()` (which raises with a hint if
  the cache is missing — the model constructor does **not** silently hit the
  network).
- Look up the row by `pl_name`, tolerant of case and internal spacing
  (e.g. `"wasp-12b"`, `"WASP-12 b"` both resolve). Raise `KeyError`/`ValueError`
  with a clear message if not found.
- Map archive columns → batman params:

  | batman param | archive column | fallback |
  |---|---|---|
  | `per` | `pl_orbper` | required — raise if NaN/missing |
  | `rp` (Rp/R\*) | `pl_ratror` | `pl_radj`·R_jup / (`st_rad`·R_sun) |
  | `a` (a/R\*) | `pl_ratdor` | `pl_orbsmax`[AU] / (`st_rad` converted to AU) |
  | `inc` (deg) | `pl_orbincl` | 90.0 |
  | `t0` | `pl_tranmid` | 0.0 |
  | `ecc` | `pl_orbeccen` | 0.0 |
  | `w` (deg) | `pl_orblper` | 90.0 |
  | `u`, `limb_dark` | not in archive | constructor args (defaults shown) |

- Unit conversions use `astropy.units`/`astropy.constants` (R_jup, R_sun, AU),
  consistent with the existing `astro.py` usage.
- Any archive value that is NaN falls back as in the table above; `per` is the
  only strictly required field.

### Exports

Add `download_exoplanet_archive` and `load_exoplanet_archive` to
`src/wcc_etc/__init__.py` `__all__` and import them. `from_planet` rides along
on the already-exported `TransitModel`.

## Caveats (to document in docstrings)

- `t0` from `pl_tranmid` is an absolute BJD. For a relative-time light curve,
  pass a `time` array spanning the real epoch, or override `t0=0`.
- PSCompPars values are a composite of literature sources and may mix
  references across parameters; this is the archive's documented behavior.

## Testing

Unit tests, no network:

- Small fixture CSV checked into `tests/` with a few real-ish rows:
  - WASP-12 b — complete row (uses `pl_ratror`, `pl_ratdor` directly).
  - A row missing `pl_ratror` and `pl_ratdor` — exercises both fallback paths.
  - A row with NaN `ecc`/`w`/`inc` — exercises NaN→default handling.
- Cover:
  - Column mapping produces expected batman params for the complete row.
  - `rp` fallback from `pl_radj`/`st_rad`.
  - `a` fallback from `pl_orbsmax`/`st_rad`.
  - NaN ecc/w/inc → defaults (0.0, 90.0, 90.0).
  - Name lookup tolerance (case/spacing).
  - Unknown planet → clear error.
  - `load_exoplanet_archive()` on a missing cache → `FileNotFoundError` with hint.
- The actual `astroquery` download is exercised only behind a network/
  availability guard (skipped by default), matching the optional-`batman`
  pattern.
