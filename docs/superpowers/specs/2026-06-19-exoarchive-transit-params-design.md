# NASA Exoplanet Archive → Transit Params (lazuli-transit package)

**Date:** 2026-06-19
**Status:** Design (revised for standalone `lazuli-transit` package)

## Goal

Make it easy to download a local CSV of NASA Exoplanet Archive planet
parameters and build a transit light-curve model for any confirmed planet,
**and** factor the instrument-agnostic transit machinery into a standalone,
separately distributable package (`lazuli-transit`) so it can later move to its
own git repo and be reused by other Lazuli instruments (e.g. the IFS) without
depending on the WCC ETC.

## Why a separate package

The transit core is pure physics + data: it depends only on numpy, astropy,
pandas, and (optionally) batman/astroquery. The WCC-specific piece —
`LightCurveSimulator`, which needs a WCC `Simulation` to get photometric SNR —
is the only thing coupling light curves to the ETC. Splitting along that seam
gives:

- **Migration ready:** `lazuli-transit` is its own installable distribution in
  a `packages/` subdirectory with its own `pyproject.toml`. Moving it to a new
  repo later is a directory move + publish; `wcc-etc` keeps depending on it.
- **IFS reuse:** the IFS imports `lazuli_transit` directly and writes its own
  instrument simulator against the same `FluxModel` interface. It never imports
  `wcc_etc`.

## Scope

- Python API only (no CLI, no notebook entry point).
- New standalone package `lazuli-transit` containing the flux/transit models
  and the exoplanet-archive download/load helpers.
- Bulk download of the **PSCompPars** table (one complete, self-consistent row
  per confirmed planet) to a local CSV cache.
- `TransitModel.from_planet(name)` maps archive columns to batman parameters
  with sensible fallbacks.
- `wcc_etc` is rewired to depend on `lazuli-transit` and re-export the moved
  names for backward compatibility.

Out of scope: per-planet TAP queries; the full PS table; limb-darkening
lookup; moving `LightCurve`/`LightCurveSimulator` out of `wcc_etc`; publishing
to PyPI; the IFS simulator itself.

## Repository layout (monorepo)

```
wcc-etc/                                  # repo root
├── pyproject.toml                        # wcc_etc; now depends on lazuli-transit
├── src/wcc_etc/
│   ├── lightcurve.py                     # KEEPS LightCurve, LightCurveSimulator;
│   │                                     #   re-exports FluxModel, TransitModel
│   └── __init__.py                       # re-exports moved names (back-compat)
├── packages/
│   └── lazuli-transit/
│       ├── pyproject.toml                # name = "lazuli-transit"
│       ├── README.md
│       └── src/lazuli_transit/
│           ├── __init__.py               # exports models + archive helpers
│           ├── models.py                 # FluxModel, TransitModel (+ from_planet)
│           └── archive.py                # download/load exoplanet archive
│       └── tests/
│           ├── test_models.py
│           ├── test_archive.py
│           └── test_portability.py       # asserts no wcc_etc import
└── tests/                                # existing wcc_etc tests (unchanged)
```

Distribution name: **`lazuli-transit`**. Import name: **`lazuli_transit`**.

## Packaging & dependencies

### `packages/lazuli-transit/pyproject.toml`

- Build backend: setuptools (matches `wcc_etc`), src-layout, `packages =
  ["lazuli_transit"]`, `package-dir = {"" = "src"}`.
- `requires-python = ">=3.11"` (matches `wcc_etc`).
- Core dependencies: `numpy`, `astropy`, `pandas`.
- Optional extras:
  - `batman = ["batman-package"]` — needed for `TransitModel.relative_flux`.
  - `archive = ["astroquery"]` — needed for `download_exoplanet_archive`.

### `wcc_etc` `pyproject.toml` changes

- Add `"lazuli-transit"` to `[project].dependencies`.
- Redefine the existing optional extras to delegate:
  ```toml
  [project.optional-dependencies]
  lightcurve = ["lazuli-transit[batman]"]
  exoarchive = ["lazuli-transit[archive]"]
  ```
  (Existing `pip install wcc-etc[lightcurve]` keeps working; batman now arrives
  transitively.)

### Local install order (monorepo)

Because `lazuli-transit` is not on PyPI, it must be installed from the local
path **before** `wcc_etc`:

```bash
pip install -e ./packages/lazuli-transit
pip install -e .
```

CI (`.github/workflows/tests.yml`, `deploy-docs.yml`) and `.readthedocs.yaml`
are updated to add the `pip install -e ./packages/lazuli-transit` step before
installing `wcc_etc`.

## Components

### `lazuli_transit/models.py`

Holds the classes currently in `wcc_etc/lightcurve.py`:

- `FluxModel` — base class, `relative_flux(time)` → normalized flux. Unchanged.
- `TransitModel(FluxModel)` — batman-backed transit model. Unchanged
  constructor and `relative_flux`/`_params` (lazy batman import + hint), **plus**
  the new `from_planet` classmethod below.

Lazy batman import hint string (verbatim, kept from current code):
> `TransitModel requires the 'batman' package. Install it with: pip install lazuli-transit[batman]`

#### `TransitModel.from_planet(name, df=None, require_transit=True, limb_dark="quadratic", u=(0.1, 0.3)) -> TransitModel`

- If `df is None`, call `load_exoplanet_archive()` from `.archive` (raises with
  a hint if the cache is missing — the constructor never silently hits the
  network).
- Look up the row by `pl_name`, tolerant of case and internal spacing
  (`"WASP-12 b"`, `"wasp-12b"` both resolve). Raise `ValueError` ("not found")
  if absent.
- If `require_transit` (default True) and the row's `tran_flag == 0`, raise
  `ValueError` ("not flagged as transiting") — a transit model is not
  meaningful for a non-transiting planet. Pass `require_transit=False` to build
  one anyway. When `tran_flag` is absent/NaN, the check is skipped.
- Map archive columns → batman params:

  | batman param | archive column | fallback |
  |---|---|---|
  | `per` | `pl_orbper` | required — raise `ValueError` if NaN/missing |
  | `rp` (Rp/R\*) | `pl_ratror` | `pl_radj`·R_jup / (`st_rad`·R_sun) |
  | `a` (a/R\*) | `pl_ratdor` | `pl_orbsmax`[AU] / (`st_rad` in R_sun) |
  | `inc` (deg) | `pl_orbincl` | 90.0 |
  | `t0` | `pl_tranmid` | 0.0 |
  | `ecc` | `pl_orbeccen` | 0.0 |
  | `w` (deg) | `pl_orblper` | 90.0 |
  | `u`, `limb_dark` | not in archive | constructor args (defaults shown) |

- Unit factors via `astropy`: R_jup/R_sun ≈ 0.10276268506540175;
  1 AU in R_sun ≈ 215.03215567054764.
- `u` accepted as any sequence; stored as a list (matches current constructor).

### `lazuli_transit/archive.py`

- `ARCHIVE_COLUMNS: list[str]` — `pl_name`, `hostname`, `pl_orbper`,
  `pl_ratror`, `pl_ratdor`, `pl_orbincl`, `pl_tranmid`, `pl_orbeccen`,
  `pl_orblper`, `pl_radj`, `pl_orbsmax`, `tran_flag`, `st_rad`, `st_teff`,
  `sy_gaiamag`, `sy_vmag`, `sy_tmag`, `sy_jmag`, `sy_hmag`, `sy_kmag`.
  Brightness bands are the high-completeness (~95%) ones — Gaia G, Johnson V,
  TESS T, 2MASS J/H/K; the Sloan ugriz bands the WCC filters map to are only
  ~52–56% populated and are intentionally excluded. `tran_flag` is fully
  populated and drives the `from_planet` transit check.
- `default_archive_path() -> pathlib.Path` — `~/.lazuli_transit/exoplanet_archive_pscomppars.csv`
  (expanded).
- `load_exoplanet_archive(path=None) -> pandas.DataFrame` — reads the cached
  CSV; raises `FileNotFoundError` (pointing at `download_exoplanet_archive`) if
  absent. No network.
- `_query_pscomppars() -> pandas.DataFrame` — lazy-imports astroquery and
  queries PSCompPars; isolated/monkeypatchable so caching is testable without
  the network. Hint on `ImportError`:
  > `download_exoplanet_archive requires 'astroquery'. Install it with: pip install lazuli-transit[archive]`
- `download_exoplanet_archive(path=None, refresh=False) -> pandas.DataFrame` —
  returns the cached DataFrame if the file exists and not `refresh`; otherwise
  calls `_query_pscomppars`, writes the CSV (creating parent dirs), returns it.

### `lazuli_transit/__init__.py`

Exports `FluxModel`, `TransitModel`, `download_exoplanet_archive`,
`load_exoplanet_archive`, `default_archive_path`, `ARCHIVE_COLUMNS`.

### `wcc_etc/lightcurve.py` (rewired)

- Remove `FluxModel`/`TransitModel` definitions; instead:
  `from lazuli_transit import FluxModel, TransitModel` at top so existing
  `from wcc_etc.lightcurve import FluxModel, TransitModel` keeps working.
- Keep `LightCurve` and `LightCurveSimulator` as-is (the WCC adapter). They
  continue to consume `FluxModel` via duck-typing (`model.relative_flux(time)`).

### `wcc_etc/__init__.py` (rewired)

- Keep `FluxModel`, `TransitModel`, `LightCurveSimulator`, `LightCurve` in
  `__all__` (now sourced via `lightcurve.py` re-export).
- Add and export `download_exoplanet_archive`, `load_exoplanet_archive`
  (imported from `lazuli_transit`) for ergonomics.

## Portability guard

A test in `lazuli-transit` asserts the package never imports `wcc_etc`: walk
every `.py` under `src/lazuli_transit/` and assert none contains `wcc_etc`
(no `import wcc_etc`, no `from wcc_etc`). This keeps the migration seam honest.

## Caveats (documented in docstrings)

- `t0` from `pl_tranmid` is absolute BJD. For a relative-time light curve, pass
  a `time` array spanning that epoch or set `t0=0` after construction.
- PSCompPars values are a composite of literature sources and may mix
  references across parameters (the archive's documented behavior).

## Testing

All default tests run with no network.

**`packages/lazuli-transit/tests/`:**
- `test_models.py`:
  - `from_planet` maps a complete fixture row (ratio columns present) correctly.
  - `rp` fallback from `pl_radj`/`st_rad`.
  - `a` fallback from `pl_orbsmax`/`st_rad`.
  - NaN `ecc`/`w`/`inc` → defaults (0.0, 90.0, 90.0).
  - name lookup is case/space-insensitive.
  - unknown planet → `ValueError("not found")`.
  - missing `pl_orbper` → `ValueError` mentioning `pl_orbper`.
  - (batman-guarded) `relative_flux` matches batman directly — mirrors the
    existing `test_lightcurve_model.py` test, moved here.
- `test_archive.py`:
  - `default_archive_path` is expanded and under `.lazuli_transit`.
  - `ARCHIVE_COLUMNS` contains required fields.
  - `load_exoplanet_archive` reads an existing CSV.
  - missing cache → `FileNotFoundError` with hint.
  - `download_exoplanet_archive` returns cache on hit without touching the
    network (monkeypatch `_query_pscomppars` to explode).
  - `download_exoplanet_archive(refresh=True)` writes the CSV using an injected
    fake query and round-trips through disk.
  - `_query_pscomppars` raises the install hint when astroquery is absent
    (simulated via `sys.modules`).
- `test_portability.py`: no `wcc_etc` reference in package source.

**`tests/` (wcc_etc):**
- Existing `test_lightcurve_*` tests continue to pass via the re-exports
  (the batman/relative_flux test may be removed here once it lives in
  `lazuli-transit`, to avoid duplication — but keeping it is harmless).
- `test_exoarchive_exports.py`: `wcc_etc.download_exoplanet_archive` and
  `wcc_etc.load_exoplanet_archive` exist and are in `__all__`.

The live astroquery download is exercised only behind a network/availability
guard (skipped by default), matching the optional-dependency pattern.
