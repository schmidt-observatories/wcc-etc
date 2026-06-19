# NASA Exoplanet Archive Transit Params Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users download a local CSV of NASA Exoplanet Archive planet parameters and build a `TransitModel` for any confirmed planet by name.

**Architecture:** A new `exoarchive.py` module downloads/caches the PSCompPars table to a local CSV (astroquery, lazily imported). `TransitModel` gains a `from_planet` classmethod that reads a row from that table and maps archive columns to batman parameters with fallbacks.

**Tech Stack:** Python, pandas, astropy (units/constants), astroquery (optional), batman (optional), pytest.

## Global Constraints

- astroquery is an **optional** dependency under extra `exoarchive`; the package must import fine without it. Lazy-import it only inside functions that call it.
- Lazy-import error hint string (verbatim): `download_exoplanet_archive requires 'astroquery'. Install it with: pip install wcc-etc[exoarchive]`
- Default cache path: `~/.wcc_etc/exoplanet_archive_pscomppars.csv` (expand `~`, create parent dir as needed).
- Persisted/selected columns: `pl_name`, `hostname`, `pl_orbper`, `pl_ratror`, `pl_ratdor`, `pl_orbincl`, `pl_tranmid`, `pl_orbeccen`, `pl_orblper`, `pl_radj`, `pl_orbsmax`, `st_rad`.
- Unit conversions via `astropy.units`/`astropy.constants` (astropy is already a core dependency).
- Follow the existing optional-dependency test pattern: `pytest.importorskip(...)` for batman/astroquery; no network in default test runs.

---

### Task 1: Cache path helper + `load_exoplanet_archive` + pyproject extra

**Files:**
- Create: `src/wcc_etc/exoarchive.py`
- Modify: `pyproject.toml` (optional-dependencies)
- Test: `tests/test_exoarchive_load.py`

**Interfaces:**
- Produces:
  - `default_archive_path() -> pathlib.Path` — returns `Path("~/.wcc_etc/exoplanet_archive_pscomppars.csv").expanduser()`.
  - `ARCHIVE_COLUMNS: list[str]` — the selected/persisted column list (verbatim from Global Constraints).
  - `load_exoplanet_archive(path=None) -> pandas.DataFrame` — reads the cached CSV; raises `FileNotFoundError` if absent.

- [ ] **Step 1: Add the optional dependency to pyproject.toml**

Modify the `[project.optional-dependencies]` block (currently has only `lightcurve`) so it reads:

```toml
[project.optional-dependencies]
lightcurve = ["batman-package"]
exoarchive = ["astroquery"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_exoarchive_load.py
import pandas as pd
import pytest

from wcc_etc.exoarchive import (
    ARCHIVE_COLUMNS,
    default_archive_path,
    load_exoplanet_archive,
)


def test_default_path_is_under_home_and_expanded():
    p = default_archive_path()
    assert p.name == "exoplanet_archive_pscomppars.csv"
    assert "~" not in str(p)
    assert ".wcc_etc" in str(p)


def test_archive_columns_contains_required_fields():
    for col in ("pl_name", "pl_orbper", "pl_ratror", "pl_ratdor", "st_rad"):
        assert col in ARCHIVE_COLUMNS


def test_load_reads_existing_csv(tmp_path):
    csv = tmp_path / "arch.csv"
    pd.DataFrame({"pl_name": ["WASP-12 b"], "pl_orbper": [1.09]}).to_csv(
        csv, index=False
    )
    df = load_exoplanet_archive(path=csv)
    assert list(df["pl_name"]) == ["WASP-12 b"]


def test_load_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_exoplanet_archive"):
        load_exoplanet_archive(path=tmp_path / "nope.csv")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_exoarchive_load.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wcc_etc.exoarchive'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/wcc_etc/exoarchive.py
"""Download and load NASA Exoplanet Archive planet parameters.

`download_exoplanet_archive` fetches the PSCompPars table (one complete row
per confirmed planet) to a local CSV cache; `load_exoplanet_archive` reads
that cache. `TransitModel.from_planet` (in lightcurve.py) consumes the result
to build transit light-curve models for real planets.
"""
from pathlib import Path

import pandas as pd

#: Identifier + transit columns we select and persist.
ARCHIVE_COLUMNS = [
    "pl_name", "hostname", "pl_orbper", "pl_ratror", "pl_ratdor",
    "pl_orbincl", "pl_tranmid", "pl_orbeccen", "pl_orblper",
    "pl_radj", "pl_orbsmax", "st_rad",
]

_ASTROQUERY_HINT = (
    "download_exoplanet_archive requires 'astroquery'. "
    "Install it with: pip install wcc-etc[exoarchive]"
)


def default_archive_path():
    """Default cache location for the downloaded PSCompPars CSV."""
    return Path("~/.wcc_etc/exoplanet_archive_pscomppars.csv").expanduser()


def load_exoplanet_archive(path=None):
    """Read the cached PSCompPars CSV into a DataFrame.

    Does not hit the network. Raises FileNotFoundError (pointing at
    download_exoplanet_archive) if the cache file does not exist.
    """
    path = Path(path) if path is not None else default_archive_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No archive cache at {path}. "
            "Run download_exoplanet_archive() first."
        )
    return pd.read_csv(path)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_exoarchive_load.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/wcc_etc/exoarchive.py tests/test_exoarchive_load.py
git commit -m "feat: add exoarchive cache loader and exoarchive optional extra"
```

---

### Task 2: `download_exoplanet_archive` (cache + lazy astroquery)

**Files:**
- Modify: `src/wcc_etc/exoarchive.py`
- Test: `tests/test_exoarchive_download.py`

**Interfaces:**
- Consumes: `default_archive_path`, `ARCHIVE_COLUMNS`, `_ASTROQUERY_HINT`, `load_exoplanet_archive` (Task 1).
- Produces: `download_exoplanet_archive(path=None, refresh=False) -> pandas.DataFrame` — returns cached DataFrame if file exists and not `refresh`; otherwise queries astroquery, writes CSV, returns DataFrame.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exoarchive_download.py
import sys
import types

import pandas as pd
import pytest

import wcc_etc.exoarchive as exo


def test_download_returns_cache_when_present_without_network(tmp_path, monkeypatch):
    csv = tmp_path / "arch.csv"
    pd.DataFrame({"pl_name": ["WASP-12 b"]}).to_csv(csv, index=False)

    # Any network attempt would import astroquery; make that explode so the
    # test proves the cache path is taken instead.
    def _boom(*a, **k):
        raise AssertionError("network/astroquery must not be touched on cache hit")

    monkeypatch.setattr(exo, "_query_pscomppars", _boom)
    df = exo.download_exoplanet_archive(path=csv, refresh=False)
    assert list(df["pl_name"]) == ["WASP-12 b"]


def test_download_writes_cache_using_injected_query(tmp_path, monkeypatch):
    csv = tmp_path / "sub" / "arch.csv"  # parent dir does not exist yet
    fake = pd.DataFrame({c: [0] for c in exo.ARCHIVE_COLUMNS})
    fake["pl_name"] = ["WASP-12 b"]
    monkeypatch.setattr(exo, "_query_pscomppars", lambda: fake)

    df = exo.download_exoplanet_archive(path=csv, refresh=True)
    assert csv.exists()
    assert list(df["pl_name"]) == ["WASP-12 b"]
    # Round-trips through the on-disk cache.
    assert list(exo.load_exoplanet_archive(path=csv)["pl_name"]) == ["WASP-12 b"]


def test_query_raises_hint_without_astroquery(monkeypatch):
    # Simulate astroquery not being importable.
    monkeypatch.setitem(sys.modules, "astroquery.ipac.nexsci.nasa_exoplanet_archive", None)
    with pytest.raises(ImportError, match="pip install wcc-etc\\[exoarchive\\]"):
        exo._query_pscomppars()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_exoarchive_download.py -v`
Expected: FAIL with `AttributeError: module 'wcc_etc.exoarchive' has no attribute 'download_exoplanet_archive'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/wcc_etc/exoarchive.py`:

```python
def _query_pscomppars():
    """Fetch the PSCompPars table via astroquery as a pandas DataFrame.

    Isolated (and monkeypatchable) so download_exoplanet_archive's caching
    logic can be tested without the network.
    """
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import (
            NasaExoplanetArchive,
        )
    except ImportError as exc:  # pragma: no cover - exercised via hint test
        raise ImportError(_ASTROQUERY_HINT) from exc

    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars", select=",".join(ARCHIVE_COLUMNS)
    )
    return table.to_pandas()


def download_exoplanet_archive(path=None, refresh=False):
    """Download the PSCompPars table to a local CSV cache and return it.

    If the cache file exists and ``refresh`` is False, load it from disk
    instead of querying the archive. Otherwise query astroquery, write the
    CSV (creating parent directories), and return the DataFrame.
    """
    path = Path(path) if path is not None else default_archive_path()
    if path.exists() and not refresh:
        return load_exoplanet_archive(path=path)

    df = _query_pscomppars()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
```

Note: the `monkeypatch.setitem(sys.modules, ..., None)` in the hint test makes Python raise `ImportError` on the `from astroquery...` line, exercising the hint branch without uninstalling astroquery.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_exoarchive_download.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/exoarchive.py tests/test_exoarchive_download.py
git commit -m "feat: add download_exoplanet_archive with local CSV caching"
```

---

### Task 3: `TransitModel.from_planet` column mapping + fallbacks

**Files:**
- Modify: `src/wcc_etc/lightcurve.py` (add classmethod to `TransitModel`)
- Test: `tests/test_transitmodel_from_planet.py`

**Interfaces:**
- Consumes: `load_exoplanet_archive` (Task 1); `TransitModel.__init__(t0, per, rp, a, inc, ecc, w, limb_dark, u)` (existing).
- Produces: `TransitModel.from_planet(name, df=None, limb_dark="quadratic", u=(0.1, 0.3)) -> TransitModel`.

Mapping (archive column → batman param), with fallbacks:
- `per` ← `pl_orbper` (required; raise `ValueError` if NaN/missing)
- `rp` ← `pl_ratror`; fallback `pl_radj * 0.10276268506540175 / st_rad` (R_jup in R_sun)
- `a` ← `pl_ratdor`; fallback `pl_orbsmax * 215.03215567054764 / st_rad` (AU in R_sun)
- `inc` ← `pl_orbincl`; fallback `90.0`
- `t0` ← `pl_tranmid`; fallback `0.0`
- `ecc` ← `pl_orbeccen`; fallback `0.0`
- `w` ← `pl_orblper`; fallback `90.0`

Name match: normalize by lowercasing and removing spaces (`"WASP-12 b"`, `"wasp-12b"` both resolve).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transitmodel_from_planet.py
import numpy as np
import pandas as pd
import pytest

from wcc_etc.lightcurve import TransitModel

# R_jup in R_sun and 1 AU in R_sun (astropy-derived constants).
RJUP_RSUN = 0.10276268506540175
AU_RSUN = 215.03215567054764


def _df():
    return pd.DataFrame([
        # Complete row: ratio columns present.
        dict(pl_name="WASP-12 b", pl_orbper=1.0914, pl_ratror=0.117,
             pl_ratdor=3.04, pl_orbincl=83.3, pl_tranmid=2456305.46,
             pl_orbeccen=0.0, pl_orblper=90.0, pl_radj=1.9, pl_orbsmax=0.0234,
             st_rad=1.66),
        # Missing ratio columns -> exercises rp and a fallbacks.
        dict(pl_name="HD 209458 b", pl_orbper=3.5247, pl_ratror=np.nan,
             pl_ratdor=np.nan, pl_orbincl=86.7, pl_tranmid=2451370.0,
             pl_orbeccen=np.nan, pl_orblper=np.nan, pl_radj=1.38,
             pl_orbsmax=0.0475, st_rad=1.19),
    ])


def test_from_planet_uses_ratio_columns_directly():
    m = TransitModel.from_planet("WASP-12 b", df=_df())
    assert m.per == pytest.approx(1.0914)
    assert m.rp == pytest.approx(0.117)
    assert m.a == pytest.approx(3.04)
    assert m.inc == pytest.approx(83.3)
    assert m.t0 == pytest.approx(2456305.46)


def test_from_planet_rp_and_a_fallbacks():
    m = TransitModel.from_planet("HD 209458 b", df=_df())
    assert m.rp == pytest.approx(1.38 * RJUP_RSUN / 1.19)
    assert m.a == pytest.approx(0.0475 * AU_RSUN / 1.19)


def test_from_planet_nan_orbital_defaults():
    m = TransitModel.from_planet("HD 209458 b", df=_df())
    assert m.ecc == pytest.approx(0.0)
    assert m.w == pytest.approx(90.0)


def test_from_planet_name_is_case_and_space_insensitive():
    m = TransitModel.from_planet("wasp-12b", df=_df())
    assert m.per == pytest.approx(1.0914)


def test_from_planet_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        TransitModel.from_planet("Kepler-999 z", df=_df())


def test_from_planet_missing_period_raises():
    df = _df()
    df.loc[df["pl_name"] == "WASP-12 b", "pl_orbper"] = np.nan
    with pytest.raises(ValueError, match="pl_orbper"):
        TransitModel.from_planet("WASP-12 b", df=df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transitmodel_from_planet.py -v`
Expected: FAIL with `AttributeError: type object 'TransitModel' has no attribute 'from_planet'`

- [ ] **Step 3: Write minimal implementation**

Add to the `TransitModel` class in `src/wcc_etc/lightcurve.py` (after `__init__`). Add `import numpy as np` already present at top of file.

```python
    @classmethod
    def from_planet(cls, name, df=None, limb_dark="quadratic", u=(0.1, 0.3)):
        """Build a TransitModel from a NASA Exoplanet Archive row.

        Looks up ``name`` in the PSCompPars table (``df``, or the local cache
        via load_exoplanet_archive if None) and maps archive columns to
        batman parameters. ``pl_orbper`` is required; other fields fall back
        to circular-orbit / direct-ratio defaults when missing. Limb-darkening
        is not in the archive, so it stays a user argument.

        Note: ``t0`` is taken from ``pl_tranmid`` (absolute BJD). For a
        relative-time light curve, pass a ``time`` array spanning that epoch
        or set ``t0=0`` after construction.
        """
        if df is None:
            from .exoarchive import load_exoplanet_archive
            df = load_exoplanet_archive()

        def _norm(s):
            return str(s).lower().replace(" ", "")

        target = _norm(name)
        matches = df[df["pl_name"].map(_norm) == target]
        if len(matches) == 0:
            raise ValueError(f"Planet {name!r} not found in archive table")
        row = matches.iloc[0]

        def _val(col, default):
            v = row.get(col)
            return default if v is None or (isinstance(v, float) and np.isnan(v)) \
                else float(v)

        per = row.get("pl_orbper")
        if per is None or (isinstance(per, float) and np.isnan(per)):
            raise ValueError(
                f"Planet {name!r} has no pl_orbper; cannot build a transit model"
            )
        per = float(per)

        # R_jup and 1 AU expressed in R_sun (astropy constants).
        import astropy.units as units
        import astropy.constants as const
        rjup_rsun = float((const.R_jup / const.R_sun).decompose().value)
        au_rsun = float((1 * units.au).to(units.R_sun).value)

        rp = _val("pl_ratror", np.nan)
        if np.isnan(rp):
            rp = _val("pl_radj", np.nan) * rjup_rsun / _val("st_rad", np.nan)

        a = _val("pl_ratdor", np.nan)
        if np.isnan(a):
            a = _val("pl_orbsmax", np.nan) * au_rsun / _val("st_rad", np.nan)

        return cls(
            t0=_val("pl_tranmid", 0.0),
            per=per,
            rp=rp,
            a=a,
            inc=_val("pl_orbincl", 90.0),
            ecc=_val("pl_orbeccen", 0.0),
            w=_val("pl_orblper", 90.0),
            limb_dark=limb_dark,
            u=u,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_transitmodel_from_planet.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/lightcurve.py tests/test_transitmodel_from_planet.py
git commit -m "feat: add TransitModel.from_planet archive-to-batman mapping"
```

---

### Task 4: Public exports

**Files:**
- Modify: `src/wcc_etc/__init__.py`
- Test: `tests/test_exoarchive_exports.py`

**Interfaces:**
- Consumes: `download_exoplanet_archive`, `load_exoplanet_archive` (Tasks 1–2).
- Produces: top-level `wcc_etc.download_exoplanet_archive`, `wcc_etc.load_exoplanet_archive`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exoarchive_exports.py
import wcc_etc


def test_exoarchive_functions_exported():
    assert hasattr(wcc_etc, "download_exoplanet_archive")
    assert hasattr(wcc_etc, "load_exoplanet_archive")
    assert "download_exoplanet_archive" in wcc_etc.__all__
    assert "load_exoplanet_archive" in wcc_etc.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_exoarchive_exports.py -v`
Expected: FAIL with `AssertionError` (attribute missing)

- [ ] **Step 3: Write minimal implementation**

In `src/wcc_etc/__init__.py`, add the two names to `__all__` (alongside the lightcurve entries) and add an import line near the existing `from .lightcurve import ...`:

```python
from .exoarchive import download_exoplanet_archive, load_exoplanet_archive
```

And add to the `__all__` list:

```python
    "download_exoplanet_archive", "load_exoplanet_archive",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_exoarchive_exports.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/__init__.py tests/test_exoarchive_exports.py
git commit -m "feat: export exoplanet archive download/load helpers"
```

---

## Self-Review Notes

- **Spec coverage:** download/cache/refresh → Task 2; load + missing-cache error → Task 1; `from_planet` mapping + all fallbacks + NaN defaults + name tolerance + unknown-planet error → Task 3; optional astroquery dep + lazy import hint → Tasks 1–2; exports → Task 4; caveats → docstring in Task 3. The skip-guarded live download is intentionally left as an optional manual check (not a default test) per the optional-dependency pattern; no separate task needed.
- **Type consistency:** `load_exoplanet_archive(path=None)`, `download_exoplanet_archive(path=None, refresh=False)`, and `TransitModel.from_planet(name, df=None, limb_dark, u)` signatures are consistent across producing/consuming tasks.
- **Constants:** `RJUP_RSUN=0.10276268506540175` and `AU_RSUN=215.03215567054764` in the Task 3 test match the astropy-derived values computed at runtime in the implementation.
