# lazuli-transit Package + Exoplanet Archive Transit Params — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract instrument-agnostic transit models into a standalone, separately distributable `lazuli-transit` package, and add NASA Exoplanet Archive download/load + `TransitModel.from_planet(name)`.

**Architecture:** A new `packages/lazuli-transit/` monorepo package (its own `pyproject.toml`, src-layout, import name `lazuli_transit`) holds `FluxModel`/`TransitModel` and the archive helpers, with zero imports from `wcc_etc`. `wcc_etc` depends on `lazuli-transit` and re-exports the moved names; its WCC-specific `LightCurveSimulator`/`LightCurve` stay put.

**Tech Stack:** Python ≥3.11, setuptools, numpy, astropy, pandas, batman (optional), astroquery (optional), pytest.

## Global Constraints

- Distribution name **`lazuli-transit`**, import name **`lazuli_transit`**.
- `lazuli_transit` source must **never** import `wcc_etc` (enforced by a test).
- Optional deps are lazy-imported with these verbatim hints:
  - batman: `TransitModel requires the 'batman' package. Install it with: pip install lazuli-transit[batman]`
  - astroquery: `download_exoplanet_archive requires 'astroquery'. Install it with: pip install lazuli-transit[archive]`
- Default cache path: `~/.lazuli_transit/exoplanet_archive_pscomppars.csv` (expand `~`, create parent dir as needed).
- Persisted/selected columns: `pl_name`, `hostname`, `pl_orbper`, `pl_ratror`, `pl_ratdor`, `pl_orbincl`, `pl_tranmid`, `pl_orbeccen`, `pl_orblper`, `pl_radj`, `pl_orbsmax`, `st_rad`.
- Local install order (monorepo): `pip install -e ./packages/lazuli-transit` **before** `pip install -e .`.
- No-network default tests; live download only behind a skip guard.

---

### Task 1: Scaffold the `lazuli-transit` package + wire the monorepo

**Files:**
- Create: `packages/lazuli-transit/pyproject.toml`
- Create: `packages/lazuli-transit/README.md`
- Create: `packages/lazuli-transit/src/lazuli_transit/__init__.py`
- Create: `packages/lazuli-transit/tests/test_portability.py`
- Modify: `pyproject.toml` (root — add dependency + redefine extras)
- Modify: `.github/workflows/tests.yml`
- Modify: `.github/workflows/deploy-docs.yml`
- Modify: `.readthedocs.yaml`

**Interfaces:**
- Produces: an importable `lazuli_transit` package (empty for now) installed editable, and a CI/build that installs it before `wcc_etc`.

- [ ] **Step 1: Create the package pyproject**

```toml
# packages/lazuli-transit/pyproject.toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lazuli-transit"
version = "0.1.0"
description = "Instrument-agnostic exoplanet transit models and NASA Exoplanet Archive helpers for Lazuli."
readme = "README.md"
requires-python = ">=3.11"
authors = [
  {name = "Gudmundur Stefansson", email = "gstefansson@schmidtsciences.org"},
]
dependencies = [
  "numpy",
  "astropy",
  "pandas",
]

[project.optional-dependencies]
batman = ["batman-package"]
archive = ["astroquery"]

[tool.setuptools]
packages = ["lazuli_transit"]
package-dir = { "" = "src" }

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create README and the package `__init__`**

```markdown
# lazuli-transit

Instrument-agnostic exoplanet transit models and NASA Exoplanet Archive
helpers for the Lazuli mission. Depends only on numpy/astropy/pandas (plus
optional `batman` and `astroquery`); it has no dependency on any specific
instrument package, so WCC, the IFS, and others can all build light-curve
simulators on top of it.
```

```python
# packages/lazuli-transit/src/lazuli_transit/__init__.py
"""Instrument-agnostic transit models + NASA Exoplanet Archive helpers."""

__version__ = "0.1.0"

__all__ = []
```

- [ ] **Step 3: Write the portability test**

```python
# packages/lazuli-transit/tests/test_portability.py
from pathlib import Path

import lazuli_transit


def test_package_imports():
    assert lazuli_transit.__version__


def test_no_wcc_etc_dependency_in_source():
    pkg_dir = Path(lazuli_transit.__file__).parent
    offenders = [p.name for p in pkg_dir.rglob("*.py") if "wcc_etc" in p.read_text()]
    assert not offenders, f"lazuli_transit must not reference wcc_etc: {offenders}"
```

- [ ] **Step 4: Install the package editable and run its test**

Run:
```bash
pip install -e ./packages/lazuli-transit
python -m pytest packages/lazuli-transit/tests -v
```
Expected: PASS (2 passed).

- [ ] **Step 5: Wire `wcc_etc` to depend on lazuli-transit**

In the root `pyproject.toml`, add `"lazuli-transit"` to `[project].dependencies` (append after `"photutils",`) and replace the optional-dependencies block:

```toml
[project.optional-dependencies]
lightcurve = ["lazuli-transit[batman]"]
exoarchive = ["lazuli-transit[archive]"]
```

- [ ] **Step 6: Update CI test workflow**

In `.github/workflows/tests.yml`, change the install + run steps to:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest
          pip install -e ./packages/lazuli-transit
          pip install -e .

      - name: Run tests
        run: |
          pytest
          pytest packages/lazuli-transit/tests
```

- [ ] **Step 7: Update docs build workflow + readthedocs**

In `.github/workflows/deploy-docs.yml`, in the "Install the package and documentation dependencies" step, add the lazuli-transit install **before** `pip install -e .`:

```yaml
          python -m pip install --upgrade pip
          pip install -e ./packages/lazuli-transit          # transit core (wcc_etc depends on it)
          pip install -e .                                  # so autodoc can import wcc_etc
          pip install -r docs/sphinx/requirements.txt
```

In `.readthedocs.yaml`, change the `python.install` list to install lazuli-transit first:

```yaml
python:
  install:
    - requirements: docs/sphinx/requirements.txt
    - method: pip
      path: packages/lazuli-transit
    - method: pip
      path: .
```

- [ ] **Step 8: Verify wcc_etc still installs and imports with the new dependency**

Run:
```bash
pip install -e .
python -c "import wcc_etc; print(wcc_etc.__version__)"
python -m pytest -q
```
Expected: install succeeds (lazuli-transit already satisfied from Step 4), import works, existing suite passes.

- [ ] **Step 9: Commit**

```bash
git add packages/lazuli-transit pyproject.toml .github/workflows/tests.yml .github/workflows/deploy-docs.yml .readthedocs.yaml
git commit -m "build: scaffold standalone lazuli-transit package and wire monorepo"
```

---

### Task 2: Move `FluxModel`/`TransitModel` into `lazuli_transit.models`

**Files:**
- Create: `packages/lazuli-transit/src/lazuli_transit/models.py`
- Modify: `packages/lazuli-transit/src/lazuli_transit/__init__.py`
- Modify: `src/wcc_etc/lightcurve.py` (remove model classes, re-export instead)
- Test: `packages/lazuli-transit/tests/test_models.py`

**Interfaces:**
- Produces: `lazuli_transit.FluxModel`, `lazuli_transit.TransitModel` (constructor `TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0, ecc=0.0, w=90.0, limb_dark="quadratic", u=(0.1, 0.3))`, methods `relative_flux(time)`).
- Consumes (later, by `wcc_etc.lightcurve`): the re-exported names.

- [ ] **Step 1: Write the failing test**

```python
# packages/lazuli-transit/tests/test_models.py
import numpy as np
import pytest

from lazuli_transit import FluxModel, TransitModel


def test_fluxmodel_base_is_abstract():
    with pytest.raises(NotImplementedError):
        FluxModel().relative_flux(np.linspace(0, 1, 5))


def test_transit_relative_flux_matches_batman_directly():
    batman = pytest.importorskip("batman")
    t = np.linspace(-0.1, 0.1, 200)
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0,
                         ecc=0.0, w=90.0, limb_dark="quadratic", u=(0.1, 0.3))
    got = model.relative_flux(t)

    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, 1.0, 0.1, 15.0
    params.inc, params.ecc, params.w = 87.0, 0.0, 90.0
    params.limb_dark, params.u = "quadratic", [0.1, 0.3]
    expected = batman.TransitModel(params, t).light_curve(params)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/lazuli-transit/tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'FluxModel' from 'lazuli_transit'`

- [ ] **Step 3: Create `models.py`**

```python
# packages/lazuli-transit/src/lazuli_transit/models.py
"""Instrument-agnostic transit / flux models.

`FluxModel` maps time to normalized relative flux; `TransitModel` wraps the
`batman` package. These models carry no dependency on any specific instrument:
a WCC or IFS simulator consumes a FluxModel through its `relative_flux(time)`
interface.
"""
import numpy as np

_BATMAN_HINT = (
    "TransitModel requires the 'batman' package. "
    "Install it with: pip install lazuli-transit[batman]"
)


class FluxModel:
    """Base class: map time -> relative flux, normalized to 1.0 out of event."""

    def relative_flux(self, time):
        raise NotImplementedError


class TransitModel(FluxModel):
    """Exoplanet transit light-curve model backed by `batman`.

    Parameters mirror batman.TransitParams: t0 (center), per (period),
    rp (Rp/R*), a (a/R*), inc (deg), ecc, w (deg), limb_dark, u (coeffs).
    """

    def __init__(self, t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0,
                 ecc=0.0, w=90.0, limb_dark="quadratic", u=(0.1, 0.3)):
        self.t0, self.per, self.rp, self.a = t0, per, rp, a
        self.inc, self.ecc, self.w = inc, ecc, w
        self.limb_dark, self.u = limb_dark, list(u)

    def _params(self):
        try:
            import batman
        except ImportError as exc:  # pragma: no cover - exercised via hint
            raise ImportError(_BATMAN_HINT) from exc
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a = self.t0, self.per, self.rp, self.a
        p.inc, p.ecc, p.w = self.inc, self.ecc, self.w
        p.limb_dark, p.u = self.limb_dark, list(self.u)
        return batman, p

    def relative_flux(self, time):
        time = np.asarray(time, dtype=float)
        batman, params = self._params()
        return batman.TransitModel(params, time).light_curve(params)
```

- [ ] **Step 4: Export the models from the package `__init__`**

Replace `packages/lazuli-transit/src/lazuli_transit/__init__.py` with:

```python
"""Instrument-agnostic transit models + NASA Exoplanet Archive helpers."""
from .models import FluxModel, TransitModel

__version__ = "0.1.0"

__all__ = ["FluxModel", "TransitModel"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest packages/lazuli-transit/tests/test_models.py -v`
Expected: PASS (the batman test is skipped if batman is not installed).

- [ ] **Step 6: Rewire `wcc_etc/lightcurve.py` to re-export the moved classes**

In `src/wcc_etc/lightcurve.py`: delete the `_BATMAN_HINT` constant and the entire `FluxModel` and `TransitModel` class definitions. Replace the module docstring + imports at the top of the file with:

```python
# src/wcc_etc/lightcurve.py
"""WCC light-curve simulation.

`LightCurveSimulator` turns a FluxModel + a Simulation into a synthetic
observed WCC light curve, using one ETC photometric SNR measurement at
baseline brightness as the per-point error. The flux models themselves
(`FluxModel`, `TransitModel`) live in the instrument-agnostic `lazuli_transit`
package and are re-exported here for backward compatibility.
"""
import numpy as np

from lazuli_transit import FluxModel, TransitModel  # noqa: F401  (re-export)
```

Leave the `LightCurve` and `LightCurveSimulator` classes in the file exactly as they are.

- [ ] **Step 7: Verify existing wcc_etc light-curve tests still pass**

Run: `python -m pytest tests/test_lightcurve_model.py tests/test_lightcurve_simulator.py tests/test_lightcurve_exports.py -v`
Expected: PASS (these import `from wcc_etc.lightcurve import ...` and resolve via the re-export).

- [ ] **Step 8: Commit**

```bash
git add packages/lazuli-transit/src/lazuli_transit src/wcc_etc/lightcurve.py packages/lazuli-transit/tests/test_models.py
git commit -m "refactor: move FluxModel/TransitModel into lazuli_transit, re-export from wcc_etc"
```

---

### Task 3: `archive.py` — cache path + columns + `load_exoplanet_archive`

**Files:**
- Create: `packages/lazuli-transit/src/lazuli_transit/archive.py`
- Test: `packages/lazuli-transit/tests/test_archive.py`

**Interfaces:**
- Produces:
  - `default_archive_path() -> pathlib.Path`
  - `ARCHIVE_COLUMNS: list[str]`
  - `load_exoplanet_archive(path=None) -> pandas.DataFrame`
  - module-level `_ASTROQUERY_HINT: str` (used in Task 4)

- [ ] **Step 1: Write the failing test**

```python
# packages/lazuli-transit/tests/test_archive.py
import pandas as pd
import pytest

from lazuli_transit.archive import (
    ARCHIVE_COLUMNS,
    default_archive_path,
    load_exoplanet_archive,
)


def test_default_path_is_under_home_and_expanded():
    p = default_archive_path()
    assert p.name == "exoplanet_archive_pscomppars.csv"
    assert "~" not in str(p)
    assert ".lazuli_transit" in str(p)


def test_archive_columns_contains_required_fields():
    for col in ("pl_name", "pl_orbper", "pl_ratror", "pl_ratdor", "st_rad"):
        assert col in ARCHIVE_COLUMNS


def test_load_reads_existing_csv(tmp_path):
    csv = tmp_path / "arch.csv"
    pd.DataFrame({"pl_name": ["WASP-12 b"], "pl_orbper": [1.09]}).to_csv(csv, index=False)
    df = load_exoplanet_archive(path=csv)
    assert list(df["pl_name"]) == ["WASP-12 b"]


def test_load_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_exoplanet_archive"):
        load_exoplanet_archive(path=tmp_path / "nope.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/lazuli-transit/tests/test_archive.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lazuli_transit.archive'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/lazuli-transit/src/lazuli_transit/archive.py
"""Download and load NASA Exoplanet Archive planet parameters.

`download_exoplanet_archive` fetches the PSCompPars table (one complete row per
confirmed planet) to a local CSV cache; `load_exoplanet_archive` reads it.
`TransitModel.from_planet` consumes the result to build transit models for real
planets.
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
    "Install it with: pip install lazuli-transit[archive]"
)


def default_archive_path():
    """Default cache location for the downloaded PSCompPars CSV."""
    return Path("~/.lazuli_transit/exoplanet_archive_pscomppars.csv").expanduser()


def load_exoplanet_archive(path=None):
    """Read the cached PSCompPars CSV into a DataFrame.

    Does not hit the network. Raises FileNotFoundError (pointing at
    download_exoplanet_archive) if the cache file does not exist.
    """
    path = Path(path) if path is not None else default_archive_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No archive cache at {path}. Run download_exoplanet_archive() first."
        )
    return pd.read_csv(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/lazuli-transit/tests/test_archive.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/lazuli-transit/src/lazuli_transit/archive.py packages/lazuli-transit/tests/test_archive.py
git commit -m "feat: add exoplanet archive cache loader to lazuli_transit"
```

---

### Task 4: `download_exoplanet_archive` (cache + lazy astroquery)

**Files:**
- Modify: `packages/lazuli-transit/src/lazuli_transit/archive.py`
- Modify: `packages/lazuli-transit/tests/test_archive.py`

**Interfaces:**
- Consumes: `default_archive_path`, `ARCHIVE_COLUMNS`, `_ASTROQUERY_HINT`, `load_exoplanet_archive` (Task 3).
- Produces:
  - `_query_pscomppars() -> pandas.DataFrame`
  - `download_exoplanet_archive(path=None, refresh=False) -> pandas.DataFrame`

- [ ] **Step 1: Add the failing tests**

Append to `packages/lazuli-transit/tests/test_archive.py`:

```python
import sys

import lazuli_transit.archive as arch


def test_download_returns_cache_when_present_without_network(tmp_path, monkeypatch):
    csv = tmp_path / "arch.csv"
    pd.DataFrame({"pl_name": ["WASP-12 b"]}).to_csv(csv, index=False)

    def _boom():
        raise AssertionError("network/astroquery must not be touched on cache hit")

    monkeypatch.setattr(arch, "_query_pscomppars", _boom)
    df = arch.download_exoplanet_archive(path=csv, refresh=False)
    assert list(df["pl_name"]) == ["WASP-12 b"]


def test_download_writes_cache_using_injected_query(tmp_path, monkeypatch):
    csv = tmp_path / "sub" / "arch.csv"  # parent dir does not exist yet
    fake = pd.DataFrame({c: [0] for c in arch.ARCHIVE_COLUMNS})
    fake["pl_name"] = ["WASP-12 b"]
    monkeypatch.setattr(arch, "_query_pscomppars", lambda: fake)

    df = arch.download_exoplanet_archive(path=csv, refresh=True)
    assert csv.exists()
    assert list(df["pl_name"]) == ["WASP-12 b"]
    assert list(arch.load_exoplanet_archive(path=csv)["pl_name"]) == ["WASP-12 b"]


def test_query_raises_hint_without_astroquery(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "astroquery.ipac.nexsci.nasa_exoplanet_archive", None
    )
    with pytest.raises(ImportError, match=r"pip install lazuli-transit\[archive\]"):
        arch._query_pscomppars()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest packages/lazuli-transit/tests/test_archive.py -v`
Expected: FAIL with `AttributeError: module 'lazuli_transit.archive' has no attribute '_query_pscomppars'` (and `download_exoplanet_archive`).

- [ ] **Step 3: Write minimal implementation**

Append to `packages/lazuli-transit/src/lazuli_transit/archive.py`:

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
    instead of querying the archive. Otherwise query astroquery, write the CSV
    (creating parent directories), and return the DataFrame.
    """
    path = Path(path) if path is not None else default_archive_path()
    if path.exists() and not refresh:
        return load_exoplanet_archive(path=path)

    df = _query_pscomppars()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest packages/lazuli-transit/tests/test_archive.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add packages/lazuli-transit/src/lazuli_transit/archive.py packages/lazuli-transit/tests/test_archive.py
git commit -m "feat: add download_exoplanet_archive with local CSV caching"
```

---

### Task 5: `TransitModel.from_planet` column mapping + fallbacks

**Files:**
- Modify: `packages/lazuli-transit/src/lazuli_transit/models.py`
- Modify: `packages/lazuli-transit/tests/test_models.py`

**Interfaces:**
- Consumes: `lazuli_transit.archive.load_exoplanet_archive` (Task 3); `TransitModel.__init__` (Task 2).
- Produces: `TransitModel.from_planet(name, df=None, limb_dark="quadratic", u=(0.1, 0.3)) -> TransitModel`.

Mapping with fallbacks (R_jup/R_sun ≈ 0.10276268506540175; 1 AU ≈ 215.03215567054764 R_sun):
- `per` ← `pl_orbper` (required; raise `ValueError` if NaN/missing)
- `rp` ← `pl_ratror`; fallback `pl_radj * RJUP_RSUN / st_rad`
- `a` ← `pl_ratdor`; fallback `pl_orbsmax * AU_RSUN / st_rad`
- `inc` ← `pl_orbincl` (fallback 90.0); `t0` ← `pl_tranmid` (0.0); `ecc` ← `pl_orbeccen` (0.0); `w` ← `pl_orblper` (90.0)
- name match: lowercase + strip spaces.

- [ ] **Step 1: Add the failing tests**

Append to `packages/lazuli-transit/tests/test_models.py`:

```python
import pandas as pd

RJUP_RSUN = 0.10276268506540175
AU_RSUN = 215.03215567054764


def _df():
    return pd.DataFrame([
        dict(pl_name="WASP-12 b", pl_orbper=1.0914, pl_ratror=0.117,
             pl_ratdor=3.04, pl_orbincl=83.3, pl_tranmid=2456305.46,
             pl_orbeccen=0.0, pl_orblper=90.0, pl_radj=1.9, pl_orbsmax=0.0234,
             st_rad=1.66),
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

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest packages/lazuli-transit/tests/test_models.py -k from_planet -v`
Expected: FAIL with `AttributeError: type object 'TransitModel' has no attribute 'from_planet'`

- [ ] **Step 3: Write minimal implementation**

Add this classmethod to `TransitModel` in `packages/lazuli-transit/src/lazuli_transit/models.py` (after `__init__`):

```python
    @classmethod
    def from_planet(cls, name, df=None, limb_dark="quadratic", u=(0.1, 0.3)):
        """Build a TransitModel from a NASA Exoplanet Archive (PSCompPars) row.

        Looks up ``name`` in ``df`` (or the local cache via
        load_exoplanet_archive if None) and maps archive columns to batman
        parameters. ``pl_orbper`` is required; other fields fall back to
        circular-orbit / direct-ratio defaults when missing. Limb-darkening is
        not in the archive, so it stays a user argument.

        Note: ``t0`` is taken from ``pl_tranmid`` (absolute BJD). For a
        relative-time light curve, pass a ``time`` array spanning that epoch or
        set ``t0=0`` after construction.
        """
        if df is None:
            from .archive import load_exoplanet_archive
            df = load_exoplanet_archive()

        def _norm(s):
            return str(s).lower().replace(" ", "")

        matches = df[df["pl_name"].map(_norm) == _norm(name)]
        if len(matches) == 0:
            raise ValueError(f"Planet {name!r} not found in archive table")
        row = matches.iloc[0]

        def _val(col, default):
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        per = row.get("pl_orbper")
        if per is None or (isinstance(per, float) and np.isnan(per)):
            raise ValueError(
                f"Planet {name!r} has no pl_orbper; cannot build a transit model"
            )

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
            per=float(per),
            rp=rp,
            a=a,
            inc=_val("pl_orbincl", 90.0),
            ecc=_val("pl_orbeccen", 0.0),
            w=_val("pl_orblper", 90.0),
            limb_dark=limb_dark,
            u=u,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest packages/lazuli-transit/tests/test_models.py -v`
Expected: PASS (all from_planet tests pass; the batman test skips if batman absent).

- [ ] **Step 5: Commit**

```bash
git add packages/lazuli-transit/src/lazuli_transit/models.py packages/lazuli-transit/tests/test_models.py
git commit -m "feat: add TransitModel.from_planet archive-to-batman mapping"
```

---

### Task 6: Public exports + full-suite regression

**Files:**
- Modify: `packages/lazuli-transit/src/lazuli_transit/__init__.py`
- Modify: `src/wcc_etc/__init__.py`
- Test: `packages/lazuli-transit/tests/test_exports.py`
- Test: `tests/test_exoarchive_exports.py`

**Interfaces:**
- Consumes: archive helpers (Tasks 3–4), models (Tasks 2, 5).
- Produces: `lazuli_transit.download_exoplanet_archive`, `lazuli_transit.load_exoplanet_archive`, `lazuli_transit.default_archive_path`, `lazuli_transit.ARCHIVE_COLUMNS`; and re-exports `wcc_etc.download_exoplanet_archive`, `wcc_etc.load_exoplanet_archive`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/lazuli-transit/tests/test_exports.py
import lazuli_transit


def test_top_level_exports():
    for name in ("FluxModel", "TransitModel", "download_exoplanet_archive",
                 "load_exoplanet_archive", "default_archive_path",
                 "ARCHIVE_COLUMNS"):
        assert hasattr(lazuli_transit, name), name
        assert name in lazuli_transit.__all__
```

```python
# tests/test_exoarchive_exports.py
import wcc_etc


def test_exoarchive_functions_exported():
    assert hasattr(wcc_etc, "download_exoplanet_archive")
    assert hasattr(wcc_etc, "load_exoplanet_archive")
    assert "download_exoplanet_archive" in wcc_etc.__all__
    assert "load_exoplanet_archive" in wcc_etc.__all__
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
python -m pytest packages/lazuli-transit/tests/test_exports.py tests/test_exoarchive_exports.py -v
```
Expected: FAIL (names not yet exported).

- [ ] **Step 3: Update `lazuli_transit/__init__.py`**

```python
"""Instrument-agnostic transit models + NASA Exoplanet Archive helpers."""
from .archive import (
    ARCHIVE_COLUMNS,
    default_archive_path,
    download_exoplanet_archive,
    load_exoplanet_archive,
)
from .models import FluxModel, TransitModel

__version__ = "0.1.0"

__all__ = [
    "FluxModel", "TransitModel",
    "download_exoplanet_archive", "load_exoplanet_archive",
    "default_archive_path", "ARCHIVE_COLUMNS",
]
```

- [ ] **Step 4: Update `src/wcc_etc/__init__.py`**

Add an import line near the existing `from .lightcurve import ...`:

```python
from lazuli_transit import download_exoplanet_archive, load_exoplanet_archive
```

And add to the `__all__` list (alongside the lightcurve entries):

```python
    "download_exoplanet_archive", "load_exoplanet_archive",
```

- [ ] **Step 5: Run the exports tests**

Run:
```bash
python -m pytest packages/lazuli-transit/tests/test_exports.py tests/test_exoarchive_exports.py -v
```
Expected: PASS.

- [ ] **Step 6: Run the full suites**

Run:
```bash
python -m pytest -q
python -m pytest packages/lazuli-transit/tests -q
```
Expected: both green, no regressions.

- [ ] **Step 7: Commit**

```bash
git add packages/lazuli-transit/src/lazuli_transit/__init__.py src/wcc_etc/__init__.py packages/lazuli-transit/tests/test_exports.py tests/test_exoarchive_exports.py
git commit -m "feat: export transit + archive helpers from lazuli_transit and wcc_etc"
```

---

## Self-Review Notes

- **Spec coverage:** standalone package + monorepo wiring → Task 1; model move + back-compat re-export → Task 2; archive load + missing-cache error → Task 3; download/cache/refresh + astroquery hint → Task 4; `from_planet` mapping/fallbacks/NaN defaults/name tolerance/errors → Task 5; exports (both packages) → Task 6; portability guard → Task 1 (`test_portability.py`); caveats → `from_planet` docstring (Task 5). Live download remains a skip-guarded manual check, no dedicated task.
- **Naming consistency:** `lazuli-transit` (dist) / `lazuli_transit` (import) used uniformly; cache dir `.lazuli_transit`; hint strings reference `lazuli-transit[batman]` / `lazuli-transit[archive]`.
- **Type consistency:** `load_exoplanet_archive(path=None)`, `download_exoplanet_archive(path=None, refresh=False)`, `TransitModel.from_planet(name, df=None, limb_dark, u)` consistent across producer/consumer tasks. `_query_pscomppars` monkeypatched by name in both download tests.
- **Constants:** test literals `RJUP_RSUN`/`AU_RSUN` match the astropy-derived values computed in `from_planet`.
- **Install-order risk:** root `pyproject.toml` declares `lazuli-transit` as a dependency that is not on PyPI; every install path (CI tests, deploy-docs, readthedocs, Task 1 Step 8) installs `packages/lazuli-transit` editable first, so resolution succeeds locally.
