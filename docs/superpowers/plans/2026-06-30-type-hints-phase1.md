# Type Hints — PR 1 (Phase 1 leaf modules + tooling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hand-written type annotations to the `wcc_etc` leaf modules (`meta`, `astro`, `io`, `utils`, `telescope`, `sensor`) and stand up the typing toolchain (lenient non-blocking mypy, `py.typed`, `typecheck` extra, CI step).

**Architecture:** Annotations are added inline. Every touched module gets `from __future__ import annotations` so annotations are lazy strings (no runtime cost, no circular-import risk). Types that are only imported lazily (synphot, sibling modules) go under a `TYPE_CHECKING` guard so mypy can resolve them without adding runtime imports. mypy is configured leniently — it checks what we annotate and ignores what we don't — and runs in CI as informational only.

**Tech Stack:** Python ≥3.11, mypy, astropy (`Quantity`), numpy (`numpy.typing.NDArray`), synphot, setuptools, GitHub Actions.

## Global Constraints

- Target Python `>=3.11`. CI runs **numpy 2.x / Python 3.11** (local is often numpy 1.26 / py313).
- Add `from __future__ import annotations` as the **first import** in every module touched.
- Annotation vocabulary: astropy quantities → `u.Quantity` (modules already do `from astropy import units as u`; use `u.Quantity` in annotations); arrays → `numpy.typing.NDArray[np.float64]`; classmethod factories that return an instance → `typing.Self`; optionals → `X | None`; config dicts → `dict[str, Any]`; file paths → `str`.
- Names imported only lazily (inside functions) or only for typing go under `if TYPE_CHECKING:` — never add a new runtime import.
- **Annotations + tooling only. No logic changes**, no reformatting, no fixing unrelated bugs.
- numpy-2 safe: no `np.trapz`, keep `arange`/`dr` args scalar (not applicable to these modules but holds repo-wide).
- mypy is **lenient and non-blocking**: never gate a merge on it.
- Legacy `src/wcc_etc/wcc_etc.py` is **excluded** — do not annotate it.
- Commit messages: plain, **no `Co-Authored-By` trailer**.
- Branch for this PR: `feat/type-hints-phase1` (already created off `main`; the design spec is already committed on it).

---

### Task 1: Typing toolchain (mypy config, py.typed, extra, CI)

Lands first so every later task can verify with mypy.

**Files:**
- Modify: `pyproject.toml`
- Create: `src/wcc_etc/py.typed`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `mypy src/wcc_etc` command (lenient, excludes `wcc_etc.py`); a `[typecheck]` extra installing `mypy`; a shipped `py.typed`; a non-blocking CI step named "Type check (mypy, non-blocking)".

- [ ] **Step 1: Add the mypy config to `pyproject.toml`**

Append this block at the end of `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
files = ["src/wcc_etc"]
ignore_missing_imports = true
disallow_untyped_defs = false
check_untyped_defs = false
no_implicit_optional = true
warn_redundant_casts = true
exclude = ['src/wcc_etc/wcc_etc\.py$']
```

- [ ] **Step 2: Add the `typecheck` optional dependency**

In `pyproject.toml`, in the existing `[project.optional-dependencies]` table (currently holding `lightcurve` and `exoarchive`), add:

```toml
typecheck = ["mypy"]
```

- [ ] **Step 3: Ship `py.typed`**

Create empty file `src/wcc_etc/py.typed` (zero bytes).

Then register it in `[tool.setuptools.package-data]`. Change the `"wcc_etc" = [ ... ]` list so its first entry is:

```toml
"wcc_etc" = [
  "py.typed",
  "data/astr_obj_models/stars/pickles_models/dat_uvk/*.fits",
```

(leave the remaining data entries unchanged).

- [ ] **Step 4: Install mypy and capture the baseline**

Run: `pip install -e .[typecheck]`
Then run: `mypy src/wcc_etc`
Expected: exits `0` ("Success: no issues found in N source files") — lenient config means the as-yet-unannotated modules produce no errors, and `wcc_etc.py` is skipped. If real errors appear from already-partially-annotated code, note them; they do not block this task.

- [ ] **Step 5: Add the non-blocking CI step**

In `.github/workflows/tests.yml`, after the "Install dependencies" step (which ends with `pip install -e .`) and before "Run tests", insert:

```yaml
      - name: Type check (mypy, non-blocking)
        continue-on-error: true
        run: |
          pip install mypy
          mypy src/wcc_etc
```

- [ ] **Step 6: Verify py.typed is packaged**

Run: `python -m build --wheel 2>/dev/null && python -c "import zipfile, glob; w=sorted(glob.glob('dist/*.whl'))[-1]; print('py.typed' , any(n.endswith('wcc_etc/py.typed') for n in zipfile.ZipFile(w).namelist()))"`
Expected: prints `py.typed True`. (If `build` is not installed: `pip install build` first. Clean up `dist/` and `build/` afterward — do not commit them.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/wcc_etc/py.typed .github/workflows/tests.yml
git commit -m "Add mypy tooling, py.typed, and non-blocking CI type check"
```

---

### Task 2: Annotate `meta.py`

**Files:**
- Modify: `src/wcc_etc/meta.py`

**Interfaces:**
- Consumes: nothing.
- Produces: typed `_MetaHolder_` base — `meta -> dict[str, Any]`, `mutable_parameters -> list[str]`, `update(...) -> KeysView[str]`. Sibling subclasses (`Telescope`, `Sensor`, `Scene`, `Simulation`) inherit these.

- [ ] **Step 1: Add imports**

At the very top of `src/wcc_etc/meta.py`, before `import warnings`:

```python
from __future__ import annotations

from typing import Any
from collections.abc import KeysView
```

- [ ] **Step 2: Annotate the class variable and methods**

Apply these exact signature changes (bodies unchanged):

```python
    _mutable_parameters: list[str] = []

    def __init__(self, meta: dict[str, Any] = {}) -> None:

    def reset(self) -> None:

    def update(self, reset: bool = False, **kwargs: Any) -> KeysView[str]:

    def describe(self) -> None:

    @property
    def meta(self) -> dict[str, Any]:

    @property
    def mutable_parameters(self) -> list[str]:
```

- [ ] **Step 3: Type-check the module**

Run: `mypy src/wcc_etc/meta.py`
Expected: `Success: no issues found in 1 source file`.

- [ ] **Step 4: Confirm no runtime breakage**

Run: `python -c "import wcc_etc.meta" && pytest tests/sensor -q`
Expected: import succeeds; tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/meta.py
git commit -m "Add type hints to meta module"
```

---

### Task 3: Annotate `astro.py`

**Files:**
- Modify: `src/wcc_etc/astro.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `get_moon_magnitude(phase, phase_type='fraction', distance_km=384400.0, m_full=-12.74) -> float | NDArray[np.float64]`.

- [ ] **Step 1: Add imports**

At the top of `src/wcc_etc/astro.py`, before `import numpy as np`:

```python
from __future__ import annotations

from numpy.typing import NDArray
```

- [ ] **Step 2: Annotate the function**

```python
def get_moon_magnitude(
    phase: float | NDArray[np.float64],
    phase_type: str = 'fraction',
    distance_km: float | None = 384400.0,
    m_full: float = -12.74,
) -> float | NDArray[np.float64]:
```

(body unchanged)

- [ ] **Step 3: Type-check**

Run: `mypy src/wcc_etc/astro.py`
Expected: `Success: no issues found in 1 source file`.

- [ ] **Step 4: Confirm no runtime breakage**

Run: `python -c "from wcc_etc.astro import get_moon_magnitude as f; print(round(float(f(1.0)), 2), round(float(f(0.5)), 2))"`
Expected: prints `-12.74 -11.99`.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/astro.py
git commit -m "Add type hints to astro module"
```

---

### Task 4: Annotate `io.py`

**Files:**
- Modify: `src/wcc_etc/io.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `resolve_bandpass(bandpass) -> SpectralElement`; `get_any_astro_name(name, retry=True) -> str | None`; `get_pickles_spectrum_filename(spectral_type, fullpath=True) -> str`; `read_config(filename, source="config") -> dict[str, Any]`; `get_sensor_config(kind, band, **kwargs) -> dict[str, Any]`; `expand_path(filename, source=None, test_extension=False) -> str`.

- [ ] **Step 1: Add imports**

At the very top of `src/wcc_etc/io.py`, before `import os`:

```python
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from synphot import SpectralElement
```

- [ ] **Step 2: Annotate the public functions**

Apply these exact signature changes (bodies unchanged):

```python
def resolve_bandpass(bandpass: str | SpectralElement) -> SpectralElement:

def get_any_astro_name(name: str, retry: bool = True) -> str | None:

def get_pickles_spectrum_filename(spectral_type: str, fullpath: bool = True) -> str:

def read_config(filename: str | dict[str, Any], source: str = "config") -> dict[str, Any]:

def get_sensor_config(kind: str, band: str, **kwargs: Any) -> dict[str, Any]:

def expand_path(filename: str, source: str | None = None, test_extension: bool = False) -> str:
```

- [ ] **Step 3: Type-check**

Run: `mypy src/wcc_etc/io.py`
Expected: `Success: no issues found` (it may report >1 file because mypy follows imports; what matters is **no errors**).

- [ ] **Step 4: Confirm no runtime breakage**

Run: `python -c "import wcc_etc.io" && pytest tests/scene tests/sensor -q`
Expected: import succeeds; tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/io.py
git commit -m "Add type hints to io module"
```

---

### Task 5: Annotate `utils.py`

**Files:**
- Modify: `src/wcc_etc/utils.py`

**Interfaces:**
- Consumes: `expand_path` (from `io`, now typed).
- Produces: `parse_element(path_or_element, wave_unit='nm') -> SpectralElement | None`; `parse_and_interpolate(input_file, xval) -> float | NDArray[np.float64]`; `list_of_quantity_to_array(quantities) -> Quantity | list[Quantity]`.

- [ ] **Step 1: Add imports**

At the very top of `src/wcc_etc/utils.py`, before `import pandas`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from astropy.units import Quantity
    from synphot import SpectralElement
```

- [ ] **Step 2: Annotate the functions**

Apply these exact signature changes (bodies unchanged):

```python
def parse_element(path_or_element: str | SpectralElement | None, wave_unit: str = 'nm') -> SpectralElement | None:

def parse_and_interpolate(input_file: str, xval: float | NDArray[np.float64]) -> float | NDArray[np.float64]:

def list_of_quantity_to_array(quantities: list[Quantity]) -> Quantity | list[Quantity]:
```

> Note: `list_of_quantity_to_array` calls `warnings.warn` but `utils.py` does not import `warnings` — a pre-existing latent bug in the unit-mismatch branch. **Do not fix it** (out of scope: annotations only). Leave it for a separate change.

- [ ] **Step 3: Type-check**

Run: `mypy src/wcc_etc/utils.py`
Expected: no errors.

- [ ] **Step 4: Confirm no runtime breakage**

Run: `python -c "import wcc_etc.utils" && pytest tests/scene -q`
Expected: import succeeds; tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/utils.py
git commit -m "Add type hints to utils module"
```

---

### Task 6: Annotate `telescope.py`

**Files:**
- Modify: `src/wcc_etc/telescope.py`

**Interfaces:**
- Consumes: `_MetaHolder_` (typed).
- Produces: typed `Telescope` — `from_config(config) -> Self`; properties `f_num -> float`, `diameter_primary -> u.Quantity`, `jitter_sigma -> u.Quantity`, `surface -> u.Quantity`, `focal_len -> u.Quantity`.

- [ ] **Step 1: Add imports**

At the very top of `src/wcc_etc/telescope.py`, before `import numpy as np`:

```python
from __future__ import annotations

from typing import Any, Self
```

- [ ] **Step 2: Annotate `__init__`, `from_config`, and properties**

Apply these exact signature changes (bodies unchanged):

```python
    def __init__(self, f_num: float, diameter_primary: float | u.Quantity,
                 jitter_sigma: float | u.Quantity = 0,
                 meta: dict[str, Any] = {}) -> None:

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:

    @property
    def f_num(self) -> float:

    @property
    def diameter_primary(self) -> u.Quantity:

    @property
    def jitter_sigma(self) -> u.Quantity:

    @property
    def surface(self) -> u.Quantity:

    @property
    def focal_len(self) -> u.Quantity:
```

- [ ] **Step 3: Type-check**

Run: `mypy src/wcc_etc/telescope.py`
Expected: no errors.

- [ ] **Step 4: Confirm no runtime breakage**

Run: `python -c "import wcc_etc.telescope" && pytest tests/sensor tests/scene -q`
Expected: import succeeds; tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/telescope.py
git commit -m "Add type hints to telescope module"
```

---

### Task 7: Annotate `sensor.py`

**Files:**
- Modify: `src/wcc_etc/sensor.py`

**Interfaces:**
- Consumes: `_MetaHolder_`, `parse_element`, `parse_and_interpolate` (all typed), `Telescope` (typed, used only as a parameter type).
- Produces: typed `Sensor` — factories `from_name`/`from_config`/`from_kind_and_band` returning `Self`; `set_bandpass(...) -> None`; `get_plate_scale(telescope) -> u.Quantity`; properties returning `u.Quantity` / `SpectralElement` / `int | None`.

- [ ] **Step 1: Add imports**

At the very top of `src/wcc_etc/sensor.py` (the file starts with a blank line then `from astropy import units as u`), add before that import:

```python
from __future__ import annotations

from typing import Any, Self, TYPE_CHECKING

if TYPE_CHECKING:
    from synphot import SpectralElement
    from .telescope import Telescope
```

- [ ] **Step 2: Annotate `__init__` and the factories**

```python
    def __init__(self, bandpass: str | SpectralElement,
                 pixel_size: float | u.Quantity,
                 read_noise: float | u.Quantity,
                 dark_current: float | u.Quantity,
                 gain: float | u.Quantity,
                 area: float | u.Quantity,
                 temperature: float | u.Quantity | None = None,
                 qe: float = 1,
                 well_depth: float | u.Quantity | None = None,
                 bit_depth: int | None = None,
                 bias_level: float | None = None,
                 meta: dict[str, Any] = {}) -> None:

    @classmethod
    def from_name(cls, name: str) -> Self:

    @classmethod
    def from_config(cls, config_or_name: dict[str, Any] | str) -> Self:

    @classmethod
    def from_kind_and_band(cls, kind: str, band: str) -> Self:
```

- [ ] **Step 3: Annotate the methods and properties**

```python
    def set_bandpass(self, bandpass: str | SpectralElement) -> None:

    def get_plate_scale(self, telescope: Telescope) -> u.Quantity:

    @property
    def bandpass(self) -> SpectralElement:

    @property
    def wavelength(self) -> u.Quantity:

    @property
    def area(self) -> u.Quantity:

    @property
    def gain(self) -> u.Quantity:

    @property
    def dark_current(self) -> u.Quantity:

    @property
    def read_noise(self) -> u.Quantity:

    @property
    def pixel_size(self) -> u.Quantity:

    @property
    def bit_depth(self) -> int | None:

    @property
    def bias_level(self) -> u.Quantity:

    @property
    def adc_max(self) -> u.Quantity:
```

- [ ] **Step 4: Type-check**

Run: `mypy src/wcc_etc/sensor.py`
Expected: no errors.

- [ ] **Step 5: Confirm no runtime breakage**

Run: `python -c "import wcc_etc.sensor" && pytest tests/sensor -q`
Expected: import succeeds; tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/sensor.py
git commit -m "Add type hints to sensor module"
```

---

### Task 8: Full verification and open PR

**Files:** none modified (verification + PR).

**Interfaces:**
- Consumes: all of Tasks 1–7.
- Produces: a pushed branch and an open PR for Phase 1 + tooling.

- [ ] **Step 1: Full type check**

Run: `mypy src/wcc_etc`
Expected: exits `0`, no errors. `wcc_etc.py` is excluded by config.

- [ ] **Step 2: Full test suite**

Run: `pytest -q`
Expected: `552 passed` (same count as before this work; warnings are pre-existing). This is the primary guard that annotations introduced no runtime regressions.

- [ ] **Step 3: Import smoke test**

Run: `python -c "import wcc_etc; print(wcc_etc.__version__)"`
Expected: prints the version with no errors.

- [ ] **Step 4: Push the branch**

```bash
git -c credential.helper='!gh auth git-credential' push https://github.com/schmidt-observatories/wcc-etc.git feat/type-hints-phase1
```

- [ ] **Step 5: Open the PR**

Write the body to a temp file (avoids shell-quoting issues), then:

```bash
gh pr create --base main --head feat/type-hints-phase1 \
  --title "Type hints (Phase 1): leaf modules + tooling" \
  --body-file <path-to-body-file>
```

PR body should state: Phase 1 of the public-API typing effort per `docs/superpowers/specs/2026-06-30-repo-type-hints-design.md`; annotates `meta`, `astro`, `io`, `utils`, `telescope`, `sensor`; adds lenient non-blocking mypy, `py.typed`, and a `typecheck` extra; legacy `wcc_etc.py` excluded; `552 passed`, `mypy` clean. Note that Phases 2 and 3 follow as separate PRs.

- [ ] **Step 6: Watch CI**

Run: `gh pr checks --watch` (or check the PR's `test` job). The pytest gate must stay green; the mypy step is informational (`continue-on-error`) and never blocks.

---

## Self-Review

**Spec coverage:**
- Scope → Phase 1 modules each have a task (meta/astro/io/utils/telescope/sensor); Phases 2–3 explicitly deferred to later plans. ✓
- Type conventions (`from __future__ import annotations`, `u.Quantity`, `NDArray`, `Self`, `X | None`, `TYPE_CHECKING` for lazy/3rd-party imports) → applied in every task. ✓
- mypy config / `typecheck` extra / `py.typed` / CI step → Task 1. ✓
- Verification (per-module mypy + tests; end-to-end full suite; wheel ships py.typed; import smoke) → Tasks 1–8. ✓
- Delivery as PR 1 of 3 → Task 8 opens the PR and notes follow-ups. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step shows exact lines. ✓

**Type consistency:** `_MetaHolder_` produces `dict[str, Any]`/`list[str]` used implicitly by subclasses; `parse_element -> SpectralElement | None` and `parse_and_interpolate -> float | NDArray[np.float64]` consumed by `sensor`; `Telescope` used as a parameter type in `sensor.get_plate_scale`. Names consistent across tasks. ✓
