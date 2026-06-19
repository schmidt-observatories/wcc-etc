# Transit Light-Curve Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulate WCC-observed exoplanet transit light curves — a `batman` transit shape with photometric noise drawn from the existing ETC SNR — and plot them in matplotlib and bokeh.

**Architecture:** One new module `src/wcc_etc/lightcurve.py` follows the pluggable-source pattern of `psfsim.py`. A `FluxModel` base maps time→relative flux; `TransitModel` wraps `batman`; `LightCurveSimulator` computes one ETC photometric SNR at baseline brightness (σ = 1/SNR, held constant) and adds Gaussian scatter; `LightCurve` carries the result and plots it. Plotting functions live in `plotting.py` to match existing conventions.

**Tech Stack:** Python, numpy, `batman-package` (optional extra, lazy-imported), matplotlib, bokeh, pytest.

## Global Constraints

- batman is an **optional dependency**: `batman-package` under `[project.optional-dependencies]` as the `lightcurve` extra. Import `batman` **lazily inside methods**, never at module top. batman-dependent tests use `pytest.importorskip("batman")`.
- CI runs **numpy 2.x / Python 3.11** (local is numpy 1.26). Do not use `np.trapz`; use `np.random.default_rng(seed)` for RNG.
- Plotting follows `plotting.py` conventions: mpl variant returns `(fig, ax)`; bokeh variant takes `return_="obj"|"html"|"components"` and ends with `_finish_bokeh(obj, return_)`. Honor `set_wcc_style()`; labels in Title Case.
- σ is computed **once** at baseline brightness and applied to every point (no per-point rescaling).
- Frequent commits: one commit per task (after its tests pass).
- Run the **full suite** (`pytest -q`) at the end of each task; all tests must pass before committing.

---

### Task 1: `FluxModel` base + `TransitModel` (batman wrapper) + optional dependency

**Files:**
- Create: `src/wcc_etc/lightcurve.py`
- Create: `tests/test_lightcurve_model.py`
- Modify: `pyproject.toml` (add `[project.optional-dependencies]` with `lightcurve` extra)

**Interfaces:**
- Consumes: nothing (new module).
- Produces:
  - `class FluxModel` with method `relative_flux(self, time) -> np.ndarray`.
  - `class TransitModel(FluxModel)`, constructor
    `TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0, ecc=0.0, w=90.0, limb_dark="quadratic", u=(0.1, 0.3))`;
    `relative_flux(time)` returns the batman light curve (ndarray, 1.0 out of transit).

- [ ] **Step 1: Add the optional dependency to `pyproject.toml`**

After the `dependencies = [...]` block in `pyproject.toml`, add:

```toml
[project.optional-dependencies]
lightcurve = ["batman-package"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_lightcurve_model.py`:

```python
import numpy as np
import pytest

from wcc_etc.lightcurve import FluxModel, TransitModel

batman = pytest.importorskip("batman")


def test_fluxmodel_base_is_abstract():
    with pytest.raises(NotImplementedError):
        FluxModel().relative_flux(np.linspace(0, 1, 5))


def test_transit_relative_flux_matches_batman_directly():
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


def test_transit_out_of_transit_is_unity_and_dip_present():
    t = np.linspace(-0.25, 0.25, 400)
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=90.0, u=(0.0, 0.0))
    flux = model.relative_flux(t)
    assert flux[0] == pytest.approx(1.0, abs=1e-6)        # far from transit
    assert flux.min() < 1.0                                # dip exists
    assert flux.min() == pytest.approx(1 - 0.1**2, abs=2e-3)  # depth ~ rp^2 (uniform LD)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_lightcurve_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wcc_etc.lightcurve'` (or `ImportError` for the names).

- [ ] **Step 4: Write the minimal implementation**

Create `src/wcc_etc/lightcurve.py`:

```python
# src/wcc_etc/lightcurve.py
"""Time-series light-curve simulation for the WCC ETC.

`FluxModel` maps time to normalized relative flux; `TransitModel` wraps the
`batman` package. `LightCurveSimulator` turns a model + a Simulation into a
synthetic observed light curve, using one ETC photometric SNR measurement at
baseline brightness as the per-point error. Future scenarios (moon transits,
Cepheids) subclass `FluxModel` and work with the simulator and plotters
unchanged.
"""
import numpy as np

_BATMAN_HINT = (
    "TransitModel requires the 'batman' package. "
    "Install it with: pip install wcc-etc[lightcurve]"
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

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_lightcurve_model.py -v`
Expected: PASS (3 passed; if batman is not installed, 2 skipped + the base-class test passes).

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/lightcurve.py tests/test_lightcurve_model.py pyproject.toml
git commit -m "Add FluxModel base and batman-backed TransitModel"
```

---

### Task 2: `LightCurve` result object + `LightCurveSimulator` (ETC bridge)

**Files:**
- Modify: `src/wcc_etc/lightcurve.py`
- Create: `tests/test_lightcurve_simulator.py`

**Interfaces:**
- Consumes: `FluxModel.relative_flux(time)` (Task 1); `Simulation.get_image_snr(time=..., n_reads=1, r_aper_mas=, ee_frac=, psf=, jitter_sigma_mas=, npix=) -> dict` with key `"snr"`.
- Produces:
  - `class LightCurve` with attributes `time, flux, flux_clean, flux_err, exptime, snr` and a method `plot(self, backend="mpl", **kw)` (plotting wired in Task 3; define the method to dispatch now).
  - `class LightCurveSimulator(sim, model)` with
    `simulate(self, time, exptime, *, r_aper_mas=None, ee_frac=None, psf=None, jitter_sigma_mas=None, npix=128, seed=None) -> LightCurve`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lightcurve_simulator.py`:

```python
import numpy as np
import pytest

from wcc_etc.lightcurve import FluxModel, LightCurve, LightCurveSimulator


class _ConstDip(FluxModel):
    """Deterministic model: 1.0 everywhere except a 0.99 block in the middle."""
    def relative_flux(self, time):
        time = np.asarray(time, dtype=float)
        f = np.ones_like(time)
        f[len(f) // 4: 3 * len(f) // 4] = 0.99
        return f


class _FakeSim:
    """Stand-in for Simulation that records the get_image_snr call."""
    def __init__(self, snr):
        self._snr = snr
        self.calls = []
    def get_image_snr(self, **kw):
        self.calls.append(kw)
        return {"snr": self._snr}


def test_simulate_sigma_is_inverse_snr_and_called_once_with_nreads1():
    sim = _FakeSim(snr=200.0)
    lc = LightCurveSimulator(sim, _ConstDip()).simulate(
        time=np.linspace(0, 1, 50), exptime=30.0, seed=0)
    assert lc.flux_err == pytest.approx(1 / 200.0)
    assert len(sim.calls) == 1                       # single baseline measurement
    assert sim.calls[0]["n_reads"] == 1
    assert sim.calls[0]["time"] == 30.0
    assert lc.snr == pytest.approx(200.0)
    assert lc.exptime == 30.0


def test_simulate_clean_matches_model_and_seed_is_reproducible():
    model = _ConstDip()
    t = np.linspace(0, 1, 200)
    lc1 = LightCurveSimulator(_FakeSim(100.0), model).simulate(t, 30.0, seed=42)
    lc2 = LightCurveSimulator(_FakeSim(100.0), model).simulate(t, 30.0, seed=42)
    np.testing.assert_array_equal(lc1.flux_clean, model.relative_flux(t))
    np.testing.assert_array_equal(lc1.flux, lc2.flux)        # reproducible


def test_simulate_noise_statistics_match_sigma():
    t = np.linspace(0, 1, 5000)
    lc = LightCurveSimulator(_FakeSim(50.0), _ConstDip()).simulate(t, 30.0, seed=1)
    resid = lc.flux - lc.flux_clean
    assert np.std(resid) == pytest.approx(1 / 50.0, rel=0.1)
    assert isinstance(lc, LightCurve)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_lightcurve_simulator.py -v`
Expected: FAIL — `ImportError: cannot import name 'LightCurve'` / `'LightCurveSimulator'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/wcc_etc/lightcurve.py`:

```python
class LightCurve:
    """Result of a light-curve simulation.

    Attributes
    ----------
    time : ndarray
    flux : ndarray            # noisy realization
    flux_clean : ndarray      # noiseless model
    flux_err : float          # per-point sigma (= 1 / snr)
    exptime : float           # per-point exposure time (s)
    snr : float               # baseline photometric SNR
    """

    def __init__(self, time, flux, flux_clean, flux_err, exptime, snr):
        self.time = np.asarray(time, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.flux_clean = np.asarray(flux_clean, dtype=float)
        self.flux_err = float(flux_err)
        self.exptime = float(exptime)
        self.snr = float(snr)

    def plot(self, backend="mpl", **kw):
        """Plot this light curve. Wired to plotting.py (lazy import to avoid
        an import cycle, mirroring SimulatedImage)."""
        from . import plotting
        if backend == "mpl":
            return plotting.plot_lightcurve_mpl(self, **kw)
        if backend == "bokeh":
            return plotting.plot_lightcurve_bokeh(self, **kw)
        raise ValueError("backend must be 'mpl' or 'bokeh'")


class LightCurveSimulator:
    """Turn a FluxModel into a WCC-observed light curve via the ETC."""

    def __init__(self, sim, model):
        self.sim = sim
        self.model = model

    def simulate(self, time, exptime, *, r_aper_mas=None, ee_frac=None,
                 psf=None, jitter_sigma_mas=None, npix=128, seed=None):
        time = np.asarray(time, dtype=float)
        result = self.sim.get_image_snr(
            time=float(exptime), n_reads=1, r_aper_mas=r_aper_mas,
            ee_frac=ee_frac, psf=psf, jitter_sigma_mas=jitter_sigma_mas,
            npix=npix)
        snr = float(result["snr"])
        sigma = 1.0 / snr
        flux_clean = self.model.relative_flux(time)
        rng = np.random.default_rng(seed)
        flux = flux_clean + rng.normal(0.0, sigma, size=time.shape)
        return LightCurve(time, flux, flux_clean, sigma, float(exptime), snr)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_lightcurve_simulator.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/lightcurve.py tests/test_lightcurve_simulator.py
git commit -m "Add LightCurve result and LightCurveSimulator ETC bridge"
```

---

### Task 3: Plotting — `plot_lightcurve_mpl` / `plot_lightcurve_bokeh`

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Create: `tests/test_lightcurve_plotting.py`

**Interfaces:**
- Consumes: `LightCurve` (Task 2) attributes `time, flux, flux_clean, flux_err`; existing `_finish_bokeh(obj, return_)` in `plotting.py`.
- Produces:
  - `plot_lightcurve_mpl(source=None, *, show_noise=True, show_model=True, time=None, flux=None, flux_clean=None, flux_err=None, ax=None, **kw) -> (fig, ax)`
  - `plot_lightcurve_bokeh(source=None, *, show_noise=True, show_model=True, time=None, flux=None, flux_clean=None, flux_err=None, width=600, height=350, return_="obj") -> bokeh obj/html/(script,div)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lightcurve_plotting.py`:

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")

from wcc_etc.lightcurve import FluxModel, LightCurveSimulator
from wcc_etc.plotting import plot_lightcurve_mpl, plot_lightcurve_bokeh


class _Dip(FluxModel):
    def relative_flux(self, time):
        return np.ones_like(np.asarray(time, dtype=float))


class _FakeSim:
    def get_image_snr(self, **kw):
        return {"snr": 100.0}


def _lc():
    t = np.linspace(0, 1, 100)
    return LightCurveSimulator(_FakeSim(), _Dip()).simulate(t, 30.0, seed=0)


def test_mpl_returns_fig_ax_and_draws_both():
    fig, ax = plot_lightcurve_mpl(_lc())
    assert ax.has_data()
    assert len(ax.lines) >= 1          # model line
    assert len(ax.collections) >= 1    # errorbar points

def test_mpl_show_model_only_has_no_errorbar_collection():
    fig, ax = plot_lightcurve_mpl(_lc(), show_noise=False, show_model=True)
    assert len(ax.lines) >= 1
    assert len(ax.collections) == 0

def test_mpl_accepts_raw_arrays():
    lc = _lc()
    fig, ax = plot_lightcurve_mpl(time=lc.time, flux=lc.flux,
                                  flux_clean=lc.flux_clean, flux_err=lc.flux_err)
    assert ax.has_data()

def test_bokeh_components_returns_script_div():
    out = plot_lightcurve_bokeh(_lc(), return_="components")
    assert isinstance(out, tuple) and len(out) == 2
    assert "<div" in out[1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_lightcurve_plotting.py -v`
Expected: FAIL — `ImportError: cannot import name 'plot_lightcurve_mpl'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `src/wcc_etc/plotting.py` (after the existing functions):

```python
def _resolve_lightcurve(source=None, *, time=None, flux=None,
                        flux_clean=None, flux_err=None):
    """Resolve (time, flux, flux_clean, flux_err) from a LightCurve or arrays."""
    if source is not None and hasattr(source, "flux_clean"):
        return source.time, source.flux, source.flux_clean, source.flux_err
    return time, flux, flux_clean, flux_err


def plot_lightcurve_mpl(source=None, *, show_noise=True, show_model=True,
                        time=None, flux=None, flux_clean=None, flux_err=None,
                        ax=None, **kw):
    """Matplotlib light-curve plot: clean model line, noisy points with error
    bars, or both. Accepts a LightCurve (positional) or raw arrays."""
    time, flux, flux_clean, flux_err = _resolve_lightcurve(
        source, time=time, flux=flux, flux_clean=flux_clean, flux_err=flux_err)
    if ax is None:
        fig, ax = plt.subplots(figsize=kw.pop("figsize", (7, 4)))
    else:
        fig = ax.figure
    if show_noise and flux is not None:
        ax.errorbar(time, flux, yerr=flux_err, fmt="o", ms=3, color="0.35",
                    ecolor="0.7", elinewidth=0.8, capsize=0, zorder=1,
                    label="Simulated")
    if show_model and flux_clean is not None:
        ax.plot(time, flux_clean, "-", color="C3", lw=1.8, zorder=2,
                label="Model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Flux")
    ax.set_title("Transit Light Curve")
    ax.legend(loc="best", frameon=False)
    return fig, ax


def plot_lightcurve_bokeh(source=None, *, show_noise=True, show_model=True,
                          time=None, flux=None, flux_clean=None, flux_err=None,
                          width=600, height=350, return_="obj"):
    """Bokeh light-curve plot. ``return_`` selects the output form (see
    _finish_bokeh). Accepts a LightCurve (positional) or raw arrays."""
    from bokeh.plotting import figure
    time, flux, flux_clean, flux_err = _resolve_lightcurve(
        source, time=time, flux=flux, flux_clean=flux_clean, flux_err=flux_err)
    p = figure(width=width, height=height, title="Transit Light Curve",
               x_axis_label="Time", y_axis_label="Relative Flux")
    if show_noise and flux is not None:
        p.scatter(time, flux, size=4, color="#595959", alpha=0.8,
                  legend_label="Simulated")
        if flux_err:
            lower = np.asarray(flux) - flux_err
            upper = np.asarray(flux) + flux_err
            p.segment(time, lower, time, upper, color="#b3b3b3", line_width=0.8)
    if show_model and flux_clean is not None:
        p.line(time, flux_clean, color="crimson", line_width=2,
               legend_label="Model")
    p.legend.location = "bottom_right"
    return _finish_bokeh(p, return_)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_lightcurve_plotting.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_lightcurve_plotting.py
git commit -m "Add matplotlib + bokeh light-curve plotting"
```

---

### Task 4: Public API exports + full-suite verification

**Files:**
- Modify: `src/wcc_etc/__init__.py`
- Create: `tests/test_lightcurve_exports.py`

**Interfaces:**
- Consumes: all classes/functions from Tasks 1-3.
- Produces: top-level names on `wcc_etc`: `FluxModel`, `TransitModel`, `LightCurveSimulator`, `LightCurve`, `plot_lightcurve_mpl`, `plot_lightcurve_bokeh`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lightcurve_exports.py`:

```python
import wcc_etc


def test_lightcurve_names_exported():
    for name in ["FluxModel", "TransitModel", "LightCurveSimulator",
                 "LightCurve", "plot_lightcurve_mpl", "plot_lightcurve_bokeh"]:
        assert hasattr(wcc_etc, name), f"missing export: {name}"
        assert name in wcc_etc.__all__
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_lightcurve_exports.py -v`
Expected: FAIL — `AssertionError: missing export: FluxModel`.

- [ ] **Step 3: Wire up the exports**

In `src/wcc_etc/__init__.py`, add the new names to `__all__` (inside the existing list):

```python
    "FluxModel", "TransitModel", "LightCurveSimulator", "LightCurve",
    "plot_lightcurve_mpl", "plot_lightcurve_bokeh",
```

Add an import for the lightcurve classes (after the `from .plotting import (...)` block):

```python
from .lightcurve import FluxModel, TransitModel, LightCurveSimulator, LightCurve
```

And add the two plotting functions to the existing `from .plotting import (...)` block:

```python
    plot_lightcurve_mpl, plot_lightcurve_bokeh,
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_lightcurve_exports.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all tests pass (existing suite + new lightcurve tests; batman tests pass if installed, otherwise skipped).

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/__init__.py tests/test_lightcurve_exports.py
git commit -m "Export light-curve API from package top level"
```

---

## Post-implementation (per feature workflow)

After all tasks pass, before the PR:

1. **Scratch demo notebook** — `notebooks_scratch/20260617_transit_lightcurve.ipynb`, built from a Python/nbformat builder and executed end-to-end via the `notebook-demo` skill. Show: a `TransitModel` curve, the ETC bridge (`get_scene` → `Simulation.from_sensor_and_scene` → `LightCurveSimulator.simulate`), and clean-vs-noisy plots in both mpl and bokeh; call `set_wcc_style()`.
2. **Update the project-status memory** — new module/API surface, test count, the constant-σ / optional-batman gotchas.
3. Push and open a PR per the pushing-and-CI workflow.

## Self-Review

- **Spec coverage:** FluxModel/TransitModel (Task 1) ✓; LightCurveSimulator single-frame SNR + constant σ + Gaussian draw (Task 2) ✓; LightCurve result (Task 2) ✓; mpl+bokeh plotting with clean/noisy/both (Task 3) ✓; optional batman extra + lazy import (Task 1, Global Constraints) ✓; exports (Task 4) ✓; testing approach (every task) ✓.
- **Placeholders:** none — every code/test step has complete content.
- **Type consistency:** `relative_flux(time)` used identically across tasks; `get_image_snr(..., n_reads=1)["snr"]` matches the real signature in `simulation.py:836`; `_finish_bokeh(obj, return_)` matches `plotting.py:238`; `LightCurve` attribute names consistent between Task 2 and Task 3's `_resolve_lightcurve`.
