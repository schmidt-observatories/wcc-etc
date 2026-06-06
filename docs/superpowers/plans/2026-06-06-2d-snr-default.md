# 2D Image SNR as the Default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Simulation.get_snr` compute the SNR via the 2D image simulation by default (returning the PSF-aware dict, vectorized over array time), preserve the old analytic Airy calculation behind a deprecated `get_snr_airy`, and cache the expensive render so repeated/swept calls are fast.

**Architecture:** A new `_image_render_bundle` helper renders the PSF and computes count *rates* once, memoized on render-affecting state and cleared on any `update()`/`set_sensor()`/`set_telescope()`. `get_image_snr` (canonical) and `get_image_exptime_for_snr` both consume the bundle; `get_image_snr` loops the cheap per-time aperture evaluation to support scalar and array time. `get_snr` delegates to `get_image_snr`; the old body moves to `get_snr_airy` with a `DeprecationWarning`.

**Tech Stack:** Python, numpy, astropy.units, pytest. Repo lives in the `wcc-etc/` subdirectory — run all `git`/`pytest` there. Local env is numpy 1.26 / py313; CI is numpy 2.x / py311 (use `scipy.integrate.trapezoid`, never `np.trapz`/`np.trapezoid` — not relevant to this plan's code but keep in mind).

**Branch:** `2d-snr-default` (already created; spec already committed).

---

## File Structure

- `src/wcc_etc/psfsim.py` — add `cache_key()` to `PSFSource`, `_ResampledPSF`, `DefocusPSF`.
- `src/wcc_etc/simulation.py` — add `_image_render_bundle`, init + clear cache fields, refactor `get_image_snr` (bundle + array time) and `get_image_exptime_for_snr` (bundle), add `get_snr_airy`, rewrite `get_snr` as a delegator.
- `src/wcc_etc/wcc_etc.py` — one-line caller fix in `get_wcc_snr_and_simulation`.
- `tests/test_psfsim.py` (or existing psf test file) — `cache_key` tests.
- `tests/test_simulation.py` — new caching / array-time / delegation / deprecation tests; fix the cross-check test.
- `tests/test_snr.py`, `tests/test_sensorfilter.py` — `.value` → `["snr"]`.
- `notebooks/01,04,05,06` — update `get_snr` cells and re-execute.

---

## Task 1: PSF `cache_key()` methods

**Files:**
- Modify: `src/wcc_etc/psfsim.py` (class `PSFSource` ~line 80, `AiryPSF` ~87, `_ResampledPSF` ~113, `DefocusPSF` ~135)
- Test: `tests/test_psfsim.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create/append `tests/test_psfsim.py`:

```python
import numpy as np
from wcc_etc.psfsim import AiryPSF, CustomPSF, DefocusPSF
from wcc_etc import DEFOCUS_1WAVE_PATH


def test_airy_cache_key_is_constant():
    assert AiryPSF().cache_key() == AiryPSF().cache_key()
    assert AiryPSF().cache_key() == ("AiryPSF",)


def test_resampled_cache_key_stable_per_object():
    data = np.ones((9, 9))
    psf = CustomPSF(data, src_um_per_pix=4.0)
    assert psf.cache_key() == psf.cache_key()              # stable across calls
    assert psf.cache_key()[0] == "CustomPSF"
    assert psf.cache_key()[1] == 4.0


def test_defocus_cache_key_uses_path():
    psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
    assert psf.cache_key()[0] == "DefocusPSF"
    assert psf.cache_key()[-1] == DEFOCUS_1WAVE_PATH
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd wcc-etc && pytest tests/test_psfsim.py -q`
Expected: FAIL — `AttributeError: 'AiryPSF' object has no attribute 'cache_key'`.

- [ ] **Step 3: Add the methods**

In `PSFSource` (base), add:

```python
    def cache_key(self):
        """Hashable key identifying this PSF for render caching."""
        return (type(self).__name__,)
```

In `_ResampledPSF`, add (after `render`):

```python
    def cache_key(self):
        return (type(self).__name__, self.src_um_per_pix, id(self._data))
```

In `DefocusPSF`, add (after `__init__`):

```python
    def cache_key(self):
        return (type(self).__name__, self.src_um_per_pix, self.path)
```

`AiryPSF` and `CustomPSF` inherit (`AiryPSF` → base; `CustomPSF` → `_ResampledPSF`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd wcc-etc && pytest tests/test_psfsim.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd wcc-etc && git add src/wcc_etc/psfsim.py tests/test_psfsim.py
git commit -m "Add cache_key() to PSF sources for render caching"
```

---

## Task 2: Render-bundle cache + invalidation; refactor `get_image_snr` (scalar)

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `__init__` (~line 121), `set_sensor` (~302), `set_telescope` (~322), `update` (~446), `get_image_snr` (~762), add `_image_render_bundle`.
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simulation.py`:

```python
def test_image_render_bundle_cached(monkeypatch):
    import wcc_etc.psfsim as psfsim
    sim = _bright_sim(16)
    calls = {"n": 0}
    orig = psfsim.AiryPSF.render
    def counting_render(self, ctx):
        calls["n"] += 1
        return orig(self, ctx)
    monkeypatch.setattr(psfsim.AiryPSF, "render", counting_render)
    a = sim.get_image_snr(time=30)["snr"]
    b = sim.get_image_snr(time=60)["snr"]
    assert calls["n"] == 1                       # rendered once, reused
    assert len(sim._image_render_bundle_cache) == 1
    assert b > a                                 # longer exposure -> higher SNR


def test_update_invalidates_render_cache():
    sim = _bright_sim(16)
    snr1 = sim.get_image_snr(time=60)["snr"]
    assert len(sim._image_render_bundle_cache) == 1
    sim.update(source__mag=20)                   # fainter source
    assert len(sim._image_render_bundle_cache) == 0
    assert len(sim._psf_profile) == 0            # stale-PSF bug fix
    snr2 = sim.get_image_snr(time=60)["snr"]
    assert snr2 < snr1


def test_update_jitter_clears_caches():
    sim = _bright_sim(16)
    sim.get_image_snr(time=60)
    sim.update(jitter_sigma=50)
    assert len(sim._image_render_bundle_cache) == 0
    assert len(sim._psf_profile) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd wcc-etc && pytest tests/test_simulation.py -k "render_bundle or invalidate or jitter_clears" -q`
Expected: FAIL — `AttributeError: 'Simulation' object has no attribute '_image_render_bundle_cache'`.

- [ ] **Step 3: Initialize and clear the caches**

In `__init__`, after `self._default_psf = None  # set by from_sensorfilter only` (line 121) add:

```python
        self._psf_profile = {}
        self._image_render_bundle_cache = {}
```

In `set_sensor`, change the line `self._psf_profile = {}  # reset the psf profile` to:

```python
        self._psf_profile = {}  # reset the psf profile
        self._image_render_bundle_cache = {}
```

In `set_telescope`, change `self._psf_profile = {} # reset the psf profile` to:

```python
        self._psf_profile = {} # reset the psf profile
        self._image_render_bundle_cache = {}
```

In `update`, after the final line `self._meta |= update_this` add:

```python
        # any parameter change can affect the PSF/count rates: drop caches
        self._psf_profile = {}
        self._image_render_bundle_cache = {}
```

- [ ] **Step 4: Add the `_image_render_bundle` helper**

Add this method to `Simulation` (place it just above `get_image_snr`, ~line 762):

```python
    def _image_render_bundle(self, psf, jitter_sigma_mas, npix, oversample):
        """
        Cached, time-independent inputs for the PSF-aware SNR/exptime path.

        Returns a dict with the rendered normalized PSF, the plate scale (mas),
        and the source/diffuse count *rates* (electrons / s). Memoized on
        render-affecting state; cleared by update()/set_sensor()/set_telescope().
        """
        from .psfsim import ImageSimulator

        cache = getattr(self, "_image_render_bundle_cache", None)
        if cache is None:
            cache = self._image_render_bundle_cache = {}
        key = (psf.cache_key(), jitter_sigma_mas, int(npix), int(oversample))
        if key in cache:
            return cache[key]

        imsim = ImageSimulator(self, npix=npix, oversample=oversample)
        ctx = imsim._context(jitter_sigma_mas=jitter_sigma_mas)
        psf_norm = psf.render(ctx)

        profile = self.psf_profile
        ee_at_aper = profile["ee_at_aper"]
        if ee_at_aper == 0:
            raise ValueError("ee_at_aper is zero; aperture radius is degenerate.")
        num_psf_pixels = profile["num_psf_pixels"]
        n_psf = num_psf_pixels.value if isinstance(num_psf_pixels, u.Quantity) else num_psf_pixels

        count_rates = self.get_countrates(units="e/s", as_dict=True)
        source_rate_total = (count_rates["source"] / ee_at_aper).to(u.electron / u.s).value
        diffuse_rate_per_pix = 0.0
        for name, rate in count_rates.items():
            if name == "source":
                continue
            diffuse_rate_per_pix += (rate / n_psf).to(u.electron / u.s).value

        bundle = {"psf_norm": psf_norm,
                  "plate_scale_mas": ctx.plate_scale_mas,
                  "source_rate_total": source_rate_total,
                  "diffuse_rate_per_pix": diffuse_rate_per_pix}
        cache[key] = bundle
        return bundle
```

- [ ] **Step 5: Refactor `get_image_snr` to use the bundle (scalar path unchanged)**

Replace the body of `get_image_snr` *below the docstring* (current lines 800-852) with:

```python
        from .psfsim import AiryPSF, aperture_snr_radial, select_aperture

        if time is None:
            time = self._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second

        n_reads = self._resolve_n_reads(n_reads)
        if psf is None:
            psf = self._default_psf if self._default_psf is not None else AiryPSF()

        b = self._image_render_bundle(psf, jitter_sigma_mas, npix, oversample)

        # default to the ETC aperture if no mode was requested
        if not optimize and r_aper_mas is None and ee_frac is None:
            r_aper_mas = self._meta.get("r_aper_mas")

        read_noise = self.sensor.read_noise.to(u.electron / u.pix).value * np.sqrt(n_reads)
        dark_rate_per_pix = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value

        t_sec = time.to(u.second).value
        source_e_total = b["source_rate_total"] * t_sec
        diffuse_per_pix = b["diffuse_rate_per_pix"] * t_sec
        dark_per_pix = dark_rate_per_pix * t_sec

        prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                                   source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
        idx = select_aperture(prof, r_aper_mas=r_aper_mas, ee_frac=ee_frac, optimize=optimize)

        return {"snr": float(prof["snr"][idx]),
                "signal_e": float(prof["signal_e"][idx]),
                "noise_e": float(prof["noise_e"][idx]),
                "enclosed_fraction": float(prof["enclosed_fraction"][idx]),
                "r_aper_mas": float(prof["r_mas"][idx]),
                "n_pix": int(prof["n_pix"][idx])}
```

(Array time is added in Task 3 — keep this scalar version now so existing scalar tests stay green.)

- [ ] **Step 6: Run tests**

Run: `cd wcc-etc && pytest tests/test_simulation.py -q`
Expected: PASS (new caching tests + all existing `get_image_snr` tests, including `test_get_image_snr_matches_get_snr_in_focus` which still compares to the 1D `get_snr` at this point).

- [ ] **Step 7: Commit**

```bash
cd wcc-etc && git add src/wcc_etc/simulation.py tests/test_simulation.py
git commit -m "Cache PSF render bundle; clear caches on update/set_*; refactor get_image_snr"
```

---

## Task 3: Array (vectorized) time support in `get_image_snr`

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_image_snr`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_simulation.py`:

```python
def test_get_image_snr_array_time():
    sim = _bright_sim(16)
    times = np.array([30., 60., 120.])
    out = sim.get_image_snr(time=times)
    assert np.shape(out["snr"]) == (3,)
    assert np.shape(out["n_pix"]) == (3,)
    assert out["n_pix"].dtype.kind == "i"
    # monotonic increasing SNR with exposure time
    assert out["snr"][0] < out["snr"][1] < out["snr"][2]
    # each element equals the scalar call
    for i, t in enumerate(times):
        assert out["snr"][i] == pytest.approx(sim.get_image_snr(time=float(t))["snr"], rel=1e-9)


def test_get_image_snr_scalar_still_dict_of_floats():
    sim = _bright_sim(16)
    out = sim.get_image_snr(time=60)
    assert isinstance(out["snr"], float)
    assert isinstance(out["n_pix"], int)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd wcc-etc && pytest tests/test_simulation.py -k "array_time" -q`
Expected: FAIL — broadcast `ValueError` (shapes `(3,)` vs the flattened PSF).

- [ ] **Step 3: Add scalar/array dispatch**

Replace the section of `get_image_snr` from `t_sec = time.to(u.second).value` through the final `return {...}` (the block added in Task 2 Step 5) with:

```python
        def _snr_at(t_sec):
            source_e_total = b["source_rate_total"] * t_sec
            diffuse_per_pix = b["diffuse_rate_per_pix"] * t_sec
            dark_per_pix = dark_rate_per_pix * t_sec
            prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                                       source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
            idx = select_aperture(prof, r_aper_mas=r_aper_mas, ee_frac=ee_frac, optimize=optimize)
            return {"snr": float(prof["snr"][idx]),
                    "signal_e": float(prof["signal_e"][idx]),
                    "noise_e": float(prof["noise_e"][idx]),
                    "enclosed_fraction": float(prof["enclosed_fraction"][idx]),
                    "r_aper_mas": float(prof["r_mas"][idx]),
                    "n_pix": int(prof["n_pix"][idx])}

        if time.isscalar:
            return _snr_at(time.to(u.second).value)

        results = [_snr_at(t) for t in time.to(u.second).value]
        out = {k: np.array([r[k] for r in results])
               for k in ("snr", "signal_e", "noise_e", "enclosed_fraction", "r_aper_mas")}
        out["n_pix"] = np.array([r["n_pix"] for r in results], dtype=int)
        return out
```

- [ ] **Step 4: Run tests**

Run: `cd wcc-etc && pytest tests/test_simulation.py -k "array_time or scalar_still" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd wcc-etc && git add src/wcc_etc/simulation.py tests/test_simulation.py
git commit -m "Support array exposure times in get_image_snr (vectorized dict)"
```

---

## Task 4: `get_snr_airy` (deprecated) + `get_snr` delegation + caller updates

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_snr` (~910)
- Modify: `src/wcc_etc/wcc_etc.py:946`
- Modify: `tests/test_snr.py:23`, `tests/test_sensorfilter.py:133`, `tests/test_simulation.py:168`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simulation.py`:

```python
def test_get_snr_airy_deprecated_matches_analytic():
    sim = _bright_sim(16)
    with pytest.warns(DeprecationWarning):
        airy = sim.get_snr_airy(60)
    signal, variance = sim.get_signal_and_variance(60)
    assert float(airy.value) == pytest.approx(float((signal / np.sqrt(variance)).value), rel=1e-12)


def test_get_snr_delegates_to_image_snr():
    sim = _bright_sim(16)
    assert sim.get_snr(60)["snr"] == pytest.approx(sim.get_image_snr(time=60)["snr"], rel=1e-12)


def test_get_snr_array_time():
    sim = _bright_sim(16)
    out = sim.get_snr(np.array([30., 60., 120.]))
    assert out["snr"][0] < out["snr"][1] < out["snr"][2]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd wcc-etc && pytest tests/test_simulation.py -k "get_snr_airy or delegates or get_snr_array" -q`
Expected: FAIL — `AttributeError: ... 'get_snr_airy'` and `get_snr(60)` returns a Quantity (no `["snr"]`).

- [ ] **Step 3: Replace `get_snr` and add `get_snr_airy`**

Replace the entire current `get_snr` method (lines 910-928) with these two methods:

```python
    def get_snr(self, time=None, psf=None, r_aper_mas=None, ee_frac=None,
                optimize=False, jitter_sigma_mas=None, n_reads=None,
                npix=128, oversample=11):
        """
        Signal-to-noise ratio via the 2D image simulation (PSF-aware default).

        Delegates to get_image_snr; see it for parameter details and the
        returned dict. `time` may be a scalar or an array (returns dict of
        arrays). For the legacy analytic Airy approximation use get_snr_airy
        (deprecated).
        """
        return self.get_image_snr(
            time=time, psf=psf, r_aper_mas=r_aper_mas, ee_frac=ee_frac,
            optimize=optimize, jitter_sigma_mas=jitter_sigma_mas,
            n_reads=n_reads, npix=npix, oversample=oversample)

    def get_snr_airy(self, time=None, n_reads=None):
        """
        DEPRECATED analytic Airy-disk SNR (the 1D approximation).

        Use get_snr, which computes the SNR via the 2D image simulation.

        Parameters
        ----------
        time : float or Quantity, optional
            Exposure time.
        n_reads : int, optional
            Number of reads. Defaults to meta['n_reads'] if set, else 1.

        Returns
        -------
        Quantity
            The SNR.
        """
        warnings.warn(
            "get_snr_airy (analytic Airy approximation) is deprecated; "
            "use get_snr, which now uses the 2D image simulation.",
            DeprecationWarning, stacklevel=2)
        signal, variance = self.get_signal_and_variance(time, n_reads=n_reads)
        return signal / np.sqrt(variance)
```

(Confirm `import warnings` is already present at the top of `simulation.py` — it is, used by `update`.)

- [ ] **Step 4: Update the in-repo callers**

`src/wcc_etc/wcc_etc.py:946` — change:

```python
    snr = simu.get_snr(texp)
```
to:
```python
    snr = simu.get_snr(texp)["snr"]
```

`tests/test_snr.py:23` — change:
```python
    snr_val = simu.get_snr(time = 60).value
```
to:
```python
    snr_val = simu.get_snr(time = 60)["snr"]
```

`tests/test_sensorfilter.py:131-134` — change:
```python
    # basic sanity: can compute an analytic SNR
    snr = sim.get_snr(60)
    assert snr.value > 0
```
to:
```python
    # basic sanity: can compute a (2D) SNR
    assert sim.get_snr(60)["snr"] > 0
```

`tests/test_simulation.py` — the cross-check `test_get_image_snr_matches_get_snr_in_focus` (currently lines 168-176) must compare the 2D path to the *analytic* path, so call `get_snr_airy`:
```python
def test_get_image_snr_matches_get_snr_in_focus():
    sim = _bright_sim(16)
    for t in [30, 300]:
        for m in [16, 20]:
            sim.update(source__mag=m)
            with pytest.warns(DeprecationWarning):
                etc = sim.get_snr_airy(t)
            etc = float(etc.value) if hasattr(etc, "value") else float(etc)
            img = sim.get_image_snr(time=t)["snr"]
            assert img == pytest.approx(etc, rel=0.03)
```

- [ ] **Step 5: Run tests**

Run: `cd wcc-etc && pytest tests/test_simulation.py tests/test_snr.py tests/test_sensorfilter.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd wcc-etc && git add src/wcc_etc/simulation.py src/wcc_etc/wcc_etc.py tests/test_simulation.py tests/test_snr.py tests/test_sensorfilter.py
git commit -m "Make get_snr default to 2D image SNR; deprecate analytic path as get_snr_airy"
```

---

## Task 5: Refactor `get_image_exptime_for_snr` onto the bundle

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_image_exptime_for_snr` (~854)
- Test: `tests/test_simulation.py` (existing exptime tests must still pass)

- [ ] **Step 1: Establish the current value (characterization test)**

Append to `tests/test_simulation.py`:

```python
def test_image_exptime_for_snr_unchanged_after_refactor():
    sim = _bright_sim(16)
    target = 20.0
    res = sim.get_image_exptime_for_snr(target)
    # round-trips: the returned time reproduces the target SNR via get_image_snr
    snr_back = sim.get_image_snr(time=res["time_s"])["snr"]
    assert snr_back == pytest.approx(target, rel=0.02)
```

- [ ] **Step 2: Run to verify it passes against the current implementation**

Run: `cd wcc-etc && pytest tests/test_simulation.py -k "exptime_for_snr_unchanged" -q`
Expected: PASS (this characterizes existing behavior before refactor).

- [ ] **Step 3: Refactor to use the bundle**

Replace the body of `get_image_exptime_for_snr` *below the docstring* (current lines 873-908) with:

```python
        from .psfsim import AiryPSF, aperture_time_for_snr

        n_reads = self._resolve_n_reads(n_reads)
        if psf is None:
            psf = self._default_psf if self._default_psf is not None else AiryPSF()

        b = self._image_render_bundle(psf, jitter_sigma_mas, npix, oversample)

        dark_rate_per_pix = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        read_noise = self.sensor.read_noise.to(u.electron / u.pix).value

        if not optimize and r_aper_mas is None and ee_frac is None:
            r_aper_mas = self._meta.get("r_aper_mas")

        return aperture_time_for_snr(b["psf_norm"], b["plate_scale_mas"],
                                     b["source_rate_total"], b["diffuse_rate_per_pix"],
                                     dark_rate_per_pix, read_noise,
                                     n_reads=n_reads, snr=snr,
                                     r_aper_mas=r_aper_mas, ee_frac=ee_frac,
                                     optimize=optimize)
```

- [ ] **Step 4: Run tests**

Run: `cd wcc-etc && pytest tests/test_simulation.py -q`
Expected: PASS (characterization test + all existing exptime tests).

- [ ] **Step 5: Full suite**

Run: `cd wcc-etc && pytest -q`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
cd wcc-etc && git add src/wcc_etc/simulation.py tests/test_simulation.py
git commit -m "Reuse render bundle in get_image_exptime_for_snr (DRY)"
```

---

## Task 6: Update and re-execute notebooks

**Files:**
- Modify + re-execute: `notebooks/01_getting_started.ipynb`, `notebooks/04_psf_and_image_snr.ipynb`, `notebooks/05_n_reads_exptime.ipynb`, `notebooks/06_source_spectra.ipynb`

`get_snr` now returns a dict (scalar or array values), so every `get_snr(...).value`
or scalar-SNR use must become `get_snr(...)["snr"]`. Cells that intentionally show
the *analytic* value should call `get_snr_airy(...)` instead.

Use the **notebook-demo** skill to edit each notebook's builder script and
re-execute. Concrete edits per notebook:

- [ ] **Step 1: `01_getting_started.ipynb`**
  - `sim.get_snr(time=60)` → `sim.get_snr(time=60)["snr"]`
  - `sim.get_snr(time=times)` (array curve) → `sim.get_snr(time=times)["snr"]`
    (now returns an array — the SNR-vs-time plot works unchanged downstream).
  - Any `print(... get_snr(60))` → `... get_snr(60)["snr"]`.
  - Update surrounding markdown: "`get_snr(time)` returns the SNR ..." → note it
    returns a PSF-aware dict; `["snr"]` is the scalar/array SNR.

- [ ] **Step 2: `04_psf_and_image_snr.ipynb`**
  - `sim.get_snr(60).value` → `sim.get_snr(60)["snr"]` (both occurrences).
  - If a cell contrasts analytic vs image SNR, use `get_snr_airy(60).value` for
    the analytic side and `get_snr(60)["snr"]` for the 2D side.

- [ ] **Step 3: `05_n_reads_exptime.ipynb`**
  - `sim.get_snr(t).value` → `sim.get_snr(t)["snr"]` (all occurrences, incl. the
    f-strings printing `get_snr(t)` and `get_snr(fixed_t)`).

- [ ] **Step 4: `06_source_spectra.ipynb`**
  - `[... get_snr(60).value for m in mags]` → `[... get_snr(60)["snr"] for m in mags]`.
  - Remaining `get_snr(60).value` → `get_snr(60)["snr"]`.

- [ ] **Step 5: Re-execute all four notebooks clean (zero error outputs)** via the notebook-demo skill's execution path.

- [ ] **Step 6: Commit**

```bash
cd wcc-etc && git add notebooks/01_getting_started.ipynb notebooks/04_psf_and_image_snr.ipynb notebooks/05_n_reads_exptime.ipynb notebooks/06_source_spectra.ipynb
git commit -m "Update example notebooks for 2D get_snr dict return"
```

---

## Final verification

- [ ] `cd wcc-etc && pytest -q` — full suite green.
- [ ] `cd wcc-etc && python -c "import warnings; warnings.simplefilter('error', DeprecationWarning); import wcc_etc, numpy as np; s=wcc_etc.Simulation.from_sensor_and_scene('sony:r', wcc_etc.get_scene(name='G5V', mag=18, background='zodi', bandpass='johnson_r', background_prop={'bandpass':'johnson_r','mag':22.5})); print(s.get_snr(60)['snr']); print(s.get_snr(np.array([30.,60.,120.]))['snr'])"` — confirms `get_snr` does NOT emit a DeprecationWarning and array time works.
- [ ] Update memory `project-status.md`: `get_snr` now defaults to the 2D image SNR (dict return, array-time aware) with a render bundle cache; `get_snr_airy` is the deprecated 1D path; the stale-PSF-after-`update` bug and the count-rate DRY duplication are resolved.
- [ ] Use `superpowers:finishing-a-development-branch` to open the PR.

## Self-review notes

- **Spec coverage:** deprecate 1D (Task 4) ✓; 2D default (Task 4) ✓; render cache (Task 2) ✓; cache invalidation + stale-PSF fix (Task 2) ✓; array-time (Task 3) ✓; DRY consolidation in exptime (Task 5) ✓; caller/test/notebook updates (Tasks 4, 6) ✓.
- **Type consistency:** bundle keys (`psf_norm`, `plate_scale_mas`, `source_rate_total`, `diffuse_rate_per_pix`) used identically in `_image_render_bundle`, `get_image_snr`, and `get_image_exptime_for_snr`. `cache_key()` defined in Task 1, consumed in Task 2. `get_image_snr` dict keys (`snr`, `signal_e`, `noise_e`, `enclosed_fraction`, `r_aper_mas`, `n_pix`) consistent scalar/array.
- **Green at every boundary:** Task 2 keeps `get_image_snr` scalar-identical; the cross-check test is only switched to `get_snr_airy` in Task 4, the same task that changes `get_snr`'s return type — so the suite never goes red between tasks.
