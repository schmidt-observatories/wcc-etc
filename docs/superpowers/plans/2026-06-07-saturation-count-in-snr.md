# Saturated-Pixel Count + Warning on the SNR / Exptime Paths — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report the number of saturated pixels in the rendered image from `get_snr`/`get_image_snr` and `get_image_exptime_for_snr` (added to their result dicts), warn when saturation occurs (with a `warn=` toggle), and emit a warning-only on the deprecated analytic paths.

**Architecture:** Extract the saturation-mask test already used by `psfsim.ImageSimulator.simulate` into a reusable helper, then call it from the SNR/exptime methods on a **per-frame clean electron image** (`time / n_reads`, no noise) formed from the data already in `_image_render_bundle`. The per-frame, clean convention matches `get_peak_pixel`/`is_saturated`. The mask itself is the rendered-image mask (full-well OR ADC clip, bias-free), which is **broader** than `is_saturated` (ADC clip only): `is_saturated == True` implies `n_saturated > 0`, not the converse. SNR/exptime math is untouched.

**Tech Stack:** Python, numpy, astropy units, pytest. Spec: `docs/superpowers/specs/2026-06-07-saturation-count-in-snr-design.md`.

**Working directory note:** The git repo is the nested `wcc-etc/` subdir. Run all `git`/`pytest` commands from `/Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc`. Branch: `saturation-count-in-snr` (already checked out). Commit messages: plain, no `Co-Authored-By` trailer.

---

## File Structure

- **Modify** `src/wcc_etc/psfsim.py`
  - Add module-level `saturation_mask_from_image_e(sensor, image_e)`.
  - Refactor `ImageSimulator.simulate_image` (~lines 341–348) to call it.
- **Modify** `src/wcc_etc/simulation.py`
  - `get_image_snr` (`_snr_at`, `_assemble_array`, the three return sites) — add `n_saturated`/`saturated` + `warn=` (~lines 819–937).
  - `get_snr` — pass `warn=` through (~lines 979–1000).
  - `get_image_exptime_for_snr` — add `n_saturated`/`saturated` + `warn=` (~lines 939–977).
  - `get_snr_airy` — warning-only + `warn=` (~lines 1002–1025).
  - `get_exptime_for_snr` — warning-only + `warn=` (~lines 1027–1034).
- **Create** `tests/test_saturation_count.py` — all new tests for this feature.

Existing test fixtures to mirror (`_sim(mag)` building a `sony:r` G5V/zodi scene): `tests/test_saturation_reads.py`, `tests/test_image_exptime.py`. Reference physics: `sony:r` mag 17 saturates a single 60-s frame; mag 12 is brighter (saturates); mag 25.4 is faint (does not saturate).

---

## Task 1: Extract the saturation-mask helper (single source of truth)

**Files:**
- Modify: `src/wcc_etc/psfsim.py` (add helper near the other module functions; refactor `simulate_image` ~341–348)
- Test: `tests/test_saturation_count.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_saturation_count.py` with:

```python
import warnings

import numpy as np
import pytest

import wcc_etc
from wcc_etc.psfsim import ImageSimulator, AiryPSF, saturation_mask_from_image_e


def _sim(mag):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_helper_matches_simulate_image_mask():
    # The extracted helper must reproduce simulate_image's saturation_mask exactly.
    sim = _sim(12)
    imsim = ImageSimulator(sim, npix=128, oversample=11)
    res = imsim.simulate_image(time=60, psf=AiryPSF(), add_noise=False)
    expected = res.saturation_mask
    got = saturation_mask_from_image_e(sim.sensor, res.image_e)
    assert np.array_equal(got, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_saturation_count.py::test_helper_matches_simulate_image_mask -v`
Expected: FAIL with `ImportError: cannot import name 'saturation_mask_from_image_e'`.

- [ ] **Step 3: Add the helper and refactor `simulate_image`**

In `src/wcc_etc/psfsim.py`, add this module-level function (place it just above `class SimulatedImage:`, near line 168, so it is importable as `wcc_etc.psfsim.saturation_mask_from_image_e`):

```python
def saturation_mask_from_image_e(sensor, image_e):
    """Boolean mask of pixels at/over the ADC full scale or the full well.

    Parameters
    ----------
    sensor : Sensor
        Provides gain, adc_max, and (optionally) meta['well_depth'].
    image_e : ndarray
        Per-pixel charge in electrons (per frame for saturation tests).

    Returns
    -------
    ndarray of bool
        True where (image_e / gain) >= adc_max, OR image_e >= well_depth
        when a well_depth is configured.
    """
    gain = sensor.gain.to(u.electron / u.ct).value
    mask = (image_e / gain) >= sensor.adc_max.to(u.ct).value
    well_depth = sensor.meta.get("well_depth")
    if well_depth is not None:
        mask = mask | (image_e >= well_depth)
    return mask
```

Then in `ImageSimulator.simulate_image`, replace the inline mask block (currently around lines 341–348):

```python
        gain = sim.sensor.gain.to(u.electron / u.ct).value
        bias_level = sim.sensor.bias_level.to(u.ct).value
        adc_max = sim.sensor.adc_max.to(u.ct).value
        well_depth = sim.sensor.meta.get("well_depth")

        saturation_mask = (image_e / gain) >= adc_max
        if well_depth is not None:
            saturation_mask = saturation_mask | (image_e >= well_depth)
```

with:

```python
        gain = sim.sensor.gain.to(u.electron / u.ct).value
        bias_level = sim.sensor.bias_level.to(u.ct).value

        saturation_mask = saturation_mask_from_image_e(sim.sensor, image_e)
```

(`gain` and `bias_level` are still passed to `SimulatedImage(...)` below, so keep them. The local `adc_max`/`well_depth` are no longer needed.)

- [ ] **Step 4: Run test to verify it passes (and nothing regressed)**

Run: `pytest tests/test_saturation_count.py::test_helper_matches_simulate_image_mask tests/test_image_simulator.py -v`
Expected: PASS for the new test and all of `test_image_simulator.py`.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_saturation_count.py
git commit -m "Extract saturation_mask_from_image_e helper; reuse in simulate_image"
```

---

## Task 2: Saturated-pixel count + warning in `get_image_snr` / `get_snr`

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_image_snr` (~819–937), `get_snr` (~979–1000)
- Test: `tests/test_saturation_count.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_saturation_count.py`:

```python
def test_get_snr_faint_no_saturation_no_warning():
    sim = _sim(25.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning -> error
        res = sim.get_snr(time=60)
    assert res["n_saturated"] == 0
    assert res["saturated"] is False


def test_get_snr_bright_counts_and_warns():
    sim = _sim(12)                               # bright: saturates a 60-s frame
    with pytest.warns(UserWarning, match="saturat"):
        res = sim.get_snr(time=60)
    assert res["n_saturated"] > 0
    assert res["saturated"] is True


def test_get_snr_warn_false_silences_but_keeps_count():
    sim = _sim(12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sim.get_snr(time=60, warn=False)
    assert res["n_saturated"] > 0
    assert res["saturated"] is True


def test_get_snr_array_time_returns_count_array():
    sim = _sim(12)
    res = sim.get_snr(time=[30, 60, 120], warn=False)
    assert res["n_saturated"].shape == (3,)
    assert res["n_saturated"].dtype.kind == "i"
    assert res["saturated"].dtype == bool
    # more exposure -> at least as many saturated pixels (monotone, per-frame)
    assert np.all(np.diff(res["n_saturated"]) >= 0)


def test_get_snr_values_unchanged_regression():
    # adding the count must not change the SNR itself
    sim = _sim(25.4)
    snr = sim.get_snr(time=60)["snr"]
    assert np.isclose(snr, sim.get_image_snr(time=60, warn=False)["snr"])


def test_get_snr_count_superset_of_is_saturated():
    # is_saturated (ADC clip only) implies n_saturated > 0 (ADC clip OR full well).
    # Not iff: n_saturated can exceed 0 via full well alone. At mag 12 both hold.
    sim = _sim(12)
    res = sim.get_snr(time=60, warn=False)
    assert bool(sim.is_saturated(60))            # mag 12 clips the ADC at 60 s
    assert res["n_saturated"] > 0                 # ... so the image mask must agree
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_saturation_count.py -k "get_snr" -v`
Expected: FAIL with `KeyError: 'n_saturated'` (and/or unexpected-warning errors).

- [ ] **Step 3: Implement in `get_image_snr` and `get_snr`**

In `src/wcc_etc/simulation.py`:

(a) Extend the `get_image_snr` import (currently line ~869) to pull in the helper:

```python
        from .psfsim import (AiryPSF, aperture_snr_radial, select_aperture,
                             saturation_mask_from_image_e)
```

(b) Add `warn=True` to the `get_image_snr` signature. Change (lines ~819–821):

```python
    def get_image_snr(self, time=None, mags=None, psf=None, r_aper_mas=None,
                      ee_frac=None, optimize=False, jitter_sigma_mas=None,
                      n_reads=None, npix=128, oversample=11):
```

to:

```python
    def get_image_snr(self, time=None, mags=None, psf=None, r_aper_mas=None,
                      ee_frac=None, optimize=False, jitter_sigma_mas=None,
                      n_reads=None, npix=128, oversample=11, warn=True):
```

(c) Replace the `_snr_at` closure (lines ~904–916) with a version that also counts saturated pixels on the per-frame clean image:

```python
        def _snr_at(t_sec, source_scale=1.0):
            source_e_total = b["source_rate_total"] * source_scale * t_sec
            diffuse_per_pix = b["diffuse_rate_per_pix"] * t_sec
            dark_per_pix = dark_rate_per_pix * t_sec
            prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                                       source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
            idx = select_aperture(prof, r_aper_mas=r_aper_mas, ee_frac=ee_frac, optimize=optimize)
            # per-frame clean electron image -> saturated-pixel count
            # (matches get_peak_pixel/is_saturated: saturation is per-frame)
            tf = t_sec / n_reads
            image_e = (b["source_rate_total"] * source_scale * tf) * b["psf_norm"] \
                      + b["diffuse_rate_per_pix"] * tf + dark_rate_per_pix * tf
            mask = saturation_mask_from_image_e(self.sensor, image_e)
            return {"snr": float(prof["snr"][idx]),
                    "signal_e": float(prof["signal_e"][idx]),
                    "noise_e": float(prof["noise_e"][idx]),
                    "enclosed_fraction": float(prof["enclosed_fraction"][idx]),
                    "r_aper_mas": float(prof["r_mas"][idx]),
                    "n_pix": int(prof["n_pix"][idx]),
                    "n_saturated": int(mask.sum()),
                    "saturated": bool(mask.any())}
```

(d) Replace `_assemble_array` (lines ~918–922) so the new keys become arrays, and add a `_maybe_warn` helper right after it:

```python
        def _assemble_array(results):
            out = {k: np.array([r[k] for r in results])
                   for k in ("snr", "signal_e", "noise_e", "enclosed_fraction", "r_aper_mas")}
            out["n_pix"] = np.array([r["n_pix"] for r in results], dtype=int)
            out["n_saturated"] = np.array([r["n_saturated"] for r in results], dtype=int)
            out["saturated"] = np.array([r["saturated"] for r in results], dtype=bool)
            return out

        def _maybe_warn(result):
            if warn and bool(np.any(result["saturated"])):
                n = result["n_saturated"]
                n_max = int(np.max(n)) if np.ndim(n) else int(n)
                warnings.warn(
                    f"peak pixel saturates the detector — up to {n_max} pixel(s) "
                    f"at or above full well / ADC clip (per-frame, n_reads={n_reads}); "
                    f"SNR is unreliable. See is_saturated/get_peak_pixel.",
                    stacklevel=2)
            return result
```

(e) Wrap all three return sites in `_maybe_warn`. The mags-array return (line ~930):

```python
            return _maybe_warn(_assemble_array([_snr_at(t_sec, s) for s in scales]))
```

The scalar-time return (line ~935):

```python
        if time.isscalar:
            return _maybe_warn(_snr_at(time.to(u.second).value, source_scale))
```

The time-array return (line ~937):

```python
        return _maybe_warn(_assemble_array(results))
```

(f) Add `warn=True` to `get_snr` and pass it through. Change the signature (lines ~979–981):

```python
    def get_snr(self, time=None, psf=None, r_aper_mas=None, ee_frac=None,
                optimize=False, jitter_sigma_mas=None, n_reads=None,
                npix=128, oversample=11, warn=True):
```

and the delegating call (lines ~997–1000):

```python
        return self.get_image_snr(
            time=time, psf=psf, r_aper_mas=r_aper_mas, ee_frac=ee_frac,
            optimize=optimize, jitter_sigma_mas=jitter_sigma_mas,
            n_reads=n_reads, npix=npix, oversample=oversample, warn=warn)
```

(`warnings` is already imported in `simulation.py` — confirm the `import warnings` at the top; it is used by `get_snr_airy`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_saturation_count.py -k "get_snr" tests/test_snr.py -v`
Expected: PASS for all new `get_snr` tests and the existing `test_snr.py` suite (regression).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_saturation_count.py
git commit -m "get_snr/get_image_snr: return n_saturated + saturated, warn on saturation"
```

---

## Task 3: Saturated-pixel count + warning in `get_image_exptime_for_snr`

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_image_exptime_for_snr` (~939–977)
- Test: `tests/test_saturation_count.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_saturation_count.py`:

```python
def test_exptime_for_snr_reports_count_keys():
    sim = _sim(25.4)                              # faint: solved time does not saturate
    res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
    assert isinstance(res["n_saturated"], int)
    assert res["saturated"] is False
    assert res["n_saturated"] == 0


def test_exptime_count_consistent_with_get_image_snr():
    # count at the solved time must equal what get_image_snr reports at that time
    sim = _sim(12)
    res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
    chk = sim.get_image_snr(time=res["time_s"], r_aper_mas=70, warn=False)
    assert res["n_saturated"] == chk["n_saturated"]
    assert res["saturated"] == chk["saturated"]


def test_exptime_for_snr_warns_when_solved_time_saturates():
    # bright source + very high target SNR -> long exposure -> saturates
    sim = _sim(12)
    with pytest.warns(UserWarning, match="saturat"):
        sim.get_image_exptime_for_snr(1e5, r_aper_mas=70)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_saturation_count.py -k "exptime" -v`
Expected: FAIL with `KeyError: 'n_saturated'`.

- [ ] **Step 3: Implement in `get_image_exptime_for_snr`**

In `src/wcc_etc/simulation.py`:

(a) Extend the method's import (currently line ~958):

```python
        from .psfsim import AiryPSF, aperture_time_for_snr, saturation_mask_from_image_e
```

(b) Add `warn=True` to the signature (lines ~939–942):

```python
    def get_image_exptime_for_snr(self, snr, psf=None, r_aper_mas=None,
                                  ee_frac=None, optimize=False,
                                  jitter_sigma_mas=None, n_reads=None,
                                  npix=128, oversample=11, warn=True):
```

(c) Replace the final `return aperture_time_for_snr(...)` block (lines ~972–977) with:

```python
        result = aperture_time_for_snr(b["psf_norm"], b["plate_scale_mas"],
                                       b["source_rate_total"], b["diffuse_rate_per_pix"],
                                       dark_rate_per_pix, read_noise,
                                       n_reads=n_reads, snr=snr,
                                       r_aper_mas=r_aper_mas, ee_frac=ee_frac,
                                       optimize=optimize)

        # saturated-pixel count at the solved time (per-frame; matches is_saturated)
        tf = result["time_s"] / n_reads
        image_e = (b["source_rate_total"] * tf) * b["psf_norm"] \
                  + b["diffuse_rate_per_pix"] * tf + dark_rate_per_pix * tf
        mask = saturation_mask_from_image_e(self.sensor, image_e)
        result["n_saturated"] = int(mask.sum())
        result["saturated"] = bool(mask.any())
        if warn and result["saturated"]:
            warnings.warn(
                f"peak pixel saturates the detector at the solved time "
                f"({result['time_s']:.3g} s) — {result['n_saturated']} pixel(s) at "
                f"or above full well / ADC clip (per-frame, n_reads={n_reads}).",
                stacklevel=2)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_saturation_count.py -k "exptime" tests/test_image_exptime.py -v`
Expected: PASS for all new exptime tests and the existing `test_image_exptime.py` suite.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_saturation_count.py
git commit -m "get_image_exptime_for_snr: report n_saturated + saturated, warn on saturation"
```

---

## Task 4: Warning-only on the deprecated analytic paths

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_snr_airy` (~1002–1025), `get_exptime_for_snr` (~1027–1034)
- Test: `tests/test_saturation_count.py`

Note: return types/values of these two methods are unchanged (bare `Quantity`); only a `UserWarning` is added, gated by `warn=`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_saturation_count.py`:

```python
import astropy.units as u


def test_get_snr_airy_warns_on_saturation_and_keeps_type():
    sim = _sim(12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val = sim.get_snr_airy(60)
    assert isinstance(val, u.Quantity)            # return type unchanged
    assert any(issubclass(w.category, UserWarning) and "saturat" in str(w.message)
               for w in caught)


def test_get_snr_airy_warn_false_no_saturation_warning():
    sim = _sim(12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.get_snr_airy(60, warn=False)
    assert not any(issubclass(w.category, UserWarning) and "saturat" in str(w.message)
                   for w in caught)


def test_get_exptime_for_snr_warns_when_solved_time_saturates():
    sim = _sim(12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = sim.get_exptime_for_snr(1e5)          # huge SNR -> long time -> saturates
    assert isinstance(t, u.Quantity)              # return type unchanged
    assert any(issubclass(w.category, UserWarning) and "saturat" in str(w.message)
               for w in caught)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_saturation_count.py -k "airy or exptime_for_snr_warns" -v`
Expected: FAIL — `get_snr_airy`/`get_exptime_for_snr` take no `warn=` kwarg (TypeError) and/or emit no saturation `UserWarning`.

- [ ] **Step 3: Implement the warning-only paths**

In `src/wcc_etc/simulation.py`:

(a) `get_snr_airy` — add `warn=True` to the signature (line ~1002):

```python
    def get_snr_airy(self, time=None, n_reads=None, warn=True):
```

The existing body ends with these two lines (after the `DeprecationWarning`):

```python
        signal, variance = self.get_signal_and_variance(time, n_reads=n_reads)
        return signal / np.sqrt(variance)
```

Insert the saturation-warning block **between** those two lines (after `get_signal_and_variance`, before `return`) so the existing `get_signal_and_variance` call is reused and not duplicated. The result is:

```python
        signal, variance = self.get_signal_and_variance(time, n_reads=n_reads)
        if warn:
            try:
                sat = self.is_saturated(time, n_reads=n_reads)
            except ValueError:
                sat = False                       # no time resolvable; skip the saturation check
            if bool(np.any(sat)):
                warnings.warn(
                    "detector saturates (per-frame); the analytic Airy SNR is "
                    "unreliable. Use get_snr for the saturated-pixel count.",
                    stacklevel=2)
        return signal / np.sqrt(variance)
```

(b) `get_exptime_for_snr` — add `warn=True` to the signature (line ~1027) and warn if the solved (finite) time saturates:

```python
    def get_exptime_for_snr(self, snr, n_reads=None, warn=True):
        """
        Exposure time (seconds, Quantity) to reach a target SNR on the analytic
        (Airy) path. Inverse of get_snr_airy. Returns inf*u.s if the source rate is 0.
        """
        from .psfsim import solve_time_for_snr
        A, B, C = self._snr_coefficients(n_reads=n_reads)
        t = solve_time_for_snr(snr, A, B, C) * u.second
        if warn and np.isfinite(t.value) and bool(np.any(self.is_saturated(t, n_reads=n_reads))):
            warnings.warn(
                f"detector saturates at the solved exposure time ({t:.3g}); "
                "use get_image_exptime_for_snr for the saturated-pixel count.",
                stacklevel=2)
        return t
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_saturation_count.py -k "airy or exptime_for_snr_warns" tests/test_exptime.py -v`
Expected: PASS for the new tests and the existing `test_exptime.py` suite.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_saturation_count.py
git commit -m "Deprecated analytic SNR/exptime paths: warn on saturation (return types unchanged)"
```

---

## Task 5: Full-suite regression + final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass (new `tests/test_saturation_count.py` plus the existing suite — in particular `test_snr.py`, `test_image_exptime.py`, `test_image_simulator.py`, `test_saturation_reads.py`, `test_exptime.py`).

- [ ] **Step 2: Sanity-check the new keys interactively (optional)**

Run:
```bash
python -c "
import warnings, wcc_etc
s = wcc_etc.get_scene(name='G5V', mag=12, background='zodi', bandpass='johnson_r', background_prop={'bandpass':'johnson_r','mag':22.5})
sim = wcc_etc.Simulation.from_sensor_and_scene('sony:r', s)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    r = sim.get_snr(time=60)
print('n_saturated=', r['n_saturated'], 'saturated=', r['saturated'])
print('warning=', [str(x.message) for x in w])
"
```
Expected: `n_saturated` > 0, `saturated=True`, and one saturation `UserWarning`.

- [ ] **Step 3: Commit (only if Step 2 added anything; otherwise skip)**

No commit needed if no files changed.

---

## Notes for the implementer

- **Do not change SNR/exptime math.** `n_reads` still enters SNR only via the read-noise term. The `/ n_reads` division appears **only** in the saturation-image construction, to match `get_peak_pixel`/`is_saturated`.
- **Per-frame, clean image.** No Poisson/read noise in the saturation count — it is a deterministic ETC quantity, consistent with `is_saturated`.
- **`warn=` never affects the returned count** — it only gates `warnings.warn`.
- Line numbers above are approximate (from the spec date); locate by the surrounding code shown in each block.
- After merge (separate follow-ups, per the feature workflow): add a demo-notebook cell showing `n_saturated` vs exposure time/brightness, and update the `project-status` memory with the new dict keys (`n_saturated`, `saturated`) and the `warn=` kwarg.
