# 2D image simulation as the default SNR

**Date:** 2026-06-06
**Status:** Approved (brainstorm)

## Problem

`Simulation.get_snr` returns the analytic Airy-disk SNR (the "1D approximation"):
it derives `signal / sqrt(variance)` from `get_signal_and_variance`, assuming a
diffraction-limited Airy PSF. The PSF-aware `get_image_snr` renders the actual
PSF on the detector grid and is the more accurate calculation, but it is opt-in.

We want the accurate 2D path to be the default, the old 1D path preserved behind
a deprecation warning, and the 2D path cached so repeated calls (e.g. time or
aperture sweeps) don't re-render the PSF every time.

## Goals

1. `get_snr` computes the SNR through the 2D image simulation by default.
2. The old 1D (analytic Airy) calculation is preserved in a deprecated method
   that raises a `DeprecationWarning`.
3. The 2D path caches the expensive, time-independent work so subsequent calls
   are fast.
4. Surgical, minimal edits.

## Non-goals

- No change to `get_exptime_for_snr` / `get_image_exptime_for_snr` *semantics*
  (the 1D inverse stays as-is; only its shared render is cached).
- No new SNR physics. The 2D path (`get_image_snr`) already exists and is
  unchanged in behavior — it reproduces the in-focus analytic result to ~1%.
- No redshift/extinction/etc.

## Design

### 1. Deprecate the 1D path — `get_snr_airy`

Move the current `get_snr(time, n_reads)` body verbatim into a new method:

```python
def get_snr_airy(self, time=None, n_reads=None):
    """
    DEPRECATED analytic Airy-disk SNR (the 1D approximation).

    Use get_snr, which now computes the SNR via the 2D image simulation.
    """
    warnings.warn(
        "get_snr_airy (analytic Airy approximation) is deprecated; "
        "use get_snr, which now uses the 2D image simulation.",
        DeprecationWarning, stacklevel=2)
    signal, variance = self.get_signal_and_variance(time, n_reads=n_reads)
    return signal / np.sqrt(variance)
```

Return type and math are unchanged (scalar dimensionless Quantity).
`get_signal_and_variance` stays as-is (still used here).

### 2. `get_snr` becomes the 2D default

`get_snr` takes the same signature as `get_image_snr` and delegates to it,
returning the **dict** (`{'snr', 'signal_e', 'noise_e', 'enclosed_fraction',
'r_aper_mas', 'n_pix'}`):

```python
def get_snr(self, time=None, psf=None, r_aper_mas=None, ee_frac=None,
            optimize=False, jitter_sigma_mas=None, n_reads=None,
            npix=128, oversample=11):
    """
    Signal-to-noise ratio via the 2D image simulation (PSF-aware).

    This is the default SNR calculation. See get_image_snr for parameter
    details. For the legacy analytic Airy approximation, see get_snr_airy
    (deprecated).
    """
    return self.get_image_snr(
        time=time, psf=psf, r_aper_mas=r_aper_mas, ee_frac=ee_frac,
        optimize=optimize, jitter_sigma_mas=jitter_sigma_mas,
        n_reads=n_reads, npix=npix, oversample=oversample)
```

`get_image_snr` remains the canonical implementation (notebooks and tests
reference it directly).

### 3. Render-cache

The expensive, time-independent part of `get_image_snr` /
`get_image_exptime_for_snr` is: building the `ImageSimulator`, the render
context, rendering `psf_norm = psf.render(ctx)`, and computing the count rates.
Per call, only the aperture profile (`aperture_snr_radial` /
`aperture_time_for_snr`) + `select_aperture` need to run.

Add a private helper that computes and memoizes the time-independent bundle:

```python
def _image_render_bundle(self, psf, jitter_sigma_mas, npix, oversample):
    """
    Cached, time-independent inputs for the PSF-aware SNR/exptime path:
    the rendered normalized PSF, the plate scale, and the source/diffuse
    count *rates* (electrons / s). Keyed on render-affecting state; cleared
    by update()/set_sensor()/set_telescope().
    """
```

Returns `dict(psf_norm, plate_scale_mas, source_rate_total, diffuse_rate_per_pix)`.

- `source_rate_total = count_rates["source"] / ee_at_aper`  (e/s)
- `diffuse_rate_per_pix = sum over non-source elements of rate / n_psf`  (e/s)
- `dark_rate_per_pix` and `read_noise` are read live from `self.sensor` in the
  callers (cheap; and a sensor change clears the cache anyway).

Cache key: `(psf.cache_key(), jitter_sigma_mas, int(npix), int(oversample))`.
Stored in `self._image_render_bundle_cache` (a dict), initialized in `__init__`
alongside `_psf_profile`.

`get_image_snr` then becomes:

```python
b = self._image_render_bundle(psf, jitter_sigma_mas, npix, oversample)
t = time.to(u.second).value  # after the existing time-normalization
source_e_total = b["source_rate_total"] * t
diffuse_per_pix = b["diffuse_rate_per_pix"] * t
dark_per_pix = (self.sensor.dark_current * time).to(u.electron / u.pix).value
read_noise = self.sensor.read_noise.to(u.electron / u.pix).value * np.sqrt(n_reads)
prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                           source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
# ...select_aperture + dict assembly unchanged
```

`get_image_exptime_for_snr` likewise pulls `psf_norm`, `plate_scale_mas`,
`source_rate_total`, `diffuse_rate_per_pix` from the bundle (it already works in
rates), removing its duplicated count-rate block.

This consolidates the count-rate / render code duplicated across
`get_image_snr` and `get_image_exptime_for_snr` (the noted DRY follow-up).

#### PSF cache keys

Add a `cache_key()` method to the PSF classes in `psfsim.py`:

- `PSFSource.cache_key(self)` — base default `(type(self).__name__,)`.
- `AiryPSF` inherits the base → constant `("AiryPSF",)`. This is what makes the
  common `psf=None` path (which builds a *fresh* `AiryPSF()` each call) cache.
- `_ResampledPSF.cache_key(self)` → `(type(self).__name__, self.src_um_per_pix,
  id(self._data))`. Caches when the same PSF object is reused (the normal
  pattern); always correct because the cache is also cleared on state changes.
- `DefocusPSF.cache_key(self)` → `(type(self).__name__, self.src_um_per_pix,
  self.path)`.

### 4. Cache invalidation (and a bug fix)

Clear `self._image_render_bundle_cache = {}` (and `self._psf_profile = {}`) at
the end of `update()`, and in `set_sensor()` / `set_telescope()` (the latter two
already reset `_psf_profile`). This fixes the known bug where `_psf_profile` is
not invalidated after `update(jitter_sigma=...)`, so `get_snr` /
`get_image_snr` / `get_peak_pixel` no longer use a stale PSF after an in-place
update.

### 5. Caller / test updates (blast radius)

`get_snr` now returns a dict, so `.value` callers change:

- `tests/test_snr.py:23` — `simu.get_snr(time=60).value` → `simu.get_snr(time=60)["snr"]`.
  Expected ~5.1 still holds (2D reproduces 1D to ~1%, within `abs=0.1`).
- `tests/test_sensorfilter.py:133` — `sim.get_snr(60); assert snr.value > 0` →
  `sim.get_snr(60)["snr"] > 0`.
- `tests/test_simulation.py:168` (`test_get_image_snr_matches_get_snr_in_focus`)
  — the 1D-vs-2D cross-check must compare against `get_snr_airy` to stay
  meaningful; wrap the call in `pytest.warns(DeprecationWarning)`.
- `src/wcc_etc/wcc_etc.py:946` (`get_wcc_snr_and_simulation`) — `simu.get_snr(texp)`
  → `simu.get_snr(texp)["snr"]` so it keeps returning a scalar.
- Notebooks `01_getting_started`, `04_psf_and_image_snr`, `05_n_reads_exptime`,
  `06_source_spectra` — update cells that call `get_snr` (use `["snr"]`, or
  `get_snr_airy` where the 1D value is the point) and re-execute.

### 6. New test coverage

- `get_snr_airy` raises `DeprecationWarning` and equals the old analytic value.
- `get_snr(time)["snr"]` equals `get_image_snr(time)["snr"]` (delegation).
- Render-cache: a second `get_snr` call with the same render-affecting params
  reuses the bundle (assert the cache dict is populated and that a sweep over
  several `time` values renders once — e.g. spy/patch `AiryPSF.render` call
  count, or assert `len(cache) == 1` after the sweep).
- Cache invalidation: after `update(jitter_sigma=...)` (or `source__mag=...`)
  the cache is empty and the new SNR reflects the change (regression test for
  the stale-PSF bug).

## Files touched

- `src/wcc_etc/simulation.py` — `get_snr` (new body), `get_snr_airy` (new),
  `_image_render_bundle` (new), refactor `get_image_snr` /
  `get_image_exptime_for_snr`, cache init + invalidation in `__init__` /
  `update` / `set_sensor` / `set_telescope`.
- `src/wcc_etc/psfsim.py` — `cache_key()` on `PSFSource` / `_ResampledPSF` /
  `DefocusPSF`.
- `src/wcc_etc/wcc_etc.py` — one-line caller fix.
- `tests/` — `test_snr.py`, `test_sensorfilter.py`, `test_simulation.py`, plus
  new cache/deprecation tests.
- `notebooks/` — 01, 04, 05, 06 re-executed.
