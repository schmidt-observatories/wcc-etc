# PSF-Aware Saturation & 2D Count-Rate Migration — Design

**Date:** 2026-06-18
**Branch:** `psf-aware-saturation`
**Status:** Approved (design), pending spec review

## Problem

`is_saturated` / `get_peak_pixel` compute the brightest-pixel value from
`compute_psf_profile`'s `peak_pixel_fraction`, which is **always an in-focus
Airy peak** — it ignores the actual PSF (`_default_psf`, e.g. the 2-wave
defocus selected by `from_sensorfilter('zwo:bb2')`). Light from a defocused
PSF is spread over many pixels, so its true peak-pixel fraction is far smaller;
the current code therefore reports a defocused, broadband config saturating
*sooner* than an in-focus narrowband one — backwards from reality.

The PSF-aware 2D path (`get_image_snr`, `ImageSimulator.simulate`) already
handles this correctly: it builds a per-frame clean electron image from the
*rendered* PSF (`_per_frame_clean_image_e` + `saturation_mask_from_image_e`)
and reports `saturated`/`n_saturated`. Only the standalone analytic
`get_peak_pixel`/`is_saturated` path is wrong.

## Goal

Migrate all non-deprecated computations (SNR, exposure time, saturation, image
simulation) to derive from the rendered PSF + PSF-independent total count
rates. After this change, `compute_psf_profile` and the Airy scalars
(`ee_at_aper`, `peak_pixel_fraction`, `psf_area`, `num_psf_pixels`) are used
**only** by the explicitly-deprecated analytic methods.

## Why the migration is clean (not a stranding)

In every 2D consumer, `get_countrates`'s `× ee_at_aper(Airy)` is immediately
divided back out to recover the PSF-independent **total** source rate; the real
enclosed energy then comes from the rendered PSF. The surface-brightness path
collapses the same way: `psf_area = num_psf_pixels × plate_scale²`
(`simulation.py:1202`), so the background round-trip (`× psf_area` then
`÷ num_psf_pixels`) reduces to **per-pixel sky = surface_brightness ×
plate_scale²**. So the 2D path needs only totals + plate scale + the rendered
PSF — no Airy profile.

## Components

### 1. `_count_rate_components()` — PSF-independent count-rate primitive
Returns a dict of rates with **no Airy aperture**:
- `source_rate_total` (e/s) — full source flux
  (`observation.countrate(area=telescope.surface)`, i.e. today's
  `count_rate_total` *without* the `× ee_at_aper`).
- `background_rate_per_pix` (e/s/pix) — per-pixel sky. For a
  surface-brightness element, computed by evaluating the observation at a
  one-pixel area (`plate_scale_arcsec²`); for a non-SB diffuse element, the
  total rate divided across pixels via plate scale. No `psf_area`/
  `num_psf_pixels`.
- `diffuse_rate_per_pix` (e/s/pix) — sum of any non-source, non-background
  elements, per pixel.

Convention (unchanged from the current bundle): the `source` element is a
point source distributed by the PSF; every other element is uniform per pixel.

### 2. `_image_render_bundle` and `ImageSimulator.simulate`
Replace their `get_countrates` + `psf_profile` (`ee_at_aper`/`num_psf_pixels`)
usage with `_count_rate_components()`. This is a net deletion of the
multiply-in/divide-out round-trip. `get_image_snr`/`get_image_exptime_for_snr`
become independent of `compute_psf_profile`.

### 3. `get_peak_pixel` / `is_saturated` — reimplemented on the rendered PSF
The brightest-pixel **rate** is
```
peak_rate = source_rate_total · psf_norm.max()
          + background_rate_per_pix + dark_rate_per_pix     # e/s
peak_e(t_frame) = peak_rate · t_frame                       # linear in time
```
`psf_norm` is the rendered (native-pixel, sum=1) PSF from
`_image_render_bundle`; `psf_norm.max()` is the *true* peak-pixel fraction of
the actual PSF. Linearity in `t_frame` handles scalar and array `time` without
looping.

New signatures (backward compatible — existing `(time, units, n_reads)` calls
still work):
```
get_peak_pixel(time=None, units="adu", n_reads=None, *,
               psf=None, jitter_sigma_mas=None, npix=128, oversample=11)
is_saturated(time=None, n_reads=None, *,
             psf=None, jitter_sigma_mas=None, npix=128, oversample=11)
```
`psf` defaults to `_default_psf` if set, else `AiryPSF()` — mirroring
`get_image_snr`. `units` keeps `'adu'` (adds bias) / `'e-'` semantics; ADU still
compares against `sensor.adc_max` in `is_saturated`.

### 4. Retire `peak_pixel_fraction`
Remove it from `compute_psf_profile`'s returned dict (no remaining consumer).
Keep `ee_at_aper`, `psf_area`, `num_psf_pixels`, `psf1d`, `ee`, `r_psf_mas`
(still used by the deprecated analytic path).

### 5. Deprecate but keep (backward compatible)
Add/keep `DeprecationWarning` on `compute_psf_profile`, `get_countrates`
(in-aperture), `get_signal_and_variance`, the analytic `get_exptime_for_snr`,
and `get_snr_airy` (already deprecated). They remain functional and become the
only users of the Airy profile. Their docstrings note the PSF-aware
replacements (`get_image_snr`, `get_image_exptime_for_snr`, the 2D
`get_peak_pixel`/`is_saturated`).

## Data flow (saturation, after migration)
`is_saturated(t, psf=…)` → `_image_render_bundle(psf, …)` gives `psf_norm` +
`_count_rate_components()` rates → `peak_rate` → `peak_e(t/n_reads)` →
ADU compare vs `sensor.adc_max`. Agrees with `get_image_snr`'s image-based
`saturated` flag by construction (same rendered PSF, same per-pixel budget).

## Testing (TDD)
- `_count_rate_components`: `source_rate_total` equals `count_rate_total`
  (the pre-`ee_at_aper` value); `background_rate_per_pix` equals
  surface_brightness flux × plate_scale² for the default zodi background
  (cross-check against the old `count_rates["background"]/num_psf_pixels`).
- Defocus peak fraction: for `zwo:bb2`, the peak-pixel fraction used by
  `get_peak_pixel` equals the rendered `psf_norm.max()` and is **much smaller**
  than the Airy value; `is_saturated` allows a **longer** unsaturated exposure
  than the pre-change Airy result.
- Consistency: `is_saturated(t, psf=…)` agrees with
  `get_image_snr(time=t, psf=…)['saturated']` across a grid of `t`.
- In-focus regression: for an in-focus `zwo:r`/default Airy config, the peak
  value matches the rendered Airy `psf_norm.max()` path; `get_snr`/
  `get_image_snr` unchanged.
- Array `time` → array peak / saturation flags (linearity).
- Deprecated methods (`get_snr_airy`, `get_countrates`,
  `get_signal_and_variance`, analytic `get_exptime_for_snr`,
  `compute_psf_profile`) emit `DeprecationWarning` and still return their prior
  values.

## Known risk to verify (not assume)
In-focus Airy peak values may shift at the ~%-level (rendered `psf_norm.max()`
vs the old analytic `render_detector_psf` peak). Existing
`tests/test_saturation_count.py` and `tests/test_saturation_reads.py`
expectations must be checked; update only where the rendered value is the
correct one, and record any numeric change.

## Validation / payoff (separate deliverable, depends on transit-lightcurve)
Add an Earth-twin transit example to the scratch demo notebook: V=13 G2V star,
`zwo:bb2` (defocused broadband) vs `zwo:r` (in-focus), showing the in-focus
config saturating / failing where the broadband+defocus mode succeeds — plus
the same Earth-radius planet around an M dwarf (~930 ppm depth) detectable with
the in-focus config. This notebook is gitignored and imports `wcc_etc.lightcurve`
(PR #33), so it is produced where both this branch and the transit-lightcurve
module are available; it is not part of this PR's tracked files.

## Out of scope (YAGNI)
- Removing the analytic Airy methods or `compute_psf_profile` (deprecate only).
- Changing `get_countrates`'s public in-aperture return semantics.
- Sub-pixel/dithered saturation modeling, charge bleed, non-linearity below
  full well.
- PSF-aware surface-brightness *sources* distributed by a PSF (background/
  diffuse stay uniform per pixel, as today).
