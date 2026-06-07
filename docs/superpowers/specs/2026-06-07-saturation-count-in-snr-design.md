# Saturated-pixel count + warning on the SNR / exptime paths

**Date:** 2026-06-07
**Branch:** `saturation-count-in-snr`
**Status:** Approved

## Goal

Report the number of saturated pixels in the rendered image from the SNR and
exposure-time methods, and raise a warning when saturation occurs. Surgical
edits only, reusing the saturation machinery already in the package.

## Background / what already exists

- `Simulation.get_peak_pixel()` / `Simulation.is_saturated()` model saturation
  as **per-frame** (integration `time / n_reads`), on the **clean** (noiseless)
  electron image, against `sensor.adc_max` (ADU clip) **and** `well_depth`
  (full well).
- `psfsim.ImageSimulator.simulate()` already builds a `saturation_mask`
  with exactly that test:
  `(image_e / gain) >= adc_max | image_e >= well_depth` (psfsim.py ~341–348).

Note: the rendered-image saturation test (`simulate`'s mask, which this feature
reuses) is **broader** than `is_saturated`. `is_saturated` flags only the ADC
clip (and adds bias), whereas the image mask flags full-well **or** ADC clip on
the bias-free electron image. For sensors whose full well is reached before the
ADC clip (e.g. `sony:r`: well ≈ 16275 e⁻ vs ADC clip ≈ 17118 e⁻), `n_saturated`
can be > 0 while `is_saturated` is False. `n_saturated` is therefore a superset:
`is_saturated == True` implies `n_saturated > 0`, but not the converse. This is
the correct definition for "saturated pixels in the image."
- `Simulation._image_render_bundle()` already provides, for the PSF-aware SNR
  path, the normalized PSF image `psf_norm` (sums to 1) plus
  `source_rate_total`, `diffuse_rate_per_pix` (electrons/s); the dark rate is
  read from the sensor. These are enough to form the per-frame clean electron
  image without a second render.

The work taps into these — no new saturation physics.

## Design

### 1. Single source of truth for the saturation mask (`psfsim.py`)

Extract the inline mask logic in `simulate_image` into a module-level helper:

```python
def saturation_mask_from_image_e(sensor, image_e):
    """Bool mask: pixels at/over the ADC clip or full well. image_e in electrons."""
    gain = sensor.gain.to(u.electron / u.ct).value
    mask = (image_e / gain) >= sensor.adc_max.to(u.ct).value
    well_depth = sensor.meta.get("well_depth")
    if well_depth is not None:
        mask = mask | (image_e >= well_depth)
    return mask
```

`simulate_image` is refactored to call this helper. Its observable behavior
(the `saturation_mask` it returns) is unchanged.

### 2. `get_image_snr` (and `get_snr`, which delegates to it)

Inside the existing `_snr_at(t_sec, source_scale)` closure, form the per-frame
clean electron image from the render bundle and count saturated pixels:

```python
tf = t_sec / n_reads          # per-frame integration; saturation is per-frame
image_e = (b["source_rate_total"] * source_scale * tf) * b["psf_norm"] \
          + b["diffuse_rate_per_pix"] * tf + dark_rate_per_pix * tf
mask = saturation_mask_from_image_e(self.sensor, image_e)
```

Add to the per-point result dict:
- `"n_saturated"`: `int(mask.sum())`
- `"saturated"`: `bool(mask.any())`

Extend `_assemble_array` so that array `time`/`mags` inputs return
`n_saturated` as an int ndarray and `saturated` as a bool ndarray.

After results are assembled, if any point saturates and `warn=True`, emit a
single `warnings.warn(...)`.

The SNR computation itself (signal/noise/aperture selection) is untouched —
`n_reads` continues to enter SNR only via the read-noise term. Per-frame
division by `n_reads` is applied **only** in the saturation calculation, to
match `get_peak_pixel`/`is_saturated`.

### 3. `get_image_exptime_for_snr`

After it solves `time_s`, evaluate the same per-frame mask at that time using
the render bundle, add `n_saturated`/`saturated` to its returned dict, and warn
(gated by `warn=`).

### 4. Deprecated analytic paths (`get_snr_airy`, `get_exptime_for_snr`)

These return a bare `Quantity`, not a dict, so adding `n_saturated` to the
result would change their return type and break the API. They receive a
**warning only** (reusing `is_saturated` at the relevant time, gated by
`warn=`); their return values are unchanged. The saturated-pixel **count** is
therefore available only on the modern image-based dict paths.

### 5. New keyword `warn=True`

Add `warn=True` to `get_snr`, `get_image_snr`, `get_image_exptime_for_snr`, and
the two analytic methods. `get_snr` passes it through to `get_image_snr`.
The count is always returned regardless of `warn`; `warn=False` only silences
the warning (useful in magnitude/time sweeps and loops).

### Warning text (example)

```
peak pixel saturates the detector — 142 / 16384 pixels at or above
full well / ADC clip (per 30.0 s frame). SNR is unreliable.
```

(`UserWarning`, `stacklevel` set so it points at the caller.)

## Testing (TDD)

1. Default-aperture SNR values for a non-saturating source are **unchanged**
   from current behavior (regression guard).
2. Bright source → `n_saturated > 0` and a warning is raised.
3. Faint source → `n_saturated == 0` and no warning.
4. `warn=False` silences the warning but `n_saturated`/`saturated` are still
   present and correct.
5. Array `time` → `n_saturated` is an int ndarray of matching length;
   `saturated` is a bool ndarray.
6. `get_image_exptime_for_snr` returns `n_saturated`/`saturated` in its dict.
7. `simulate_image`'s `saturation_mask` is byte-for-byte unchanged after the
   helper refactor.
8. Deprecated analytic paths warn when saturated and `warn=True`, do not warn
   when `warn=False`, and their return type/values are unchanged.

## Out of scope

- No change to SNR or exposure-time math.
- No new saturation model (reuses adc_max + well_depth).
- No noise in the saturation count (clean per-frame image, matching
  `get_peak_pixel`).

## Follow-ups (after merge, per feature workflow)

- Demo notebook cell showing `n_saturated` rising with exposure time / source
  brightness.
- Update project-status memory with the new dict keys and `warn=` kwarg.
