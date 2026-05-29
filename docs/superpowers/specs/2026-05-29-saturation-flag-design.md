# Saturation Flag — Design

**Date:** 2026-05-29
**Branch:** `saturation`
**Status:** Approved (pending spec review)

## Goal

Add a saturation flag to the WCC ETC that warns when a source saturates the
detector for a given exposure time. Saturation is defined as the **brightest
pixel exceeding the analog-to-digital converter (ADC) full scale (ADU clip)**,
which can occur before the physical full well when gain is set high.

## Definitions and modeling choices

These were settled during brainstorming:

1. **Trigger condition — ADC clip (ADU).** The flag fires when the peak-pixel
   value in ADU reaches the digitizer ceiling `adc_max`, *not* the electron
   full well.
2. **ADU ceiling — per-sensor `bit_depth`.** `adc_max = 2**bit_depth - 1`.
   Bit depth differs per sensor: ZWO is 16-bit, the qcmos sensor is 12-bit.
3. **Peak-pixel flux — detector-grid PSF rendering.** The PSF is rendered onto
   the actual detector pixel grid (not approximated from the encircled-energy
   curve), so the brightest-pixel fraction is physically faithful. This path is
   built to extend to defocused PSFs later.
4. **Peak-pixel budget — source + background + dark + bias.** The value
   compared against `adc_max` includes the source PSF peak, the sky background
   per pixel, the dark current per pixel, and an additive bias/offset level.

## Component 1 — Sensor config and properties

### Config fields (`src/wcc_etc/data/config/*.toml`)

- `bit_depth` — integer. `zwo.toml` (Sony/IMX) → `16`, `qcmos.toml` (hwk) →
  `12`. (Confirmed: "hwk" is the `qcmos` config, 12-bit; Sony/IMX is 16-bit.)
- `bias_level` — additive offset in ADU. Set to `0` explicitly in both configs;
  `from_config` still defaults it to `0` when absent.

### `Sensor` class (`src/wcc_etc/sensor.py`)

- `from_config` reads `bit_depth` and `bias_level` (with `bias_level`
  defaulting to 0).
- Add `bit_depth` and `bias_level` to `_mutable_parameters` so they are tunable
  via `sim.update(sensor__bit_depth=..., sensor__bias_level=...)`.
- New properties:
  - `bit_depth` → int (from meta).
  - `bias_level` → Quantity in ADU (`* u.ct`), default 0.
  - `adc_max` → Quantity in ADU, `(2**bit_depth - 1) * u.ct`.

## Component 2 — Detector-grid PSF renderer (`src/wcc_etc/airy.py`)

### New function: `render_detector_psf(...)`

Renders the (optionally jittered) Airy PSF and returns the PSF sampled at the
**detector pixel scale**, normalized to total flux, with its peak **centered on
a pixel** (worst case, so the peak-pixel fraction is conservative for a
saturation warning).

- Inputs mirror the existing PSF functions: `wavelength`, `jitter_sigma_mas`,
  `fnum`, `D`, `pixel_size`, plus an `oversample` factor.
- Renders on an **oversampled** grid (e.g. `oversample = 11`) and bins down to
  detector pixels, so each detector pixel value is the *integral* of the PSF
  over the pixel area rather than a single point sample.
- The grid is built with an **odd** number of detector pixels so a pixel sits
  exactly on the PSF peak (worst-case brightest pixel).
- Defocus extensibility: the broadening/convolution step is the natural hook
  for a future defocus kernel; the function is structured so a defocus term can
  be added there without changing callers.
- Returns the detector-pixel PSF (2D, summing to 1). The **peak-pixel
  fraction** is `psf_detector.max()` — the fraction of *total* source energy
  landing in the brightest pixel.

### Refactor

`get_airy_and_ee_curve_pixel_grid` is refactored to share the rendering core
where practical, to avoid duplicated PSF-construction logic.

### `Simulation.compute_psf_profile()` (`src/wcc_etc/simulation.py`)

Add a `peak_pixel_fraction` entry to the returned profile dict, computed via
`render_detector_psf` using the same telescope/sensor parameters already used
for the Airy curve.

## Component 3 — Saturation logic (`Simulation`)

API mirrors the existing `get_snr(time)` / `get_signal_and_variance(time)`
pattern (same call style, units handling, and array-`time` vectorization).

### `get_peak_pixel(time, units="adu")`

Returns the peak-pixel value for the given exposure time(s). Computation:

1. **Source peak (e-):**
   `(count_rates["source"] / ee_at_aper) * peak_pixel_fraction * time`.
   Dividing by `ee_at_aper` recovers the *total* source flux from the
   aperture-integrated value (since `count_rates["source"]` is already
   multiplied by `ee_at_aper` in `get_countrates`); multiplying by
   `peak_pixel_fraction` gives the brightest-pixel source contribution.
2. **Background per pixel (e-):**
   `(background-in-aperture countrate / num_psf_pixels) * time`
   (approximates a locally uniform background across the PSF aperture).
   If the scene has no background element, this term is 0.
3. **Dark per pixel (e-):** `sensor.dark_current * time`.
4. Sum the three electron terms, convert to ADU via `/ sensor.gain`, then add
   `sensor.bias_level` (ADU).

`units="e-"` returns the pre-conversion electron sum (without bias, which is an
ADU-domain quantity); `units="adu"` (default) returns the full ADU value
including bias.

### `is_saturated(time)`

Returns `get_peak_pixel(time, "adu") >= sensor.adc_max`. Works element-wise for
array `time` (returns a boolean array).

## Component 4 — Tests

- `tests/test_sensor.py`:
  - `adc_max` math for both bit depths (65535 for 16-bit, 4095 for 12-bit).
  - `bias_level` defaults to 0 when absent and reads from config when present.
  - `bit_depth` / `bias_level` are updatable via `sensor.update(...)`.
- `tests/test_simulation.py`:
  - `peak_pixel_fraction` is in `(0, 1]`.
  - `get_peak_pixel` increases with `time` and includes background/dark/bias
    contributions (a bright background or large bias raises the value).
  - `is_saturated` is `False` at short `time` and `True` at long `time` for a
    bright source; flips at the expected threshold.
  - Array-`time` input returns a boolean array of matching shape.

## Out of scope (YAGNI)

- `saturation_time()` — solving for the exposure time at which the peak pixel
  reaches `adc_max`. Useful in an ETC but not requested. **Confirm during spec
  review whether to include it.**
- Electron full-well saturation (the alternative trigger that was not chosen).
- A bias-level config for sensors that genuinely have no offset (defaults to 0).

## Open items to confirm during spec review

1. ~~"hwk" sensor maps to the `qcmos` config (bit_depth 12)?~~ **Confirmed:**
   hwk = qcmos = 12-bit; Sony/IMX (zwo) = 16-bit. Both `bit_depth` and
   `bias_level` are now set explicitly in the config files.
2. ~~Include the optional `saturation_time()` helper, or leave it out?~~
   **Confirmed:** left out of this work (remains in Out of scope above).
