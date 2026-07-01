# WCC ETC public-release code review

Date: 2026-06-24

Scope reviewed:

- Primary public API shown in `README.md`: `get_scene`, `Simulation.from_sensor_and_scene`, `Simulation.get_snr`, `get_image_snr`, `get_image_exptime_for_snr`, `get_peak_pixel`, and `is_saturated`.
- Core science modules: `scene.py`, `simulation.py`, `sensor.py`, `telescope.py`, `psfsim.py`, `airy.py`, `astro.py`, `io.py`, detector/telescope config files, and the science/performance tests.
- Legacy `wcc_etc.WCCETC` class was reviewed as public import surface because `wcc_etc/__init__.py` imports `from .wcc_etc import *`.

Verification run:

- `pytest -q`
- Result: `309 passed, 58 warnings in 99.25s`
- Targeted science subset also passed: `50 passed, 6 warnings in 77.69s`

Passing tests are not enough for public release. The most serious risks below are calibration/assumption issues that current tests either do not check or intentionally regress against the current implementation.

## Executive summary

The newer 2D PSF-aware ETC path is in much better shape than the legacy analytic path. It correctly treats source counts, sky/background, diffuse host light, dark current, and read noise in the aperture-level CCD equation, and it uses pivot wavelength rather than peak-throughput wavelength for broadband PSF rendering. The sky-noise tests are a good model for science tests because they compare against a closed-form CCD equation and a Monte Carlo image scatter check.

I would not release this publicly yet. The 3-5 items that need to be fixed first are:

1. Resolve detector gain unit conventions and qCMOS calibration source conflicts.
2. Reconcile saturation logic: ADC, full well, and bias are currently handled inconsistently across public methods.
3. Remove, hide, or hard-deprecate the legacy `WCCETC` API and deprecated analytic methods from the public release surface.
4. Replace regression-style science-performance expectations with externally validated benchmark cases and tolerances.
5. Document and validate the astrophysical background / throughput assumptions, especially zodiacal-light default, magnitude system, aperture policy, and detector noise inflation.

## Release blockers / major findings

### 1. Detector gain conventions are ambiguous and could break qCMOS saturation and ADU outputs

Severity: high.

Relevant code/data:

- `src/wcc_etc/sensor.py:265-270`: `Sensor.gain` always returns `meta["gain"] * electron / ct`, i.e. e-/ADU.
- `src/wcc_etc/data/config/qcmos.toml:5`: `gain = 0.112` with comment `ADU/e` and `NOTE: Gabor's note suggest gain is ADU/e rather than e/ADU`.
- `src/wcc_etc/data/sensors/qCMOS/HWK4123_sensor_parameters.txt:9`: `gain,8.9,ADU/e,"as measured by MIT"`.
- `src/wcc_etc/psfsim.py:183-184`: saturation threshold uses `image_e / gain >= adc_max`, which assumes e-/ADU.
- `src/wcc_etc/psfsim.py:203-205` and `src/wcc_etc/simulation.py:747`: ADU conversion assumes e-/ADU.

The numerical values `8.9 ADU/e` and `0.112 e-/ADU` are reciprocals, so the current TOML value may be numerically right while the comments and source file are mislabeled. That is still a release blocker because a future maintainer, user, or calibration update can easily invert it. If `0.112` were interpreted literally as ADU/e by users, qCMOS ADC saturation thresholds would be wrong by about `(8.9 / 0.112) ~= 79.5` in electron-space behavior.

Current tests do not catch this. `tests/test_sensor.py` asserts that synthetic gains are e-/ct, and `tests/test_saturation_count.py` derives expected ADC behavior from the same `Sensor.gain` property. Those tests prove internal consistency, not calibration correctness.

Required fix:

- Normalize detector config to one explicit internal convention, preferably `gain_e_per_adu`.
- Rename config keys or add a parser that accepts explicit units (`gain_e_per_adu`, `gain_adu_per_e`) and converts once.
- Update qCMOS config comments and source documentation so all files say the same thing.
- Add a test using the qCMOS calibration file value: e.g. if MIT reports `8.9 ADU/e`, assert the internal `Sensor.gain` is approximately `1 / 8.9 e-/ADU`, and assert the ADC electron threshold is `adc_max / 8.9`.

### 2. Saturation criteria are inconsistent across public methods

Severity: high.

Relevant code:

- `src/wcc_etc/simulation.py:711-748`: `get_peak_pixel(..., units="adu")` computes source + background + dark, divides by gain, adds `bias_level`.
- `src/wcc_etc/simulation.py:750-764`: `is_saturated` compares that biased ADU peak only against `sensor.adc_max`.
- `src/wcc_etc/psfsim.py:167-188`: `saturation_mask_from_image_e` flags ADC saturation from electron image and full-well saturation, but does not include bias in the ADC comparison.
- `src/wcc_etc/simulation.py:755-759`: comment acknowledges the mismatch: ADC+bias vs image full-well+bias handling is a known follow-up.
- `src/wcc_etc/simulation.py:817-827`: `_per_frame_clean_image_e` excludes host/diffuse components by design.
- `src/wcc_etc/psfsim.py:338-342`: image simulation includes diffuse/host flux in the image but comments that saturation budget excludes it.

For public ETC use, saturation is a core scientific/product result. The current API can disagree depending on which method users call:

- `is_saturated` is ADC-only and includes bias.
- `get_image_snr(...)[saturated]` is ADC-or-full-well and ignores bias in the ADC threshold.
- host/diffuse flux is rendered in images and contributes shot noise, but is intentionally excluded from saturation.

Some of these choices may be intentional, but they should not be exposed as ambiguous behavior. A bright host/background can physically fill pixels and affect full well / ADC; excluding host from saturation needs a documented science justification, not only an inline comment.

Required fix:

- Create one detector-limit function that accepts a per-frame electron image and applies both limits consistently:
  - full well: `image_e >= well_depth_e`
  - ADC: `image_e / gain_e_per_adu + bias_adu >= adc_max_adu`
- Have `is_saturated`, `get_peak_pixel`, `get_image_snr`, and `ImageSimulator.simulate` all call the same function.
- Decide whether diffuse/host counts should saturate pixels. If excluded for a mission-specific reason, expose the choice as a named parameter and document the default.
- Add tests where bias alone changes the ADC threshold and where full well is lower than ADC full scale.

### 3. Deprecated analytic and legacy APIs remain too easy to use

Severity: high for public release, medium for internal use.

Relevant code:

- `src/wcc_etc/__init__.py:17`: imports `from .wcc_etc import *`.
- `src/wcc_etc/wcc_etc.py`: legacy class and functions remain importable.
- `src/wcc_etc/wcc_etc.py:710-721`: legacy SNR methods contain explicit notes saying gain is not correctly accounted for.
- `src/wcc_etc/simulation.py:537-544`, `640-663`, `1064-1099`, `1101-1129`: deprecated analytic methods remain available and are still used in tests.
- `tests/test_exptime.py:35-59` and `tests/test_simulation.py:328+`: tests assert deprecated analytic round trips.

This is risky for a public astronomical audience because users will discover and call public names from tab completion, old examples, notebooks, or generated docs. The old analytic path has known limitations:

- Airy in-aperture approximation rather than rendered PSF.
- Historical background EE-factor issue.
- Old comments about gain handling.
- Saturation warnings are bolted on rather than part of a unified detector model.

Required fix:

- Stop wildcard-importing `wcc_etc.py` in `__init__.py`, or move it behind an explicit compatibility namespace such as `wcc_etc.legacy`.
- Remove legacy methods from public docs and top-level API examples.
- For public release, make deprecated analytic methods raise stronger warnings or keep them only as private/test utilities.
- Convert tests that depend on deprecated methods into private unit tests of the algebra, not release-facing behavior checks.

### 4. Scientific performance tests include regression constants but not enough external validation

Severity: high.

Good tests:

- `tests/test_sky_background_noise.py`: strong structure. It checks the CCD equation, Monte Carlo scatter, and qualitative sky-brightness behavior.
- `tests/test_source_physics.py`: blackbody/flat/power-law/emission-line shape checks compare against independent analytic expectations.
- `tests/test_psf_wavelength.py`: catches the important pivot-vs-wpeak error.

Weak or risky tests:

- `tests/test_snr.py:6-34`: asserts `SNR ~= 4.95` for a 25.4 AB-mag source in 60 s. The comments explain why this changed, but the expected value is still a project-internal regression number. It is not tied to an external ETC, calibration note, hand calculation in the test, or versioned benchmark table.
- `tests/test_count_rate_components.py:14-37` and `tests/test_twod_countrate_migration.py:30-43`: compare new count-rate components to deprecated legacy behavior divided by `ee_at_aper`. This is useful for migration, but it partly inherits the legacy implementation as a reference.
- `tests/test_peak_pixel_fraction.py:19-22` and `tests/test_psf_aware_saturation.py:14-19`: assert only broad inequalities for defocus peak fraction. They do not verify the Zemax PSF pixel scale, wavelength dependence, normalization after resampling, or expected encircled energy.
- Many saturation tests compute expected masks using the same formula as the implementation; they do not validate external detector thresholds.

Required fix:

- Add benchmark files or tests with hand-computed/reference expected values for:
  - count rate for an AB-flat source through each released bandpass,
  - qCMOS and Sony gain/read-noise/dark-current/well-depth thresholds,
  - zodiacal surface brightness per pixel for a known plate scale,
  - PSF peak-pixel fraction and encircled-energy radii for in-focus and 1/2-wave defocus.
- For each public sensor/filter, add at least one benchmark row with source rate, sky rate per pixel, dark/read variance, aperture radius, SNR, and saturation threshold.
- Keep regression tests, but label them as regression tests and do not use them as the only scientific-performance validation.

### 5. Several astrophysical assumptions are hard-coded or under-documented

Severity: medium-high.

Relevant code/data:

- `src/wcc_etc/scene.py:326-330`: default zodi/background is `mag=22.5`, `surface_brightness=True`, `bandpass="johnson_v"`.
- `src/wcc_etc/data/config/lazuli.toml:8-9`: `zodi_mag_r=22.5` with comment `Need trace of where 22.5 originates from`.
- `src/wcc_etc/sensor.py:146`: Sony read noise is multiplied by 2 to allow for unmodelled noise sources.
- `src/wcc_etc/data/config/qcmos.toml:3`: qCMOS read noise is already doubled in the config comment.
- `src/wcc_etc/data/config/qcmos.toml:6`: qCMOS dark-current/temperature comments indicate unresolved temperature assumptions.

The default sky background and noise inflation materially affect public ETC predictions. The code has comments saying the provenance needs tracing or assumptions need confirmation. That is acceptable during development, but not for public release.

Required fix:

- Document the adopted zodiacal surface brightness source, band, magnitude system, and whether `22.5` is V, r, AB, or Vega.
- Make the default background explicit in the README/tutorials and encourage users to set `magsys` and `bandpass`.
- Replace the hard-coded Sony `read_noise * 2` with an explicit config field such as `read_noise_margin_factor`, and apply it consistently or not at all.
- Confirm qCMOS dark-current temperature and publish which temperature the default represents.

## Other correctness and simplification observations

### Surface-brightness handling is mostly right, but area is not validated

`SceneElement.get_mag` correctly converts mag/arcsec^2 to integrated magnitude via `mag - 2.5 log10(area)` (`scene.py:513-525`). However, if `area` is `None`, zero, or negative for a surface-brightness source, it will produce a runtime error, `nan`, or `inf` rather than a clear validation error. Because background and host flux are core ETC terms, this should raise `ValueError` when `area is None` or `area <= 0`.

### Source name resolution uses substring matching

`io.get_any_astro_name` uses `str.contains(name)` (`io.py:113`) and then retries with `name + "."`. This can accidentally match unintended spectra if names overlap. For release, exact mapping for Pickles spectral types and exact basename matching should be preferred. If substring search remains, it should be opt-in and case handling should be documented.

### Wavelength choice is scientifically improved

`Sensor.wavelength` uses `bandpass.pivot()` (`sensor.py:237-255`), and tests protect this. This is a good fix: `wpeak()` is arbitrary for broad flat-topped filters and biases PSF size.

### The 2D SNR equation is internally coherent

The core 2D SNR path computes:

- source signal: total source electrons times enclosed PSF fraction,
- noise variance: source signal + `(sky + host/diffuse + dark + read_noise^2) * n_pix`,
- read noise scaled by `sqrt(n_reads)` in fixed-time SNR and `n_reads * RN^2` in exposure-time solving.

Relevant code:

- `src/wcc_etc/simulation.py:918-930`
- `src/wcc_etc/psfsim.py:923-955`
- `src/wcc_etc/psfsim.py:985-1021`

This is the right high-level CCD-equation structure for aperture photometry, assuming the calibration inputs are correct.

### Aperture representation is pixel-discrete

`_radial_cumulative` sorts whole pixels by centroid radius and treats `n_pix` as whole-pixel aperture area (`psfsim.py:901-920`). This is acceptable for a fast ETC, but public docs should state that aperture radii are snapped to rendered detector pixels. It is not exact fractional-pixel aperture photometry.

### Packaging is probably okay for data, but pyproject package-data is misleading

`MANIFEST.in` recursively includes `src/wcc_etc/data *`, which is good. `pyproject.toml` package-data patterns are not obviously recursive for nested sensor directories, but `include-package-data = true` plus `MANIFEST.in` should include the files in source distributions. Before release, verify both sdist and wheel contain all throughput, sensor, PSF, Pickles, and Brown files.

## Are any scientific performance tests wrong?

I did not find a currently failing test that encodes an obviously false formula in the newer 2D path. The sky-noise, source-physics, and pivot-wavelength tests are scientifically well motivated.

However, several tests are scientifically insufficient for release:

- `test_snrs_25p4_mag_60s` is a regression number, not an independently validated performance benchmark.
- qCMOS detector tests do not validate the calibration-source gain convention.
- saturation tests validate formula consistency but not the physically unified detector-limit model, especially bias/full-well interactions.
- defocus tests check qualitative peak reduction, not quantitative PSF/EE benchmarks.
- some tests still depend on deprecated analytic methods, which should not be used as science references.

Recommended public-release test additions:

1. A benchmark table of source electron rates and sky electron/pixel rates for each released sensor/filter, computed from an independent script/notebook and checked into `tests/data`.
2. Detector-limit tests for Sony and qCMOS with explicit e-/ADU, ADU/e, full-well, ADC, and bias values.
3. PSF benchmark tests for in-focus, 1-wave, and 2-wave defocus: normalization, peak-pixel fraction, EE50/EE80/EE90 radii, and truncation loss on the default grid.
4. A benchmark that reproduces a hand CCD-equation SNR row end-to-end without using deprecated methods.
5. Tests that fail if undocumented noise margin factors are silently applied.

## Recommended pre-release checklist

1. Fix detector gain units and add calibration-source tests.
2. Unify saturation handling and decide/document whether host/diffuse flux contributes to saturation.
3. Remove legacy `WCCETC` from top-level imports and public documentation.
4. Replace or supplement regression SNR constants with independent benchmark cases.
5. Document zodiacal-light, magnitude-system, throughput, detector-temperature, and read-noise margin assumptions.
6. Build and inspect both sdist and wheel contents.
7. Run notebooks from a clean environment after installing the built wheel, not only editable mode.

