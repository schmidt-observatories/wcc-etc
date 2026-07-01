# WCC ETC — Pre-Release Scientific Code Review (fresh independent pass)

**Reviewer:** Claude (Opus 4.8)
**Date:** 2026-06-24
**Branch reviewed:** `fix/snr-test-pin-after-both-fixes` (current working tree)
**Scope:** `src/wcc_etc/` and `tests/`, focused on the scientific correctness of the
count-rate / SNR / exposure-time / saturation calculations for the Lazuli Wide-field
Context Camera ETC, ahead of public release to the astronomical community.
**Test status at review:** `309 passed` (full suite, 86 s), all green — but the green suite does **not** catch blockers #1, #3, #4, or #5 below (see §4).

> **What's new in this pass.** The two top issues from the prior reviews —
> *(a)* the sky/zodiacal background dropped from the SNR/exptime noise, and
> *(b)* the PSF rendered at peak-transmission instead of pivot wavelength — have been
> **fixed and merged** (commits `66e6e30`, `308061f`). I re-verified both fixes and they
> are **correct** (see §2). This review concentrates on what those reviews did **not**
> catch, plus the items that are still open. The single most important *new* finding is a
> detector-gain unit contradiction for the qCMOS that biases its reported ADU and
> saturation by ~80× (§3, blocker #1).

---

## 1. Executive summary

The architecture remains sound: clean `Telescope`/`Sensor`/`Scene`/`Simulation`
separation, a genuine 2-D PSF-aware image path, spectrally-correct count rates via
`synphot.Observation` over the full bandpass, parametric source spectra with independent
physics tests, and a sensible deprecation story for the old analytic Airy code. The two
recent fixes are real improvements and are correctly implemented.

However, several issues remain — including one new, high-impact, detector-specific error —
that should be resolved before a public release. The headline-noise math
(`aperture_snr_radial` / `aperture_time_for_snr`) is **correct** and I verified the quadratic
inversion and the noise-budget structure independently. The remaining risks are concentrated
in **detector calibration constants, the absolute flux scale, and the broken legacy module**,
not in the core SNR algebra.

---

## 2. Verification of the two recent fixes (both correct ✅)

**Sky background in the noise budget (`66e6e30`).** `get_image_snr._snr_at` and
`get_image_exptime_for_snr` now add `background_rate_per_pix` to the per-pixel noise
(simulation.py:920-927 and :1016-1023). I traced `_count_rate_components` →
`_image_render_bundle` → `aperture_snr_radial`: the sky shot noise is now correctly folded
into both the SNR and the exposure-time solve, and `tests/test_sky_background_noise.py`
guards it with a closed-form CCD check, a Monte-Carlo cross-check, a "brighter sky lowers
SNR" check, and an exptime round-trip. **Correct.**

**PSF at the pivot wavelength (`308061f`).** `Sensor.wavelength` now returns
`bandpass.pivot()` instead of `wpeak()` (sensor.py:237-255). Pivot is the photometrically
meaningful effective wavelength and is the right monochromatic-PSF reference; this removes
the 17–28% PSF-size bias on the broadband modes. **Correct.** (Caveat unchanged from prior
review: the PSF is still monochromatic and not source-spectrum-weighted, and the *defocus*
PSF is still wavelength-frozen — see §3 blocker #5.)

The noise-budget structure of `aperture_snr_radial`/`aperture_time_for_snr` was also
re-verified: enclosed source counts under the source-shot term, `(sky+diffuse+dark)·n_pix`
under the background-shot term, `n_reads·RN²·n_pix` for read noise, signal and noise share
the same `n_pix`, and the time-for-SNR quadratic recovers the target SNR exactly. No errors.

---

## 3. The five things to fix before release

### 🔴 #1 — NEW / CRITICAL: qCMOS detector gain has a units contradiction; ADU and saturation are off by ~80×

**Where:** `data/config/qcmos.toml:4` vs `sensor.py:265-269` and its consumers
(`simulation.py:594`, `:704`, `:747`; `psfsim.py:180-187`).

The config carries:
```toml
gain = 0.112   # ADU/e # NOTE: Gabor's note suggest gain is ADU/e rather than e/ADU. Ask
```
but the code **defines and uses `gain` as electrons-per-ADU** (`@property gain → "e-/ct"`,
sensor.py:267) and converts electrons → ADU by **dividing**: `count_rate /= self.sensor.gain`
(simulation.py:594) and `peak_e / self.sensor.gain` for the ADC-clip test. Those two
conventions are reciprocals. If the config label is right (0.112 ADU/e⁻ ≈ 8.93 e⁻/ADU), the
number the code needs in its slot is **8.93, not 0.112**.

**Verified impact (arithmetic):**
- Reported ADU rate uses `e⁻ / 0.112 = 8.93·e⁻`, whereas the correct ADU rate is
  `e⁻ · 0.112`. **Reported ADU is high by ≈ 8.93/0.112 ≈ 80×.**
- ADC-clip saturation triggers when `e⁻/gain ≥ adc_max = 4095`, i.e. at
  `e⁻ ≥ 0.112·4095 ≈ 459 e⁻` — **6.6 % of the 7000 e⁻ full well.** So the ADC clip, not the
  well, sets saturation, and it does so ~80× too early. With the reciprocal value (8.93)
  the ADC ceiling becomes `8.93·4095 ≈ 36,600 e⁻`, the 7000 e⁻ well correctly becomes the
  binding limit, and the `image_e >= well_depth` branch stops being dead code.

The author flagged this in-line (`# ... Ask`), so it is a known-uncertain value, not a typo —
but it cannot ship to the public unresolved, because it directly corrupts the qCMOS (a
primary WCC detector) ADU outputs, the "brightest observable star," dynamic range, and every
saturation/peak-pixel warning.

**Fix.** Resolve the convention with Gabor and lock it: almost certainly set
`qcmos.toml gain = 8.93` (= 1/0.112) e⁻/ADU and delete the "Ask". Add a one-line sanity
assertion in `Sensor.from_config` that `gain · adc_max` is within a small factor of
`well_depth` (the ADC range should be comparable to the well); that single guard would have
caught this. Confirm the ZWO `path_gain_curve` is also in e⁻/ADU.

---

### 🔴 #2 — STILL OPEN / CRITICAL: a broken, self-contradictory public API ships via `from .wcc_etc import *`

**Where:** `__init__.py:17` does `from .wcc_etc import *`; `wcc_etc.py` has no `__all__`, so
its full surface lands in the importable package namespace. (`__init__.__all__` only narrows
`from wcc_etc import *`; `wcc_etc.WCCETC`, etc. are still attribute-accessible and show in
`dir()`.) Confirmed still present on the current branch:

- **`calc_moon_scatter_countrate_per_pixel`** (wcc_etc.py:1002) calls `trapezoid(ff, ww)` at
  line 1029, but `trapezoid` is never imported (only `from numpy import sqrt`) → **`NameError`
  on any call.**
- **`utils.list_of_quantity_to_array`** calls `warnings.warn(...)` at utils.py:88 while
  `warnings` is never imported (utils.py imports only `pandas`, `numpy`) → the mixed-unit
  branch **raises `NameError` instead of warning.**
- **`WCCETC.calc_SNR`** ships with the docstring *"NOTE — Does not correctly account for
  gain"* (wcc_etc.py:710, :720), alongside `calc_SNR_one_frame_with_gain` whose noise uses a
  hard-coded `n_b = n_pix * 100000  # assuming background is well known` fudge (line 747).
- **`WCCETC.simulate_2D_psf`** raises `NotImplementedError` (line 801).
- **`import toml`** (third-party, wcc_etc.py:1) is pulled in only for this dead module; the
  rest of the codebase uses stdlib `tomllib`.

**Impact.** A public user exploring `wcc_etc.<TAB>` finds names that crash or carry
self-admitted scientific errors — a credibility and support problem for a mission-planning
tool.

**Fix.** Keep the one genuinely useful helper (`get_wcc_snr_and_simulation`, which wraps the
modern `Simulation`) — move it to `simulation.py` or a thin `convenience.py` — then stop
`import *`-ing the rest. Add `import warnings` to `utils.py` regardless (it's a live
`NameError` reachable from the analytic path). Delete or quarantine `WCCETC`,
`calc_moon_scatter_countrate_per_pixel`, and the `toml` dependency.

---

### 🟠 #3 — The absolute flux scale is undefended: no independent count-rate/zeropoint test, and the collecting area ignores the central obstruction

Two related issues that together mean the *absolute* electrons/second — the most basic ETC
output — is neither tested against ground truth nor physically complete.

**(a) No independent zeropoint test (coverage gap).** Every test that touches the count rate
is relative or self-referential: `test_source_physics.py` checks only the *ratio* of two
rates (Pogson 100×); `test_count_rate_components.py` / `test_twod_countrate_migration.py`
compare two code paths that share the same `synphot` call; `test_scene.py` round-trips a
magnitude through synphot's own `Observation.effstim`. **A wrong collecting area, throughput
normalization, or zeropoint would pass the entire suite green.** This is exactly the class of
error an ETC most needs to guard.

**(b) Collecting area is unobstructed.** `telescope.py:111-115` returns `π·(D/2)²` with
`D = 3.065 m` and no central obstruction; `airy.get_airy_psf` is the clear-aperture Airy
pattern. A 3-m-class space telescope loses ~10–30 % of geometric area to the secondary +
spider. Ignoring it overstates every count rate (and `SNR ∝ √area`), and gives a PSF with too
little energy in the wings (overstated EE in a fixed aperture, understated diffraction-spike
contribution to saturation). The code and configs record no obstruction at all.

**Fix.** (a) Add one test that pins the source electron rate for a fixed
magnitude/filter/sensor against a *hand* calculation (AB mag → Fν → photons s⁻¹ cm⁻² over the
band × collecting area × throughput), tolerance ~5 %. This is the single test that catches a
wrong area/throughput/zeropoint. (b) Add an `obstruction_fraction` (or secondary diameter)
config key and use `π/4·(D² − d²)`, *or*, if the bundled throughput already accounts for the
obscured area, state that explicitly in `surface`'s docstring and confirm the clear-aperture
PSF is accurate enough for the required EE/saturation tolerance. Right now the obstruction is
neither modeled nor documented.

---

### 🟠 #4 — STILL OPEN: read noise is silently doubled, by two different mechanisms

**Where:** `sensor.py:146` applies `... * 2 # multiply by 2 to allow for unmodelled noise
sources` in *code* for the ZWO gain-curve path; `qcmos.toml:2` bakes the doubled value into
the *config* (`read_noise = 0.56 # 0.28 # ... doubling to be conservative`).

Doubling the RMS quadruples the read-noise *variance*. As an internal margin this is
defensible, but for a public ETC it is surprising, non-physical, and — worse — applied two
*different* ways (code for one detector, data for the other), so anyone "fixing" one could
silently double-double the other. A user comparing against the qCMOS datasheet (0.28 e⁻) or
another ETC gets disagreeing answers with no visible reason.

**Fix.** Carry the true datasheet read noise in both configs and apply a single, documented,
configurable `read_noise_margin` (default 2.0) in one place in `from_config`; surface it in
any `describe()`/docs output.

---

### 🟠 #5 — Sky/zodiacal normalization: encircled-energy applied to the sky (legacy path), plus unresolved band/magsys provenance

**(a) `get_countrates` multiplies the sky background by the PSF encircled-energy fraction.**
`simulation.py:591` does `count_rate = count_rate_total * ee_at_aper` for **every** scene
element, including the surface-brightness sky. A uniform background fills the aperture — it
does not concentrate like a point source — so multiplying it by `ee_at_aper (<1)` *under-counts*
the sky. This is in the **deprecated** 1-D analytic path (`get_countrates` warns; the headline
2-D path via `_count_rate_components` correctly uses a per-pixel rate and is **not** affected),
but `get_countrates` is still a public method a user can call directly and will return a wrong
sky rate (and, for the default `units="adu/s"`, also inherits the gain issue of #1 on qCMOS).

**(b) Zodi band/magsys provenance.** `scene.py:327` hardcodes the default zodi as
`{"mag": 22.5, "surface_brightness": True, "bandpass": "johnson_v"}` with no explicit
`magsys`, so 22.5 is taken as a **Vega V** surface brightness — while `lazuli.toml` carries
`zodi_mag_r = 22.5` (an **r-band** label) with the comment *"L3-0016 has v-band mag of 22.1.
Need trace of where 22.5 originates from."* So the number's band, its magnitude system, and
the L3 value (22.1) are all unreconciled. Post-sky-fix the SNR is now (correctly) sensitive to
this level, so it matters.

**Fix.** (a) In `get_countrates`, do not apply `ee_at_aper` to surface-brightness elements
(sky count rate = per-pixel rate × n_pix in aperture); or formally retire the method. (b) Make
the zodi config explicit — `{"mag": <correct value>, "magsys": "abmag", "bandpass": <matching
band>, "surface_brightness": True}` — with the band label matching the value's true band, and
document the L3 provenance.

---

## 4. Are any of the scientific performance tests wrong?

No test asserts anything *physically false*, and the physics tests are strong. The issues are
**coverage** and **over-stated guard claims**:

- **`test_source_physics.py` — exemplary, keep.** Planck ratio + Wien's law, flat Fν → λ⁻² in
  Fλ, power-law slope, Gaussian line peak/FWHM/flux-ratio, Pogson 100× — all against
  closed-form analytic truth with tight tolerances. This is the model the rest of the suite
  should follow.
- **`test_psf_wavelength.py` — good.** Pins pivot (not `wpeak`) and proves the two differ by
  >10 % for the broadband filter, so it actually catches the bug it guards.
- **`test_snr.py::test_snrs_25p4_mag_60s` (the 4.95 pin) — weak guard, misleading docstring.**
  The pin is correctly *derived* (it is the post-fix code output, and the pivot fix did move
  it), but this config (AB 25.4, sky 22.5) is **read-noise dominated** (RN²·n_pix ≈ 528 e²
  vs sky ≈ 34 e²): dropping the sky term moves SNR only 4.95 → 5.08 (~2.5 %), barely inside the
  `abs=0.1` tolerance. So despite its docstring, it is a near-useless guard for the sky fix
  (that job is correctly done by `test_sky_background_noise.py`). **Fix:** re-point the
  docstring to call it a faint-source/pivot snapshot, or add a sky-dominated companion pin.
- **`test_sky_background_noise.py` — strong, but its independence claim is over-stated.** The
  Monte-Carlo cross-check is genuinely independent of the analytic noise *formula*, but both it
  and the closed-form check pull rates from the same `_count_rate_components` primitive, so
  neither would catch a wrong *absolute sky rate* — only a dropped sky *term*. Tolerances are
  well chosen. **Fix:** soften the "none of these re-uses…" docstring claim; the absolute-rate
  gap is covered by the new zeropoint test recommended in §3(a).
- **`test_airy.py` peak-fraction 0.1333 / saturation snapshots — fine as regression guards**,
  but they are code-output snapshots (~4 % above the analytic continuous-Airy central fraction
  of ≈0.128, because of truncated-window renormalization), not independent physics. Acceptable;
  just don't read them as analytic validation.
- **`test_saturation_count.py` — good and independent** (hand-computes ADC/well thresholds; cross-checks
  against `ImageSimulator.simulate`). Note: once #1 is fixed, the qCMOS saturation thresholds
  these tests pin will change — update them deliberately as part of the gain fix.

**Bottom line on tests:** the suite is solid on spectral physics, PSF geometry, saturation
*logic*, and noise-formula *assembly*. The one real exposure is the **absence of any
independent absolute-flux/zeropoint test** (§3a).

---

## 5. Lower-severity findings & simplifications

- **PSF realism: the defocus PSF is wavelength-frozen at 500 nm.** `AiryPSF` scales with
  `ctx.wavelength_m`, but `DefocusPSF`/`_ResampledPSF.render` (psfsim.py:~132) only resamples
  the bundled monochromatic 500 nm Zemax data by the pixel-scale ratio; changing the simulation
  wavelength leaves the defocus peak-pixel fraction unchanged. Defocused fiducial configs in,
  e.g., the r band are therefore modeled with 500 nm defocus structure. Either rescale for
  wavelength or document the fixed-500 nm approximation and bound the per-band EE error.
- **`FitsImg.aperture_photometry` noise budget (peripheral path).** In this helper (not the
  headline SNR path), the source shot term `shot_var = net_flux` is taken in image units (ADU
  when `gain≠1`) while the final error divides by `gain`, so the source-Poisson term is wrong
  by a factor of `gain` unless data are already in electrons; and the empirical `bkg_std` term
  is added on top of explicit `read_var`/`dark_var`, double-counting read noise. Convert to
  electrons up front and pick *either* the empirical *or* the analytic noise model, not both.
  Also clamp `shot_var = max(net_flux, 0)`.
- **`radial_data.py` uses `== None` on possibly-array arguments** (lines ~99, 103, 110); with a
  non-default `x`/`y`/`working_mask`/`rmax` this raises under array truthiness. Use `is None`.
- **`get_ee_value_at_radius` (airy.py:45-49)** has no `bounds_error=False`; an aperture radius
  beyond the computed EE grid raises `ValueError` instead of returning EE≈1. Add
  `bounds_error=False, fill_value=(0.0, ee[-1])`.
- **qCMOS dark-current temperature is unreconciled** (qcmos.toml: measured at −20 °C, applied
  at an assumed 0 °C with no scaling; the comment flags the discrepancy). Dark is usually
  subdominant for the WCC, but confirm the operating temperature and either supply a dark-vs-T
  curve or apply a documented scaling.
- **Lunar brightness model is linear-in-illuminated-fraction** (`astro.get_moon_magnitude`):
  `−2.5·log10(frac)` gives ~0.75 mag dimming at quarter phase vs the real ~2.5 mag (no
  opposition surge / phase function). Fine as a rough helper, but label it approximate; the
  moon-scatter path it feeds is also broken (#2, `trapezoid`).
- **`force='extrap'` on stellar-template `Observation`s** (scene.py:593) will linearly
  extrapolate flux beyond a template's wavelength coverage into a broad WCC band. Prefer
  `force='taper'` for empirical templates, or validate band ⊂ template coverage.
- **`eval(f"self.{origin...}")` in `Simulation._fullkey_to_value`** — keys are internal so not a
  security issue, but `getattr`-chaining is clearer and avoids `eval` in a trusted library.
- **`is_saturated` vs `saturation_mask_from_image_e`** use slightly different criteria (ADC+bias
  vs ADC-without-bias-OR-well); the docstring acknowledges this is an open reconciliation. With
  `bias_level=0` they mostly agree today, but once #1 makes the well binding they may diverge —
  unify them.
- **`Sensor.get_plate_scale`** hardcodes `206265` with a `# why 206265` comment; replace with
  `(1*u.rad).to(u.arcsec)` for clarity (no numeric change). The formula is correct.

**Not an issue (checked and dismissed):** the `_radial_cumulative` "partial annulus"
quantization — signal, noise, and `n_pix` all use the *same* sorted pixel set, so the discrete
aperture (the N nearest pixels) is internally self-consistent; it is a sampling definition, not
an error. The surface-brightness → integrated-magnitude formula
(`mu − 2.5·log10(area)`) is dimensionally correct in the sub-arcsec-pixel regime.

---

## 6. Repo hygiene for a public release

`build/` is checked in (including an obsolete `source.py` no longer in `src/`); there are
numerous `.DS_Store` files, a `.code_review_report.md.swp` swap file, a stray
`vega_spectrum.png` at repo root, and the `.superpowers/` / `docs/superpowers/` planning dirs.
Prune these and add them to `.gitignore` before publishing. Also drop the third-party `toml`
dependency once the dead `wcc_etc.py` module is removed (#2).

---

## 7. Bottom line — what to block release on

1. **#1 (qCMOS gain units)** — resolve the e⁻/ADU vs ADU/e⁻ contradiction and add the
   `gain·adc_max ≈ well_depth` sanity assertion. ~80× error on a primary detector's ADU and
   saturation; highest-impact *new* finding.
2. **#2 (broken `import *` surface)** — `trapezoid`/`warnings` `NameError`s and the
   self-contradictory `WCCETC` class are crash-prone, self-admitted-wrong public API.
3. **#3 (absolute flux scale)** — add the independent zeropoint test and model/document the
   central obstruction; without these the most basic ETC number is untested and biased.
4. **#4 (read-noise doubling)** and **#5 (sky normalization / zodi provenance)** — make the
   margin explicit and reconcile the zodi band/magsys so public numbers are defensible.

The two recent fixes (sky background, pivot wavelength) are correct, and the core SNR/exptime
algebra is sound. The remaining problems are concentrated in detector calibration constants,
the absolute flux scale, and the legacy module — all fixable without touching the architecture.
