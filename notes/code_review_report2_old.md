# WCC ETC — Pre-Release Scientific Code Review (independent pass)

**Reviewer:** Claude (Opus 4.8)
**Date:** 2026-06-24
**Scope:** `src/wcc_etc/` and `tests/` on **`main`** (after merging both
`fix/sky-background-snr-noise` and `fix/psf-pivot-wavelength`), focused on
scientific correctness of the count-rate / SNR / exposure-time and saturation
calculations for the Lazuli Wide-field Context Camera ETC, ahead of public release.
**Test status at review:** `309 passed` (full suite, ~90 s), all green, including
the new sky-background regression guards.

> **Update — this pass now reflects merged `main`.** The review was first run on a
> stale branch (`fix/psf-pivot-wavelength`) that lacked the sky-background fix, so
> the original #1 below reproduced as a live 2.3× bug. Both fix branches have since
> been merged to `main` and verified: **#1 is RESOLVED on `main`** (sky shot noise
> is now in the SNR/exptime noise budget, and SNR correctly drops when the sky
> brightens). The remaining findings (#2–#4) were re-checked against the current
> `main` and still stand. The PSF-wavelength fix (pivot, not peak transmission) is
> also in.

---

## 1. Executive summary

The architecture is sound: clean `Telescope`/`Sensor`/`Scene`/`Simulation`
separation, a genuine 2-D PSF-aware image path, spectrally-correct count rates
via `synphot.Observation` over the full bandpass, parametric source spectra with
independent physics tests, and a sensible deprecation story for the old analytic
Airy code. The recent move to render the PSF at the **pivot** wavelength (rather
than peak transmission) is the right call and is already committed.

**The one critical scientific error — sky (zodiacal) background dropped from the
SNR/exptime noise — has now been FIXED and merged to `main`** (`fix/sky-background-snr-noise`,
commit `66e6e30`). It is retained as #1 below for the record, with its before/after
evidence, because it is the most important calculation in the tool and now carries
independent regression guards. The four items still open before release are #2–#5.

Below are the five items in priority order (with #1 marked resolved), then the
science-test assessment and lower-severity findings.

---

## 2. The five things to fix before release

### ✅ #1 — RESOLVED on `main`: the sky/zodiacal background is now in the SNR and exposure-time noise

> **Status: FIXED** by `fix/sky-background-snr-noise` (commit `66e6e30`), merged to
> `main`. `get_image_snr._snr_at` and `get_image_exptime_for_snr` now add
> `background_rate_per_pix` to the per-pixel noise term, and
> `tests/test_sky_background_noise.py` guards it with a closed-form CCD check, a
> Monte-Carlo cross-check, and a "brighter sky lowers SNR" check. Verified
> post-merge: the sky-limited case below now returns the correct ~22.1 (not 50.1),
> and brightening the sky 18→16 mag/arcsec² drops SNR 22.1 → 9.6. The description
> below is the pre-fix behavior, kept for the record.

**Where (the original bug):**
- `simulation.py::get_image_snr._snr_at`, line 923 — only `diffuse_rate_per_pix` is passed.
- `simulation.py::get_image_exptime_for_snr`, line 1013 — same omission.
- The noise is then assembled in `psfsim.py::aperture_snr_radial` (line 953) and
  `aperture_time_for_snr` (line 1003).

`_count_rate_components()` (simulation.py:603) correctly splits the scene into
three rates: `source_rate_total`, `background_rate_per_pix` (the sky/zodi), and
`diffuse_rate_per_pix` (host + anything that is neither source nor background).
The render bundle (`_image_render_bundle`) carries all three. But the SNR/exptime
path forwards only the **diffuse** term into the per-pixel noise:

```python
# get_image_snr._snr_at  (simulation.py:921-926)
source_e_total = b["source_rate_total"] * source_scale * t_sec
diffuse_per_pix = b["diffuse_rate_per_pix"] * t_sec          # <-- background_rate_per_pix dropped
dark_per_pix    = dark_rate_per_pix * t_sec
prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                           source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
```

For the default scene (point source + zodi, **no host**), `diffuse_rate_per_pix == 0`,
so the **entire sky background is excluded from the noise**. Meanwhile the
Monte-Carlo image path (`ImageSimulator.simulate`, psfsim.py:336–346) and the
saturation path (`get_peak_pixel`, `_per_frame_clean_image_e`) *do* include
`background_rate_per_pix`. So the rendered image and the SNR formula are built
from two different noise models — the rendered image is right, the headline SNR
is wrong.

**Evidence (reproduced live during this review):**

Scene: `G5V`, AB = 24.0, `sony:r`, 300 s, sky brightened to 18 mag/arcsec² in the
default aperture:

| Quantity | Value |
|---|---|
| `get_snr(...)["snr"]` (reported) | **50.07** |
| Correct SNR (same code, sky added to noise) | **22.13** |
| Sky electrons in aperture | 14,720 e⁻ |
| Source electrons in aperture | 2,994 e⁻ |
| **Overestimate** | **≈ 2.3×** |

The aperture holds ~5× more sky charge than source charge, yet none of that sky
shot noise enters the SNR. A second check is even more diagnostic: brightening the
sky from 18 → 16 mag/arcsec² (6× more sky flux) leaves the reported SNR **exactly
unchanged** (50.07 → 50.07). The noise model is completely blind to the sky.

**Impact.** SNR is overestimated and the required exposure time is *under*estimated
for any faint, sky-limited target — precisely the regime the WCC ETC exists to plan
(fiducial-observation timing, filter trade studies, L3 performance numbers). For
bright/read-noise-limited targets the error is small, which is exactly why it has
gone unnoticed.

**Fix (≈2 lines).** Fold the background per-pixel rate in wherever the diffuse term
feeds the noise:

```python
# get_image_snr._snr_at
diffuse_per_pix = (b["diffuse_rate_per_pix"] + b["background_rate_per_pix"]) * t_sec

# get_image_exptime_for_snr -> aperture_time_for_snr(...)
b["diffuse_rate_per_pix"] + b["background_rate_per_pix"],   # in place of diffuse alone
```

Better still, make the omission *impossible to recur*: give `aperture_snr_radial`
and `aperture_time_for_snr` an explicit `sky_per_pix` argument distinct from the
diffuse term, so a caller cannot silently forget it. Then add the regression tests
in §3.

---

### 🔴 #2 — Broken, self-contradictory public API is shipped via `from .wcc_etc import *`

**Where:** `__init__.py:17` does `from .wcc_etc import *`, and `wcc_etc.py` has no
`__all__`, so its entire public surface lands in the top-level namespace that
external users will `dir()` and call. Several members raise on use or are flagged
as wrong by their own docstrings:

- **`calc_moon_scatter_countrate_per_pixel`** (wcc_etc.py:1002) returns
  `trapezoid(ff, ww)` at line 1029 (and again ~1049), but `trapezoid` is **never
  imported** — only `from numpy import sqrt`. Any call raises `NameError`.
- **`utils.list_of_quantity_to_array`** calls `warnings.warn(...)` at utils.py:88
  while `warnings` is never imported. The mixed-unit branch raises `NameError`
  instead of warning. (This is reachable from the deprecated analytic path.)
- **`WCCETC.calc_SNR`** (wcc_etc.py:708) ships with the docstring *"NOTE — Does not
  correctly account for gain"*, alongside `calc_SNR_one_frame_with_gain` whose
  noise uses a hard-coded `n_b = n_pix * 100000  # assuming background is well known`
  fudge (line 747).
- `WCCETC.simulate_2D_psf` raises `NotImplementedError` (line 801).
- `import toml` (third-party) at the top of `wcc_etc.py` pulls a dependency that
  exists only for this dead module — the rest of the codebase uses stdlib `tomllib`.

**Impact.** A public astronomical user exploring `wcc_etc.<TAB>` finds class and
function names that either crash (`NameError`/`NotImplementedError`) or carry
self-admitted scientific errors. That is a credibility and support-burden problem
for a tool meant to be trusted for mission planning.

**Fix.** The genuinely useful helper in this module is
`get_wcc_snr_and_simulation` (wcc_etc.py:874), which wraps the modern `Simulation`.
Keep that (move it to `simulation.py` or a thin `convenience.py`), stop
`import *`-ing the rest, and either delete or clearly quarantine `WCCETC`,
`calc_moon_scatter_countrate_per_pixel`, and friends. At minimum: add `warnings`
to `utils.py`, and do not export the broken names. (Add `import *` discipline via
an explicit `__all__` in any module re-exported at top level.)

---

### 🟠 #3 — Read noise is silently doubled, inconsistently, in two different places

**Where:**
- `sensor.py::from_config`, line 146:
  `read_noise = parse_and_interpolate(...) * 2 # multiply by 2 to allow for unmodelled noise sources`
  (applied in *code*, for the ZWO gain-curve path).
- `qcmos.toml`: `read_noise = 0.56 # 0.28 # ... doubling to be conservative` —
  the already-doubled value is baked into the *config*.

Doubling the RMS quadruples the read-noise *variance*. As an internal engineering
margin this is defensible, but for a **public** ETC it is surprising, non-physical,
and applied two different ways (code for one sensor, data for the other). A user
comparing against the qCMOS datasheet (0.28 e⁻) or against another ETC will get
disagreeing answers with no visible reason.

**Fix.** Make the margin a single, documented, configurable field — e.g.
`read_noise_margin = 2.0` applied once in `from_config` — and surface it in any
`describe()`/docs output. Carry the true datasheet read noise in the config.

---

### 🟠 #4 — PSF realism: defocus PSF is wavelength-frozen, and the aperture is unobstructed

Two related limitations that bias the encircled-energy / aperture / saturation
numbers, especially for the broadband mode and the defocused fiducial configs.

**(a) `DefocusPSF` ignores wavelength entirely.** `AiryPSF` correctly scales with
`ctx.wavelength_m`. But `DefocusPSF`/`_ResampledPSF.render` (psfsim.py:132) only
resamples the bundled Zemax Huygens data by the *pixel-scale* ratio
(`src_um_per_pix / pixel_size_um`); the data itself is monochromatic **500 nm**
(see the file names / header comment, psfsim.py:26). I verified this directly:
halving the simulation wavelength changes the Airy peak-pixel fraction
(0.140 → 0.283) but leaves the defocus peak-pixel fraction **exactly unchanged**
(0.0027 → 0.0027). So a defocused observation in, say, the `r` band (pivot ≈ 614 nm)
or any NIR band is modeled with the 500 nm defocus structure — wrong diffraction
scale and wrong EE. This directly feeds the `from_sensorfilter` defocus modes used
for fiducial observations.

**(b) Collecting area and Airy PSF assume an unobstructed circular aperture.**
`telescope.py::surface` returns `π·(D/2)²` with no central obstruction, and
`airy.get_airy_psf` is the clear-aperture Airy pattern. A telescope of this class
has a secondary + spider: ignoring the obscuration (i) overstates the geometric
collecting area (a linear error on every count rate and on SNR), and (ii) gives a
PSF with too little energy in the wings (overstated EE in a fixed aperture,
understated diffraction-spike contribution to saturation). The two errors partially
cancel in SNR but **not** in EE/saturation. The code and configs record no
obstruction at all.

**Fix.** (a) Either rescale the defocus PSF for wavelength, or document loudly that
defocus PSFs are fixed-500 nm approximations and bound the resulting EE error per
band. (b) Fold the obstruction into `surface` and use an obscured (annular) Airy —
or, if the bundled throughput already accounts for the obscured area, state that
explicitly in `surface`'s docstring and confirm the clear-aperture PSF is accurate
enough for the required EE/saturation tolerance.

---

### 🟡 #5 — Mostly ADDRESSED: the SNR test coverage gap that let #1 hide

> **Status: largely closed.** `fix/sky-background-snr-noise` added
> `tests/test_sky_background_noise.py`, which now provides exactly the independent
> guards this finding called for (closed-form CCD noise incl. sky, Monte-Carlo
> scatter cross-check, brighter-sky-lowers-SNR, and an exptime round-trip). The
> headline pin `test_snrs_25p4_mag_60s` was also updated: merging both fixes moved
> the correct AB-mag value from the stale `5.1` to `4.95` (sky noise now counts and
> the PSF is at the pivot wavelength), confirming the old value was bug-contaminated.

The original gap, kept for the record: the headline SNR regression pinned
`get_snr ≈ 5.1` at AB = 25.4 in 60 s — a **read-noise/dark-limited** point where the
missing sky term was only a few percent, so the §1 bug stayed inside the `abs=0.1`
tolerance. Every other SNR test either compared `get_snr` to `get_image_snr` (same
code path — structurally unable to detect the bug) or checked saturation (a
different, correct path). No test compared the analytic SNR noise to an independent
sky-inclusive ground truth — which is how a 2.3× error shipped green.

Remaining nice-to-have: the Monte-Carlo cross-check in `test_sky_background_noise.py`
ties the analytic path to the rendered-image path; consider extending it to the
`optimize`/`ee_frac` aperture modes too, not just the fixed default aperture.

---

## 3. Are any of the scientific performance tests wrong?

- **`test_source_physics.py` — good, keep.** Planck ratio, Wien's law, F_ν/F_λ
  conversion, Gaussian line peak/FWHM, two-line flux ratio, and the Pogson
  5-mag → 100× scaling are all checked against independent analytic truth. These
  are the model tests to emulate.
- **`test_snr.py::test_snrs_25p4_mag_60s` — enshrines a value produced by the §1
  bug.** It is in a regime where the bug is sub-tolerance, so it neither catches
  nor breaks on the fix. Replace the magic `5.1` with a value derived from the CCD
  equation *including sky*, and tighten the tolerance.
- **`test_saturation_count.py` / `test_psf_aware_saturation.py` — good and
  self-consistent.** They exercise the path that *does* include background, and are
  in fact what makes the SNR-path inconsistency visible.

No errors found in the *physics* tests themselves; the gap is one of **coverage**
(no independent or Monte-Carlo SNR cross-check), not of incorrect assertions.

---

## 4. Lower-severity findings & simplifications

- **Zodi band/magsys provenance.** `get_scene_element` hardcodes the default zodi as
  `{"mag": 22.5, "surface_brightness": True, "bandpass": "johnson_v"}` (scene.py:327),
  while `lazuli.toml` carries `zodi_mag_r = 22.5` with the comment *"L3-0016 has
  v-band mag of 22.1. Need trace of where 22.5 originates from."* So an
  r-band-labelled value is applied as a V-band surface brightness, and the L3 number
  (22.1) is different again. Reconcile the band, value, and provenance before
  release — these set the sky level that §1 will (correctly, post-fix) make the SNR
  sensitive to.
- **qCMOS gain comment is alarming but the value appears self-consistent.**
  `qcmos.toml` has `gain = 0.112 # NOTE: ... ADU/e rather than e/ADU. Ask`. The code
  treats `gain` as e⁻/ADU (`image_e / gain → ADU`); 0.112 e⁻/ADU is the reciprocal
  of ~8.9 ADU/e⁻, so it is consistent with the code's convention *if* the detector
  conversion gain really is ~8.9 ADU/e⁻. But with a 12-bit ADC this caps the
  measurable signal at 4095 × 0.112 ≈ 459 e⁻ — far below the 7000 e⁻ well — so the
  ADC clip, not the well, sets saturation. That may be correct for a high-conversion-
  gain qCMOS mode, but it is worth an explicit confirmation and a cleaned-up comment
  (drop the "Ask") so a public reader does not think the value is unverified.
- **`is_saturated` vs. `saturation_mask_from_image_e`.** `is_saturated`/`get_peak_pixel`
  test the ADC clip + bias; `saturation_mask_from_image_e` tests ADC-clip-without-bias
  OR full-well. The docstring already flags that the two criteria are not fully
  reconciled — acceptable to ship, but document it in the user guide so saturation
  flags are interpreted consistently.
- **`eval(f"self.{origin...}")` in `Simulation._fullkey_to_value`** (simulation.py:377).
  Keys are internal so it is not a security issue, but `getattr`-chaining is clearer
  and avoids `eval` in a library others will read and trust.
- **`get_moon_magnitude`** (astro.py) is a crude fraction × inverse-square model with
  no opposition surge or phase-function physics; fine as a rough helper, but label it
  as approximate in the docs if it is exposed publicly.
- **Even-`npix` notice** prints to `stderr` on first use (`render_detector_psf`).
  Consider a `warnings`-based once-only message so it is suppressible in notebooks.

## 5. Repo hygiene for a public release

- `build/` is checked in (including an obsolete `source.py` that no longer exists in
  `src/`), as are numerous `.DS_Store` files, a stray `vega_spectrum.png` at repo
  root, the `.code_review_report.md.swp` swap file, and the `.superpowers/` /
  `docs/superpowers/` planning dirs. Prune these and add them to `.gitignore` before
  publishing.

---

## 6. Bottom line

**#1 (the critical sky-background error) is now fixed and merged to `main`, with
regression guards** — the most important calculation in the tool is correct and
protected. The remaining release blockers are **#2** (broken `import *` surface —
`trapezoid`/`warnings` NameErrors and the self-contradictory `WCCETC` class, all
still present on `main`) and **#3** (silent, inconsistent read-noise doubling), both
about not shipping crash-prone or surprising public API. **#4** (defocus-PSF
wavelength freeze + unobstructed aperture) is needed for the scientific
defensibility of the broadband and defocused-fiducial numbers. **#5** is largely
closed by the new sky-background test module. The architecture and the
source-physics tests are solid; the remaining problems are concentrated and fixable.
