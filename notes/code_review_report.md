# WCC ETC — Pre-Release Code Review

**Reviewer:** Claude (Opus 4.8)
**Date:** 2026-06-24
**Scope:** `src/wcc_etc/` and `tests/`, with emphasis on scientific correctness of the
exposure-time / SNR calculations for the Lazuli Wide-field Context Camera.
**Test status at review:** `297 passed` (full suite, 32 s). All green — but see §3, the
green suite hides the most important bug.

---

## 1. Executive summary

The package is in good overall shape: clean separation of `Telescope` / `Sensor` /
`Scene` / `Simulation`, a real 2-D PSF-aware image path, physically-motivated parametric
spectra with independent physics tests, and a sensible deprecation story for the old
analytic Airy code.

However, **there is one critical scientific bug that must be fixed before public
release**: the headline SNR and exposure-time methods omit the sky (zodiacal) background
from the noise budget. This is the single most important quantity an ETC has to get right,
and it is currently wrong by up to a factor of ~2 in the sky-limited regime that the WCC
actually operates in for faint sources. Below are the 5 things I would block release on,
in priority order, followed by lower-severity findings and a tests assessment.

---

## 2. The 5 things to fix before release

### 🔴 #1 — CRITICAL: sky/zodiacal background is dropped from the SNR & exposure-time noise

**Where:** `simulation.py::get_image_snr._snr_at` (≈ line 921–926) and
`simulation.py::get_image_exptime_for_snr` (≈ line 1012–1013). `get_snr` (the documented
public entry point) delegates to `get_image_snr`, so it inherits the bug.

**What happens.** `_count_rate_components()` correctly splits the scene into three rates:
`source_rate_total`, `background_rate_per_pix` (the sky/zodi), and `diffuse_rate_per_pix`
(host + anything else). The render bundle keeps all three. But the SNR path passes **only
`diffuse_rate_per_pix`** into `aperture_snr_radial`:

```python
# get_image_snr._snr_at
diffuse_per_pix = b["diffuse_rate_per_pix"] * t_sec        # <-- sky (background) missing
prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                           source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
```

For a normal scene (point source + zodi, no host) `diffuse_rate_per_pix == 0`, so the
**entire sky background is excluded from the noise**. The Monte-Carlo image path
(`ImageSimulator.simulate`) and the saturation path (`get_peak_pixel` /
`_per_frame_clean_image_e`) both *do* include `background_rate_per_pix` — so the rendered
image and the SNR formula are computed from different noise models.

**Evidence (reproduced during review):**

| Case | reported `get_snr` | correct (sky in noise) | legacy `get_snr_airy` | error |
|---|---|---|---|---|
| G5V, AB=25.4, zodi 22.5, 60 s, sony:r | 5.14 | 5.01 | — | +2.5% |
| G5V, AB=24.0, sky 18.0, 300 s, sony:r | **44.16** | **20.69** | 22.13 | **+113%** |

In the second (sky-dominated) case the aperture holds ~10,700 sky e⁻ vs ~2,400 source e⁻,
yet none of that sky shot noise enters the SNR. Note the *deprecated* analytic Airy path
(`get_snr_airy`, which sums all scene count rates) gets it essentially right (22.1) — i.e.
the regression that everyone is told to use is the broken one.

**Impact.** SNR is overestimated and required exposure time is *under*estimated for any
faint, sky-limited target — exactly the regime the WCC ETC is used to plan. This will
propagate into fiducial-observation timing, filter trade studies, and any L3 performance
numbers derived from the tool.

**Fix (≈2 lines).** Include the background per-pixel rate everywhere the diffuse term is
used in the noise:

```python
# get_image_snr._snr_at
diffuse_per_pix = (b["diffuse_rate_per_pix"] + b["background_rate_per_pix"]) * t_sec

# get_image_exptime_for_snr
result = aperture_time_for_snr(
    b["psf_norm"], b["plate_scale_mas"], b["source_rate_total"],
    b["diffuse_rate_per_pix"] + b["background_rate_per_pix"],   # <-- add background
    dark_rate_per_pix, read_noise, n_reads=n_reads, snr=snr, ...)
```

(Either combine here, or better: have `aperture_snr_radial` /`aperture_time_for_snr`
take an explicit `sky_per_pix` argument so the omission can't recur silently.)
**Add a regression test** that cross-checks `get_snr` against the closed-form CCD equation
*including* sky, and against the scatter of `ImageSimulator.simulate` frames — see §4.

---

### 🟠 #2 — PSF is monochromatic at the *peak-throughput* wavelength, not the effective wavelength

**Where:** `sensor.py::Sensor.wavelength` → `self.bandpass.wpeak()`, consumed by every PSF
render (`ImageSimulator._context`, `_compute_psf_profile_impl`).

`wpeak()` returns the wavelength of **maximum transmission**, which for a roughly
flat-topped filter is essentially arbitrary within the band. The diffraction PSF scales
linearly with λ, so this directly biases the PSF width, the encircled-energy curve, the
optimal aperture, and the peak-pixel fraction used for saturation. Measured during review:

| band | `wpeak` (PSF λ used) | `pivot` | `avgwave` | bias in λ |
|---|---|---|---|---|
| sony:r | 560 nm | 614 nm | 615 nm | −9% |
| sony:bb (broadband) | 485 nm | 582 nm | 592 nm | **−17%** |
| qcmos:bb | 480 nm | 613 nm | 626 nm | **−22%** |

The broadband filter — a primary WCC mode — gets a PSF ~17–22% too small, so EE inside a
fixed aperture is overstated and the saturation peak-pixel fraction is overstated.

**Fix.** Use `bandpass.pivot()` (or the source-weighted effective wavelength) for the
representative monochromatic PSF λ. Ideally document that the PSF is monochromatic and note
the polychromatic limitation; the source spectrum also weights the in-band photon
distribution and is currently ignored for the PSF.

---

### 🟠 #3 — Collecting area and Airy PSF assume an unobstructed circular aperture

**Where:** `telescope.py::Telescope.surface` = `π·(D/2)²` (D = 3.065 m, no central
obstruction), and `airy.get_airy_psf` is the unobstructed circular-aperture Airy pattern.

A space telescope of this class has a secondary mirror / spider. Ignoring the central
obstruction (a) overestimates the geometric collecting area (linear error on every count
rate and hence on SNR), and (b) gives the wrong PSF — an obstructed aperture pushes more
energy into the wings, lowering EE inside a fixed aperture and raising the diffraction-spike
contribution to saturation. The two errors partly cancel in SNR but not in EE/saturation.

**Fix.** Either fold the obstruction into the collecting area and use an obscured-aperture
PSF (annular Airy), or — if the bundled throughput curves already account for the
obstructed area — state that explicitly in the docs and in `surface`'s docstring, and
confirm the PSF approximation is acceptable for the required accuracy. Right now neither the
code nor the config records the obstruction at all.

---

### 🟡 #4 — Read noise is silently doubled

**Where:** `sensor.py::from_config` multiplies the interpolated read noise by 2 for the
ZWO/gain-setting path (`... * 2 # multiply by 2 to allow for unmodelled noise sources`);
the qCMOS config carries the already-doubled value (`read_noise = 0.56  # 0.28, doubling to
be conservative`).

Doubling the RMS quadruples the read-noise *variance*. This is a defensible engineering
margin, but for a **public** ETC it is surprising and non-physical: users comparing against
other ETCs or against the datasheet (0.28 e⁻ for the qCMOS) will get answers that disagree
and won't know why. It is also applied inconsistently (in code for one sensor, in the config
value for the other).

**Fix.** Make the margin explicit and configurable — e.g. a documented
`read_noise_margin = 2.0` field in the config, applied in one place — and surface it in
`describe()`/docs. Do not bake it silently into both the code and the data.

---

### 🟡 #5 — The legacy `wcc_etc.py` module is half-deprecated, partly broken, and fully re-exported

**Where:** `wcc_etc.py` is exported wholesale via `from .wcc_etc import *` in
`__init__.py`, so its public surface ships to users.

Problems found:
- `calc_moon_scatter_countrate_per_pixel` calls `trapezoid(...)` (lines ~1029, 1049) but
  `trapezoid` is never imported → **`NameError` on any call**.
- `WCCETC.calc_SNR` is shipped with a docstring that literally says *"Does not correctly
  account for gain"*, alongside a second `calc_SNR_one_frame_with_gain` with a hard-coded
  `n_b = n_pix * 100000` "background is well known" fudge.
- `simulate_2D_psf` raises `NotImplementedError` ("removed from the deprecated module").
- `__init__(self, config: [str, dict], ...)` — `[str, dict]` is a list literal, not a valid
  type annotation (harmless but sloppy for a public API).
- `import toml` (the third-party package) while the rest of the codebase uses stdlib
  `tomllib` — an extra dependency pulled in only for dead code.

The genuinely useful helper here is `get_wcc_snr_and_simulation`, which wraps the modern
`Simulation`. **Fix:** keep that helper (move it to `simulation.py` or a thin
`convenience.py`), and delete or clearly quarantine the rest. At minimum, stop
`import *`-ing the broken `WCCETC` class and moon-scatter function into the top-level
namespace before release.

---

## 3. Why the test suite is green despite #1

The SNR regression test (`tests/test_snr.py::test_snrs_25p4_mag_60s`) pins
`get_snr ≈ 5.1` at AB = 25.4 — a *read-noise-limited* point where the missing sky term is
only ~2.5%, so the bug stays inside the `abs=0.1` tolerance. Every other SNR test either
(a) compares `get_snr` to `get_image_snr` (same code path — can't detect the bug), or
(b) checks saturation counts (a different, correct path). **No test compares the analytic
SNR noise against an independent sky-inclusive ground truth, and none compares it against
the Monte-Carlo image scatter.** That gap is exactly why a factor-of-2 error ships green.

---

## 4. Are any of the scientific performance tests wrong?

- `test_source_physics.py` — **good.** Planck ratio, Wien's law, F_ν/F_λ, Gaussian
  line peak/FWHM, two-line flux ratio, and Pogson 5-mag→100× scaling are all checked against
  independent analytic truth. Keep these.
- `test_snr.py::test_snrs_25p4_mag_60s` — **enshrines a value produced by the bug.** It
  passes today and would still (barely) pass after the #1 fix (5.01 is within 0.1 of 5.1),
  so it neither catches the bug nor breaks on the fix. Replace the magic 5.1 with a
  derivation from the CCD equation including sky, and tighten the tolerance.
- `test_saturation_count.py` / `test_psf_aware_saturation.py` — **good and self-consistent**;
  they exercise the path that *does* include background. (They are also what makes the
  inconsistency with the SNR path visible.)

**Recommended new tests:**
1. `get_snr` noise vs. closed-form `sqrt(S + (sky+dark+RN²)·n_pix)` with a deliberately
   bright sky — must agree to <1%.
2. `get_snr` vs. the standard deviation of N `ImageSimulator.simulate(add_noise=True)`
   aperture fluxes — must agree to within Monte-Carlo error.
3. A parametrized check that SNR *decreases* when only the background magnitude is made
   brighter (it currently does not change at all).

---

## 5. Lower-severity findings & simplifications

- **Zodi band label mismatch.** `lazuli.toml` defines `zodi_mag_r = 22.5` (r-band) with a
  comment that L3-0016 is V = 22.1, yet the default zodi `SceneElement` normalizes in
  `johnson_v`. So an r-band number is applied as a V-band surface brightness. Reconcile the
  band and the value, and document the provenance (the comment already flags it as untraced).
- **`is_saturated` vs image-mask reconciliation.** `is_saturated`/`get_peak_pixel` test the
  ADC clip + bias, while `saturation_mask_from_image_e` tests ADC-clip-without-bias OR
  full-well. The docstring acknowledges the two criteria aren't fully reconciled — fine to
  ship, but call it out in user docs so saturation flags are interpreted consistently.
- **`utils.list_of_quantity_to_array`** references `warnings` without importing it; the
  warning branch (mixed units) would itself raise `NameError`. Add `import warnings`.
- **qCMOS gain comment is alarming but the value is correct.** `qcmos.toml` has
  `gain = 0.112 # ... ADU/e rather than e/ADU. Ask`. I verified: the sensor file measures
  8.9 ADU/e, and 1/8.9 = 0.112, so 0.112 e⁻/ADU is the right number for the code's
  convention (`image_e / gain → ADU`). Saturation is self-consistent. **Just delete the
  "Ask" comment** so a public reader doesn't think the value is suspect.
- **Even-`npix` stderr note** (`render_detector_psf`) prints to `stderr` on first use. Fine,
  but consider a `warnings`-based once-only message so it's suppressible.
- **`eval(f"self.{origin...}")` in `Simulation._fullkey_to_value`.** Keys are internal, so
  not a security issue, but `getattr` chaining would be cleaner and avoids `eval` in a
  library that others will read and trust.
- **Repo hygiene for a public release:** `build/` (including an obsolete `source.py`),
  numerous `.DS_Store` files, the `.superpowers/` and `docs/superpowers/` planning dirs, and
  the stray `vega_spectrum.png` at repo root are checked in. Add to `.gitignore` / prune
  before publishing.

---

## 6. Bottom line

Fix **#1 (sky background in the SNR/exptime noise)** without question — it's a ~2-line
change with a factor-of-2 scientific impact, plus the two regression tests in §4 so it can
never silently regress again. Address **#2 (PSF wavelength)** and **#3 (obstruction)** for
scientific defensibility of the broadband and saturation numbers. **#4 (read-noise margin)**
and **#5 (legacy module)** are about not shipping surprising or broken surface area to the
public astronomical community. The architecture and the source-physics tests are solid; the
problems are concentrated and fixable.
