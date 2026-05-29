# PSF Simulator on a Realistic Detector Grid — Design

**Date:** 2026-05-29
**Branch:** `psfs`
**Status:** Approved (pending spec review)

## Goal

Build a robust PSF simulator that renders a point source onto a realistic
detector grid (Sony/IMX and HWK/qCMOS) with noise. It must model:

- (a) Diffraction-limited Airy PSF (default)
- (b) +1 wave defocus (`data/psfs/CAD_1-waves-defocus_500nm_Huygens-PSF-Data_Linear.txt`)
- (c) +2 wave defocus (`data/psfs/CAD_2-waves-defocus_500nm_Huygens-PSF-Data_Linear.txt`)
- (d) A hook for other custom PSFs (e.g. a future scattering PSF)

It evaluates the image on the actual detector pixel grid for both sensors
(different physical pixel sizes), defaulting to small/fast grids.

## Decisions (from brainstorming)

1. **Integrate with `Sensor`/`Telescope`.** The simulator is constructed from a
   sensor name (`sony:r`, `qcmos:r`) and a `Scene`, pulling pixel size, dark
   current, read noise, gain, `adc_max`, diameter, f-number, and jitter from the
   same configs the ETC uses.
2. **Noise (v1):** Poisson shot (source + sky), Poisson dark, Gaussian read
   noise; sky background from the scene; saturation/well clipping. Nonlinearity
   and flat-fielding are deferred to v2.
3. **PSF-source objects** with a shared interface (`AiryPSF`, `DefocusPSF`,
   `CustomPSF`), each rendering a normalized PSF onto the detector grid.
4. **Refactor/replace in place:** the new design replaces `PSFSimulator` and
   `CustomPSF` in `psfsim.py`. `FitsImg`/`FitsImgList`/photometry and helpers
   are kept. Existing scratch notebooks using the old API will need updating.
5. **Source flux from the ETC scene (magnitude):** the source electron rate is
   computed like `Simulation.get_countrates`, and the sky background comes from
   the scene's background element.
6. **Default grid `npix=300`** for both sensors (see grid-size analysis).

### Architecture choice

**Compose on top of `Simulation`** (approach A). `ImageSimulator` builds/holds a
`Simulation` (telescope + sensor + scene) and reuses its physics
(`get_countrates`, `psf_profile`, sensor noise params, `adc_max`/well) rather
than re-deriving them. PSF-source objects handle the 2D rendering. This avoids
duplicating the countrate/units logic that the saturation feature already
established and tested. (Rejected: standalone re-derivation; adding
`simulate_image()` onto the scalar ETC `Simulation`.)

## Grid-size analysis (justifies npix=300)

Encircled energy measured directly from the two Huygens files (256×256, 4 µm
spacing, ±510 µm half-extent). The defocused PSF is the limiting case; the Airy
core is far smaller.

| PSF | 99% EE diameter | Sony (3.76 µm) | HWK (4.6 µm) |
|---|---|---|---|
| 1-wave defocus | 289 µm | 77 pix | 63 pix |
| 2-wave defocus | 385 µm | 102 pix | 84 pix |

Energy on the data's outermost border is ~1.6e-5, so the 1024 µm Zemax data area
already contains the PSF (not truncated). A 300×300 grid (±150 pix) therefore
captures ≥99% for both sensors and both defocus levels, with margin. `npix` is
overridable.

## Components

### 1. PSF-source classes (`psfsim.py`, replacing `CustomPSF`)

A shared interface. Each renders a **normalized (sum = 1)** PSF onto a detector
context. The context bundles what every source may need:

```
DetectorPSFContext:
    npix:               int          # grid size (square)
    pixel_size_um:      float        # detector pixel size (microns)
    plate_scale_mas:    float        # detector plate scale (mas/pix)
    wavelength_m:       float        # band peak wavelength (for Airy)
    diameter_m:         float        # primary diameter
    fnum:               float        # focal ratio
    jitter_sigma_mas:   float        # jitter (0 = none)
    center:             (cx, cy) | None   # sub-pixel center; None = grid center
    oversample:         int          # sub-pixel sampling for accurate binning
```

Interface:

```
class PSFSource:
    def render(self, ctx: DetectorPSFContext) -> np.ndarray:
        """Return an (npix, npix) array summing to 1, peak at `center`."""
```

- **`AiryPSF`** — diffraction-limited, default. Delegates to
  `airy.render_detector_psf` using `ctx` optics (wavelength, fnum, D,
  pixel_size, jitter, npix, oversample). Already produces a normalized,
  peak-centered detector-grid PSF.
- **`DefocusPSF(path, src_um_per_pix=4.0)`** — loads a Zemax Huygens text file:
  - Auto-detect the header by skipping leading lines that start with `#`
    (fixes the current `skiprows=22` default; these files have 14 header lines).
  - Read as utf-16, whitespace/tab-delimited, into a 2D float array of relative
    intensity.
  - Resample from `src_um_per_pix` (4 µm) to `ctx.pixel_size_um` via
    `zoom_factor = src_um_per_pix / pixel_size_um` (physical-micron ratio — this
    is what makes Sony 3.76 µm and HWK 4.6 µm differ correctly).
  - Recenter onto the `npix` grid with a sub-pixel shift to `center`, crop/pad,
    clip negatives, normalize to sum = 1, apply jitter (Gaussian convolution).
  - Module constants `DEFOCUS_1WAVE_PATH`, `DEFOCUS_2WAVE_PATH` point at the
    bundled files for convenience: `DefocusPSF(DEFOCUS_1WAVE_PATH)`.
- **`CustomPSF(array_or_path, src_um_per_pix)`** — generic hook (ndarray or file)
  using the same resample/center/normalize path. For future PSFs (scattering).

Caveat (documented in `DefocusPSF`): the Huygens files are monochromatic
(500 nm). v1 uses them as-is rather than rescaling the blur to the observing
band.

### 2. `ImageSimulator` (`psfsim.py`, replacing `PSFSimulator`)

```
ImageSimulator.from_sensor_and_scene(sensor, scene, npix=300, r_aper_mas=70)
ImageSimulator(simulation, npix=300)         # also accept a prebuilt Simulation
```

- Holds a `Simulation` (`telescope`, `sensor`, `scene`) for all physics.
- Builds a `DetectorPSFContext` from the simulation: `pixel_size_um` and
  `plate_scale_mas` from the sensor/telescope, `wavelength_m` from the sensor
  band, `diameter_m`/`fnum` from the telescope, `jitter_sigma_mas` from the
  telescope (overridable per call).

`simulate(time, psf=None, jitter_sigma_mas=None, center=None, add_noise=True, seed=None)`:

1. `psf = psf or AiryPSF()`. Render `psf_norm = psf.render(ctx)` (sum = 1).
2. **Source:** total source electrons
   `S = count_rates['source'] / ee_at_aper * time` (recover total flux, as in
   `get_peak_pixel`); clean source image `= S * psf_norm`.
3. **Sky background (flat per pixel):** if the scene has a background,
   `bkg_per_pix = count_rates['background'] / num_psf_pixels * time`; add to
   every pixel.
4. **Dark (flat per pixel):** `dark_per_pix = dark_current * time`; add to every
   pixel.
5. `image_clean = source_image + bkg_per_pix + dark_per_pix` (electrons,
   noiseless).
6. If `add_noise`: draw Poisson on `image_clean`, add Gaussian read noise
   (`read_noise` rms), using `np.random.default_rng(seed)` for reproducibility →
   `image_e`. Else `image_e = image_clean`.
7. **Saturation mask:** `image_e >= well_depth` OR `(image_e / gain) >= adc_max`.

Returns a small result object (`SimulatedImage`) with:
- `image_e` — noisy electrons (or clean if `add_noise=False`),
- `image_clean` — noiseless electrons,
- `saturation_mask` — boolean array,
- `to_adu()` — electrons → ADU via `gain` (+ `bias_level`),
- `to_fitsimg()` — wrap `image_e` in a `FitsImg` to reuse its photometry,
  plotting, and radial-profile methods,
- `npix`, `pixel_scale_mas`, and the `psf` used (for provenance).

### 3. Reuse / cleanup

- Lean on `airy.py` primitives (`render_detector_psf`, `gaussian_kernel_2d`,
  `calc_plate_scale_from_flength`, `load_custom_psf`); do not duplicate PSF math.
- Keep `FitsImg`, `FitsImgList`, photometry, `howell_center`, `apply_jitter`.
- In the code being replaced, remove the `sys.exit()`-in-library pattern and the
  hardcoded support-data paths that are no longer needed by the new classes.

## Data flow

```
sensor name + Scene
      │
      ▼
  Simulation (telescope + sensor + scene)   ── get_countrates(), psf_profile, sensor params
      │
      ▼
  ImageSimulator  ── builds DetectorPSFContext
      │
      ├── PSFSource.render(ctx)  ── normalized PSF (sum=1)   [AiryPSF | DefocusPSF | CustomPSF]
      │
      ▼
  distribute source electrons + sky + dark  →  image_clean
      │
      ▼
  Poisson + read noise (seeded)             →  image_e
      │
      ▼
  saturation mask (well / adc_max)          →  SimulatedImage
```

## Error handling

- File loading: explicit utf-16 read, auto-skip `#` header lines, raise a clear
  error if the parsed array is not 2D or is empty.
- Resampling: clip negative pixels (interpolation artifacts) before
  normalization; guard divide-by-zero on the normalization sum.
- `simulate`: reuse `Simulation`'s existing time handling (default to
  `meta['time']`, raise if none, accept scalar `Quantity`/seconds).
- Degenerate aperture (`ee_at_aper == 0`) already guarded in the shared physics.

## Testing

- **PSF-source render:** each returns an `(npix, npix)` array summing to ~1 with
  a finite, centered peak. Defocus is broader (larger FWHM / lower peak) than
  Airy; 2-wave is broader than 1-wave. The same physical defocus PSF occupies
  more Sony pixels than HWK pixels (pixel-size dependence).
- **Encircled energy:** ≥99% of the defocus energy falls within the default 300
  grid for both sensors.
- **Flux conservation:** with `add_noise=False`, `image_clean` sums to
  (source electrons captured within the grid) + (sky+dark)×npix², within the EE
  fraction.
- **Noise:** Poisson mean of `image_e` ≈ `image_clean`; read-noise standard
  deviation in a source-free corner ≈ `read_noise`; `seed` makes runs
  reproducible.
- **Saturation:** a bright source flips the saturation mask; mask agrees with
  `image_e/gain >= adc_max`.
- **Integration:** build and simulate from both `sony:r` and `qcmos:r`;
  confirm pixel scales and grid footprints differ.

## Out of scope (v2 / YAGNI)

- Nonlinearity scaling and flat-fielding (need per-sensor data; only qCMOS
  nonlinearity table exists).
- Polychromatic / band-weighted defocus (defocus files are monochromatic 500 nm).
- Multiple sources / full field scenes (single point source at a chosen center).

## Open items to confirm during spec review

1. ~~`SimulatedImage` as a lightweight result object vs returning a `FitsImg`?~~
   **Resolved:** `simulate()` returns a `SimulatedImage` (carries clean + noisy
   electrons, saturation mask, ADU conversion) with a `.to_fitsimg()` bridge for
   photometry/plotting/radial profiles.
