# PSF Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust PSF simulator that renders a point source (Airy / +1-wave / +2-wave defocus / custom) onto a realistic detector grid for the Sony and HWK sensors, with shot+dark+read noise, sky background, and saturation, driven by source magnitude through the existing ETC.

**Architecture:** New PSF-source objects (`AiryPSF`, `DefocusPSF`, `CustomPSF`) render a normalized PSF onto a `DetectorPSFContext`, reusing `airy.py` primitives. A new `ImageSimulator` composes an ETC `Simulation` for all physics (countrates, sensor noise params, `adc_max`/well, plate scale) and produces a `SimulatedImage`. Replaces the legacy `PSFSimulator`/`CustomPSF` in `psfsim.py`; keeps `FitsImg`/photometry.

**Tech Stack:** Python, numpy, scipy (`ndimage.zoom`/`shift`, `signal.fftconvolve`), astropy.units, synphot. Tests with pytest.

---

## File Structure

- `src/wcc_etc/psfsim.py` — remove legacy `PSFSimulator`, `CustomPSF`, `get_micron_to_mas`; add `DetectorPSFContext`, module helpers (`center_crop_or_pad`, `normalize_psf`, `recenter`, `load_huygens_psf`), `PSFSource` + `AiryPSF` + `_ResampledPSF` + `DefocusPSF` + `CustomPSF`, `SimulatedImage`, `ImageSimulator`, and the `DEFOCUS_1WAVE_PATH`/`DEFOCUS_2WAVE_PATH` constants. Keep `FitsImg`, `FitsImgList`, `howell_center`, `apply_jitter`, `apply_nl_scaling`, flat/tiff helpers, `calc_hwhm`/`calc_hwzm`.
- `src/wcc_etc/wcc_etc.py` — neutralize the one deprecated `simulate_psf` method that referenced `PSFSimulator`.
- `src/wcc_etc/__init__.py` — export the new public classes.
- `tests/test_psfsource.py` (new) — PSF-source + helper tests.
- `tests/test_image_simulator.py` (new) — `ImageSimulator` + `SimulatedImage` tests.

**Facts the implementer needs (verified against the codebase):**
- `psfsim.py` already imports `numpy as np`, `astropy.units as u`, `from scipy.ndimage import zoom, shift`, `from scipy.signal import fftconvolve`, `from . import airy`.
- `airy.render_detector_psf(wavelength, fnum, D, pixel_size, jitter_sigma_mas=0, n_pixels=21, oversample=11, verbose=False)` → `(psf_detector, pscale_mas)`; `psf_detector` sums to 1, peak centered, **forced odd** `n_pixels`. `wavelength` in m, `pixel_size` in µm, `D` in m.
- `psfsim.apply_jitter(data, jitter_mas, pixel_scale)` convolves with a Gaussian (`pixel_scale` in **mas/pixel**) and rescales to preserve the input sum.
- `Simulation.from_sensor_and_scene(sensor, scene)`; `Simulation.get_countrates(units="e/s", as_dict=True)` → dict in **electron/s** within the aperture (always `"source"`; `"background"` if the scene has one). `Simulation.psf_profile` has `ee_at_aper` (float) and `num_psf_pixels` (Quantity; `.value` is the pixel count).
- `Sensor`: `wavelength` (Quantity), `pixel_size` (µm/pix Quantity), `gain` (electron/ct), `dark_current` (electron/(s·pix)), `read_noise` (electron/pix), `adc_max` (ct), `bias_level` (ct), `get_plate_scale(telescope)` → arcsec/pix Quantity. Full-well is in `sensor.meta["well_depth"]` (electrons) or absent.
- `Telescope`: `f_num` (float), `diameter_primary` (m Quantity), `jitter_sigma` (mas Quantity).
- Defocus files at `src/wcc_etc/data/psfs/CAD_{1,2}-waves-defocus_500nm_Huygens-PSF-Data_Linear.txt`: utf-16, 14 `#`-prefixed header lines, then 256×256 tab-separated floats, 4 µm source spacing.

**Commit hygiene for every task:** plain commit messages, NO `Co-Authored-By`/Claude trailer. Never stage anything other than the files named in the task.

---

## Task 1: Remove legacy PSFSimulator and CustomPSF

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Modify: `src/wcc_etc/wcc_etc.py`

- [ ] **Step 1: Confirm the only in-package use of the legacy classes**

Run: `grep -rn "PSFSimulator\|CustomPSF\|get_micron_to_mas" src/`
Expected: definitions in `psfsim.py`, plus one use in `wcc_etc.py` inside `simulate_psf`. (Notebooks are out of scope.)

- [ ] **Step 2: Delete the legacy classes from `psfsim.py`**

Remove the entire `class PSFSimulator(object):` block, the entire `class CustomPSF(object):` block, and the `def get_micron_to_mas(...)` function. Keep everything else (`howell_center`, `apply_jitter`, `tiff_to_fits`, `create_master_flat_from_fits`, `apply_nl_scaling`, `FitsImgList`, `FitsImg`, `calc_hwzm`, `calc_hwhm`, and all imports).

- [ ] **Step 3: Neutralize the deprecated `simulate_psf` in `wcc_etc.py`**

In `src/wcc_etc/wcc_etc.py`, replace the body of `simulate_psf` (the block from `self.PSF = psfsim.PSFSimulator(` through `return self.PSF.data_flat`) with:

```python
        raise NotImplementedError(
            "simulate_psf has been removed from the deprecated wcc_etc module. "
            "Use wcc_etc.ImageSimulator.from_sensor_and_scene(...).simulate(...) instead."
        )
```

- [ ] **Step 4: Verify the package still imports and the suite still passes**

Run: `python -c "import wcc_etc"`
Expected: imports without error (deprecation warning from `wcc_etc.py` is fine).
Run: `pytest -q`
Expected: PASS (existing 37 tests; nothing referenced the removed classes).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py src/wcc_etc/wcc_etc.py
git commit -m "Remove legacy PSFSimulator/CustomPSF ahead of new PSF simulator"
```

---

## Task 2: Detector context and shared PSF helpers

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Test: `tests/test_psfsource.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_psfsource.py`:

```python
import os
import numpy as np
import pytest
from wcc_etc.psfsim import (
    DetectorPSFContext, center_crop_or_pad, normalize_psf, recenter,
    load_huygens_psf, DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH,
)


def test_normalize_psf_sums_to_one_and_clips_negatives():
    a = np.array([[-1.0, 1.0], [2.0, 4.0]])
    out = normalize_psf(a)
    assert out.min() >= 0.0
    assert out.sum() == pytest.approx(1.0)


def test_normalize_psf_raises_on_nonpositive():
    with pytest.raises(ValueError):
        normalize_psf(np.zeros((3, 3)))


def test_center_crop_or_pad_crops_to_size_preserving_center():
    a = np.zeros((5, 5)); a[2, 2] = 1.0
    out = center_crop_or_pad(a, 3)
    assert out.shape == (3, 3)
    assert out[1, 1] == 1.0  # center preserved


def test_center_crop_or_pad_pads_to_size():
    a = np.zeros((3, 3)); a[1, 1] = 1.0
    out = center_crop_or_pad(a, 5)
    assert out.shape == (5, 5)
    assert out[2, 2] == 1.0


def test_recenter_moves_peak():
    a = np.zeros((11, 11)); a[5, 5] = 1.0
    out = recenter(a, (7.0, 5.0))  # (cx, cy) -> column 7, row 5
    assert np.unravel_index(np.argmax(out), out.shape) == (5, 7)


def test_load_huygens_psf_shape_and_finite():
    assert os.path.exists(DEFOCUS_1WAVE_PATH)
    data = load_huygens_psf(DEFOCUS_1WAVE_PATH)
    assert data.shape == (256, 256)
    assert np.all(np.isfinite(data))
    assert data.sum() > 0


def test_detector_context_defaults():
    ctx = DetectorPSFContext(npix=64, pixel_size_um=3.76, plate_scale_mas=20.0,
                             wavelength_m=0.6e-6, diameter_m=3.0, fnum=15.0)
    assert ctx.jitter_sigma_mas == 0.0
    assert ctx.center is None
    assert ctx.oversample == 11
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_psfsource.py -v`
Expected: FAIL with `ImportError` (names not defined yet).

- [ ] **Step 3: Implement the context, helpers, and defocus paths**

Add to `src/wcc_etc/psfsim.py` (near the top, after the imports add `import os` if not present — it is already imported — and `from dataclasses import dataclass`):

```python
from dataclasses import dataclass

# Bundled Zemax Huygens defocus PSF data (monochromatic, 500 nm, 4 um spacing)
_PSF_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "psfs")
DEFOCUS_1WAVE_PATH = os.path.join(_PSF_DATA_DIR, "CAD_1-waves-defocus_500nm_Huygens-PSF-Data_Linear.txt")
DEFOCUS_2WAVE_PATH = os.path.join(_PSF_DATA_DIR, "CAD_2-waves-defocus_500nm_Huygens-PSF-Data_Linear.txt")


@dataclass
class DetectorPSFContext:
    """Detector + optics parameters a PSF source needs to render onto the grid."""
    npix: int
    pixel_size_um: float
    plate_scale_mas: float
    wavelength_m: float
    diameter_m: float
    fnum: float
    jitter_sigma_mas: float = 0.0
    center: tuple = None
    oversample: int = 11


def normalize_psf(psf):
    """Clip negatives and normalize a 2D PSF so it sums to 1."""
    psf = np.clip(np.asarray(psf, dtype=float), 0.0, None)
    total = psf.sum()
    if total <= 0:
        raise ValueError("PSF total is non-positive; cannot normalize.")
    return psf / total


def center_crop_or_pad(img, npix, fill=0.0):
    """Center-crop or zero-pad a 2D array to (npix, npix), preserving the center."""
    img = np.asarray(img, dtype=float)
    ny, nx = img.shape
    out = np.full((npix, npix), fill, dtype=float)
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    y0 = int(round(cy - (npix - 1) / 2.0))
    x0 = int(round(cx - (npix - 1) / 2.0))
    y1, x1 = y0 + npix, x0 + npix
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(ny, y1), min(nx, x1)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def recenter(psf, center):
    """Sub-pixel shift a grid-centered PSF so its center lands at (cx, cy)."""
    npix = psf.shape[0]
    grid_center = (npix - 1) / 2.0
    cx, cy = float(center[0]), float(center[1])
    return shift(psf, shift=(cy - grid_center, cx - grid_center),
                 order=3, mode="constant", cval=0.0)


def load_huygens_psf(path, encoding="utf-16"):
    """Load a Zemax Huygens PSF text file into a 2D float array of intensities."""
    with open(path, encoding=encoding) as fh:
        rows = [ln for ln in fh.read().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    data = np.array([[float(x) for x in ln.split()] for ln in rows], dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError(f"Huygens PSF file did not parse to a 2D array: {path}")
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_psfsource.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_psfsource.py
git commit -m "Add DetectorPSFContext and shared PSF helpers"
```

---

## Task 3: PSFSource base and AiryPSF

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Test: `tests/test_psfsource.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_psfsource.py`:

```python
from wcc_etc.psfsim import PSFSource, AiryPSF


def _airy_ctx(npix=64, pixel_size_um=3.76):
    return DetectorPSFContext(
        npix=npix, pixel_size_um=pixel_size_um, plate_scale_mas=20.0,
        wavelength_m=0.6e-6, diameter_m=3.0, fnum=15.0,
        jitter_sigma_mas=0.0, oversample=11)


def test_psfsource_base_is_abstract():
    with pytest.raises(NotImplementedError):
        PSFSource().render(_airy_ctx())


def test_airy_render_shape_normalized_centered():
    psf = AiryPSF().render(_airy_ctx(npix=64))
    assert psf.shape == (64, 64)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    assert abs(cy - 31.5) <= 1 and abs(cx - 31.5) <= 1  # peak near center


def test_airy_recenter_shifts_peak():
    ctx = _airy_ctx(npix=65)
    ctx.center = (40.0, 32.0)  # (cx, cy)
    psf = AiryPSF().render(ctx)
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    assert abs(cx - 40) <= 1 and abs(cy - 32) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_psfsource.py -k "airy or abstract" -v`
Expected: FAIL with `ImportError` for `PSFSource`/`AiryPSF`.

- [ ] **Step 3: Implement**

Add to `src/wcc_etc/psfsim.py` (after the helpers from Task 2):

```python
class PSFSource:
    """Base class: render a normalized (sum=1) PSF onto a DetectorPSFContext."""

    def render(self, ctx):
        raise NotImplementedError("Subclasses must implement render(ctx).")


class AiryPSF(PSFSource):
    """Diffraction-limited Airy PSF rendered on the detector grid (default)."""

    def render(self, ctx):
        psf, _ = airy.render_detector_psf(
            wavelength=ctx.wavelength_m, fnum=ctx.fnum, D=ctx.diameter_m,
            pixel_size=ctx.pixel_size_um, jitter_sigma_mas=ctx.jitter_sigma_mas,
            n_pixels=ctx.npix, oversample=ctx.oversample)
        if psf.shape[0] != ctx.npix:  # render_detector_psf forces odd n_pixels
            psf = center_crop_or_pad(psf, ctx.npix)
        if ctx.center is not None:
            psf = recenter(psf, ctx.center)
        return normalize_psf(psf)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_psfsource.py -k "airy or abstract" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_psfsource.py
git commit -m "Add PSFSource base and AiryPSF"
```

---

## Task 4: Resampled PSFs — DefocusPSF and CustomPSF

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Test: `tests/test_psfsource.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_psfsource.py`:

```python
from wcc_etc.psfsim import DefocusPSF, CustomPSF
from scipy.ndimage import zoom as _zoom


def _grid_ctx(pixel_size_um, npix=300):
    return DetectorPSFContext(
        npix=npix, pixel_size_um=pixel_size_um, plate_scale_mas=20.0,
        wavelength_m=0.6e-6, diameter_m=3.0, fnum=15.0,
        jitter_sigma_mas=0.0, oversample=11)


def test_defocus_render_normalized_shape():
    psf = DefocusPSF(DEFOCUS_1WAVE_PATH).render(_grid_ctx(3.76))
    assert psf.shape == (300, 300)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)


def test_two_wave_is_broader_than_one_wave():
    p1 = DefocusPSF(DEFOCUS_1WAVE_PATH).render(_grid_ctx(3.76))
    p2 = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(3.76))
    assert p2.max() < p1.max()  # more defocus -> more spread -> lower peak


def test_same_psf_spreads_over_more_sony_pixels_than_hwk():
    sony = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(3.76))   # smaller pixels
    hwk = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(4.6))     # larger pixels
    assert sony.max() < hwk.max()


def test_defocus_99pct_contained_in_default_grid_sony():
    data = DefocusPSF(DEFOCUS_2WAVE_PATH)._data
    z = np.clip(_zoom(data, 4.0 / 3.76, order=1), 0.0, None)
    crop = center_crop_or_pad(z, 300)
    assert crop.sum() / z.sum() >= 0.99


def test_custom_psf_from_array():
    arr = np.zeros((51, 51)); arr[25, 25] = 1.0
    psf = CustomPSF(arr, src_um_per_pix=3.76).render(_grid_ctx(3.76, npix=64))
    assert psf.shape == (64, 64)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_psfsource.py -k "defocus or custom or wave or contained" -v`
Expected: FAIL with `ImportError` for `DefocusPSF`/`CustomPSF`.

- [ ] **Step 3: Implement**

Add to `src/wcc_etc/psfsim.py` (after `AiryPSF`):

```python
class _ResampledPSF(PSFSource):
    """A PSF defined as a sampled image at a known source pixel scale (microns)."""

    def __init__(self, data, src_um_per_pix):
        self._data = np.asarray(data, dtype=float)
        if self._data.ndim != 2:
            raise ValueError(f"PSF data must be 2D, got shape {self._data.shape}")
        self.src_um_per_pix = float(src_um_per_pix)

    def render(self, ctx):
        zoom_factor = self.src_um_per_pix / ctx.pixel_size_um
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive (check pixel sizes).")
        zoomed = zoom(self._data, zoom_factor, order=1, mode="constant", cval=0.0)
        psf = normalize_psf(center_crop_or_pad(zoomed, ctx.npix))
        if ctx.jitter_sigma_mas and ctx.jitter_sigma_mas > 0:
            psf = normalize_psf(apply_jitter(psf, ctx.jitter_sigma_mas, ctx.plate_scale_mas))
        if ctx.center is not None:
            psf = normalize_psf(recenter(psf, ctx.center))
        return psf


class DefocusPSF(_ResampledPSF):
    """A defocused PSF loaded from a Zemax Huygens text file."""

    def __init__(self, path, src_um_per_pix=4.0, encoding="utf-16"):
        super().__init__(load_huygens_psf(path, encoding), src_um_per_pix)
        self.path = path


class CustomPSF(_ResampledPSF):
    """A custom PSF from an ndarray or a Huygens-format text file (future hook)."""

    def __init__(self, source, src_um_per_pix, encoding="utf-16"):
        data = source if isinstance(source, np.ndarray) else load_huygens_psf(source, encoding)
        super().__init__(data, src_um_per_pix)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_psfsource.py -v`
Expected: PASS (all PSF-source tests).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_psfsource.py
git commit -m "Add DefocusPSF and CustomPSF resampled PSF sources"
```

---

## Task 5: SimulatedImage result object

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Test: `tests/test_image_simulator.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_simulator.py`:

```python
import numpy as np
import pytest
from wcc_etc.psfsim import SimulatedImage, AiryPSF, FitsImg


def _make_simimg():
    image_e = np.array([[100.0, 200.0], [300.0, 400.0]])
    clean = image_e.copy()
    sat = np.array([[False, False], [False, True]])
    return SimulatedImage(image_e=image_e, image_clean=clean, saturation_mask=sat,
                          gain=2.0, bias_level=100.0, npix=2,
                          pixel_scale_mas=20.0, psf=AiryPSF())


def test_simulated_image_to_adu():
    s = _make_simimg()
    adu = s.to_adu()
    assert np.allclose(adu, s.image_e / 2.0 + 100.0)


def test_simulated_image_to_fitsimg():
    s = _make_simimg()
    f = s.to_fitsimg()
    assert isinstance(f, FitsImg)
    assert np.allclose(f.data, s.image_e)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_image_simulator.py -v`
Expected: FAIL with `ImportError` for `SimulatedImage`.

- [ ] **Step 3: Implement**

Add to `src/wcc_etc/psfsim.py` (after the PSF-source classes):

```python
@dataclass
class SimulatedImage:
    """Result of an ImageSimulator.simulate() call."""
    image_e: np.ndarray          # detector image in electrons (noisy unless add_noise=False)
    image_clean: np.ndarray      # noiseless electrons
    saturation_mask: np.ndarray  # bool: pixels at/over adc_max or full well
    gain: float                  # electron / ct
    bias_level: float            # ct
    npix: int
    pixel_scale_mas: float
    psf: "PSFSource"

    def to_adu(self):
        """Electrons -> ADU via gain, plus the bias level."""
        return self.image_e / self.gain + self.bias_level

    def to_fitsimg(self):
        """Wrap the electron image in a FitsImg for photometry/plotting."""
        return FitsImg(data=self.image_e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_image_simulator.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_image_simulator.py
git commit -m "Add SimulatedImage result object"
```

---

## Task 6: ImageSimulator

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Test: `tests/test_image_simulator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_simulator.py`:

```python
import astropy.units as u
import wcc_etc
from wcc_etc.psfsim import ImageSimulator, DefocusPSF, DEFOCUS_2WAVE_PATH


def _scene(mag=15):
    return wcc_etc.get_scene(
        name='G5V', mag=mag, host=None, background="zodi",
        bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})


def test_image_simulator_clean_flux_conservation():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=128)
    res = imsim.simulate(time=10, add_noise=False)
    assert res.image_clean.shape == (128, 128)

    sim = imsim.sim
    t = 10 * u.second
    cr = sim.get_countrates(units="e/s", as_dict=True)
    ee = sim.psf_profile["ee_at_aper"]
    npp = sim.psf_profile["num_psf_pixels"].value
    source_e = (cr["source"] / ee * t).to(u.electron).value
    bkg = (cr["background"] * t / npp).to(u.electron).value
    dark = (sim.sensor.dark_current * t).to(u.electron / u.pix).value
    expected = source_e + (bkg + dark) * 128 * 128
    # Airy is fully contained in a 128 grid -> source sums to ~source_e
    assert res.image_clean.sum() == pytest.approx(expected, rel=0.02)


def test_image_simulator_source_scales_with_time():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=128)
    peak10 = imsim.simulate(time=10, add_noise=False).image_clean.max()
    peak100 = imsim.simulate(time=100, add_noise=False).image_clean.max()
    assert peak100 > peak10


def test_image_simulator_noise_is_seed_reproducible():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=64)
    a = imsim.simulate(time=10, add_noise=True, seed=1).image_e
    b = imsim.simulate(time=10, add_noise=True, seed=1).image_e
    c = imsim.simulate(time=10, add_noise=True, seed=2).image_e
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_image_simulator_read_noise_in_blank_corner():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(20), npix=128)
    res = imsim.simulate(time=1, add_noise=True, seed=0)
    corner = res.image_e[:16, :16]  # far from the centered source
    rn = imsim.sim.sensor.read_noise.to(u.electron / u.pix).value
    assert np.std(corner) == pytest.approx(rn, rel=0.5)


def test_image_simulator_saturation_flips_with_brightness():
    faint = ImageSimulator.from_sensor_and_scene('sony:r', _scene(20), npix=64)
    bright = ImageSimulator.from_sensor_and_scene('sony:r', _scene(6), npix=64)
    assert not faint.simulate(time=1, add_noise=False).saturation_mask.any()
    assert bright.simulate(time=100, add_noise=False).saturation_mask.any()


def test_image_simulator_sensor_pixel_scales_differ():
    sony = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=64)
    hwk = ImageSimulator.from_sensor_and_scene('qcmos:r', _scene(15), npix=64)
    rs = sony.simulate(time=10, add_noise=False)
    rh = hwk.simulate(time=10, add_noise=False)
    assert rs.pixel_scale_mas != pytest.approx(rh.pixel_scale_mas)


def test_image_simulator_runs_with_defocus_psf():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=300)
    res = imsim.simulate(time=10, psf=DefocusPSF(DEFOCUS_2WAVE_PATH), add_noise=False)
    assert res.image_clean.shape == (300, 300)
    assert np.isfinite(res.image_clean).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_image_simulator.py -k ImageSimulator -v`
Expected: FAIL with `ImportError` for `ImageSimulator`.

- [ ] **Step 3: Implement**

Add to `src/wcc_etc/psfsim.py`. At the top of the file add the import `from .simulation import Simulation` (placed with the other `from .` imports). Then add the class (after `SimulatedImage`):

```python
class ImageSimulator:
    """Render a point source onto a detector grid with noise, driven by an ETC Simulation."""

    def __init__(self, simulation, npix=300, oversample=11):
        self.sim = simulation
        self.npix = int(npix)
        self.oversample = int(oversample)

    @classmethod
    def from_sensor_and_scene(cls, sensor, scene, npix=300, oversample=11):
        sim = Simulation.from_sensor_and_scene(sensor, scene)
        return cls(sim, npix=npix, oversample=oversample)

    def _context(self, jitter_sigma_mas=None, center=None):
        sim = self.sim
        plate_scale_mas = sim.sensor.get_plate_scale(sim.telescope).to("arcsec/pix").value * 1000.0
        if jitter_sigma_mas is None:
            jitter_sigma_mas = sim.telescope.jitter_sigma.to("mas").value
        return DetectorPSFContext(
            npix=self.npix,
            pixel_size_um=sim.sensor.pixel_size.value,
            plate_scale_mas=plate_scale_mas,
            wavelength_m=sim.sensor.wavelength.to("m").value,
            diameter_m=sim.telescope.diameter_primary.to("m").value,
            fnum=sim.telescope.f_num,
            jitter_sigma_mas=jitter_sigma_mas,
            center=center,
            oversample=self.oversample)

    def simulate(self, time=None, psf=None, jitter_sigma_mas=None, center=None,
                 add_noise=True, seed=None):
        """Simulate a detector image for the given exposure time and PSF."""
        sim = self.sim
        if time is None:
            time = sim._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second

        if psf is None:
            psf = AiryPSF()
        ctx = self._context(jitter_sigma_mas=jitter_sigma_mas, center=center)
        psf_norm = psf.render(ctx)  # sum = 1

        profile = sim.psf_profile
        ee_at_aper = profile["ee_at_aper"]
        num_psf_pixels = profile["num_psf_pixels"]
        n_pix = num_psf_pixels.value if isinstance(num_psf_pixels, u.Quantity) else num_psf_pixels

        count_rates = sim.get_countrates(units="e/s", as_dict=True)
        # total source electrons (recover total flux from the aperture EE), spread by the PSF
        source_e_total = (count_rates["source"] / ee_at_aper * time).to(u.electron).value
        source_image = source_e_total * psf_norm

        # sky background per pixel (uniform across the grid); 0 if no background element
        if "background" in count_rates:
            bkg_per_pix = (count_rates["background"] * time / n_pix).to(u.electron).value
        else:
            bkg_per_pix = 0.0
        # dark current per pixel (uniform)
        dark_per_pix = (sim.sensor.dark_current * time).to(u.electron / u.pix).value

        image_clean = source_image + bkg_per_pix + dark_per_pix

        if add_noise:
            rng = np.random.default_rng(seed)
            image_e = rng.poisson(np.clip(image_clean, 0.0, None)).astype(float)
            read_noise = sim.sensor.read_noise.to(u.electron / u.pix).value
            image_e = image_e + rng.normal(0.0, read_noise, size=image_e.shape)
        else:
            image_e = image_clean.copy()

        gain = sim.sensor.gain.to(u.electron / u.ct).value
        bias_level = sim.sensor.bias_level.to(u.ct).value
        adc_max = sim.sensor.adc_max.to(u.ct).value
        well_depth = sim.sensor.meta.get("well_depth")

        saturation_mask = (image_e / gain) >= adc_max
        if well_depth is not None:
            saturation_mask = saturation_mask | (image_e >= well_depth)

        return SimulatedImage(
            image_e=image_e, image_clean=image_clean, saturation_mask=saturation_mask,
            gain=gain, bias_level=bias_level, npix=self.npix,
            pixel_scale_mas=ctx.plate_scale_mas, psf=psf)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_image_simulator.py -v`
Expected: PASS (all `ImageSimulator` + `SimulatedImage` tests).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_image_simulator.py
git commit -m "Add ImageSimulator that renders detector images with noise"
```

---

## Task 7: Public exports and full-suite verification

**Files:**
- Modify: `src/wcc_etc/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_image_simulator.py`:

```python
def test_public_exports_available():
    import wcc_etc
    for name in ["ImageSimulator", "AiryPSF", "DefocusPSF", "CustomPSF", "SimulatedImage"]:
        assert hasattr(wcc_etc, name), f"{name} not exported from wcc_etc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_image_simulator.py::test_public_exports_available -v`
Expected: FAIL (`ImageSimulator` not on the `wcc_etc` namespace).

- [ ] **Step 3: Implement**

In `src/wcc_etc/__init__.py`, add after the existing `from .simulation import Simulation` line:

```python
from .psfsim import (
    ImageSimulator, SimulatedImage,
    AiryPSF, DefocusPSF, CustomPSF,
    DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH,
)
```

- [ ] **Step 4: Run the focused test and the full suite**

Run: `pytest tests/test_image_simulator.py::test_public_exports_available -v`
Expected: PASS.
Run: `pytest -q`
Expected: PASS (all prior tests plus the new `test_psfsource.py` and `test_image_simulator.py`).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/__init__.py tests/test_image_simulator.py
git commit -m "Export ImageSimulator and PSF sources from wcc_etc"
```

---

## Self-Review notes

- **Spec coverage:** Airy/defocus/custom PSFs (Tasks 3–4); detector grid for both sensors via `_context` pixel size + plate scale, validated Sony≠HWK (Task 6); npix=300 default + EE containment test (Task 4); noise = shot+dark+read with seed, sky background, saturation vs adc_max/well (Task 6); flux from scene magnitude via `get_countrates` (Task 6); `SimulatedImage` + `to_fitsimg` (Task 5); replace legacy in place (Task 1); reuse `airy.py`/`apply_jitter` (Tasks 3–4). Nonlinearity/flat correctly deferred (not in any task).
- **Placeholder scan:** none — every code step has full code.
- **Type consistency:** `DetectorPSFContext` fields match every `render` use; `render(ctx)` returns sum-1 arrays consumed by `ImageSimulator.simulate`; `SimulatedImage` fields match construction in `simulate` and the Task-5 tests; `gain`/`bias_level`/`adc_max` use `u.ct`, electrons via `u.electron`/`u.pix`. `DefocusPSF`/`CustomPSF` share `_ResampledPSF.render`.
- **Note for executor:** `apply_jitter` rescales to preserve total, so re-normalizing after it is belt-and-suspenders but harmless and keeps sum exactly 1 after numerical drift.
