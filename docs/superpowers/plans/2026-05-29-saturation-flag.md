# Saturation Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a saturation flag to the WCC ETC that warns when a source's brightest pixel exceeds the detector ADC full scale (ADU clip) for a given exposure time.

**Architecture:** Add per-sensor `bit_depth`/`bias_level` config + `Sensor` properties (`adc_max`). Add a reusable detector-grid PSF renderer in `airy.py` that returns the brightest-pixel energy fraction. Surface it through `Simulation.compute_psf_profile()`, then add `get_peak_pixel(time, units)` and `is_saturated(time)` methods on `Simulation`, mirroring the existing `get_snr` / `get_signal_and_variance` pattern.

**Tech Stack:** Python, numpy, scipy (`fftconvolve`), astropy.units, synphot. Tests with pytest.

---

## File Structure

- `src/wcc_etc/data/config/zwo.toml`, `qcmos.toml` — **already updated** (commit `6aed87f`) with `bit_depth` and `bias_level`. No further change.
- `src/wcc_etc/sensor.py` — add `bit_depth`/`bias_level` to `__init__`, `from_config`, `_mutable_parameters`; add `bit_depth`, `bias_level`, `adc_max` properties.
- `src/wcc_etc/airy.py` — add `render_detector_psf(...)` (reusable detector-grid renderer; defocus-extensible).
- `src/wcc_etc/simulation.py` — add `peak_pixel_fraction` to `compute_psf_profile()`; add `get_peak_pixel()` and `is_saturated()`.
- `tests/test_sensor.py`, `tests/test_airy.py` (new), `tests/test_simulation.py` — tests.

**Conventions confirmed from the codebase:** ADU is represented with `u.ct`; `gain` is `electron/ct`, so `electrons / gain` → ADU. `count_rates` from `get_countrates(units="e/s")` are in `electron/s`. `psf_profile` exposes `ee_at_aper` and `num_psf_pixels` (a `Quantity` whose `.value` is the pixel count in the aperture).

---

## Task 1: Sensor bit_depth, bias_level, adc_max

**Files:**
- Modify: `src/wcc_etc/sensor.py`
- Test: `tests/test_sensor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sensor.py`:

```python
def test_sensor_bit_depth_and_adc_max():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=16)
    assert s.bit_depth == 16
    assert s.adc_max == 65535 * u.ct


def test_sensor_bias_level_defaults_to_zero():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=12)
    assert s.bias_level == 0 * u.ct
    assert s.adc_max == 4095 * u.ct


def test_sensor_bias_level_from_value():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=16, bias_level=100)
    assert s.bias_level == 100 * u.ct


def test_bit_depth_and_bias_level_are_updatable():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=16)
    s.update(bit_depth=12, bias_level=50)
    assert s.adc_max == 4095 * u.ct
    assert s.bias_level == 50 * u.ct


def test_from_config_reads_bit_depth_and_bias_level():
    cfg = {
        "throughput": None, "pixel_size": 4, "sensor_area": 50,
        "gain": 1.5, "read_noise": 2.5, "dark_current": 0.01,
        "well_depth": 30000, "bit_depth": 16, "bias_level": 5,
    }
    s = Sensor.from_config(cfg)
    assert s.bit_depth == 16
    assert s.bias_level == 5 * u.ct
    assert s.adc_max == 65535 * u.ct
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sensor.py -k "bit_depth or bias_level or adc_max" -v`
Expected: FAIL (`Sensor.__init__` got an unexpected keyword `bit_depth`, or `AttributeError: 'Sensor' object has no attribute 'adc_max'`).

- [ ] **Step 3: Add constructor params and mutable parameters**

In `src/wcc_etc/sensor.py`, update `_mutable_parameters`:

```python
    _mutable_parameters = ["bandpass", "bandpass_name",
                            "pixel_size", "read_noise", "dark_current",
                            "gain", "area", "temperature",
                            "bit_depth", "bias_level"]
```

Update the `__init__` signature to add the two new keyword parameters (place them after `well_depth=None`):

```python
    def __init__(self, bandpass,
                 pixel_size,
                 read_noise,
                 dark_current,
                 gain,
                 area,
                 temperature=None,
                 qe= 1, # part of the total throughput for now.
                 well_depth=None,
                 bit_depth=None,
                 bias_level=None,
                meta={}):
```

The existing `init_parameters` comprehension already captures any non-`None` local into `meta`, so `bit_depth` and `bias_level` are stored automatically when provided.

- [ ] **Step 4: Read the fields in `from_config`**

In `from_config`, after the gain/well_depth block and before the `return cls(...)`, add:

```python
        # ADC properties
        bit_depth = config.get("bit_depth")
        bias_level = config.get("bias_level", 0)
```

Then add the two new keywords to the `return cls(...)` call:

```python
        return cls(bandpass=bandpass,
                     pixel_size=pixel_size,
                     read_noise=read_noise,
                     dark_current=dark_current,
                     gain=gain,
                     area=sensor_area,
                     temperature=sensor_temp,
                     well_depth=well_depth,
                     bit_depth=bit_depth,
                     bias_level=bias_level,
                     qe=1, # forced qe=1 as included in total throughput
                    meta=config)
```

- [ ] **Step 5: Add properties**

Add these properties to `Sensor` (next to the other `@property` definitions, after `pixel_size`):

```python
    @property
    def bit_depth(self):
        """
        The ADC bit depth (int), or None if not configured.
        """
        return self.meta.get("bit_depth")

    @property
    def bias_level(self):
        """
        The additive bias/offset level in ADU (u.ct). Defaults to 0.
        """
        return self.meta.get("bias_level", 0) * u.ct

    @property
    def adc_max(self):
        """
        The ADC full-scale (clip ceiling) in ADU (u.ct): 2**bit_depth - 1.
        """
        bit_depth = self.bit_depth
        if bit_depth is None:
            raise ValueError("bit_depth is not set; cannot compute adc_max")
        return (2**bit_depth - 1) * u.ct
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_sensor.py -v`
Expected: PASS (all sensor tests, including the pre-existing ones).

- [ ] **Step 7: Commit**

```bash
git add src/wcc_etc/sensor.py tests/test_sensor.py
git commit -m "Add bit_depth, bias_level, adc_max to Sensor"
```

---

## Task 2: Detector-grid PSF renderer

**Files:**
- Modify: `src/wcc_etc/airy.py`
- Test: `tests/test_airy.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_airy.py`:

```python
import numpy as np
import pytest
from wcc_etc.airy import render_detector_psf


def test_render_detector_psf_shape_and_normalization():
    psf, pscale = render_detector_psf(
        wavelength=0.6e-6, fnum=15, D=3, pixel_size=3.76,
        jitter_sigma_mas=0, n_pixels=21, oversample=11)
    assert psf.shape == (21, 21)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)
    assert pscale > 0


def test_render_detector_psf_peak_is_centered():
    psf, _ = render_detector_psf(
        wavelength=0.6e-6, fnum=15, D=3, pixel_size=3.76,
        jitter_sigma_mas=0, n_pixels=21, oversample=11)
    center = (psf.shape[0] // 2, psf.shape[1] // 2)
    assert np.unravel_index(np.argmax(psf), psf.shape) == center


def test_render_detector_psf_peak_fraction_in_unit_interval():
    psf, _ = render_detector_psf(
        wavelength=0.6e-6, fnum=15, D=3, pixel_size=3.76,
        jitter_sigma_mas=0, n_pixels=21, oversample=11)
    assert 0.0 < psf.max() <= 1.0


def test_jitter_reduces_peak_fraction():
    psf0, _ = render_detector_psf(
        wavelength=0.6e-6, fnum=15, D=3, pixel_size=3.76,
        jitter_sigma_mas=0, n_pixels=21, oversample=11)
    psf_j, _ = render_detector_psf(
        wavelength=0.6e-6, fnum=15, D=3, pixel_size=3.76,
        jitter_sigma_mas=50, n_pixels=21, oversample=11)
    assert psf_j.max() < psf0.max()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_airy.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_detector_psf'`.

- [ ] **Step 3: Implement `render_detector_psf`**

Add to `src/wcc_etc/airy.py` (the module already imports `numpy as np`, `fftconvolve`, and defines `get_airy_psf`, `gaussian_kernel_2d`, `calc_plate_scale_from_flength`):

```python
def render_detector_psf(wavelength, fnum, D, pixel_size,
                        jitter_sigma_mas=0, n_pixels=21, oversample=11,
                        verbose=False):
    """
    Render the (optionally jittered) Airy PSF onto the detector pixel grid.

    The PSF is computed on a grid oversampled by ``oversample`` per detector
    pixel, optionally convolved with a Gaussian jitter kernel, then binned down
    to detector pixels. The grid uses an odd number of detector pixels so the
    PSF peak sits on the central pixel (worst case), making the brightest-pixel
    fraction conservative for a saturation check. The result is normalized so
    the rendered window sums to 1.

    Defocus extension: a defocus kernel can be convolved alongside the jitter
    kernel at the oversampled stage without changing this function's interface.

    INPUT:
        wavelength       - wavelength in m
        fnum             - f-number (focal ratio)
        D                - primary diameter in m
        pixel_size       - detector pixel size in microns
        jitter_sigma_mas - jitter sigma in mas (0 = none)
        n_pixels         - detector pixels per side (forced odd)
        oversample       - sub-pixel sampling factor per detector pixel
        verbose          - print diagnostics

    OUTPUT:
        psf_detector     - (n_pixels, n_pixels) array summing to 1; its max
                           is the brightest-pixel energy fraction
        pscale_mas       - detector plate scale in mas/pixel
    """
    # Strip units to plain floats
    wavelength = float(wavelength.value) if hasattr(wavelength, 'unit') else float(wavelength)
    fnum = float(fnum.value) if hasattr(fnum, 'unit') else float(fnum)
    D = float(D.value) if hasattr(D, 'unit') else float(D)
    pixel_size = float(pixel_size.value) if hasattr(pixel_size, 'unit') else float(pixel_size)
    jitter_sigma_mas = float(jitter_sigma_mas.value) if hasattr(jitter_sigma_mas, 'unit') else float(jitter_sigma_mas)
    n_pixels = int(n_pixels)
    oversample = int(oversample)

    # Force odd pixel count so a pixel is centered on the PSF peak
    if n_pixels % 2 == 0:
        n_pixels += 1

    # Detector and oversampled plate scales (mas/pix)
    pscale_mas = calc_plate_scale_from_flength(fnum * D, pixel_size) * 1000.0
    fine_pscale_mas = pscale_mas / oversample

    fine_n = n_pixels * oversample  # odd * odd = odd -> a sample lands on r=0
    half = (fine_n - 1) / 2.0
    coord_mas = (np.arange(fine_n) - half) * fine_pscale_mas

    mas_per_radian = 206265.0 * 1000.0
    coord_rad = coord_mas / mas_per_radian
    xx, yy = np.meshgrid(coord_rad, coord_rad, indexing='xy')
    rr = np.sqrt(xx**2 + yy**2)

    # Raw Airy intensity (do not normalize on the truncated grid)
    psf_fine = get_airy_psf(D, rr, wavelength, normalize=False)

    # Jitter broadening at the oversampled scale
    if jitter_sigma_mas != 0:
        sigma_pix_fine = jitter_sigma_mas / fine_pscale_mas
        ker = gaussian_kernel_2d(sigma_pix_fine)
        psf_fine = fftconvolve(psf_fine, ker, mode='same')

    # Bin oversample x oversample blocks into detector pixels
    psf_detector = psf_fine.reshape(n_pixels, oversample,
                                    n_pixels, oversample).sum(axis=(1, 3))

    # Normalize the rendered window to 1 (conservative: window truncates wings)
    total = psf_detector.sum()
    if total > 0:
        psf_detector = psf_detector / total

    if verbose:
        print(f"render_detector_psf: pscale={pscale_mas:.3f} mas/pix, "
              f"n_pixels={n_pixels}, oversample={oversample}, "
              f"peak_fraction={psf_detector.max():.4f}")

    return psf_detector, pscale_mas
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_airy.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/airy.py tests/test_airy.py
git commit -m "Add detector-grid PSF renderer (render_detector_psf)"
```

---

## Task 3: Expose peak_pixel_fraction in compute_psf_profile

**Files:**
- Modify: `src/wcc_etc/simulation.py`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_simulation.py`:

```python
import wcc_etc


def _bright_sim(mag=20, sensor="sony:r"):
    scene = wcc_etc.get_scene(
        name='G5V', mag=mag, host=None, background="zodi",
        bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene(sensor, scene)


def test_psf_profile_has_peak_pixel_fraction():
    sim = _bright_sim()
    profile = sim.psf_profile
    assert "peak_pixel_fraction" in profile
    assert 0.0 < profile["peak_pixel_fraction"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation.py::test_psf_profile_has_peak_pixel_fraction -v`
Expected: FAIL with `KeyError: 'peak_pixel_fraction'`.

- [ ] **Step 3: Implement**

In `src/wcc_etc/simulation.py`, edit `compute_psf_profile`. Update the import line at the top of the method:

```python
        from .airy import get_airy_and_ee_curve, render_detector_psf
```

After the `psf_area` line (just before the `return {...}`), add:

```python
        # brightest-pixel energy fraction on the detector grid
        psf_detector, _ = render_detector_psf(
            wavelength=wavelength,
            fnum=self.telescope.f_num,
            D=self.telescope.diameter_primary.value,
            pixel_size=self.sensor.pixel_size.value,
            jitter_sigma_mas=self.telescope.jitter_sigma.to("mas").value,
            verbose=False)
        peak_pixel_fraction = float(psf_detector.max())
```

Add the entry to the returned dict:

```python
        return {"wavelength": wavelength,
                "r_psf_mas": r_psf_mas,
                "psf1d": psf1d,
                "ee": ee,
                "ee_at_aper": ee_at_aper,
                "num_psf_pixels": num_psf_pixels,
                "psf_area": psf_area,
                "peak_pixel_fraction": peak_pixel_fraction
                }
```

(Note: `wavelength` is already `self.sensor.wavelength.to("m")` in this method, in meters — pass it straight through.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation.py::test_psf_profile_has_peak_pixel_fraction -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_simulation.py
git commit -m "Expose peak_pixel_fraction in compute_psf_profile"
```

---

## Task 4: Simulation.get_peak_pixel

**Files:**
- Modify: `src/wcc_etc/simulation.py`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simulation.py`:

```python
import astropy.units as u


def test_get_peak_pixel_increases_with_time():
    sim = _bright_sim(mag=15)
    p10 = sim.get_peak_pixel(10, units="adu")
    p100 = sim.get_peak_pixel(100, units="adu")
    assert p100 > p10


def test_get_peak_pixel_adu_units_are_ct():
    sim = _bright_sim(mag=15)
    p = sim.get_peak_pixel(10, units="adu")
    assert p.unit == u.ct


def test_get_peak_pixel_includes_bias():
    sim = _bright_sim(mag=15)
    base = sim.get_peak_pixel(10, units="adu")
    sim.update(sensor__bias_level=100)
    biased = sim.get_peak_pixel(10, units="adu")
    assert biased.value == pytest.approx(base.value + 100, rel=1e-6)


def test_get_peak_pixel_electrons_excludes_bias():
    sim = _bright_sim(mag=15)
    e = sim.get_peak_pixel(10, units="e-")
    assert e.unit == u.electron
    assert e.value > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulation.py -k get_peak_pixel -v`
Expected: FAIL with `AttributeError: 'Simulation' object has no attribute 'get_peak_pixel'`.

- [ ] **Step 3: Implement**

Add this method to `Simulation` in `src/wcc_etc/simulation.py`, immediately after `get_signal_and_variance` (and before `get_snr`):

```python
    def get_peak_pixel(self, time=None, units="adu"):
        """
        Get the brightest-pixel value for a given exposure time.

        The peak pixel combines the source PSF peak, the per-pixel sky
        background, the per-pixel dark current, and (for ADU) the additive bias
        level. Used to test ADC-clip saturation against ``sensor.adc_max``.

        Parameters
        ----------
        time : float or Quantity or array_like, optional
            Exposure time(s) in seconds. Defaults to self.meta['time'].
        units : str, optional
            'adu' (default, includes bias) or 'e-'/'e'/'electron' (excludes bias).

        Returns
        -------
        Quantity
            The peak-pixel value, in ADU (u.ct) or electrons.
        """
        if time is None:
            time = self._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second

        profile = self.psf_profile
        peak_fraction = profile["peak_pixel_fraction"]
        ee_at_aper = profile["ee_at_aper"]
        num_psf_pixels = profile["num_psf_pixels"]
        n_pix = num_psf_pixels.value if isinstance(num_psf_pixels, u.Quantity) else num_psf_pixels

        # count rates within the aperture, in electron/s
        count_rates = self.get_countrates(units="e/s", as_dict=True)

        # source: recover total flux (divide out aperture EE), take peak fraction
        source_peak = (count_rates["source"] / ee_at_aper * peak_fraction * time).to(u.electron)

        # background per pixel (uniform across the aperture); 0 if no background
        if "background" in count_rates:
            bkg_peak = (count_rates["background"] * time / n_pix).to(u.electron)
        else:
            bkg_peak = 0 * u.electron

        # dark current per pixel (dark_current is electron/(s*pix))
        dark_peak = (self.sensor.dark_current * time).value * u.electron

        peak_e = source_peak + bkg_peak + dark_peak  # electrons in the brightest pixel

        if units in ["e", "e-", "electron"]:
            return peak_e

        if units.lower() == "adu":
            return (peak_e / self.sensor.gain).to(u.ct) + self.sensor.bias_level

        raise ValueError(f"unknown units {units=}. 'adu' or electron/'e-' expected.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulation.py -k get_peak_pixel -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_simulation.py
git commit -m "Add Simulation.get_peak_pixel"
```

---

## Task 5: Simulation.is_saturated

**Files:**
- Modify: `src/wcc_etc/simulation.py`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simulation.py`:

```python
def test_is_saturated_flips_with_time():
    sim = _bright_sim(mag=8)  # bright star on a 16-bit sensor (adc_max=65535)
    assert sim.is_saturated(0.001) == False
    assert sim.is_saturated(1000) == True


def test_is_saturated_accepts_array_time():
    sim = _bright_sim(mag=8)
    result = sim.is_saturated(np.array([0.001, 1000.0]))
    assert np.shape(result) == (2,)
    assert bool(result[0]) is False
    assert bool(result[1]) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulation.py -k is_saturated -v`
Expected: FAIL with `AttributeError: 'Simulation' object has no attribute 'is_saturated'`.

- [ ] **Step 3: Implement**

Add this method to `Simulation` in `src/wcc_etc/simulation.py`, immediately after `get_peak_pixel`:

```python
    def is_saturated(self, time=None):
        """
        Whether the brightest pixel reaches the ADC full scale (ADU clip).

        Parameters
        ----------
        time : float or Quantity or array_like, optional
            Exposure time(s) in seconds. Defaults to self.meta['time'].

        Returns
        -------
        bool or ndarray of bool
            True where the peak pixel (in ADU) >= sensor.adc_max.
        """
        peak_adu = self.get_peak_pixel(time, units="adu")
        return peak_adu >= self.sensor.adc_max
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulation.py -k is_saturated -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all tests, including the pre-existing `test_snr.py` and `test_sensor.py`).

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_simulation.py
git commit -m "Add Simulation.is_saturated saturation flag"
```

---

## Self-Review notes

- **Spec coverage:** Sensor config/properties (Task 1), detector-grid renderer (Task 2), `peak_pixel_fraction` in profile (Task 3), `get_peak_pixel` with source+background+dark+bias budget (Task 4), `is_saturated` ADU-clip flag + array support (Task 5). Config files already carry `bit_depth`/`bias_level` (committed). `saturation_time()` intentionally omitted (out of scope).
- **Type consistency:** `adc_max`, `bias_level`, and ADU `get_peak_pixel` all use `u.ct`; `gain` is `electron/ct` so `electrons / gain → ct`. `peak_pixel_fraction` is a plain float. `is_saturated` compares two `u.ct` quantities.
- **Conservative-by-design:** renderer forces an odd grid (peak centered on a pixel) and normalizes within the truncated window, both of which err toward flagging saturation — appropriate for a warning.
