# ImageSimulator Convenience Plotting Functions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add convenience plotting helpers for `SimulatedImage` (single image, 3-panel row, radial profile, encircled energy) in both matplotlib and bokeh, where bokeh variants can render in a notebook or return embeddable HTML for the Flask web portal.

**Architecture:** A new `src/wcc_etc/plotting.py` module holds standalone functions. Each plot type has separate `_mpl` and `_bokeh` functions. Every function accepts either a `SimulatedImage` (positional `source`) or explicit raw arrays (keywords), resolved through one shared helper. Thin convenience methods on `SimulatedImage` dispatch to the right backend via a `backend=` kwarg. Bokeh functions take `return_='obj'|'html'|'components'`.

**Tech Stack:** numpy, matplotlib (`Agg` in tests), astropy.visualization (stretch/normalize), bokeh 3.6.2, existing `wcc_etc.radial_data.radial_data` and `wcc_etc.airy.psf_to_encircled_energy`.

---

## File Structure

- **Create:** `src/wcc_etc/plotting.py` — all eight plotting functions + 4 internal helpers (`_resolve_inputs`, `_make_norm`, `_image_extent`, `_saturation_overlay`, `_finish_bokeh`).
- **Modify:** `src/wcc_etc/psfsim.py` — add 4 convenience methods to the `SimulatedImage` dataclass (`plot_image`, `plot_image_row`, `plot_radial`, `plot_encircled_energy`).
- **Modify:** `src/wcc_etc/__init__.py` — export the public plotting functions.
- **Create:** `tests/test_plotting.py` — all tests.

### Shared signatures (defined in Task 1, used everywhere)

```python
# All plot functions resolve their data through this:
_resolve_inputs(source=None, *, image_e=None, image_clean=None,
                saturation_mask=None, pixel_scale_mas=None)
    -> (image_e, image_clean, saturation_mask, pixel_scale_mas)

_make_norm(data, stretch)            # stretch in {'log','hist','linear'} -> ImageNormalize or None
_image_extent(ny, nx, pixel_scale_mas, units)   # units in {'pix','mas'} -> extent list or None
_saturation_overlay(saturation_mask)  # -> np.ma.MaskedArray of ones, masked where NOT saturated
_finish_bokeh(obj, return_)          # return_ in {'obj','html','components'}
```

---

## Task 1: Module scaffold and shared helpers

**Files:**
- Create: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plotting.py
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from wcc_etc.psfsim import SimulatedImage, AiryPSF
from wcc_etc import plotting


def _make_simimg(npix=2):
    image_e = np.array([[100.0, 200.0], [300.0, 9e9]])
    clean = np.array([[100.0, 200.0], [300.0, 400.0]])
    sat = np.array([[False, False], [False, True]])
    return SimulatedImage(image_e=image_e, image_clean=clean, saturation_mask=sat,
                          gain=2.0, bias_level=100.0, npix=npix,
                          pixel_scale_mas=20.0, psf=AiryPSF())


def test_resolve_inputs_from_simimg():
    s = _make_simimg()
    ie, ic, sat, ps = plotting._resolve_inputs(s)
    assert np.array_equal(ie, s.image_e)
    assert np.array_equal(ic, s.image_clean)
    assert np.array_equal(sat, s.saturation_mask)
    assert ps == 20.0


def test_resolve_inputs_from_arrays():
    ie, ic, sat, ps = plotting._resolve_inputs(
        image_e=np.zeros((2, 2)), image_clean=np.ones((2, 2)),
        saturation_mask=np.zeros((2, 2), bool), pixel_scale_mas=5.0)
    assert ps == 5.0
    assert np.array_equal(ic, np.ones((2, 2)))


def test_saturation_overlay_masks_only_unsaturated():
    mask = np.array([[False, True], [True, False]])
    ov = plotting._saturation_overlay(mask)
    assert np.array_equal(ov.mask, ~mask)


def test_image_extent_mas_vs_pix():
    assert plotting._image_extent(4, 4, 10.0, "pix") is None
    ext = plotting._image_extent(4, 4, 10.0, "mas")
    assert ext == [-20.0, 20.0, -20.0, 20.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wcc_etc.plotting'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/wcc_etc/plotting.py
"""Convenience plotting helpers for SimulatedImage (matplotlib + bokeh).

Every function accepts either a SimulatedImage (positional `source`) or explicit
raw arrays as keywords, so the Flask web portal can pass arrays directly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from astropy.visualization import LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from .radial_data import radial_data
from .airy import psf_to_encircled_energy
from .psfsim import calc_hwhm


def _resolve_inputs(source=None, *, image_e=None, image_clean=None,
                    saturation_mask=None, pixel_scale_mas=None):
    """Resolve (image_e, image_clean, saturation_mask, pixel_scale_mas) from a
    SimulatedImage or explicit arrays."""
    if source is not None and hasattr(source, "image_e"):
        return (source.image_e, source.image_clean,
                source.saturation_mask, source.pixel_scale_mas)
    return image_e, image_clean, saturation_mask, pixel_scale_mas


def _make_norm(data, stretch):
    """Build an astropy ImageNormalize for the given stretch ('log'|'hist'|'linear')."""
    if stretch == "hist":
        return ImageNormalize(stretch=HistEqStretch(np.asarray(data)))
    if stretch == "log":
        return ImageNormalize(np.asarray(data), stretch=LogStretch())
    return None  # linear: no normalization object


def _image_extent(ny, nx, pixel_scale_mas, units):
    """imshow extent centered on the grid, in mas, or None for pixel units."""
    if units == "mas" and pixel_scale_mas:
        hx = nx / 2.0 * pixel_scale_mas
        hy = ny / 2.0 * pixel_scale_mas
        return [-hx, hx, -hy, hy]
    return None


def _saturation_overlay(saturation_mask):
    """A masked array of ones, masked everywhere the pixel is NOT saturated, so an
    imshow of it colors only the saturated pixels."""
    mask = np.asarray(saturation_mask, dtype=bool)
    return np.ma.masked_where(~mask, np.ones(mask.shape, dtype=float))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add plotting module scaffold and shared helpers"
```

---

## Task 2: `plot_image_mpl` (single panel + saturation overlay)

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
import matplotlib.figure
import matplotlib.axes


def test_plot_image_mpl_returns_fig_ax():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, stretch="linear")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_image_mpl_noise_selects_image_e():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, noise=True, stretch="linear")
    # the main image is the first AxesImage; its data is image_e
    main = ax.get_images()[0]
    assert np.array_equal(main.get_array().data, s.image_e)


def test_plot_image_mpl_no_noise_selects_image_clean():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, noise=False, stretch="linear")
    main = ax.get_images()[0]
    assert np.array_equal(main.get_array().data, s.image_clean)


def test_plot_image_mpl_saturation_overlay_adds_second_image():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, show_saturation=True, stretch="linear")
    # base image + overlay = 2 AxesImages
    assert len(ax.get_images()) == 2
    overlay = ax.get_images()[1].get_array()
    assert np.array_equal(np.ma.getmaskarray(overlay), ~s.saturation_mask)


def test_plot_image_mpl_equal_aspect():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, stretch="linear")
    assert ax.get_aspect() in (1.0, "equal")


def test_plot_image_mpl_accepts_raw_arrays():
    fig, ax = plotting.plot_image_mpl(
        image_e=np.ones((4, 4)), image_clean=np.zeros((4, 4)),
        saturation_mask=np.zeros((4, 4), bool), pixel_scale_mas=10.0,
        stretch="linear")
    assert isinstance(fig, matplotlib.figure.Figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_image_mpl`
Expected: FAIL — `AttributeError: module 'wcc_etc.plotting' has no attribute 'plot_image_mpl'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/wcc_etc/plotting.py
def plot_image_mpl(source=None, *, noise=True, show_saturation=False,
                   image_e=None, image_clean=None, saturation_mask=None,
                   pixel_scale_mas=None, stretch="log", cmap="viridis",
                   vmin=None, vmax=None, colorbar=True, title="",
                   units="pix", sat_color="red", sat_alpha=0.6, ax=None):
    """Plot a single simulated detector image with equal x/y scale.

    noise=True shows the noisy image_e; noise=False shows the noiseless
    image_clean. show_saturation overlays the saturation mask. units='mas'
    labels the axes in milliarcsec using pixel_scale_mas. Returns (fig, ax)."""
    ie, ic, sat, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    data = ie if noise else ic
    ny, nx = data.shape
    extent = _image_extent(ny, nx, ps, units)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    norm = _make_norm(data, stretch)
    im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm,
                   vmin=vmin, vmax=vmax, extent=extent, aspect="equal")

    if show_saturation and sat is not None:
        ax.imshow(_saturation_overlay(sat), origin="lower",
                  cmap=ListedColormap([sat_color]), alpha=sat_alpha,
                  extent=extent, aspect="equal")

    ax.set_xlabel("X [mas]" if extent else "X [pix]")
    ax.set_ylabel("Y [mas]" if extent else "Y [pix]")
    ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)
    return fig, ax
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_image_mpl`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add plot_image_mpl with saturation overlay"
```

---

## Task 3: `plot_image_row_mpl` (3-panel row)

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

Panels, left to right: (1) PSF + noise (`image_e`), (2) PSF no noise (`image_clean`), (3) saturation mask alone. Panels 1 and 2 share `vmin/vmax`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
def test_plot_image_row_mpl_three_axes():
    s = _make_simimg()
    fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
    assert len(axes) == 3
    for ax in axes:
        assert ax.get_aspect() in (1.0, "equal")


def test_plot_image_row_mpl_panel_data():
    s = _make_simimg()
    fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
    assert np.array_equal(axes[0].get_images()[0].get_array().data, s.image_e)
    assert np.array_equal(axes[1].get_images()[0].get_array().data, s.image_clean)
    # panel 3 shows the boolean mask as a float/int image
    assert np.array_equal(
        np.asarray(axes[2].get_images()[0].get_array()).astype(bool),
        s.saturation_mask)


def test_plot_image_row_mpl_shared_color_scale():
    s = _make_simimg()
    fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
    im0, im1 = axes[0].get_images()[0], axes[1].get_images()[0]
    assert im0.get_clim() == im1.get_clim()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_image_row_mpl`
Expected: FAIL — `AttributeError: ... has no attribute 'plot_image_row_mpl'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/wcc_etc/plotting.py
def plot_image_row_mpl(source=None, *, image_e=None, image_clean=None,
                       saturation_mask=None, pixel_scale_mas=None,
                       stretch="log", cmap="viridis", units="pix",
                       sat_cmap="gray", figsize=(15, 5), axes=None):
    """Three panels: PSF+noise, PSF (no noise), and the saturation mask.

    The two image panels share a common color scale (computed from the noisy
    image). Returns (fig, axes) where axes has length 3."""
    ie, ic, sat, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    ny, nx = ie.shape
    extent = _image_extent(ny, nx, ps, units)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].figure

    norm = _make_norm(ie, stretch)
    vmin = float(np.min(ie)) if norm is None else None
    vmax = float(np.max(ie)) if norm is None else None

    titles = ["PSF + noise", "PSF (no noise)", "Saturation mask"]
    for ax, data, title in zip(axes[:2], [ie, ic], titles[:2]):
        ax.imshow(data, origin="lower", cmap=cmap, norm=norm,
                  vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("X [mas]" if extent else "X [pix]")
        ax.set_ylabel("Y [mas]" if extent else "Y [pix]")

    axes[2].imshow(np.asarray(sat, dtype=float), origin="lower",
                   cmap=sat_cmap, extent=extent, aspect="equal")
    axes[2].set_title(titles[2])
    axes[2].set_xlabel("X [mas]" if extent else "X [pix]")
    axes[2].set_ylabel("Y [mas]" if extent else "Y [pix]")
    return fig, axes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_image_row_mpl`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add plot_image_row_mpl 3-panel view"
```

---

## Task 4: `plot_radial_mpl` (radial profile)

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
def _gaussian_simimg(npix=41, sigma=4.0, scale=20.0):
    c = (npix - 1) / 2
    yy, xx = np.mgrid[0:npix, 0:npix]
    g = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma ** 2)))
    return SimulatedImage(image_e=g.copy(), image_clean=g.copy(),
                          saturation_mask=np.zeros_like(g, bool),
                          gain=1.0, bias_level=0.0, npix=npix,
                          pixel_scale_mas=scale, psf=AiryPSF())


def test_plot_radial_mpl_returns_fig_ax_and_decreasing():
    s = _gaussian_simimg()
    fig, ax, (r, prof) = plotting.plot_radial_mpl(s, units="pix")
    assert isinstance(fig, matplotlib.figure.Figure)
    # Gaussian centered profile: peak at small r, decreasing outward overall
    assert prof[0] > prof[-1]


def test_plot_radial_mpl_units_scale_x_axis():
    s = _gaussian_simimg(scale=20.0)
    _, _, (r_pix, _) = plotting.plot_radial_mpl(s, units="pix")
    _, _, (r_mas, _) = plotting.plot_radial_mpl(s, units="mas")
    assert np.allclose(r_mas, r_pix * 20.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_radial_mpl`
Expected: FAIL — `AttributeError: ... has no attribute 'plot_radial_mpl'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/wcc_etc/plotting.py
def plot_radial_mpl(source=None, *, noise=False, image_e=None, image_clean=None,
                    saturation_mask=None, pixel_scale_mas=None, units="mas",
                    annulus_width=1, show_hwhm=True, title="", ax=None):
    """Azimuthally-averaged radial profile of the simulated image.

    Uses image_clean by default (noise=True uses image_e). units='mas' scales the
    radius by pixel_scale_mas. Returns (fig, ax, (radius, profile))."""
    ie, ic, _, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    data = ie if noise else ic
    rd = radial_data(np.asarray(data), annulus_width=annulus_width)
    r = np.asarray(rd.r, dtype=float)
    prof = np.asarray(rd.mean, dtype=float)
    if units == "mas" and ps:
        r = r * ps

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(r, prof)
    if show_hwhm:
        hwhm = calc_hwhm(r, prof)
        if len(hwhm):
            ymin, ymax = ax.get_ylim()
            ax.vlines(hwhm[0], ymin, ymax, color="orange", linestyle="--", lw=1,
                      label="HWHM={:.2f}".format(hwhm[0]))
            ax.legend(loc="upper right")
    ax.set_xlabel("Radius [mas]" if (units == "mas" and ps) else "Radius [pix]")
    ax.set_ylabel("Azimuthally-averaged signal")
    ax.set_title(title)
    ax.grid(lw=0.5, alpha=0.3)
    return fig, ax, (r, prof)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_radial_mpl`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add plot_radial_mpl radial profile"
```

---

## Task 5: `plot_encircled_energy_mpl`

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
def test_plot_ee_mpl_monotonic_to_one():
    s = _gaussian_simimg()
    fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(s, units="pix")
    assert np.all(np.diff(ee) >= -1e-9)        # non-decreasing
    assert ee[-1] == pytest.approx(1.0, abs=1e-6)


def test_plot_ee_mpl_target_marker_returns_radius():
    s = _gaussian_simimg()
    fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(
        s, units="mas", ee_target=0.8)
    idx = np.searchsorted(ee, 0.8)
    assert 0 < idx < len(r)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_ee_mpl`
Expected: FAIL — `AttributeError: ... has no attribute 'plot_encircled_energy_mpl'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/wcc_etc/plotting.py
def plot_encircled_energy_mpl(source=None, *, noise=False, image_e=None,
                              image_clean=None, saturation_mask=None,
                              pixel_scale_mas=None, units="mas",
                              ee_target=None, title="", ax=None):
    """Encircled-energy curve (normalized to 1) of the simulated image.

    Uses image_clean by default. units='mas' uses the mas radius from
    psf_to_encircled_energy; units='pix' divides by pixel_scale_mas. ee_target
    (e.g. 0.8) draws the enclosing-radius marker. Returns (fig, ax, (radius, ee))."""
    ie, ic, _, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    data = ie if noise else ic
    scale = ps if ps else 1.0
    r_mas, _psf1d, ee = psf_to_encircled_energy(np.asarray(data), scale, scale)
    r = r_mas if (units == "mas" and ps) else r_mas / scale
    if ee[-1] > 0:
        ee = ee / ee[-1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(r, ee)
    if ee_target is not None:
        idx = int(np.searchsorted(ee, ee_target))
        if 0 < idx < len(r):
            ax.axvline(r[idx], color="red", linestyle="--", lw=1,
                       label="EE={:.2f} @ {:.1f}".format(ee_target, r[idx]))
            ax.axhline(ee_target, color="gray", linestyle=":", lw=1)
            ax.legend(loc="lower right")
    ax.set_xlabel("Radius [mas]" if (units == "mas" and ps) else "Radius [pix]")
    ax.set_ylabel("Encircled energy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(lw=0.5, alpha=0.3)
    return fig, ax, (r, ee)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_ee_mpl`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add plot_encircled_energy_mpl"
```

---

## Task 6: Bokeh return helper + `plot_image_bokeh`

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

`_finish_bokeh` converts a bokeh object to the requested return form. `plot_image_bokeh` uses `match_aspect=True` to enforce equal x/y scale.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
from bokeh.models import Plot


def test_finish_bokeh_obj_html_components():
    s = _gaussian_simimg()
    obj = plotting.plot_image_bokeh(s, return_="obj")
    assert isinstance(obj, Plot)

    html = plotting.plot_image_bokeh(s, return_="html")
    assert isinstance(html, str) and "<script" in html

    comp = plotting.plot_image_bokeh(s, return_="components")
    assert isinstance(comp, tuple) and len(comp) == 2
    assert all(isinstance(x, str) for x in comp)


def test_plot_image_bokeh_bad_return_raises():
    s = _gaussian_simimg()
    with pytest.raises(ValueError):
        plotting.plot_image_bokeh(s, return_="nope")


def test_plot_image_bokeh_accepts_raw_arrays():
    obj = plotting.plot_image_bokeh(
        image_e=np.ones((8, 8)), image_clean=np.zeros((8, 8)),
        saturation_mask=np.zeros((8, 8), bool), pixel_scale_mas=10.0,
        return_="obj")
    assert isinstance(obj, Plot)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k bokeh`
Expected: FAIL — `AttributeError: ... has no attribute 'plot_image_bokeh'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/wcc_etc/plotting.py
def _finish_bokeh(obj, return_):
    """Return a bokeh object as the figure ('obj'), a standalone HTML string
    ('html'), or an (script, div) components tuple ('components')."""
    if return_ == "obj":
        return obj
    if return_ == "html":
        from bokeh.embed import file_html
        from bokeh.resources import CDN
        return file_html(obj, CDN)
    if return_ == "components":
        from bokeh.embed import components
        return components(obj)
    raise ValueError("return_ must be 'obj', 'html', or 'components'")


def plot_image_bokeh(source=None, *, noise=True, image_e=None, image_clean=None,
                     saturation_mask=None, pixel_scale_mas=None, units="pix",
                     palette="Viridis256", title="", width=400, height=400,
                     return_="obj"):
    """Bokeh single-image plot with equal x/y scale (match_aspect=True).

    return_ selects the output form (see _finish_bokeh)."""
    from bokeh.plotting import figure
    ie, ic, _sat, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    data = np.asarray(ie if noise else ic, dtype=float)
    ny, nx = data.shape
    if units == "mas" and ps:
        x0, y0, dw, dh = -nx / 2.0 * ps, -ny / 2.0 * ps, nx * ps, ny * ps
        axis_label = "mas"
    else:
        x0, y0, dw, dh = 0, 0, nx, ny
        axis_label = "pix"

    p = figure(width=width, height=height, match_aspect=True, title=title,
               x_axis_label="X [{}]".format(axis_label),
               y_axis_label="Y [{}]".format(axis_label))
    p.image(image=[data], x=x0, y=y0, dw=dw, dh=dh, palette=palette)
    return _finish_bokeh(p, return_)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k bokeh`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add bokeh return helper and plot_image_bokeh"
```

---

## Task 7: `plot_image_row_bokeh` (3-panel row)

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
from bokeh.models import LayoutDOM


def test_plot_image_row_bokeh_obj_is_layout():
    s = _gaussian_simimg()
    obj = plotting.plot_image_row_bokeh(s, return_="obj")
    assert isinstance(obj, LayoutDOM)


def test_plot_image_row_bokeh_html_and_components():
    s = _gaussian_simimg()
    html = plotting.plot_image_row_bokeh(s, return_="html")
    assert isinstance(html, str) and "<script" in html
    script, div = plotting.plot_image_row_bokeh(s, return_="components")
    assert isinstance(script, str) and isinstance(div, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_image_row_bokeh`
Expected: FAIL — `AttributeError: ... has no attribute 'plot_image_row_bokeh'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/wcc_etc/plotting.py
def plot_image_row_bokeh(source=None, *, image_e=None, image_clean=None,
                         saturation_mask=None, pixel_scale_mas=None, units="pix",
                         palette="Viridis256", sat_palette="Greys256",
                         width=300, height=300, return_="obj"):
    """Bokeh 3-panel row: PSF+noise, PSF (no noise), saturation mask.

    return_ selects the output form (see _finish_bokeh)."""
    from bokeh.plotting import figure
    from bokeh.layouts import row
    ie, ic, sat, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    ie = np.asarray(ie, dtype=float)
    ic = np.asarray(ic, dtype=float)
    sat = np.asarray(sat, dtype=float)
    ny, nx = ie.shape
    if units == "mas" and ps:
        x0, y0, dw, dh, lbl = -nx / 2.0 * ps, -ny / 2.0 * ps, nx * ps, ny * ps, "mas"
    else:
        x0, y0, dw, dh, lbl = 0, 0, nx, ny, "pix"

    panels = []
    for data, title, pal in [(ie, "PSF + noise", palette),
                             (ic, "PSF (no noise)", palette),
                             (sat, "Saturation mask", sat_palette)]:
        p = figure(width=width, height=height, match_aspect=True, title=title,
                   x_axis_label="X [{}]".format(lbl),
                   y_axis_label="Y [{}]".format(lbl))
        p.image(image=[data], x=x0, y=y0, dw=dw, dh=dh, palette=pal)
        panels.append(p)
    return _finish_bokeh(row(*panels), return_)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k plot_image_row_bokeh`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add plot_image_row_bokeh 3-panel view"
```

---

## Task 8: `plot_radial_bokeh` and `plot_encircled_energy_bokeh`

**Files:**
- Modify: `src/wcc_etc/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
def test_plot_radial_bokeh_obj_and_components():
    s = _gaussian_simimg()
    assert isinstance(plotting.plot_radial_bokeh(s, return_="obj"), Plot)
    script, div = plotting.plot_radial_bokeh(s, return_="components")
    assert isinstance(script, str) and isinstance(div, str)


def test_plot_ee_bokeh_obj_and_html():
    s = _gaussian_simimg()
    assert isinstance(plotting.plot_encircled_energy_bokeh(s, return_="obj"), Plot)
    html = plotting.plot_encircled_energy_bokeh(s, return_="html")
    assert isinstance(html, str) and "<script" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k "radial_bokeh or ee_bokeh"`
Expected: FAIL — `AttributeError: ... has no attribute 'plot_radial_bokeh'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/wcc_etc/plotting.py
def plot_radial_bokeh(source=None, *, noise=False, image_e=None, image_clean=None,
                      saturation_mask=None, pixel_scale_mas=None, units="mas",
                      annulus_width=1, title="", width=500, height=350,
                      return_="obj"):
    """Bokeh radial profile. return_ selects the output form (see _finish_bokeh)."""
    from bokeh.plotting import figure
    ie, ic, _sat, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    data = ie if noise else ic
    rd = radial_data(np.asarray(data), annulus_width=annulus_width)
    r = np.asarray(rd.r, dtype=float)
    prof = np.asarray(rd.mean, dtype=float)
    use_mas = units == "mas" and ps
    if use_mas:
        r = r * ps
    p = figure(width=width, height=height, title=title,
               x_axis_label="Radius [mas]" if use_mas else "Radius [pix]",
               y_axis_label="Azimuthally-averaged signal")
    p.line(r, prof, line_width=2)
    return _finish_bokeh(p, return_)


def plot_encircled_energy_bokeh(source=None, *, noise=False, image_e=None,
                                image_clean=None, saturation_mask=None,
                                pixel_scale_mas=None, units="mas", ee_target=None,
                                title="", width=500, height=350, return_="obj"):
    """Bokeh encircled-energy curve. return_ selects the output form."""
    from bokeh.plotting import figure
    ie, ic, _sat, ps = _resolve_inputs(
        source, image_e=image_e, image_clean=image_clean,
        saturation_mask=saturation_mask, pixel_scale_mas=pixel_scale_mas)
    data = ie if noise else ic
    scale = ps if ps else 1.0
    r_mas, _psf1d, ee = psf_to_encircled_energy(np.asarray(data), scale, scale)
    use_mas = units == "mas" and ps
    r = r_mas if use_mas else r_mas / scale
    if ee[-1] > 0:
        ee = ee / ee[-1]
    p = figure(width=width, height=height, title=title, y_range=(0, 1.02),
               x_axis_label="Radius [mas]" if use_mas else "Radius [pix]",
               y_axis_label="Encircled energy")
    p.line(r, ee, line_width=2)
    if ee_target is not None:
        from bokeh.models import Span
        idx = int(np.searchsorted(ee, ee_target))
        if 0 < idx < len(r):
            p.add_layout(Span(location=r[idx], dimension="height",
                              line_color="red", line_dash="dashed"))
            p.add_layout(Span(location=ee_target, dimension="width",
                              line_color="gray", line_dash="dotted"))
    return _finish_bokeh(p, return_)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k "radial_bokeh or ee_bokeh"`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/plotting.py tests/test_plotting.py
git commit -m "Add plot_radial_bokeh and plot_encircled_energy_bokeh"
```

---

## Task 9: `SimulatedImage` convenience methods

**Files:**
- Modify: `src/wcc_etc/psfsim.py` (add methods to the `SimulatedImage` dataclass, after `to_fitsimg`, around line 169)
- Test: `tests/test_plotting.py`

Methods dispatch to the `_mpl`/`_bokeh` functions by `backend=`. Import is done lazily inside the methods to avoid a circular import (`plotting` imports from `psfsim`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_plotting.py
def test_simimg_method_dispatch_mpl():
    s = _gaussian_simimg()
    fig, ax = s.plot_image(backend="mpl", stretch="linear")
    assert isinstance(fig, matplotlib.figure.Figure)
    fig2, axes = s.plot_image_row(backend="mpl", stretch="linear")
    assert len(axes) == 3
    fig3, ax3, (r, prof) = s.plot_radial(backend="mpl", units="pix")
    assert len(r) == len(prof)
    fig4, ax4, (re, ee) = s.plot_encircled_energy(backend="mpl", units="pix")
    assert ee[-1] == pytest.approx(1.0, abs=1e-6)


def test_simimg_method_dispatch_bokeh():
    s = _gaussian_simimg()
    assert isinstance(s.plot_image(backend="bokeh", return_="obj"), Plot)
    assert isinstance(s.plot_radial(backend="bokeh", return_="obj"), Plot)


def test_simimg_method_bad_backend_raises():
    s = _gaussian_simimg()
    with pytest.raises(ValueError):
        s.plot_image(backend="nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k "simimg_method"`
Expected: FAIL — `AttributeError: 'SimulatedImage' object has no attribute 'plot_image'`.

- [ ] **Step 3: Write minimal implementation**

Add these methods inside the `SimulatedImage` dataclass in `src/wcc_etc/psfsim.py`, immediately after the existing `to_fitsimg` method (around line 169):

```python
    def plot_image(self, backend="mpl", **kwargs):
        """Plot this image (single panel). backend='mpl' or 'bokeh'."""
        from . import plotting
        if backend == "mpl":
            return plotting.plot_image_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_image_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")

    def plot_image_row(self, backend="mpl", **kwargs):
        """Plot the 3-panel row (PSF+noise, PSF, saturation mask)."""
        from . import plotting
        if backend == "mpl":
            return plotting.plot_image_row_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_image_row_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")

    def plot_radial(self, backend="mpl", **kwargs):
        """Plot the azimuthally-averaged radial profile."""
        from . import plotting
        if backend == "mpl":
            return plotting.plot_radial_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_radial_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")

    def plot_encircled_energy(self, backend="mpl", **kwargs):
        """Plot the encircled-energy curve."""
        from . import plotting
        if backend == "mpl":
            return plotting.plot_encircled_energy_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_encircled_energy_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k "simimg_method"`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_plotting.py
git commit -m "Add SimulatedImage convenience plotting methods"
```

---

## Task 10: Public exports and full suite

**Files:**
- Modify: `src/wcc_etc/__init__.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Inspect current exports**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && grep -n "import\|__all__" src/wcc_etc/__init__.py`
Note the existing import style (e.g. `from .psfsim import ...`). Follow it.

- [ ] **Step 2: Write the failing test**

```python
# append to tests/test_plotting.py
def test_plotting_functions_exported():
    import wcc_etc
    for name in ["plot_image_mpl", "plot_image_bokeh", "plot_image_row_mpl",
                 "plot_image_row_bokeh", "plot_radial_mpl", "plot_radial_bokeh",
                 "plot_encircled_energy_mpl", "plot_encircled_energy_bokeh"]:
        assert hasattr(wcc_etc, name), f"{name} not exported from wcc_etc"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q -k exported`
Expected: FAIL — `AssertionError: plot_image_mpl not exported from wcc_etc`.

- [ ] **Step 4: Add the export line**

Add to `src/wcc_etc/__init__.py` (place near the other `from .` imports, matching existing style):

```python
from .plotting import (
    plot_image_mpl, plot_image_bokeh,
    plot_image_row_mpl, plot_image_row_bokeh,
    plot_radial_mpl, plot_radial_bokeh,
    plot_encircled_energy_mpl, plot_encircled_energy_bokeh,
)
```

If the file defines `__all__`, append those eight names to it as well.

- [ ] **Step 5: Run the export test, then the full suite**

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest tests/test_plotting.py -q`
Expected: PASS (all plotting tests).

Run: `cd /Users/gudmundurstefansson/Dropbox/mypylib/notebooks/GIT/wcc-etc4/wcc-etc && python -m pytest -q`
Expected: PASS — no regressions across the existing suite.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/__init__.py tests/test_plotting.py
git commit -m "Export plotting functions from wcc_etc"
```

---

## Post-implementation (separate, after plan approval to proceed)

- Demo notebook + project-status update per the team workflow (see `wcc-etc-feature-workflow` memory). Not part of this TDD plan; do as a follow-up once tests are green.

---

## Self-Review Notes

- **Spec coverage:** (a) Task 2; (b) Tasks 3 & 7; (c) Tasks 4 & 8; (d) Tasks 5 & 8; both backends covered; `return_` contract Task 6; module+methods structure Tasks 1 & 9; mas/pix units in every relevant task; no-stray-prints satisfied (new module never calls `print`). ✓
- **Placeholder scan:** none — every code step is complete. ✓
- **Type consistency:** `_resolve_inputs` returns the same 4-tuple everywhere; `_finish_bokeh(obj, return_)` and `return_` values (`'obj'/'html'/'components'`) consistent; mpl plot funcs return `(fig, ax)` / `(fig, axes)` / `(fig, ax, (x, y))` consistently; `psf_to_encircled_energy` unpacked as `(r_mas, psf1d, ee)` matching its real signature. ✓
