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
