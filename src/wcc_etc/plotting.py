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
