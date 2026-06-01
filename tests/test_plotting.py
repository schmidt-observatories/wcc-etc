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
