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


def test_make_norm_returns_expected_types():
    from astropy.visualization.mpl_normalize import ImageNormalize
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert plotting._make_norm(data, "linear") is None
    assert isinstance(plotting._make_norm(data, "log"), ImageNormalize)
    assert isinstance(plotting._make_norm(data, "hist"), ImageNormalize)


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
