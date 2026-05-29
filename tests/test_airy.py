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
