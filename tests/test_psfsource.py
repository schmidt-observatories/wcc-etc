import os
import numpy as np
import pytest
from wcc_etc.psfsim import (
    DetectorPSFContext, center_crop_or_pad, normalize_psf, recenter,
    load_huygens_psf, DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH,
    PSFSource, AiryPSF,
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


def test_center_crop_or_pad_even_source_to_odd_output_centers():
    a = np.zeros((6, 6)); a[3, 3] = 1.0  # one of the 4 central pixels of a 6x6
    out = center_crop_or_pad(a, 5)
    assert out.shape == (5, 5)
    assert np.unravel_index(np.argmax(out), out.shape) == (2, 2)  # round-half-up centering


def test_load_huygens_psf_second_file():
    assert os.path.exists(DEFOCUS_2WAVE_PATH)
    data = load_huygens_psf(DEFOCUS_2WAVE_PATH)
    assert data.shape == (256, 256)
    assert np.all(np.isfinite(data))
    assert data.sum() > 0


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
