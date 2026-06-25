import os

import numpy as np
import pytest
from scipy.ndimage import zoom as _zoom

from wcc_etc.psfsim import (
    DEFOCUS_1WAVE_PATH,
    DEFOCUS_2WAVE_PATH,
    AiryPSF,
    CustomPSF,
    DefocusPSF,
    DetectorPSFContext,
    PSFSource,
    center_crop_or_pad,
    load_huygens_psf,
    normalize_psf,
    recenter,
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
    a = np.zeros((5, 5))
    a[2, 2] = 1.0
    out = center_crop_or_pad(a, 3)
    assert out.shape == (3, 3)
    assert out[1, 1] == 1.0  # center preserved


def test_center_crop_or_pad_pads_to_size():
    a = np.zeros((3, 3))
    a[1, 1] = 1.0
    out = center_crop_or_pad(a, 5)
    assert out.shape == (5, 5)
    assert out[2, 2] == 1.0


def test_recenter_moves_peak():
    a = np.zeros((11, 11))
    a[5, 5] = 1.0
    out = recenter(a, (7.0, 5.0))  # (cx, cy) -> column 7, row 5
    assert np.unravel_index(np.argmax(out), out.shape) == (5, 7)


def test_load_huygens_psf_shape_and_finite():
    assert os.path.exists(DEFOCUS_1WAVE_PATH)
    data = load_huygens_psf(DEFOCUS_1WAVE_PATH)
    assert data.shape == (256, 256)
    assert np.all(np.isfinite(data))
    assert data.sum() > 0


def test_detector_context_defaults():
    ctx = DetectorPSFContext(
        npix=64,
        pixel_size_um=3.76,
        plate_scale_mas=20.0,
        wavelength_m=0.6e-6,
        diameter_m=3.0,
        fnum=15.0,
    )
    assert ctx.jitter_sigma_mas == 0.0
    assert ctx.center is None
    assert ctx.oversample == 11


def test_center_crop_or_pad_even_source_to_odd_output_centers():
    a = np.zeros((6, 6))
    a[3, 3] = 1.0  # one of the 4 central pixels of a 6x6
    out = center_crop_or_pad(a, 5)
    assert out.shape == (5, 5)
    assert np.unravel_index(np.argmax(out), out.shape) == (
        2,
        2,
    )  # round-half-up centering


def test_load_huygens_psf_second_file():
    assert os.path.exists(DEFOCUS_2WAVE_PATH)
    data = load_huygens_psf(DEFOCUS_2WAVE_PATH)
    assert data.shape == (256, 256)
    assert np.all(np.isfinite(data))
    assert data.sum() > 0


def _airy_ctx(npix=64, pixel_size_um=3.76):
    return DetectorPSFContext(
        npix=npix,
        pixel_size_um=pixel_size_um,
        plate_scale_mas=20.0,
        wavelength_m=0.6e-6,
        diameter_m=3.0,
        fnum=15.0,
        jitter_sigma_mas=0.0,
        oversample=11,
    )


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


def _grid_ctx(pixel_size_um, npix=300):
    return DetectorPSFContext(
        npix=npix,
        pixel_size_um=pixel_size_um,
        plate_scale_mas=20.0,
        wavelength_m=0.6e-6,
        diameter_m=3.0,
        fnum=15.0,
        jitter_sigma_mas=0.0,
        oversample=11,
    )


def test_defocus_render_normalized_shape():
    psf = DefocusPSF(DEFOCUS_1WAVE_PATH).render(_grid_ctx(3.76))
    assert psf.shape == (300, 300)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)


def test_two_wave_is_broader_than_one_wave():
    p1 = DefocusPSF(DEFOCUS_1WAVE_PATH).render(_grid_ctx(3.76))
    p2 = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(3.76))
    assert p2.max() < p1.max()  # more defocus -> more spread -> lower peak


def test_same_psf_spreads_over_more_sony_pixels_than_hwk():
    sony = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(3.76))  # smaller pixels
    hwk = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(4.6))  # larger pixels
    assert sony.max() < hwk.max()


def test_defocus_99pct_contained_in_default_grid_sony():
    data = DefocusPSF(DEFOCUS_2WAVE_PATH)._data
    z = np.clip(_zoom(data, 4.0 / 3.76, order=1), 0.0, None)
    crop = center_crop_or_pad(z, 300)
    assert crop.sum() / z.sum() >= 0.99


def test_custom_psf_from_array():
    arr = np.zeros((51, 51))
    arr[25, 25] = 1.0
    psf = CustomPSF(arr, src_um_per_pix=3.76).render(_grid_ctx(3.76, npix=64))
    assert psf.shape == (64, 64)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)


def test_solve_time_for_snr_scalar_roundtrips():
    from wcc_etc.psfsim import solve_time_for_snr

    A, B, C, snr = 10.0, 5.0, 100.0, 25.0
    t = solve_time_for_snr(snr, A, B, C)
    # forward SNR at t must recover the target
    assert abs(A * t / (B * t + C) ** 0.5 - snr) < 1e-9


def test_solve_time_for_snr_array_and_zero_signal():
    import numpy as np

    from wcc_etc.psfsim import solve_time_for_snr

    A = np.array([10.0, 0.0, 4.0])
    B = np.array([5.0, 5.0, 2.0])
    C = np.array([100.0, 100.0, 50.0])
    t = solve_time_for_snr(30.0, A, B, C)
    assert np.isinf(t[1])  # zero source -> infinite time
    assert np.allclose(
        A[[0, 2]] * t[[0, 2]] / np.sqrt(B[[0, 2]] * t[[0, 2]] + C[[0, 2]]), 30.0
    )


def test_aperture_time_for_snr_matches_forward():
    import numpy as np

    from wcc_etc.psfsim import aperture_snr_radial, aperture_time_for_snr

    # simple centered gaussian-ish PSF
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - n // 2) ** 2 + (yy - n // 2) ** 2
    psf = np.exp(-r2 / (2 * 3.0**2))
    psf /= psf.sum()
    plate, src_rate, diff_rate, dark_rate, rn = 50.0, 200.0, 0.5, 0.1, 3.0

    # pick a fixed aperture radius, find the time for SNR=50, then check forward
    res = aperture_time_for_snr(
        psf,
        plate,
        src_rate,
        diff_rate,
        dark_rate,
        rn,
        n_reads=1,
        snr=50.0,
        r_aper_mas=300.0,
    )
    t = res["time_s"]
    prof = aperture_snr_radial(
        psf, plate, src_rate * t, diff_rate * t, dark_rate * t, rn
    )
    # SNR at res's radius and that time reproduces the target
    idx = np.searchsorted(prof["r_mas"], res["r_aper_mas"], side="right") - 1
    assert abs(prof["snr"][idx] - 50.0) < 0.05


def test_aperture_time_for_snr_optimize_is_minimum():
    import numpy as np

    from wcc_etc.psfsim import aperture_time_for_snr

    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - n // 2) ** 2 + (yy - n // 2) ** 2
    psf = np.exp(-r2 / (2 * 3.0**2))
    psf /= psf.sum()
    fixed = aperture_time_for_snr(
        psf, 50.0, 200.0, 0.5, 0.1, 3.0, n_reads=1, snr=50.0, r_aper_mas=300.0
    )
    best = aperture_time_for_snr(
        psf, 50.0, 200.0, 0.5, 0.1, 3.0, n_reads=1, snr=50.0, optimize=True
    )
    assert best["time_s"] <= fixed["time_s"] + 1e-9


def test_aperture_time_for_snr_n_reads_scaling():
    import numpy as np

    from wcc_etc.psfsim import _radial_cumulative, aperture_time_for_snr

    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - n // 2) ** 2 + (yy - n // 2) ** 2
    psf = np.exp(-r2 / (2 * 3.0**2))
    psf /= psf.sum()
    plate, src_rate, diff_rate, dark_rate, rn = 50.0, 200.0, 0.5, 0.1, 3.0
    N, S = 3, 40.0
    res = aperture_time_for_snr(
        psf,
        plate,
        src_rate,
        diff_rate,
        dark_rate,
        rn,
        n_reads=N,
        snr=S,
        r_aper_mas=300.0,
    )
    t = res["time_s"]
    # reconstruct per-radius coefficients at the selected aperture and verify the
    # forward SNR with the n_reads-scaled read-noise term C = N*rn**2*n_pix
    r_mas, enclosed, n_pix = _radial_cumulative(psf, plate)
    idx = int(
        np.clip(np.searchsorted(r_mas, 300.0, side="right") - 1, 0, r_mas.size - 1)
    )
    A = src_rate * enclosed[idx]
    B = A + (diff_rate + dark_rate) * n_pix[idx]
    C = N * rn**2 * n_pix[idx]
    assert np.isclose(A * t / np.sqrt(B * t + C), S, rtol=1e-9)
    # more reads -> longer time for the same target/aperture
    res1 = aperture_time_for_snr(
        psf,
        plate,
        src_rate,
        diff_rate,
        dark_rate,
        rn,
        n_reads=1,
        snr=S,
        r_aper_mas=300.0,
    )
    assert res["time_s"] > res1["time_s"]
