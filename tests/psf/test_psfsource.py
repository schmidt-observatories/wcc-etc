"""Tests for PSF utility functions, PSFSource render, DefocusPSF, and time-solver helpers."""

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


class TestPSFUtilities:
    def test_normalize_clips_negatives(self):
        a = np.array([[-1.0, 1.0], [2.0, 4.0]])
        out = normalize_psf(a)
        assert out.min() >= 0.0

    def test_normalize_sums_to_one(self):
        a = np.array([[-1.0, 1.0], [2.0, 4.0]])
        out = normalize_psf(a)
        assert out.sum() == pytest.approx(1.0)

    def test_normalize_raises_on_nonpositive(self):
        with pytest.raises(ValueError):
            normalize_psf(np.zeros((3, 3)))

    def test_center_crop_output_shape(self):
        a = np.zeros((5, 5))
        a[2, 2] = 1.0
        out = center_crop_or_pad(a, 3)
        assert out.shape == (3, 3)

    def test_center_crop_preserves_center(self):
        a = np.zeros((5, 5))
        a[2, 2] = 1.0
        out = center_crop_or_pad(a, 3)
        assert out[1, 1] == 1.0

    def test_center_pad_output_shape(self):
        a = np.zeros((3, 3))
        a[1, 1] = 1.0
        out = center_crop_or_pad(a, 5)
        assert out.shape == (5, 5)

    def test_center_pad_preserves_center(self):
        a = np.zeros((3, 3))
        a[1, 1] = 1.0
        out = center_crop_or_pad(a, 5)
        assert out[2, 2] == 1.0

    def test_even_to_odd_output_shape(self):
        a = np.zeros((6, 6))
        a[3, 3] = 1.0
        out = center_crop_or_pad(a, 5)
        assert out.shape == (5, 5)

    def test_even_to_odd_peak_location(self):
        a = np.zeros((6, 6))
        a[3, 3] = 1.0
        out = center_crop_or_pad(a, 5)
        assert np.unravel_index(np.argmax(out), out.shape) == (2, 2)

    def test_recenter_moves_peak(self):
        a = np.zeros((11, 11))
        a[5, 5] = 1.0
        out = recenter(a, (7.0, 5.0))
        assert np.unravel_index(np.argmax(out), out.shape) == (5, 7)

    def test_load_huygens_psf_1wave_file_exists(self):
        assert os.path.exists(DEFOCUS_1WAVE_PATH)

    def test_load_huygens_psf_1wave_shape(self):
        data = load_huygens_psf(DEFOCUS_1WAVE_PATH)
        assert data.shape == (256, 256)

    def test_load_huygens_psf_1wave_is_finite(self):
        data = load_huygens_psf(DEFOCUS_1WAVE_PATH)
        assert np.all(np.isfinite(data))

    def test_load_huygens_psf_1wave_is_positive(self):
        data = load_huygens_psf(DEFOCUS_1WAVE_PATH)
        assert data.sum() > 0

    def test_load_huygens_psf_2wave_file_exists(self):
        assert os.path.exists(DEFOCUS_2WAVE_PATH)

    def test_load_huygens_psf_2wave_shape(self):
        data = load_huygens_psf(DEFOCUS_2WAVE_PATH)
        assert data.shape == (256, 256)

    def test_load_huygens_psf_2wave_is_finite(self):
        data = load_huygens_psf(DEFOCUS_2WAVE_PATH)
        assert np.all(np.isfinite(data))

    def test_load_huygens_psf_2wave_is_positive(self):
        data = load_huygens_psf(DEFOCUS_2WAVE_PATH)
        assert data.sum() > 0

    def test_detector_context_default_jitter(self):
        ctx = DetectorPSFContext(
            npix=64,
            pixel_size_um=3.76,
            plate_scale_mas=20.0,
            wavelength_m=0.6e-6,
            diameter_m=3.0,
            fnum=15.0,
        )
        assert ctx.jitter_sigma_mas == 0.0

    def test_detector_context_default_center(self):
        ctx = DetectorPSFContext(
            npix=64,
            pixel_size_um=3.76,
            plate_scale_mas=20.0,
            wavelength_m=0.6e-6,
            diameter_m=3.0,
            fnum=15.0,
        )
        assert ctx.center is None

    def test_detector_context_default_oversample(self):
        ctx = DetectorPSFContext(
            npix=64,
            pixel_size_um=3.76,
            plate_scale_mas=20.0,
            wavelength_m=0.6e-6,
            diameter_m=3.0,
            fnum=15.0,
        )
        assert ctx.oversample == 11


class TestPSFSource:
    def test_base_is_abstract(self):
        with pytest.raises(NotImplementedError):
            PSFSource().render(_airy_ctx())

    def test_airy_render_shape(self):
        psf = AiryPSF().render(_airy_ctx(npix=64))
        assert psf.shape == (64, 64)

    def test_airy_render_normalized(self):
        psf = AiryPSF().render(_airy_ctx(npix=64))
        assert psf.sum() == pytest.approx(1.0, abs=1e-6)

    def test_airy_render_centered(self):
        psf = AiryPSF().render(_airy_ctx(npix=64))
        cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
        assert abs(cy - 31.5) <= 1 and abs(cx - 31.5) <= 1

    def test_airy_recenter_shifts_peak_x(self):
        ctx = _airy_ctx(npix=65)
        ctx.center = (40.0, 32.0)
        psf = AiryPSF().render(ctx)
        cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
        assert abs(cx - 40) <= 1

    def test_airy_recenter_shifts_peak_y(self):
        ctx = _airy_ctx(npix=65)
        ctx.center = (40.0, 32.0)
        psf = AiryPSF().render(ctx)
        cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
        assert abs(cy - 32) <= 1

    def test_custom_psf_output_shape(self):
        arr = np.zeros((51, 51))
        arr[25, 25] = 1.0
        psf = CustomPSF(arr, src_um_per_pix=3.76).render(_grid_ctx(3.76, npix=64))
        assert psf.shape == (64, 64)

    def test_custom_psf_normalized(self):
        arr = np.zeros((51, 51))
        arr[25, 25] = 1.0
        psf = CustomPSF(arr, src_um_per_pix=3.76).render(_grid_ctx(3.76, npix=64))
        assert psf.sum() == pytest.approx(1.0, abs=1e-6)


class TestDefocusPSF:
    def test_defocus_render_shape(self):
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH).render(_grid_ctx(3.76))
        assert psf.shape == (300, 300)

    def test_defocus_render_normalized(self):
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH).render(_grid_ctx(3.76))
        assert psf.sum() == pytest.approx(1.0, abs=1e-6)

    def test_two_wave_broader_than_one_wave(self):
        p1 = DefocusPSF(DEFOCUS_1WAVE_PATH).render(_grid_ctx(3.76))
        p2 = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(3.76))
        assert p2.max() < p1.max()

    def test_smaller_pixels_spread_over_more_pixels(self):
        sony = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(3.76))
        hwk = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_grid_ctx(4.6))
        assert sony.max() < hwk.max()

    def test_99pct_contained_in_default_grid(self):
        data = DefocusPSF(DEFOCUS_2WAVE_PATH)._data
        z = np.clip(_zoom(data, 4.0 / 3.76, order=1), 0.0, None)
        crop = center_crop_or_pad(z, 300)
        assert crop.sum() / z.sum() >= 0.99


class TestSolveTimeForSnr:
    def test_scalar_roundtrips(self):
        from wcc_etc.psfsim import solve_time_for_snr

        A, B, C, snr = 10.0, 5.0, 100.0, 25.0
        t = solve_time_for_snr(snr, A, B, C)
        assert abs(A * t / (B * t + C) ** 0.5 - snr) < 1e-9

    def test_zero_signal_gives_infinite_time(self):
        from wcc_etc.psfsim import solve_time_for_snr

        A = np.array([10.0, 0.0, 4.0])
        B = np.array([5.0, 5.0, 2.0])
        C = np.array([100.0, 100.0, 50.0])
        t = solve_time_for_snr(30.0, A, B, C)
        assert np.isinf(t[1])

    def test_nonzero_signal_roundtrips(self):
        from wcc_etc.psfsim import solve_time_for_snr

        A = np.array([10.0, 0.0, 4.0])
        B = np.array([5.0, 5.0, 2.0])
        C = np.array([100.0, 100.0, 50.0])
        t = solve_time_for_snr(30.0, A, B, C)
        assert np.allclose(
            A[[0, 2]] * t[[0, 2]] / np.sqrt(B[[0, 2]] * t[[0, 2]] + C[[0, 2]]), 30.0
        )

    def test_aperture_time_for_snr_matches_forward(self):
        from wcc_etc.psfsim import aperture_snr_radial, aperture_time_for_snr

        n = 41
        yy, xx = np.mgrid[0:n, 0:n]
        r2 = (xx - n // 2) ** 2 + (yy - n // 2) ** 2
        psf = np.exp(-r2 / (2 * 3.0**2))
        psf /= psf.sum()
        res = aperture_time_for_snr(
            psf, 50.0, 200.0, 0.5, 0.1, 3.0, n_reads=1, snr=50.0, r_aper_mas=300.0
        )
        t = res["time_s"]
        prof = aperture_snr_radial(psf, 50.0, 200.0 * t, 0.5 * t, 0.1 * t, 3.0)
        idx = np.searchsorted(prof["r_mas"], res["r_aper_mas"], side="right") - 1
        assert abs(prof["snr"][idx] - 50.0) < 0.05

    def test_aperture_time_for_snr_optimize_is_minimum(self):
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

    def _make_gaussian_psf(self):
        n = 41
        yy, xx = np.mgrid[0:n, 0:n]
        r2 = (xx - n // 2) ** 2 + (yy - n // 2) ** 2
        psf = np.exp(-r2 / (2 * 3.0**2))
        psf /= psf.sum()
        return psf

    def test_n_reads_scaling_snr_equation(self):
        from wcc_etc.psfsim import _radial_cumulative, aperture_time_for_snr

        psf = self._make_gaussian_psf()
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
        r_mas, enclosed, n_pix = _radial_cumulative(psf, plate)
        idx = int(
            np.clip(np.searchsorted(r_mas, 300.0, side="right") - 1, 0, r_mas.size - 1)
        )
        A = src_rate * enclosed[idx]
        B = A + (diff_rate + dark_rate) * n_pix[idx]
        C = N * rn**2 * n_pix[idx]
        assert np.isclose(A * t / np.sqrt(B * t + C), S, rtol=1e-9)

    def test_more_reads_requires_more_time(self):
        from wcc_etc.psfsim import aperture_time_for_snr

        psf = self._make_gaussian_psf()
        plate, src_rate, diff_rate, dark_rate, rn = 50.0, 200.0, 0.5, 0.1, 3.0
        S = 40.0
        res = aperture_time_for_snr(
            psf,
            plate,
            src_rate,
            diff_rate,
            dark_rate,
            rn,
            n_reads=3,
            snr=S,
            r_aper_mas=300.0,
        )
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
