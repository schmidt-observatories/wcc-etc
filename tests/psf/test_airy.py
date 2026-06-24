"""Tests for render_detector_psf and psf_to_encircled_energy."""

import numpy as np
import pytest
from wcc_etc.airy import render_detector_psf, psf_to_encircled_energy


def _ee_radius(r_mas, ee, frac):
    return float(np.interp(frac, ee, r_mas))


class TestAiryPSF:
    def _render(self, **kw):
        defaults = dict(
            wavelength=0.6e-6,
            fnum=15,
            D=3,
            pixel_size=3.76,
            jitter_sigma_mas=0,
            n_pixels=21,
            oversample=11,
        )
        defaults.update(kw)
        return render_detector_psf(**defaults)

    def test_psf_shape(self):
        psf, pscale = self._render()
        assert psf.shape == (21, 21)

    def test_psf_normalization(self):
        psf, pscale = self._render()
        assert psf.sum() == pytest.approx(1.0, abs=1e-6)

    def test_plate_scale_positive(self):
        psf, pscale = self._render()
        assert pscale > 0

    def test_peak_is_centered(self):
        psf, _ = self._render()
        center = (psf.shape[0] // 2, psf.shape[1] // 2)
        assert np.unravel_index(np.argmax(psf), psf.shape) == center

    def test_peak_fraction_in_unit_interval(self):
        psf, _ = self._render()
        assert 0.0 < psf.max() <= 1.0

    def test_jitter_reduces_peak_fraction(self):
        psf0, _ = self._render(jitter_sigma_mas=0)
        psf_j, _ = self._render(jitter_sigma_mas=50)
        assert psf_j.max() < psf0.max()

    def test_peak_fraction_reference_value(self):
        psf, _ = self._render()
        assert psf.max() == pytest.approx(0.1333, abs=0.001)

    def test_even_npix_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            render_detector_psf(
                wavelength=0.6e-6,
                fnum=15,
                D=3,
                pixel_size=3.76,
                jitter_sigma_mas=0,
                n_pixels=128,
                oversample=5,
            )

    def test_even_npix_output_shape(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            psf, _ = render_detector_psf(
                wavelength=0.6e-6,
                fnum=15,
                D=3,
                pixel_size=3.76,
                jitter_sigma_mas=0,
                n_pixels=128,
                oversample=5,
            )
        assert psf.shape == (129, 129)

    def test_even_npix_normalization(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            psf, _ = render_detector_psf(
                wavelength=0.6e-6,
                fnum=15,
                D=3,
                pixel_size=3.76,
                jitter_sigma_mas=0,
                n_pixels=128,
                oversample=5,
            )
        assert psf.sum() == pytest.approx(1.0, abs=1e-6)


def _build_ee_reference():
    from wcc_etc.psfsim import center_crop_or_pad, howell_center

    psf, pscale = render_detector_psf(
        wavelength=0.6e-6,
        fnum=15,
        D=3.0,
        pixel_size=3.74,
        jitter_sigma_mas=0,
        n_pixels=256,
        oversample=11,
    )
    psf = center_crop_or_pad(psf, 256)
    xc, yc = howell_center(psf)
    r_mas, _psf1d, ee = psf_to_encircled_energy(psf, pscale, pscale)
    y, x = np.indices(psf.shape)
    r_ref = np.sqrt(((x - xc) * pscale) ** 2 + ((y - yc) * pscale) ** 2)
    edges = np.arange(0, r_ref.max() + pscale, pscale)
    hist, _ = np.histogram(r_ref.ravel(), bins=edges, weights=psf.ravel())
    ee_ref = np.cumsum(hist)
    ee_ref /= ee_ref[-1]
    r_ref_centers = 0.5 * (edges[1:] + edges[:-1])
    tol_mas = 0.05 * pscale
    return r_mas, ee, r_ref_centers, ee_ref, tol_mas


class TestEncircledEnergy:
    def test_ee50_radius_matches_centroid_reference(self):
        r_mas, ee, r_ref_centers, ee_ref, tol_mas = _build_ee_reference()
        got = _ee_radius(r_mas, ee, 0.5)
        ref = _ee_radius(r_ref_centers, ee_ref, 0.5)
        assert abs(got - ref) < tol_mas, (
            f"EE50 radius {got:.4f} mas differs from centroid-based "
            f"reference {ref:.4f} mas by more than {tol_mas:.4f} mas"
        )

    def test_ee90_radius_matches_centroid_reference(self):
        r_mas, ee, r_ref_centers, ee_ref, tol_mas = _build_ee_reference()
        got = _ee_radius(r_mas, ee, 0.9)
        ref = _ee_radius(r_ref_centers, ee_ref, 0.9)
        assert abs(got - ref) < tol_mas, (
            f"EE90 radius {got:.4f} mas differs from centroid-based "
            f"reference {ref:.4f} mas by more than {tol_mas:.4f} mas"
        )
