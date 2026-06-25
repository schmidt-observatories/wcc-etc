import numpy as np
import pytest

from wcc_etc.airy import psf_to_encircled_energy, render_detector_psf


def test_render_detector_psf_shape_and_normalization():
    psf, pscale = render_detector_psf(
        wavelength=0.6e-6,
        fnum=15,
        D=3,
        pixel_size=3.76,
        jitter_sigma_mas=0,
        n_pixels=21,
        oversample=11,
    )
    assert psf.shape == (21, 21)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)
    assert pscale > 0


def test_render_detector_psf_peak_is_centered():
    psf, _ = render_detector_psf(
        wavelength=0.6e-6,
        fnum=15,
        D=3,
        pixel_size=3.76,
        jitter_sigma_mas=0,
        n_pixels=21,
        oversample=11,
    )
    center = (psf.shape[0] // 2, psf.shape[1] // 2)
    assert np.unravel_index(np.argmax(psf), psf.shape) == center


def test_render_detector_psf_peak_fraction_in_unit_interval():
    psf, _ = render_detector_psf(
        wavelength=0.6e-6,
        fnum=15,
        D=3,
        pixel_size=3.76,
        jitter_sigma_mas=0,
        n_pixels=21,
        oversample=11,
    )
    assert 0.0 < psf.max() <= 1.0


def test_jitter_reduces_peak_fraction():
    psf0, _ = render_detector_psf(
        wavelength=0.6e-6,
        fnum=15,
        D=3,
        pixel_size=3.76,
        jitter_sigma_mas=0,
        n_pixels=21,
        oversample=11,
    )
    psf_j, _ = render_detector_psf(
        wavelength=0.6e-6,
        fnum=15,
        D=3,
        pixel_size=3.76,
        jitter_sigma_mas=50,
        n_pixels=21,
        oversample=11,
    )
    assert psf_j.max() < psf0.max()


def test_render_detector_psf_peak_fraction_reference_value():
    psf, _ = render_detector_psf(
        wavelength=0.6e-6,
        fnum=15,
        D=3,
        pixel_size=3.76,
        jitter_sigma_mas=0,
        n_pixels=21,
        oversample=11,
    )
    assert psf.max() == pytest.approx(0.1333, abs=0.001)


def _ee_radius(r_mas, ee, frac):
    """Interpolate the curve-of-growth to the radius enclosing `frac` of the energy."""
    return float(np.interp(frac, ee, r_mas))


def test_psf_to_encircled_energy_centers_on_true_centroid():
    """Regression: the EE/profile reducer must measure radii from the rendered PSF's
    true centroid, not the array's n//2 index.

    Rendered through the real path at an even npix (256), render_detector_psf forces
    the grid odd (257, peak on the exact center pixel) and AiryPSF's center_crop_or_pad
    brings it back to 256 — landing the centroid on an integer pixel (127.0), a full
    pixel away from n//2 (128). With the old `cy, cx = ny//2, nx//2` the curve-of-growth
    is biased outward by ~1 px; EE50/EE90 must instead match a reference computed about
    the measured centroid to well under 0.05 px.
    """
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
    psf = center_crop_or_pad(psf, 256)  # 257 -> 256; true centroid lands at 127.0

    # True centroid of the rendered PSF (howell_center returns (xc, yc)).
    xc, yc = howell_center(psf)

    # Curve of growth from the function under test.
    r_mas, _psf1d, ee = psf_to_encircled_energy(psf, pscale, pscale)

    # Reference curve of growth about the TRUE centroid, identical binning so the
    # ONLY difference is the assumed center.
    y, x = np.indices(psf.shape)
    r_ref = np.sqrt(((x - xc) * pscale) ** 2 + ((y - yc) * pscale) ** 2)
    edges = np.arange(0, r_ref.max() + pscale, pscale)
    hist, _ = np.histogram(r_ref.ravel(), bins=edges, weights=psf.ravel())
    ee_ref = np.cumsum(hist)
    ee_ref /= ee_ref[-1]
    r_ref_centers = 0.5 * (edges[1:] + edges[:-1])

    tol_mas = 0.05 * pscale  # sub-0.05 px
    for frac in (0.5, 0.9):
        got = _ee_radius(r_mas, ee, frac)
        ref = _ee_radius(r_ref_centers, ee_ref, frac)
        assert abs(got - ref) < tol_mas, (
            f"EE{int(frac * 100)} radius {got:.4f} mas differs from centroid-based "
            f"reference {ref:.4f} mas by more than {tol_mas:.4f} mas "
            f"({abs(got - ref) / pscale:.3f} px) — reducer is centered off the PSF centroid"
        )


def test_render_detector_psf_even_npix_no_warning():
    # Even n_pixels is rounded up to odd; this must NOT raise a UserWarning
    # (it used to warn on every call and flood notebooks). Treat any warning as an error.
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
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)
