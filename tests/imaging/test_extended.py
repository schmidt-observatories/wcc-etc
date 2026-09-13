"""Sersic host rendering: analytic normalization and grid placement."""

import numpy as np
import pytest
from astropy.modeling.models import Sersic2D
from scipy.integrate import quad

from scipy.special import gammainc, gammaincinv

from wcc_etc.extended import render_sersic, sersic_total_over_amplitude

PLATE = 0.016869  # arcsec/pix, sony/zwo IMX455 on the WCC


def enclosed_fraction(n, r_eff_pix, r_pix):
    """Analytic flux fraction inside the Sersic isophote of semi-major axis r."""
    bn = gammaincinv(2 * n, 0.5)
    return gammainc(2 * n, bn * (r_pix / r_eff_pix) ** (1.0 / n))


def centroid(img, r=6):
    """Flux-weighted (x, y) centre within r pixels of the brightest pixel.

    Windowed because Airy wings cut by the grid edge bias a whole-image centroid
    by a few hundredths of a pixel; a half-pixel placement error still shows.
    """
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    yy, xx = np.indices(img.shape)
    w = img * (np.hypot(xx - ix, yy - iy) <= r)
    return (xx * w).sum() / w.sum(), (yy * w).sum() / w.sum()


class TestTotalOverAmplitude:
    """The analytic F/I_e matches an independent numeric integral."""

    @pytest.mark.parametrize("n", [1.0, 4.0])
    def test_matches_radial_integral(self, n):
        """Analytic F/I_e == 2 pi int I(r) r dr by quad (no cusp sampling error)."""
        r_eff = 20.0
        cut = Sersic2D(amplitude=1.0, r_eff=r_eff, n=n)
        num = 2 * np.pi * quad(lambda r: r * cut(r, 0.0), 0, np.inf)[0]
        assert sersic_total_over_amplitude(n, r_eff) == pytest.approx(num, rel=1e-6)

    def test_ellipticity_scales_area(self):
        """An ellipticity e shrinks the total by (1 - e), the minor/major axis ratio."""
        assert sersic_total_over_amplitude(1.0, 20.0, 0.3) == pytest.approx(
            0.7 * sersic_total_over_amplitude(1.0, 20.0), rel=1e-12
        )


class TestRenderSersic:
    """render_sersic places a normalized profile on the detector grid."""

    def test_total_mode_sums_to_one_when_contained(self):
        """A compact n=1 profile well inside the grid carries (almost) unit flux."""
        img = render_sersic({"r_eff": 0.1}, PLATE, 128)
        assert img.sum() == pytest.approx(1.0, rel=1e-2)

    def test_total_mode_loses_light_off_grid(self):
        """A profile larger than the grid keeps < 1 total flux; no renormalizing."""
        img = render_sersic({"r_eff": 5.0}, PLATE, 128)
        assert img.sum() < 0.5

    def test_sb_mode_is_one_at_r_eff(self):
        """total=False returns I/I_e: the pixel at r_eff along +x reads ~1."""
        img = render_sersic({"r_eff": 0.5, "n": 1.0}, PLATE, 128, total=False)
        c = (128 - 1) / 2
        assert img[int(round(c)), int(round(c + 0.5 / PLATE))] == pytest.approx(
            1.0, rel=5e-2
        )

    def test_dx_shifts_peak(self):
        """dx (arcsec) moves the brightest pixel by dx / plate_scale pixels in x."""
        img = render_sersic({"r_eff": 0.1, "dx": 0.3}, PLATE, 128)
        _, ix = np.unravel_index(np.argmax(img), img.shape)
        assert ix == pytest.approx(63 + 0.3 / PLATE, abs=0.5)

    def test_center_moves_profile(self):
        """center=(cx, cy) is the source position the profile is attached to."""
        img = render_sersic({"r_eff": 0.1}, PLATE, 128, center=(40.0, 90.0))
        iy, ix = np.unravel_index(np.argmax(img), img.shape)
        assert (ix, iy) == (40, 90)


# Compact cusps a uniform sub-pixel grid cannot integrate (issue #84, R4):
# (n, r_eff arcsec, extra profile keys). r_eff/pixel runs 0.18-0.6.
COMPACT = [
    (4.0, 0.01, {}),
    (4.0, 0.003, {}),
    (8.0, 0.005, {}),
    (4.0, 0.01, {"dx": 0.5 * PLATE, "dy": 0.3 * PLATE}),  # cusp on a pixel edge
    (4.0, 0.01, {"ellip": 0.6, "pa": 30.0}),
    (1.0, 0.01, {}),
    (0.5, 0.01, {}),
]


class TestCuspIntegration:
    """The pixels around a Sersic cusp are integrated, not sampled."""

    @pytest.mark.parametrize("n,r_eff,extra", COMPACT)
    def test_on_grid_flux_never_exceeds_total(self, n, r_eff, extra):
        """A non-negative unit-total profile cannot put more than 1 on a finite grid."""
        img = render_sersic({"r_eff": r_eff, "n": n, **extra}, PLATE, 65)
        assert img.sum() <= 1.0 + 1e-9

    @pytest.mark.parametrize("n,r_eff,extra", COMPACT)
    def test_on_grid_flux_reaches_inscribed_disc_bound(self, n, r_eff, extra):
        """The grid holds the disc of radius R (cusp to nearest edge), so the sum is
        at least the analytic flux inside the isophote of semi-major axis R."""
        img = render_sersic({"r_eff": r_eff, "n": n, **extra}, PLATE, 65)
        cx = 32 + extra.get("dx", 0.0) / PLATE
        cy = 32 + extra.get("dy", 0.0) / PLATE
        r_in = min(cx + 0.5, 64.5 - cx, cy + 0.5, 64.5 - cy)
        assert img.sum() >= enclosed_fraction(n, r_eff / PLATE, r_in) - 1e-4

    def test_cusp_pixel_matches_polar_integral(self):
        """The pixel holding a centred n=4 cusp equals the profile integrated in
        polar coordinates over the unit square (independent quadrature)."""
        r_eff_pix = 0.01 / PLATE
        amp = 1.0 / sersic_total_over_amplitude(4.0, r_eff_pix)
        cut = Sersic2D(amplitude=amp, r_eff=r_eff_pix, n=4.0)

        def wedge(phi):
            r_edge = 0.5 / max(abs(np.cos(phi)), abs(np.sin(phi)))
            return quad(lambda r: r * cut(r, 0.0), 0, r_edge, epsabs=0, epsrel=1e-10)[0]

        polar = quad(wedge, 0, 2 * np.pi, epsabs=0, epsrel=1e-10, limit=200)[0]
        img = render_sersic({"r_eff": 0.01, "n": 4.0}, PLATE, 65)
        assert img[32, 32] == pytest.approx(polar, rel=1e-5)

    def test_total_independent_of_oversample(self):
        """With the cusp integrated exactly, oversample no longer moves the total."""
        prof = {"r_eff": 0.01, "n": 4.0}
        a = render_sersic(prof, PLATE, 65, oversample=11).sum()
        b = render_sersic(prof, PLATE, 65, oversample=31).sum()
        assert a == pytest.approx(b, abs=1e-4)


# --- integration: ImageSimulator and the aperture functions ------------------

from tests.helpers import make_scene  # noqa: E402
from wcc_etc.psfsim import (  # noqa: E402
    CustomPSF,
    ImageSimulator,
    aperture_snr_radial,
)


def host_and_psf(npix, center=None, psf=None, dx=0.0):
    """(PSF-convolved compact host image, rendered PSF) for a G5V host at dx."""
    host = {
        "mag": 17,
        "bandpass": "johnson_r",
        "profile": "sersic",
        "r_eff": 0.02,
        "dx": dx,
    }
    scene = make_scene(mag=20, host="G5V", host_prop=host)
    imsim = ImageSimulator.from_sensor_and_scene("sony:r", scene, npix=npix)
    ctx = imsim._context(jitter_sigma_mas=0.0, center=center)
    psf_norm = (psf or imsim.default_psf).render(ctx)
    return imsim.sim.extended_rate_image(psf_norm, ctx), psf_norm


class TestHostSourceAlignment:
    """A zero-offset host sits on the source, whatever the grid parity or PSF (#84, R5)."""

    @pytest.mark.parametrize("npix", [64, 65])
    @pytest.mark.parametrize("center", [None, (40.0, 32.25)])
    def test_zero_offset_host_sits_on_source(self, npix, center):
        """Host and PSF centroids agree to well under a pixel on odd and even grids."""
        host, psf = host_and_psf(npix, center)
        assert centroid(host) == pytest.approx(centroid(psf), abs=0.02)

    def test_even_custom_psf_host_sits_on_source(self):
        """An even-sized custom PSF (half-integer natural centre) still carries the host."""
        yy, xx = np.indices((50, 50))
        arr = np.exp(-((xx - 24.5) ** 2 + (yy - 24.5) ** 2) / (2 * 1.5**2))
        psf = CustomPSF(arr, src_um_per_pix=3.76, wavelength_scaling="none")
        host, psf_norm = host_and_psf(64, (40.0, 32.25), psf=psf)
        assert centroid(host) == pytest.approx(centroid(psf_norm), abs=0.02)

    def test_dx_offset_recovered(self):
        """dx (arcsec) moves the host centroid dx / plate_scale pixels off the source."""
        host, psf = host_and_psf(64, dx=0.3)
        assert centroid(host)[0] - centroid(psf)[0] == pytest.approx(
            0.3 / PLATE, abs=0.05
        )


class TestSimulateIncludesHost:
    def test_clean_image_carries_extended_charge(self):
        """simulate(add_noise=False) adds extended_rate_image * t to the clean image."""
        host = {"mag": 17, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 0.1}
        scene = make_scene(mag=20, host="G5V", host_prop=host)
        imsim = ImageSimulator.from_sensor_and_scene("sony:r", scene, npix=128)
        hostless = ImageSimulator.from_sensor_and_scene(
            "sony:r", make_scene(mag=20), npix=128
        )
        diff = (
            imsim.simulate(10.0, add_noise=False).image_clean
            - hostless.simulate(10.0, add_noise=False).image_clean
        )
        b = imsim.sim._image_render_bundle(imsim.default_psf, None, 128, 11)
        assert np.allclose(diff, 10.0 * b["extended_rate_image"], rtol=1e-6, atol=1e-9)


class TestDiffuseImage:
    def test_uniform_image_matches_scalar(self):
        """A constant 2D diffuse image reproduces the scalar per-pixel result."""
        psf = np.zeros((31, 31))
        psf[15, 15] = 1.0
        scalar = aperture_snr_radial(psf, 17.0, 1e4, 3.0, 0.1, 2.0)["snr"]
        image = aperture_snr_radial(psf, 17.0, 1e4, np.full((31, 31), 3.0), 0.1, 2.0)
        assert np.allclose(scalar, image["snr"])
