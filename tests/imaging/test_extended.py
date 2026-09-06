"""Sersic host rendering: analytic normalization and grid placement."""

import numpy as np
import pytest
from astropy.modeling.models import Sersic2D
from scipy.integrate import quad

from wcc_etc.extended import render_sersic, sersic_total_over_amplitude

PLATE = 0.016869  # arcsec/pix, sony/zwo IMX455 on the WCC


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
        assert ix == pytest.approx(63.5 + 0.3 / PLATE, abs=1.0)

    def test_center_moves_profile(self):
        """center=(cx, cy) is the source position the profile is attached to."""
        img = render_sersic({"r_eff": 0.1}, PLATE, 128, center=(40.0, 90.0))
        iy, ix = np.unravel_index(np.argmax(img), img.shape)
        assert (ix, iy) == (40, 90)
