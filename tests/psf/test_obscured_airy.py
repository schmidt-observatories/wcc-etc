"""Tests for the central obscuration in the Airy PSF and the simulation (#99)."""

import numpy as np
import pytest
from scipy.special import j1

import wcc_etc
from tests.helpers import make_scene
from wcc_etc.airy import get_airy_psf

D = 3.0
WAVELENGTH = 0.6e-6
# SNR captured on main (clear pupil) before the obscuration was introduced.
SNR_MAIN_SONY_BB_MAG20_10S = 76.220469819267


def airy_amplitude(x, eps):
    """Annular-pupil amplitude on a grid of x > 0 (eps = 0 is the clear pupil)."""
    amp = 2.0 * j1(x) / x
    if eps > 0:
        amp = (amp - eps**2 * 2.0 * j1(eps * x) / (eps * x)) / (1.0 - eps**2)
    return amp


def fine_rr():
    """Radial grid (rad) sampling the core at ~1/20 lambda/D out to ~4 lambda/D."""
    r = np.linspace(-4, 4, 161) * WAVELENGTH / D
    xx, yy = np.meshgrid(r, r)
    return np.hypot(xx, yy)


class TestGetAiryPsfObscured:
    def test_eps_zero_matches_clear_pupil_formula(self):
        """obscuration=0 reproduces (2 J1(x)/x)**2 exactly."""
        rr = fine_rr()
        x = (2.0 * np.pi / WAVELENGTH) * D * rr / 2.0
        nz = x != 0
        expected = np.ones_like(x)
        expected[nz] = (2.0 * j1(x[nz]) / x[nz]) ** 2
        psf = get_airy_psf(D, rr, WAVELENGTH, normalize=False)
        assert np.array_equal(psf, expected)

    def test_on_axis_unnormalized_value_is_one(self):
        """The obscured PSF is still unity on axis (x=0 limit)."""
        psf = get_airy_psf(
            D, np.array([0.0]), WAVELENGTH, normalize=False, obscuration=0.3
        )
        assert psf[0] == 1.0

    def test_obscured_peak_pixel_fraction_is_lower(self):
        """An annular pupil pushes energy into the rings, lowering the normalized peak."""
        rr = fine_rr()
        clear = get_airy_psf(D, rr, WAVELENGTH, normalize=True)
        obscured = get_airy_psf(D, rr, WAVELENGTH, normalize=True, obscuration=0.3)
        assert obscured.max() < clear.max()

    def test_first_dark_ring_moves_inward(self):
        """The first zero of the eps=0.3 amplitude lies inside the clear-pupil 3.8317."""
        x = np.linspace(1e-3, 5.0, 50001)
        amp = airy_amplitude(x, 0.3)
        first_zero = x[np.flatnonzero(np.diff(np.sign(amp)))[0]]
        assert first_zero < 3.8317


class TestSimulationObscuration:
    @pytest.fixture
    def sim(self):
        return wcc_etc.Simulation.from_sensor_and_scene("sony:bb", make_scene(mag=20))

    def test_snr_regression_clear_pupil(self, sim):
        """The default (obscuration=0) SNR is unchanged from main."""
        assert sim.get_snr(10)["snr"] == pytest.approx(SNR_MAIN_SONY_BB_MAG20_10S)

    def test_update_sets_telescope_obscuration(self, sim):
        """sim.update(obscuration=...) reaches the telescope."""
        sim.update(obscuration=0.3)
        assert sim.telescope.obscuration == 0.3

    def test_peak_pixel_drops_with_obscuration(self, sim):
        """The rendered PSF peak is lower with a central obscuration."""
        peak_clear = sim.get_peak_pixel(1.0, units="e-")
        sim.update(obscuration=0.3)
        assert sim.get_peak_pixel(1.0, units="e-") < peak_clear

    def test_source_rate_scales_with_clear_area(self, sim):
        """The total source count rate drops by 1 - eps**2 = 0.91."""
        rate_clear = sim._count_rate_components()["source_rate_total"]
        sim.update(obscuration=0.3)
        rate_obscured = sim._count_rate_components()["source_rate_total"]
        assert rate_obscured / rate_clear == pytest.approx(0.91)
