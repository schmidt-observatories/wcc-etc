"""Dispersed image, saturation and per-band SNR of the grating ETC."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from tests.helpers import make_scene
from wcc_etc.grating import GratingSimulation


class TestRateImage:
    def test_conserves_flux(self, blue):
        """The image holds the source rate times the efficiency of its orders."""
        point, _ = blue.spectral_rates()
        expected = sum(
            trapezoid(point * blue.efficiency(m), blue.wave_nm) for m in blue.orders
        )
        assert blue.rate_image().sum() == pytest.approx(expected, rel=0.02)

    def test_first_order_lands_on_its_trace(self, blue):
        """The brightest first-order column lies between 350 and 700 nm."""
        _, _, x0, _ = blue.canvas()
        column = blue.rate_image(orders=(1,)).sum(axis=0).argmax() - x0
        assert blue.offset_px(350) < column < blue.offset_px(700)

    def test_zeroth_order_at_origin(self, blue):
        """The zeroth-order image sits on the canvas origin pixel."""
        _, _, x0, y0 = blue.canvas()
        image = blue.rate_image(orders=(0,))
        assert np.unravel_index(image.argmax(), image.shape) == (y0, x0)

    def test_sky_sees_whole_passband(self, blue):
        """Zodiacal light fills every pixel of the slitless image."""
        assert blue.background_rate_per_pix() > 0


class TestSimulate:
    def test_blue_disperses_along_rows(self, blue):
        """The blue half image is wider than tall."""
        ny, nx = blue.simulate(10, add_noise=False).image_e.shape
        assert nx > ny

    def test_red_disperses_along_columns(self, red):
        """The red half image is taller than wide."""
        ny, nx = red.simulate(10, add_noise=False).image_e.shape
        assert ny > nx

    def test_seed_is_reproducible(self, blue):
        """The same seed gives the same noisy image."""
        a, b = blue.simulate(10, seed=1), blue.simulate(10, seed=1)
        assert np.array_equal(a.image_e, b.image_e)

    def test_bright_star_saturates(self):
        """A V = 6 star saturates a 1 s frame."""
        sim = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=6))
        assert sim.is_saturated(1.0)

    def test_reads_lower_the_peak(self, blue):
        """The peak pixel is per frame, so more reads lower it."""
        assert blue.get_peak_pixel(10, n_reads=10) < blue.get_peak_pixel(10)


class TestBandSNR:
    def test_blue_bands(self, blue):
        """The blue half measures u, g and r."""
        assert list(blue.get_band_snr(1e-3)) == ["u", "g", "r"]

    def test_red_bands(self, red):
        """The red half measures i and z."""
        assert list(red.get_band_snr(1e-3)) == ["i", "z"]

    def test_snr_grows_with_time(self, blue):
        """A longer exposure gives a higher SNR."""
        assert blue.get_band_snr(4e-3)["g"] > blue.get_band_snr(1e-3)["g"]

    def test_exptime_round_trip(self, blue):
        """The time solved for SNR = 10 gives SNR = 10."""
        t = blue.get_band_exptime_for_snr(10)["r"]
        assert blue.get_band_snr(t)["r"] == pytest.approx(10, rel=1e-6)

    def test_hwk_runs(self, blue_hwk):
        """The same grating works on the HWK4123 sensor."""
        assert blue_hwk.get_band_snr(1e-3)["g"] > 0

    def test_profiled_host_not_supported(self):
        """An extended host is rejected rather than silently dropped."""
        from tests.helpers import make_host_scene

        with pytest.raises(NotImplementedError):
            GratingSimulation.from_sensor_and_scene("zwo", make_host_scene())
