"""Dispersed image, saturation and per-band SNR of the grating ETC."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

import wcc_etc
from tests.helpers import make_scene
from wcc_etc.grating import GratingSimulation
from wcc_etc.psfsim import PSFSource


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


def line_grating(wave_aa, fwhm=2.0):
    """Blue-half grating looking at one emission line at ``wave_aa``."""
    line = {"wave": wave_aa, "flux": 1e-15, "fwhm": fwhm}
    scene = wcc_etc.get_scene("emission", mag=None, lines=[line], background=None)
    return GratingSimulation.from_sensor_and_scene("zwo", scene)


def line_rate(wave_aa, fwhm=2.0):
    """Pre-grating e-/s of a single emission line at ``wave_aa``."""
    return line_grating(wave_aa, fwhm).spectral_rates()[0].sum()


class HorizontalPSF(PSFSource):
    """An 11 x 1 pixel bar along detector x."""

    def render(self, ctx):
        psf = np.zeros((ctx.npix, ctx.npix))
        c = ctx.npix // 2
        psf[c, c - 5 : c + 6] = 1 / 11
        return psf


class TestReviewRegressions:
    def test_narrow_line_flux_independent_of_position(self):
        """A 2 A line keeps its flux on and between the 1 nm bin centres."""
        assert line_rate(5005.0) == pytest.approx(line_rate(5000.0), rel=0.02)

    def test_unresolved_host_does_not_add_signal(self):
        """A bright point-like host lowers, not raises, the target SNR."""
        host = {"mag": 16, "bandpass": "johnson_r"}
        with_host = GratingSimulation.from_sensor_and_scene(
            "zwo", make_scene(mag=22, host="G5V", host_prop=host)
        )
        alone = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=22))
        assert with_host.get_band_snr(1)["g"] < alone.get_band_snr(1)["g"]

    def test_set_scene_invalidates_cache(self):
        """Replacing the scene changes the SNR to that of a fresh simulation."""
        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=22))
        g.get_band_snr(1)
        g.sim.set_scene(make_scene(mag=26))
        fresh = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=26))
        assert g.get_band_snr(1)["g"] == pytest.approx(fresh.get_band_snr(1)["g"])

    def test_band_snr_needs_first_order(self):
        """Band extraction without order +1 in the image is refused."""
        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene(), orders=(0, 2))
        with pytest.raises(ValueError, match="order \\+1"):
            g.get_band_snr(1)

    def test_solver_flags_saturated_bands(self):
        """A solved exposure that saturates the peak pixel warns."""
        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=16))
        with pytest.warns(UserWarning, match="saturates at the solved time"):
            g.get_band_exptime_for_snr(1000)

    def test_red_half_keeps_psf_orientation(self):
        """A bar PSF along detector x stays along x on the red half."""
        g = GratingSimulation.from_sensor_and_scene(
            "zwo", make_scene(), half="red", orders=(0,), psf=HorizontalPSF()
        )
        image = g.oriented(g.rate_image())
        lit = image > 0.01 * image.max()
        assert lit.any(axis=0).sum() > lit.any(axis=1).sum()

    def test_boundary_pixels_weigh_variance_squared(self):
        """Half-covered pixels carry a quarter of the read-noise variance."""
        scale = GratingSimulation.from_sensor_and_scene(
            "zwo", make_scene()
        ).plate_scale_mas
        half, full = (
            GratingSimulation.from_sensor_and_scene(
                "zwo", make_scene(), aperture_mas=k * scale
            ).band_coefficients()["g"][2]
            for k in (0.25, 0.5)
        )
        assert half / full == pytest.approx(0.25)

    def test_zero_reads_rejected(self):
        """n_reads = 0 raises instead of returning a finite SNR."""
        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene())
        with pytest.raises(ValueError, match="n_reads"):
            g.get_band_snr(1, n_reads=0)


class TestAdversarialReviewRegressions:
    @pytest.mark.parametrize("wave_aa", [5000.0, 5000.25])
    def test_very_narrow_line_matches_synphot(self, wave_aa):
        """A 0.2 A line gives the synphot-integrated rate wherever it falls."""
        g = line_grating(wave_aa, fwhm=0.2)
        expected = g.sim._count_rate_components()["source_rate_total"]
        assert g.spectral_rates()[0].sum() == pytest.approx(expected, rel=0.02)

    @pytest.mark.parametrize(
        "method, kwargs",
        [
            ("update_scene", {"source__mag": 26}),
            ("update_sensor", {"read_noise": 5.0}),
            ("update_telescope", {"jitter": 30.0}),
        ],
    )
    def test_component_update_invalidates_cache(self, method, kwargs):
        """Updating a component through the simulation drops the cached rates."""
        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=22))
        g.get_band_snr(1)
        getattr(g.sim, method)(**kwargs)
        assert "rates" not in g.cache

    def test_update_scene_matches_fresh_simulation(self):
        """A warm grating follows update_scene to the new magnitude."""
        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=22))
        g.get_band_snr(1)
        g.sim.update_scene(source__mag=26)
        fresh = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=26))
        assert g.get_band_snr(1)["g"] == pytest.approx(fresh.get_band_snr(1)["g"])

    def test_gratings_on_one_simulation_do_not_share_images(self, blue):
        """A second grating on the same simulation renders its own canvas."""
        GratingSimulation(blue.sim, half="red").rate_image()
        other = GratingSimulation(blue.sim, half="blue")
        assert other.rate_image().shape == other.canvas()[:2]

    def test_single_saturated_frame_is_masked(self):
        """A pixel that saturates in one frame is masked in the stack."""
        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene(mag=10))
        sat = g.sim.sensor.saturation_e.value
        n = 10
        t = n * (sat - np.sqrt(sat)) / g.get_peak_pixel(1.0).value
        image = g.simulate(t, n_reads=n, seed=0)
        assert image.saturation_mask.any() and image.image_e.max() / n < sat

    def test_set_scene_with_profiled_host_raises(self):
        """An extended host installed after construction is still refused."""
        from tests.helpers import make_host_scene

        g = GratingSimulation.from_sensor_and_scene("zwo", make_scene())
        g.sim.set_scene(make_host_scene())
        with pytest.raises(NotImplementedError):
            g.get_band_snr(1)
