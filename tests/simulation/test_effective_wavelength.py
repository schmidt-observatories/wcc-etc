"""Tests for the source-weighted wavelength that drives PSF rendering (issue #65)."""

import astropy.units as u
import pytest

from tests.helpers import make_broadband_simulation, make_scene


class TestSimulationEffectiveWavelength:
    def test_returns_a_wavelength_quantity(self, g5v_sim):
        """The effective wavelength carries length units."""
        assert g5v_sim.effective_wavelength.unit.physical_type == "length"

    def test_differs_from_the_filter_pivot(self, m5v_sim):
        """A red source shifts the effective wavelength off the filter pivot."""
        assert m5v_sim.effective_wavelength != m5v_sim.sensor.wavelength

    def test_red_source_is_redder_than_pivot(self, m5v_sim):
        """M5V through the broad band lands well red of the 582 nm pivot."""
        assert m5v_sim.effective_wavelength.to_value(u.nm) == pytest.approx(716, abs=5)

    def test_blue_source_is_bluer_than_pivot(self, o5v_sim):
        """O5V through the broad band lands well blue of the 582 nm pivot."""
        assert o5v_sim.effective_wavelength.to_value(u.nm) == pytest.approx(541, abs=5)

    def test_solar_type_sits_near_the_pivot(self, g5v_sim):
        """G5V is close to (but not at) the pivot wavelength."""
        assert g5v_sim.effective_wavelength.to_value(u.nm) == pytest.approx(595, abs=5)

    def test_orders_by_spectral_type(self, o5v_sim, g5v_sim, m5v_sim):
        """Effective wavelength increases monotonically from O5V to M5V."""
        assert (
            o5v_sim.effective_wavelength
            < g5v_sim.effective_wavelength
            < m5v_sim.effective_wavelength
        )

    def test_falls_back_to_pivot_without_a_source(self):
        """With no scene there is no SED, so the filter pivot is used."""
        sim = make_broadband_simulation()
        sim.set_scene(None)
        assert sim.effective_wavelength == sim.sensor.wavelength

    def test_updating_the_source_updates_the_wavelength(self, g5v_sim):
        """Changing the source spectral type invalidates the cached wavelength."""
        before = g5v_sim.effective_wavelength
        g5v_sim.set_scene(make_scene(name="M5V", mag=12))
        assert g5v_sim.effective_wavelength > before

    def test_updating_the_source_updates_the_rendered_psf(self, g5v_sim, m5v_sim):
        """The render cache is invalidated too, not just the wavelength cache."""
        g5v_sim.peak_pixel_fraction()  # prime the render cache
        g5v_sim.set_scene(make_scene(name="M5V", mag=12))
        assert g5v_sim.peak_pixel_fraction() == pytest.approx(
            m5v_sim.peak_pixel_fraction()
        )

    def test_changing_the_sensor_updates_the_wavelength(self, g5v_sim):
        """A different band on the same source moves the effective wavelength."""
        before = g5v_sim.effective_wavelength
        g5v_sim.set_sensor(make_broadband_simulation(sensor="sony:r").sensor)
        assert g5v_sim.effective_wavelength > before


class TestSourceWeightedPSF:
    def test_red_source_has_a_lower_peak_pixel_fraction(self, g5v_sim, m5v_sim):
        """A redder source spreads a wider PSF, so less flux lands on one pixel."""
        assert m5v_sim.peak_pixel_fraction() < g5v_sim.peak_pixel_fraction()

    def test_blue_source_has_a_higher_peak_pixel_fraction(self, g5v_sim, o5v_sim):
        """A bluer source concentrates more flux on the brightest pixel."""
        assert o5v_sim.peak_pixel_fraction() > g5v_sim.peak_pixel_fraction()

    def test_m5v_peak_fraction_matches_the_source_weighted_value(self, m5v_sim):
        """M5V renders at 716 nm, giving ~0.076 rather than the 0.103 at pivot."""
        assert m5v_sim.peak_pixel_fraction() == pytest.approx(0.076, abs=0.004)

    def test_spectral_types_give_distinct_psf_widths(self, o5v_sim, g5v_sim, m5v_sim):
        """O5V, G5V and M5V no longer share one PSF in the same band."""
        fractions = {sim.peak_pixel_fraction() for sim in (o5v_sim, g5v_sim, m5v_sim)}
        assert len(fractions) == 3

    def test_red_source_saturates_later(self, g5v_sim, m5v_sim):
        """The wider red PSF lowers the peak pixel, delaying saturation."""
        assert m5v_sim.get_peak_pixel(10, units="e-") < g5v_sim.get_peak_pixel(
            10, units="e-"
        )
