"""Tests for peak_pixel_fraction on Simulation objects."""

import pytest
import wcc_etc
from tests.helpers import make_scene


class TestPeakPixelFraction:
    def test_peak_pixel_fraction_in_unit_interval(self):
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", make_scene(name="G2V", bandpass="johnson_v"))
        frac = sim.peak_pixel_fraction()
        assert 0.0 < frac <= 1.0

    def test_peak_pixel_fraction_matches_render_bundle(self):
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", make_scene(name="G2V", bandpass="johnson_v"))
        frac = sim.peak_pixel_fraction()
        b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        assert frac == pytest.approx(float(b["psf_norm"].max()), rel=1e-9)

    def test_defocus_peak_fraction_below_in_focus(self):
        sim_def = wcc_etc.Simulation.from_sensorfilter("zwo:bb2", make_scene(name="G2V", bandpass="johnson_v"))
        sim_foc = wcc_etc.Simulation.from_sensorfilter("zwo:r", make_scene(name="G2V", bandpass="johnson_v"))
        assert sim_def.peak_pixel_fraction() < 0.5 * sim_foc.peak_pixel_fraction()
