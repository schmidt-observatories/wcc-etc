"""Tests for peak_pixel_fraction on Simulation objects."""

import pytest
import wcc_etc


def _scene(mag=15.0):
    return wcc_etc.get_scene(
        name="G2V",
        mag=mag,
        background="zodi",
        bandpass="johnson_v",
        background_prop={"bandpass": "johnson_v", "mag": 22.5},
    )


class TestPeakPixelFraction:
    def test_in_unit_interval_and_matches_render(self):
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", _scene())
        frac = sim.peak_pixel_fraction()
        assert 0.0 < frac <= 1.0
        b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        assert frac == pytest.approx(float(b["psf_norm"].max()), rel=1e-9)

    def test_defocus_peak_fraction_below_in_focus(self):
        sim_def = wcc_etc.Simulation.from_sensorfilter("zwo:bb2", _scene())
        sim_foc = wcc_etc.Simulation.from_sensorfilter("zwo:r", _scene())
        assert sim_def.peak_pixel_fraction() < 0.5 * sim_foc.peak_pixel_fraction()
