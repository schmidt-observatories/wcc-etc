"""Tests for saturation behavior with n_reads splitting."""

import numpy as np
import wcc_etc


def _sim(mag):
    scene = wcc_etc.get_scene(
        name="G5V",
        mag=mag,
        background="zodi",
        bandpass="johnson_r",
        background_prop={"bandpass": "johnson_r", "mag": 22.5},
    )
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


class TestSaturationWithReads:
    def test_n_reads_one_matches_baseline(self):
        sim = _sim(12)
        base = sim.get_peak_pixel(60, units="e-").value
        assert np.isclose(sim.get_peak_pixel(60, units="e-", n_reads=1).value, base)

    def test_peak_pixel_scales_inverse_with_reads(self):
        sim = _sim(12)
        p1 = sim.get_peak_pixel(60, units="e-", n_reads=1).value
        p2 = sim.get_peak_pixel(60, units="e-", n_reads=2).value
        assert np.isclose(p2, p1 / 2.0, rtol=1e-6)

    def test_more_reads_can_unsaturate_bright_star(self):
        sim = _sim(17)
        assert sim.is_saturated(60, n_reads=1)
        assert not sim.is_saturated(60, n_reads=100)
