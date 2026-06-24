"""Tests for saturation behavior with n_reads splitting."""

import numpy as np
from tests.helpers import make_simulation


class TestSaturationWithReads:
    def test_n_reads_one_matches_baseline(self):
        sim = make_simulation(mag=12)
        base = sim.get_peak_pixel(60, units="e-").value
        assert np.isclose(sim.get_peak_pixel(60, units="e-", n_reads=1).value, base)

    def test_peak_pixel_scales_inverse_with_reads(self):
        sim = make_simulation(mag=12)
        p1 = sim.get_peak_pixel(60, units="e-", n_reads=1).value
        p2 = sim.get_peak_pixel(60, units="e-", n_reads=2).value
        assert np.isclose(p2, p1 / 2.0, rtol=1e-6)

    def test_one_read_saturates(self):
        sim = make_simulation(mag=17)
        assert sim.is_saturated(60, n_reads=1)

    def test_many_reads_unsaturate(self):
        sim = make_simulation(mag=17)
        assert not sim.is_saturated(60, n_reads=100)
