"""Tests for get_image_exptime_for_snr (PSF-aware exposure-time inverse)."""

import numpy as np
import pytest
from tests.conftest import make_simulation


class TestImageExptimeForSnr:
    @pytest.fixture
    def sim(self):
        return make_simulation()

    def test_n_reads_one_matches_baseline(self, sim):
        base = sim.get_image_snr(time=60)["snr"]
        assert np.isclose(sim.get_image_snr(time=60, n_reads=1)["snr"], base)

    def test_roundtrips_fixed_aperture(self, sim):
        target = 80.0
        res = sim.get_image_exptime_for_snr(target, r_aper_mas=70)
        got = sim.get_image_snr(time=res["time_s"], r_aper_mas=70)["snr"]
        assert np.isclose(got, target, rtol=2e-3)

    def test_matches_analytic_for_infocus_default_aperture(self, sim):
        t_img = sim.get_image_exptime_for_snr(50.0)["time_s"]
        t_ana = sim.get_exptime_for_snr(50.0).value
        assert np.isclose(t_img, t_ana, rtol=0.02)
