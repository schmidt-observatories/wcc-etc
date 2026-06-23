"""Tests that legacy analytic methods emit DeprecationWarning."""

import warnings
import pytest
from tests.conftest import make_simulation


class TestAnalyticDeprecation:
    @pytest.fixture
    def sim(self):
        return make_simulation(name="G2V", bandpass="johnson_v")

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.get_countrates(units="e/s"),
            lambda s: s.compute_psf_profile(),
            lambda s: s.get_signal_and_variance(1.0),
            lambda s: s.get_exptime_for_snr(50.0),
        ],
    )
    def test_method_warns(self, sim, call):
        with pytest.warns(DeprecationWarning):
            call(sim)

    def test_peak_pixel_fraction_retired(self, sim):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = sim.compute_psf_profile()
        assert "peak_pixel_fraction" not in prof
