"""Tests for _count_rate_components and its relationship to the legacy get_countrates."""

import warnings
import astropy.units as u
import pytest
from tests.conftest import make_simulation


class TestCountRateComponents:
    @pytest.fixture
    def sim(self):
        return make_simulation(name="G2V", bandpass="johnson_v")

    def _legacy(self, sim):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sim.get_countrates(units="e/s", as_dict=True)

    def test_source_total_matches_pre_ee_countrate(self, sim):
        comp = sim._count_rate_components()
        legacy = self._legacy(sim)
        ee = sim.psf_profile["ee_at_aper"]
        legacy_total = (legacy["source"] / ee).to(u.electron / u.s).value
        assert comp["source_rate_total"] == pytest.approx(legacy_total, rel=1e-6)

    def test_background_per_pix_removes_spurious_ee_factor(self, sim):
        comp = sim._count_rate_components()
        legacy = self._legacy(sim)
        prof = sim.psf_profile
        n_psf = (
            prof["num_psf_pixels"].value
            if hasattr(prof["num_psf_pixels"], "value")
            else prof["num_psf_pixels"]
        )
        ee = prof["ee_at_aper"]
        legacy_bkg_per_pix = (legacy["background"] / n_psf).to(u.electron / u.s).value
        assert comp["background_rate_per_pix"] == pytest.approx(
            legacy_bkg_per_pix / ee, rel=1e-4
        )
        assert comp["background_rate_per_pix"] > 0
