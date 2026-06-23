"""Tests verifying the 2D count-rate migration: _image_render_bundle uses
_count_rate_components (not get_countrates / psf_profile)."""

import inspect
import warnings
import numpy as np
import pytest
import astropy.units as u
import wcc_etc
from tests.conftest import make_simulation


class TestTwodCountrateMigration:
    @pytest.fixture
    def sim(self):
        return make_simulation()

    def test_bundle_uses_count_rate_components_not_psf_profile(self):
        src = inspect.getsource(wcc_etc.Simulation._image_render_bundle)
        assert "_count_rate_components" in src
        assert "ee_at_aper" not in src
        assert "get_countrates" not in src

    def test_bundle_rates_match_components(self, sim):
        comp = sim._count_rate_components()
        b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        assert b["source_rate_total"] == pytest.approx(
            comp["source_rate_total"], rel=1e-6
        )
        assert b["background_rate_per_pix"] == pytest.approx(
            comp["background_rate_per_pix"], rel=1e-6
        )
        assert b["diffuse_rate_per_pix"] == pytest.approx(
            comp["diffuse_rate_per_pix"], rel=1e-6
        )

    def test_bundle_background_higher_than_legacy_airy(self, sim):
        b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy = sim.get_countrates(units="e/s", as_dict=True)
        prof = sim.psf_profile
        n_psf = (
            prof["num_psf_pixels"].value
            if hasattr(prof["num_psf_pixels"], "value")
            else prof["num_psf_pixels"]
        )
        ee = prof["ee_at_aper"]
        legacy_bkg_per_pix = (legacy["background"] / n_psf).to(u.electron / u.s).value
        assert b["background_rate_per_pix"] > legacy_bkg_per_pix
        assert b["background_rate_per_pix"] == pytest.approx(
            legacy_bkg_per_pix / ee, rel=1e-4
        )

    def test_bright_source_snr_essentially_unchanged(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr = sim.get_image_snr(time=1.0)["snr"]
        assert snr > 0 and np.isfinite(snr)
