"""Tests verifying the 2D count-rate migration: _image_render_bundle uses
_count_rate_components (not get_countrates / psf_profile)."""

import inspect
import warnings

import astropy.units as u
import numpy as np
import pytest

import wcc_etc
from tests.helpers import make_simulation


class TestTwodCountrateMigration:
    def test_bundle_uses_count_rate_components(self):
        src = inspect.getsource(wcc_etc.Simulation._image_render_bundle)
        assert "_count_rate_components" in src

    def test_bundle_does_not_use_ee_at_aper(self):
        src = inspect.getsource(wcc_etc.Simulation._image_render_bundle)
        assert "ee_at_aper" not in src

    def test_bundle_does_not_call_get_countrates(self):
        src = inspect.getsource(wcc_etc.Simulation._image_render_bundle)
        assert "get_countrates" not in src

    def test_bundle_source_rate_matches_components(self, sim):
        comp = sim._count_rate_components()
        b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        assert b["source_rate_total"] == pytest.approx(
            comp["source_rate_total"], rel=1e-6
        )

    def test_bundle_background_rate_matches_components(self, sim):
        comp = sim._count_rate_components()
        b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        assert b["background_rate_per_pix"] == pytest.approx(
            comp["background_rate_per_pix"], rel=1e-6
        )

    def test_bundle_diffuse_rate_matches_components(self, sim):
        comp = sim._count_rate_components()
        b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        assert b["diffuse_rate_per_pix"] == pytest.approx(
            comp["diffuse_rate_per_pix"], rel=1e-6
        )

    def test_bundle_background_higher_than_legacy(self, sim):
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
        legacy_bkg_per_pix = (legacy["background"] / n_psf).to(u.electron / u.s).value
        assert b["background_rate_per_pix"] > legacy_bkg_per_pix

    def test_bundle_background_matches_corrected_legacy(self, sim):
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
        assert b["background_rate_per_pix"] == pytest.approx(
            legacy_bkg_per_pix / ee, rel=1e-4
        )

    def test_bright_source_snr_is_positive(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr = sim.get_image_snr(time=1.0)["snr"]
        assert snr > 0

    def test_bright_source_snr_is_finite(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr = sim.get_image_snr(time=1.0)["snr"]
        assert np.isfinite(snr)
