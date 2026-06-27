"""Tests for PSF-aware saturation: defocus peak fraction and is_saturated consistency."""

import warnings

import astropy.units as u
import numpy as np
import pytest

import wcc_etc
from tests.helpers import make_scene


class TestPsfAwareSaturation:
    def test_defocus_peak_fraction_far_below_airy(self):
        sim_def = wcc_etc.Simulation.from_sensorfilter(
            "zwo:bb2", make_scene(name="G2V", bandpass="johnson_v", mag=15.0)
        )
        b = sim_def._image_render_bundle(sim_def._default_psf, None, 128, 11)
        airy_b = sim_def._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
        assert b["psf_norm"].max() < 0.2 * airy_b["psf_norm"].max()

    def test_is_saturated_agrees_with_image_snr_flag(self):
        sim = wcc_etc.Simulation.from_sensorfilter(
            "zwo:bb2", make_scene(name="G2V", bandpass="johnson_v", mag=13.0)
        )
        for t in [0.05, 0.1, 0.5, 2.0]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flag_snr = bool(sim.get_image_snr(time=t)["saturated"])
                flag_sat = bool(sim.is_saturated(t))
            assert flag_sat == flag_snr, f"disagreement at t={t}"

    def test_peak_pixel_array_shape(self):
        sim = wcc_etc.Simulation.from_sensorfilter(
            "zwo:r", make_scene(name="G2V", bandpass="johnson_v", mag=16.0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p2 = (
                sim.get_peak_pixel(np.array([1.0, 2.0]), units="e-")
                .to(u.electron)
                .value
            )
        assert p2.shape == (2,)

    def test_peak_pixel_scalar_matches_array_element(self):
        sim = wcc_etc.Simulation.from_sensorfilter(
            "zwo:r", make_scene(name="G2V", bandpass="johnson_v", mag=16.0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1 = sim.get_peak_pixel(1.0, units="e-").to(u.electron).value
            p2 = (
                sim.get_peak_pixel(np.array([1.0, 2.0]), units="e-")
                .to(u.electron)
                .value
            )
        assert p2[0] == pytest.approx(p1, rel=1e-6)

    def test_peak_pixel_scales_linearly_with_time(self):
        sim = wcc_etc.Simulation.from_sensorfilter(
            "zwo:r", make_scene(name="G2V", bandpass="johnson_v", mag=16.0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1 = sim.get_peak_pixel(1.0, units="e-").to(u.electron).value
            p2 = (
                sim.get_peak_pixel(np.array([1.0, 2.0]), units="e-")
                .to(u.electron)
                .value
            )
        assert p2[1] == pytest.approx(2 * p1, rel=1e-6)
