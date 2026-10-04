"""Reference SNR values for specific sensor/scene configurations."""

import numpy as np
import pytest

import wcc_etc
from tests.helpers import make_scene
from wcc_etc.io import _SENSORFILTER_FOCUS, _SENSORFILTER_IMPLEMENTED
from wcc_etc.simulation import DEFAULT_R_APER_MAS


class TestSNRReferenceValues:
    def test_sony_r_25p4_abmag_60s(self):
        """Verify reference SNR for a 25.4 AB-mag star in 60s on sony:r.

        Allow a small tolerance.
        """
        scene = wcc_etc.get_scene(
            name="G5V",
            mag=25.4,
            magsys="abmag",
            host=None,
            background="zodi",
            bandpass="johnson_r",
            background_prop={"bandpass": "johnson_r", "mag": 22.5, "magsys": "abmag"},
        )

        # Define the sensor and filter combination
        sensor_and_filter = "sony:r"
        simu = wcc_etc.Simulation.from_sensor_and_scene(sensor_and_filter, scene)

        snr_val = simu.get_snr(time=60)["snr"]

        print(
            f"SNR for 25.4 AB-mag star in 60s with {sensor_and_filter}: {snr_val:.2f}"
        )

        # AB-mag SNR (the scene pins magsys="abmag", so this is unaffected by the
        # vegamag default). Expected ~4.95: the noise budget now includes the sky
        # shot noise and the PSF is rendered at the pivot wavelength (not peak
        # transmission). The independent closed-form/Monte-Carlo cross-checks of the
        # sky term live in test_sky_background_noise.py.
        assert snr_val == pytest.approx(4.95, abs=0.1)


DEFOCUSED = sorted(
    label
    for label, focus in _SENSORFILTER_FOCUS.items()
    if focus != "0wave" and _SENSORFILTER_IMPLEMENTED[label]
)


class TestDefaultAperture:
    """With no aperture argument, the aperture follows the PSF (#95)."""

    def test_in_focus_default_is_airy_core(self):
        """An in-focus sensorfilter keeps the 70 mas Airy-core aperture."""
        sim = wcc_etc.Simulation.from_sensorfilter("zwo:r", make_scene(mag=16))
        assert sim.default_aperture(sim.default_psf) == (DEFAULT_R_APER_MAS, False)

    @pytest.mark.parametrize("label", DEFOCUSED)
    def test_defocused_default_matches_optimize(self, label):
        """Every defocused sensorfilter gives the SNR-optimized aperture by default."""
        sim = wcc_etc.Simulation.from_sensorfilter(label, make_scene(mag=16))
        default, best = sim.get_snr(10), sim.get_snr(10, optimize=True)
        assert default["snr"] == best["snr"]

    @pytest.mark.parametrize("label", DEFOCUSED)
    def test_defocused_default_encloses_most_of_the_flux(self, label):
        """The default aperture holds at least 80% of a bright source's flux."""
        sim = wcc_etc.Simulation.from_sensorfilter(label, make_scene(mag=16))
        assert sim.get_snr(10)["enclosed_fraction"] >= 0.8

    def test_defocused_exptime_default_matches_optimize(self):
        """The exposure-time inverse follows the same PSF-aware default."""
        sim = wcc_etc.Simulation.from_sensorfilter("zwo:r+1", make_scene(mag=16))
        default = sim.get_image_exptime_for_snr(100)["time_s"]
        best = sim.get_image_exptime_for_snr(100, optimize=True)["time_s"]
        assert default == best

    def test_explicit_simulation_aperture_wins(self):
        """A Simulation built with r_aper_mas keeps that radius for any PSF."""
        sim = wcc_etc.Simulation.from_sensorfilter("zwo:r+1", make_scene(mag=16))
        sim.update(r_aper_mas=DEFAULT_R_APER_MAS)
        pinned = sim.get_snr(10, r_aper_mas=DEFAULT_R_APER_MAS)["snr"]
        assert np.isclose(sim.get_snr(10)["snr"], pinned)

    def test_meta_has_no_aperture_unless_set(self):
        """r_aper_mas is only in meta when a caller set it."""
        sim = wcc_etc.Simulation.from_sensorfilter("zwo:r", make_scene(mag=16))
        assert "r_aper_mas" not in sim.meta
