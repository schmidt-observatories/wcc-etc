"""One per-frame saturation ceiling shared by every saturation API (#64)."""

import warnings

import astropy.units as u
import numpy as np
import pytest

import wcc_etc
from tests.helpers import make_scene, make_simulation
from wcc_etc.psfsim import AiryPSF, ImageSimulator, saturation_mask_from_image_e


def sony():
    return make_simulation(sensor="sony:r").sensor


class TestSensorSaturationE:
    def test_sony_binds_on_well(self):
        """16-bit Sony: the well (~16.3 ke-) is below the ADC ceiling (~17.1 ke-)."""
        s = sony()
        assert s.saturation_e.to(u.electron).value == pytest.approx(
            s.meta["well_depth"]
        )

    def test_sony_limit_is_well(self):
        assert sony().saturation_limit == "well"

    def test_qcmos_high_gain_binds_on_adc(self):
        """12-bit at 7.42 ADU/e clips at 552 e-, below the 7500 e- well."""
        s = make_simulation(sensor="qcmos:r").sensor
        assert s.saturation_e.to(u.electron).value == pytest.approx(
            4095 / 7.42, rel=1e-3
        )

    def test_qcmos_high_gain_limit_is_adc(self):
        assert make_simulation(sensor="qcmos:r").sensor.saturation_limit == "adc"

    def test_qcmos_low_gain_limit_is_well(self):
        s = wcc_etc.Sensor.from_name("qcmos:r", gain_mode="low")
        assert s.saturation_limit == "well"

    def test_bias_reduces_adc_headroom(self):
        """Bias eats ADC range: the ceiling drops by bias * gain."""
        s = make_simulation(sensor="qcmos:r").sensor
        before = s.saturation_e.to(u.electron).value
        s.meta["bias_level"] = 100
        after = s.saturation_e.to(u.electron).value
        assert before - after == pytest.approx(100 * s.gain.value)

    def test_no_well_depth_falls_back_to_adc(self):
        s = sony()
        s.meta["well_depth"] = None
        assert s.saturation_e.to(u.electron).value == pytest.approx(
            s.adc_max.value * s.gain.value
        )


class TestMaskUsesCeiling:
    def test_sony_between_well_and_adc_is_saturated(self):
        s = sony()
        mid = 0.5 * (s.meta["well_depth"] + s.adc_max.value * s.gain.value)
        assert saturation_mask_from_image_e(s, np.array([mid]))[0]

    def test_mask_respects_bias(self):
        s = make_simulation(sensor="qcmos:r").sensor
        s.meta["bias_level"] = 1000
        just_below_unbiased = s.adc_max.value * s.gain.value - 1.0
        assert saturation_mask_from_image_e(s, np.array([just_below_unbiased]))[0]


class TestCrossApiAgreement:
    """is_saturated, the image mask, and simulate agree on the same clean frame."""

    @pytest.fixture
    def sim(self):
        return wcc_etc.Simulation.from_sensor_and_scene(
            "sony:r", make_scene(name="G2V", bandpass="johnson_v", mag=13.0)
        )

    @pytest.mark.parametrize("t", [0.02, 0.05, 0.1, 0.5])
    def test_is_saturated_matches_simulate_mask(self, sim, t):
        imsim = ImageSimulator(sim, npix=128, oversample=11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = imsim.simulate(time=t, psf=AiryPSF(), add_noise=False)
            flag = bool(sim.is_saturated(t, psf=AiryPSF()))
        assert flag == bool(img.saturation_mask.any())

    def test_is_saturated_flags_sony_in_well_adc_gap(self, sim):
        """Peak between the well and the ADC ceiling must read as saturated."""
        s = sim.sensor
        well = s.meta["well_depth"]
        adc_e = s.adc_max.value * s.gain.value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate = sim.get_peak_pixel(1.0, units="e-").to(u.electron).value
            t = 0.5 * (well + adc_e) / rate
            peak = sim.get_peak_pixel(t, units="e-").to(u.electron).value
            assert well < peak < adc_e
            assert bool(sim.is_saturated(t))
