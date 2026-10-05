"""qCMOS gain modes and ADC ceiling vs full well (#94, #61)."""

import astropy.units as u
import pytest

import wcc_etc
from tests.helpers import make_scene, make_simulation


def ceiling_e(sensor):
    """Electrons at which the ADC clips: (adc_max - bias) * gain."""
    return ((sensor.adc_max - sensor.bias_level) * sensor.gain).to(u.electron).value


class TestQcmosGainModes:
    def test_default_mode_is_high_gain(self):
        s = make_simulation(sensor="qcmos:r").sensor
        assert s.gain.to(u.electron / u.ct).value == pytest.approx(1 / 7.42, rel=1e-3)

    def test_low_gain_mode_gain(self):
        s = wcc_etc.Sensor.from_name("qcmos:r", gain_mode="low")
        assert s.gain.to(u.electron / u.ct).value == pytest.approx(1 / 0.242, rel=1e-3)

    def test_low_gain_mode_read_noise_includes_margin(self):
        s = wcc_etc.Sensor.from_name("qcmos:r", gain_mode="low")
        assert s.read_noise.to(u.electron / u.pix).value == pytest.approx(2.33 * 2.0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="gain_mode"):
            wcc_etc.Sensor.from_name("qcmos:r", gain_mode="medium")

    def test_from_sensorfilter_threads_gain_mode(self):
        sim = wcc_etc.Simulation.from_sensorfilter(
            "qcmos:bb", make_scene(), gain_mode="low"
        )
        assert sim.sensor.meta["gain_mode"] == "low"


class TestAdcCeiling:
    def test_sony_adc_covers_full_well(self):
        """16-bit Sony: the ADC never clips before the well."""
        s = make_simulation(sensor="sony:r").sensor
        assert ceiling_e(s) >= s.meta["well_depth"]

    def test_qcmos_high_gain_adc_clips_before_well(self):
        """12-bit at 7.42 ADU/e clips at 552 e-, below the 7500 e- well."""
        s = make_simulation(sensor="qcmos:r").sensor
        assert ceiling_e(s) == pytest.approx(4095 / 7.42, rel=1e-3)

    def test_qcmos_low_gain_adc_covers_full_well(self):
        """12-bit at 0.242 ADU/e reaches 16,900 e-, so the 7500 e- well binds."""
        s = wcc_etc.Sensor.from_name("qcmos:r", gain_mode="low")
        assert ceiling_e(s) >= s.meta["well_depth"]

    def test_qcmos_well_depth(self):
        s = make_simulation(sensor="qcmos:r").sensor
        assert s.meta["well_depth"] == 7500


class TestFaintStarSaturation:
    """V=24 G2V, 300 s, qcmos:bb peaks at ~1400 e-: above the high-gain ADC clip, below the well."""

    def test_saturates_in_high_gain(self):
        scene = make_scene(name="G2V", mag=24, bandpass="johnson_v", host=None)
        sim = wcc_etc.Simulation.from_sensor_and_scene("qcmos:bb", scene)
        assert sim.get_snr(300)["n_saturated"] > 0

    def test_does_not_saturate_in_low_gain(self):
        scene = make_scene(name="G2V", mag=24, bandpass="johnson_v", host=None)
        sim = wcc_etc.Simulation.from_sensor_and_scene(
            "qcmos:bb", scene, gain_mode="low"
        )
        assert sim.get_snr(300)["n_saturated"] == 0
