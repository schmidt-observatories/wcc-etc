"""ADC ceiling vs full well for the bundled sensors (#94, #61)."""

import astropy.units as u
import pytest

import wcc_etc
from tests.helpers import make_scene, make_simulation


class TestAdcCeiling:
    @pytest.mark.parametrize("sensor", ["sony:r", "qcmos:r"])
    def test_adc_ceiling_in_electrons_covers_full_well(self, sensor):
        """(2**bit_depth - 1 - bias) * gain must reach the well, else the ADC clips first."""
        s = make_simulation(sensor=sensor).sensor
        ceiling_e = (s.adc_max - s.bias_level) * s.gain
        assert ceiling_e >= s.meta["well_depth"] * u.electron

    def test_qcmos_gain_is_inverse_of_measured_adu_per_e(self):
        """MIT measured 8.9 ADU/e; the config stores e/ADU."""
        s = make_simulation(sensor="qcmos:r").sensor
        assert s.gain.to(u.electron / u.ct).value == pytest.approx(1 / 8.9, rel=0.01)

    def test_faint_qcmos_star_is_not_saturated(self):
        """V=25 G2V, 300 s, qcmos:bb peaks at ~560 e- (5010 ADU): below the 7000 e- well, above a 12-bit ADC."""
        scene = make_scene(name="G2V", mag=25, bandpass="johnson_v", host=None)
        sim = wcc_etc.Simulation.from_sensor_and_scene("qcmos:bb", scene)
        assert sim.get_snr(300)["n_saturated"] == 0
