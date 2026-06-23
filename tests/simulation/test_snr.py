"""Reference SNR values for specific sensor/scene configurations."""

import pytest
import wcc_etc


class TestSNRReferenceValues:
    def test_sony_r_25p4_abmag_60s(self):
        scene = wcc_etc.get_scene(
            name="G5V",
            mag=25.4,
            magsys="abmag",
            host=None,
            background="zodi",
            bandpass="johnson_r",
            background_prop={"bandpass": "johnson_r", "mag": 22.5, "magsys": "abmag"},
        )
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        snr = sim.get_snr(time=60)["snr"]
        assert snr == pytest.approx(4.95, abs=0.1)
