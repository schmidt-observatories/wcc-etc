"""The bundled hsiao07 SN Ia template is a 2-column phase-0 slice that loads (#102, #62)."""

import warnings

import wcc_etc
from tests.helpers import make_scene


class TestHsiao07:
    def test_scene_loads(self):
        assert wcc_etc.get_scene("hsiao07", mag=20, host=None) is not None

    def test_snr_is_positive(self):
        scene = make_scene(name="hsiao07", mag=20, host=None)
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:bb", scene)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert sim.get_snr(60)["snr"] > 0
