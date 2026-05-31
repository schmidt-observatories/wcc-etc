import numpy as np
import wcc_etc


def _sim(mag=15):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_n_reads_is_mutable_parameter_and_defaults_to_one():
    sim = _sim()
    assert sim.meta.get("n_reads") == 1
    assert any(k.endswith("n_reads") or k == "n_reads" for k in sim.mutable_parameters)
    sim.update(n_reads=4)
    assert sim.meta["n_reads"] == 4
