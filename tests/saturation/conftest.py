import pytest
import wcc_etc


def _make_scene(mag, bandpass="johnson_r"):
    return wcc_etc.get_scene(
        name="G5V",
        mag=mag,
        background="zodi",
        bandpass=bandpass,
        background_prop={"bandpass": bandpass, "mag": 22.5},
    )


@pytest.fixture
def bright_sim():
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", _make_scene(12))


@pytest.fixture
def faint_sim():
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", _make_scene(25.4))
