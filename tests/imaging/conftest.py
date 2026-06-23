import pytest
import wcc_etc
from wcc_etc.psfsim import ImageSimulator


def _make_scene(mag=15, name="G5V", bandpass="johnson_r"):
    return wcc_etc.get_scene(
        name=name,
        mag=mag,
        host=None,
        background="zodi",
        bandpass=bandpass,
        background_prop={"bandpass": bandpass, "mag": 22.5},
    )


@pytest.fixture
def imsim():
    return ImageSimulator.from_sensor_and_scene("sony:r", _make_scene(), npix=128)


@pytest.fixture
def bright_imsim():
    return ImageSimulator.from_sensor_and_scene("sony:r", _make_scene(mag=6), npix=64)


@pytest.fixture
def faint_imsim():
    return ImageSimulator.from_sensor_and_scene("sony:r", _make_scene(mag=20), npix=128)
