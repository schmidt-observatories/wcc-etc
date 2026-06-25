import pytest

from tests.helpers import make_scene
from wcc_etc.psfsim import ImageSimulator


@pytest.fixture
def imsim():
    return ImageSimulator.from_sensor_and_scene("sony:r", make_scene(), npix=128)


@pytest.fixture
def bright_imsim():
    return ImageSimulator.from_sensor_and_scene("sony:r", make_scene(mag=6), npix=64)


@pytest.fixture
def faint_imsim():
    return ImageSimulator.from_sensor_and_scene("sony:r", make_scene(mag=20), npix=128)
