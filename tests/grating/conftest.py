import pytest

from tests.helpers import make_scene
from wcc_etc.grating import GratingSimulation


@pytest.fixture(scope="module")
def blue():
    return GratingSimulation.from_sensor_and_scene("zwo", make_scene(), half="blue")


@pytest.fixture(scope="module")
def red():
    return GratingSimulation.from_sensor_and_scene("zwo", make_scene(), half="red")


@pytest.fixture(scope="module")
def blue_hwk():
    return GratingSimulation.from_sensor_and_scene("qcmos", make_scene(), half="blue")
