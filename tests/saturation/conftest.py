import pytest

import wcc_etc
from tests.helpers import make_scene


@pytest.fixture
def bright_sim():
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", make_scene(mag=12))


@pytest.fixture
def faint_sim():
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", make_scene(mag=25.4))
