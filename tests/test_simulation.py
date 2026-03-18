import numpy as np
import pytest
from wcc_etc.simulation import (
    calculate_bg_normalization_magnitude,
    Simulation,
)


def test_calculate_bg_normalization_magnitude_basic():
    # area = 1 arcsec^2 should give same magnitude
    bg = 20.0
    area = 1.0
    assert calculate_bg_normalization_magnitude(bg, area) == pytest.approx(bg)


def test_calculate_bg_normalization_magnitude_scaling():
    bg = 20.0
    area = 10.0
    expected = bg - 2.5 * np.log10(area)
    assert calculate_bg_normalization_magnitude(bg, area) == pytest.approx(expected)


def test_fullkey_to_element_and_key():
    # fully qualified
    element, key = Simulation._fullkey_to_element_and_key("sensor__gain")
    assert element == "sensor"
    assert key == "gain"

    # single key returns (None, key)
    element2, key2 = Simulation._fullkey_to_element_and_key("time")
    assert element2 is None
    assert key2 == "time"


def test_get_parameter_time_default_and_mutable_parameters():
    # create a Simulation with a custom time and check get_parameter
    sim = Simulation(telescope=None, sensor=None, scene=None, time=123, r_aper_mas=70)
    val = sim.get_parameter("time")
    assert isinstance(val, list)
    assert val[0] == 123

    # mutable parameters should contain the defaults
    mp = sim.mutable_parameters
    assert "time" in mp
    assert "r_aper_mas" in mp


def test_has_element_and_setting_attribute():
    sim = Simulation(telescope=None, sensor=None, scene=None)
    assert sim.has_element("telescope") is False
    # set a dummy attribute and test
    sim._telescope = object()
    assert sim.has_element("telescope") is True
