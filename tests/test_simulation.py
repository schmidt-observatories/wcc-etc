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


import wcc_etc
import astropy.units as u


def _bright_sim(mag=20, sensor="sony:r"):
    scene = wcc_etc.get_scene(
        name='G5V', mag=mag, host=None, background="zodi",
        bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene(sensor, scene)


def test_psf_profile_has_peak_pixel_fraction():
    sim = _bright_sim()
    profile = sim.psf_profile
    assert "peak_pixel_fraction" in profile
    assert 0.0 < profile["peak_pixel_fraction"] <= 1.0


def test_get_peak_pixel_increases_with_time():
    sim = _bright_sim(mag=15)
    p10 = sim.get_peak_pixel(10, units="adu")
    p100 = sim.get_peak_pixel(100, units="adu")
    assert p100 > p10


def test_get_peak_pixel_adu_units_are_ct():
    sim = _bright_sim(mag=15)
    p = sim.get_peak_pixel(10, units="adu")
    assert p.unit == u.ct


def test_get_peak_pixel_includes_bias():
    sim = _bright_sim(mag=15)
    base = sim.get_peak_pixel(10, units="adu")
    sim.update(sensor__bias_level=100)
    biased = sim.get_peak_pixel(10, units="adu")
    assert biased.value == pytest.approx(base.value + 100, rel=1e-6)


def test_get_peak_pixel_electrons_excludes_bias():
    sim = _bright_sim(mag=15)
    e = sim.get_peak_pixel(10, units="e-")
    assert e.unit == u.electron
    assert e.value > 0


def test_get_peak_pixel_adu_matches_electron_conversion():
    sim = _bright_sim(mag=15)
    e = sim.get_peak_pixel(10, units="e-")
    adu = sim.get_peak_pixel(10, units="adu")
    expected = (e / sim.sensor.gain).to(u.ct) + sim.sensor.bias_level
    assert adu.value == pytest.approx(expected.value, rel=1e-9)


def test_get_peak_pixel_brighter_background_increases_value():
    sim = _bright_sim(mag=15)
    base = sim.get_peak_pixel(100, units="e-")
    sim.update(background__mag=18)  # lower mag = brighter background
    brighter = sim.get_peak_pixel(100, units="e-")
    assert brighter.value > base.value


def test_get_peak_pixel_accepts_array_time():
    sim = _bright_sim(mag=15)
    vals = sim.get_peak_pixel(np.array([10.0, 100.0]), units="adu")
    assert np.shape(vals) == (2,)
    assert vals[1] > vals[0]


def test_get_peak_pixel_host_increases_value():
    # a scene with a host element should yield a larger peak pixel than without
    scene_no_host = wcc_etc.get_scene(
        name='G5V', mag=15, host=None, background="zodi",
        bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})
    sim_no_host = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene_no_host)

    scene_host = wcc_etc.get_scene(
        name='G5V', mag=15, host='G5V', host_prop={"mag": 16, "bandpass": 'johnson_r'},
        background="zodi", bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})
    sim_host = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene_host)

    assert sim_host.get_peak_pixel(100, units="e-").value > sim_no_host.get_peak_pixel(100, units="e-").value


def test_is_saturated_flips_with_time():
    sim = _bright_sim(mag=8)  # bright star on a 16-bit sensor (adc_max=65535)
    assert sim.is_saturated(0.001) == False
    assert sim.is_saturated(1000) == True


def test_is_saturated_accepts_array_time():
    sim = _bright_sim(mag=8)
    result = sim.is_saturated(np.array([0.001, 1000.0]))
    assert np.shape(result) == (2,)
    assert bool(result[0]) is False
    assert bool(result[1]) is True
