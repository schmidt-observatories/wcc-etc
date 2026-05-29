import pytest
import astropy.units as u
from wcc_etc.sensor import Sensor


def test_sensor_property_units_and_values():
    s = Sensor(bandpass=None,
               pixel_size=5,   # um
               read_noise=3,   # e-
               dark_current=0.1, # e-/s
               gain=2.0,       # e-/ct
               area=100 * u.mm**2,
               temperature=None)

    assert s.pixel_size == 5 * u.um / u.pix
    assert s.read_noise == 3 * u.electron / u.pix
    assert s.dark_current == 0.1 * (u.electron / (u.s * u.pix))
    assert s.gain == 2.0 * (u.electron / u.ct)
    assert s.area == 100 * u.mm**2


def test_get_plate_scale_matches_manual_calculation():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=1, dark_current=0.0, gain=1.0, area=1*u.mm**2)

    class DummyTel:
        pass

    tel = DummyTel()
    tel.diameter_primary = 1.0 * u.m
    tel.f_num = 10.0

    plate = s.get_plate_scale(tel)

    expected = (s.pixel_size.to("m/pix") / tel.diameter_primary.to("m") / tel.f_num * 206265 * u.arcsec)
    # compare as quantities
    assert plate == expected


def test_from_config_minimal():
    cfg = {
        "throughput": None,
        "pixel_size": 4,
        "sensor_area": 50,
        "gain": 1.5,
        "read_noise": 2.5,
        "dark_current": 0.01,
        "well_depth": 30000,
    }

    s = Sensor.from_config(cfg)
    # area converted to mm^2
    assert s.area == 50 * u.mm**2
    assert s.pixel_size == 4 * u.um / u.pix
    assert s.gain == 1.5 * (u.electron / u.ct)


def test_sensor_bit_depth_and_adc_max():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=16)
    assert s.bit_depth == 16
    assert s.adc_max == 65535 * u.ct


def test_sensor_bias_level_defaults_to_zero():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=12)
    assert s.bias_level == 0 * u.ct
    assert s.adc_max == 4095 * u.ct


def test_sensor_bias_level_from_value():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=16, bias_level=100)
    assert s.bias_level == 100 * u.ct


def test_bit_depth_and_bias_level_are_updatable():
    s = Sensor(bandpass=None, pixel_size=5, read_noise=3, dark_current=0.1,
               gain=2.0, area=100 * u.mm**2, bit_depth=16)
    s.update(bit_depth=12, bias_level=50)
    assert s.adc_max == 4095 * u.ct
    assert s.bias_level == 50 * u.ct


def test_from_config_reads_bit_depth_and_bias_level():
    cfg = {
        "throughput": None, "pixel_size": 4, "sensor_area": 50,
        "gain": 1.5, "read_noise": 2.5, "dark_current": 0.01,
        "well_depth": 30000, "bit_depth": 16, "bias_level": 5,
    }
    s = Sensor.from_config(cfg)
    assert s.bit_depth == 16
    assert s.bias_level == 5 * u.ct
    assert s.adc_max == 65535 * u.ct
