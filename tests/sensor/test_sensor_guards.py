"""Sensor input guards: qe, sensor_temp, and clamped lookup tables (#102)."""

import warnings

import astropy.units as u
import pytest

import wcc_etc
from wcc_etc.sensor import Sensor
from wcc_etc.utils import parse_and_interpolate


def zwo_sensor_config():
    return wcc_etc.get_sensor_config("zwo", "r")["sensor"]


class TestQeGuard:
    def test_qe_not_one_raises(self):
        """QE lives in the throughput file; a separate qe is never applied."""
        with pytest.raises(ValueError, match="throughput"):
            Sensor(
                bandpass=None,
                pixel_size=5,
                read_noise=3,
                dark_current=0.1,
                gain=2.0,
                area=100 * u.mm**2,
                qe=0.8,
            )

    def test_qe_one_is_accepted(self):
        """from_config passes qe=1 and must keep working."""
        s = Sensor(
            bandpass=None,
            pixel_size=5,
            read_noise=3,
            dark_current=0.1,
            gain=2.0,
            area=100 * u.mm**2,
            qe=1,
        )
        assert s.meta["qe"] == 1


class TestSensorTempGuard:
    def test_missing_sensor_temp_raises_value_error(self):
        """A zwo config without sensor_temp fails clearly, not with AttributeError."""
        cfg = zwo_sensor_config()
        del cfg["sensor_temp"]
        with pytest.raises(ValueError, match="sensor_temp is required"):
            Sensor.from_config(cfg)


class TestTableRangeWarning:
    def test_dark_current_outside_table_warns(self):
        """A temperature beyond the ZWO table range warns that it is clamped."""
        cfg = zwo_sensor_config()
        with pytest.warns(UserWarning, match="table range"):
            parse_and_interpolate(cfg["path_dark_current"], 500.0)

    def test_warning_names_the_table_file(self):
        """The warning says which table was clamped."""
        cfg = zwo_sensor_config()
        with pytest.warns(UserWarning, match="Dark_Current"):
            parse_and_interpolate(cfg["path_dark_current"], 500.0)

    def test_inside_table_does_not_warn(self):
        """The bundled sensor_temp is inside the table: silent."""
        cfg = zwo_sensor_config()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parse_and_interpolate(cfg["path_dark_current"], cfg["sensor_temp"])
