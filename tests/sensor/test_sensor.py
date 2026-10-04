"""Tests for the Sensor class: property units, from_config, bit depth."""

import astropy.units as u
import pytest

from wcc_etc.sensor import Sensor
from wcc_etc.utils import parse_and_interpolate


class TestSensor:
    def _make(self, **kw):
        defaults = dict(
            bandpass=None,
            pixel_size=5,
            read_noise=3,
            dark_current=0.1,
            gain=2.0,
            area=100 * u.mm**2,
        )
        defaults.update(kw)
        return Sensor(**defaults)

    def test_pixel_size_unit(self):
        s = self._make()
        assert s.pixel_size == 5 * u.um / u.pix

    def test_read_noise_unit(self):
        s = self._make()
        assert s.read_noise == 3 * u.electron / u.pix

    def test_dark_current_unit(self):
        s = self._make()
        assert s.dark_current == 0.1 * (u.electron / (u.s * u.pix))

    def test_gain_unit(self):
        s = self._make()
        assert s.gain == 2.0 * (u.electron / u.ct)

    def test_area_unit(self):
        s = self._make()
        assert s.area == 100 * u.mm**2

    def test_get_plate_scale_matches_manual_calculation(self):
        s = self._make(pixel_size=5, area=1 * u.mm**2)

        class DummyTel:
            diameter_primary = 1.0 * u.m
            f_num = 10.0

        plate = s.get_plate_scale(DummyTel())
        expected = (
            s.pixel_size.to("m/pix")
            / DummyTel.diameter_primary.to("m")
            / DummyTel.f_num
            * 206265
            * u.arcsec
        )
        assert plate == expected

    def test_from_config_area(self):
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
        assert s.area == 50 * u.mm**2

    def test_from_config_pixel_size(self):
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
        assert s.pixel_size == 4 * u.um / u.pix

    def test_from_config_gain(self):
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
        assert s.gain == 1.5 * (u.electron / u.ct)

    def test_bit_depth_value(self):
        s = self._make(bit_depth=16)
        assert s.bit_depth == 16

    def test_adc_max_value(self):
        s = self._make(bit_depth=16)
        assert s.adc_max == 65535 * u.ct

    def test_bias_level_default_is_zero(self):
        s = self._make(bit_depth=12)
        assert s.bias_level == 0 * u.ct

    def test_adc_max_for_12bit(self):
        s = self._make(bit_depth=12)
        assert s.adc_max == 4095 * u.ct

    def test_bias_level_from_value(self):
        s = self._make(bit_depth=16, bias_level=100)
        assert s.bias_level == 100 * u.ct

    def test_update_bit_depth_changes_adc_max(self):
        s = self._make(bit_depth=16)
        s.update(bit_depth=12, bias_level=50)
        assert s.adc_max == 4095 * u.ct

    def test_update_bias_level(self):
        s = self._make(bit_depth=16)
        s.update(bit_depth=12, bias_level=50)
        assert s.bias_level == 50 * u.ct

    def test_from_config_bit_depth(self):
        cfg = {
            "throughput": None,
            "pixel_size": 4,
            "sensor_area": 50,
            "gain": 1.5,
            "read_noise": 2.5,
            "dark_current": 0.01,
            "well_depth": 30000,
            "bit_depth": 16,
            "bias_level": 5,
        }
        s = Sensor.from_config(cfg)
        assert s.bit_depth == 16

    def test_from_config_bias_level(self):
        cfg = {
            "throughput": None,
            "pixel_size": 4,
            "sensor_area": 50,
            "gain": 1.5,
            "read_noise": 2.5,
            "dark_current": 0.01,
            "well_depth": 30000,
            "bit_depth": 16,
            "bias_level": 5,
        }
        s = Sensor.from_config(cfg)
        assert s.bias_level == 5 * u.ct

    def test_from_config_adc_max(self):
        cfg = {
            "throughput": None,
            "pixel_size": 4,
            "sensor_area": 50,
            "gain": 1.5,
            "read_noise": 2.5,
            "dark_current": 0.01,
            "well_depth": 30000,
            "bit_depth": 16,
            "bias_level": 5,
        }
        s = Sensor.from_config(cfg)
        assert s.adc_max == 65535 * u.ct

    def test_adc_max_raises_without_bit_depth(self):
        s = self._make()
        with pytest.raises(ValueError):
            s.adc_max


class TestReadNoiseMargin:
    """The read-noise margin is an explicit config key, not a hidden factor (#97)."""

    def _config(self, **kw):
        return dict(
            throughput=None,
            pixel_size=5,
            sensor_area=100,
            read_noise=2.5,
            dark_current=0.1,
            gain=2.0,
            well_depth=50000,
            **kw,
        )

    def test_margin_defaults_to_one(self):
        """Without the key the config read noise is used as is."""
        s = Sensor.from_config(self._config())
        assert s.read_noise == 2.5 * u.electron / u.pix

    def test_margin_scales_config_read_noise(self):
        """read_noise_margin multiplies the configured read noise."""
        s = Sensor.from_config(self._config(read_noise_margin=2.0))
        assert s.read_noise == 5.0 * u.electron / u.pix

    def test_zwo_read_noise_is_margin_times_datasheet(self):
        """The ZWO curve value times the config margin, nothing hidden."""
        s = Sensor.from_name("zwo:r")
        datasheet = parse_and_interpolate(
            s.meta["path_read_noise"], s.meta["gain_setting"]
        )
        expected = datasheet * s.meta["read_noise_margin"]
        assert s.read_noise.to(u.electron / u.pix).value == pytest.approx(expected)

    def test_qcmos_read_noise_unchanged(self):
        """qCMOS still carries 2 x 0.28 e- after moving the factor to config."""
        s = Sensor.from_name("qcmos:bb")
        assert s.read_noise.to(u.electron / u.pix).value == pytest.approx(0.56)

    def test_get_snr_reports_effective_read_noise(
        self,
    ):
        """get_snr exposes the per-pixel read noise it used."""
        from tests.helpers import make_simulation

        sim = make_simulation(mag=16)
        rn = sim.sensor.read_noise.to(u.electron / u.pix).value
        assert sim.get_snr(10)["read_noise_e"] == pytest.approx(rn)
