"""Tests for the Telescope class."""

import numpy as np
import astropy.units as u
from wcc_etc.telescope import Telescope


class TestTelescope:
    def test_properties_unit_conversion_and_values(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0, jitter_sigma=5)
        assert tel.diameter_primary == 2.0 * u.m
        assert tel.jitter_sigma == 5 * u.mas
        expected_surface = np.pi * (0.5 * (2.0 * u.m)) ** 2
        assert tel.surface == expected_surface
        assert tel.focal_len == tel.diameter_primary * tel.f_num

    def test_from_config_creates_telescope(self):
        cfg = {"f_num": 5.6, "diameter_primary": 1.2, "jitter_sigma": 2}
        tel = Telescope.from_config(cfg)
        assert tel.f_num == 5.6
        assert tel.diameter_primary == 1.2 * u.m
