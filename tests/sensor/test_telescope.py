"""Tests for the Telescope class."""

import numpy as np
import astropy.units as u
from wcc_etc.telescope import Telescope


class TestTelescope:
    def test_diameter_primary_unit(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0, jitter_sigma=5)
        assert tel.diameter_primary == 2.0 * u.m

    def test_jitter_sigma_unit(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0, jitter_sigma=5)
        assert tel.jitter_sigma == 5 * u.mas

    def test_surface_area(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0, jitter_sigma=5)
        expected_surface = np.pi * (0.5 * (2.0 * u.m)) ** 2
        assert tel.surface == expected_surface

    def test_focal_length(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0, jitter_sigma=5)
        assert tel.focal_len == tel.diameter_primary * tel.f_num

    def test_from_config_f_num(self):
        cfg = {"f_num": 5.6, "diameter_primary": 1.2, "jitter_sigma": 2}
        tel = Telescope.from_config(cfg)
        assert tel.f_num == 5.6

    def test_from_config_diameter(self):
        cfg = {"f_num": 5.6, "diameter_primary": 1.2, "jitter_sigma": 2}
        tel = Telescope.from_config(cfg)
        assert tel.diameter_primary == 1.2 * u.m
