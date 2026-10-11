"""Tests for the Telescope class."""

import astropy.units as u
import numpy as np
import pytest

from wcc_etc.io import read_config
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


class TestTelescopeObscuration:
    def test_default_obscuration_is_zero(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0)
        assert tel.obscuration == 0.0

    def test_surface_scales_by_one_minus_eps_squared(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0, obscuration=0.3)
        clear_area = np.pi * (0.5 * (2.0 * u.m)) ** 2
        assert tel.surface.to_value("m2") == pytest.approx(
            clear_area.to_value("m2") * 0.91
        )

    def test_from_lazuli_config_reads_obscuration(self):
        cfg = read_config("lazuli")["telescope"]
        tel = Telescope.from_config(cfg)
        assert tel.obscuration == cfg.get("obscuration", 0.0)

    def test_update_obscuration(self):
        tel = Telescope(f_num=8.0, diameter_primary=2.0)
        tel.update(obscuration=0.3)
        assert tel.obscuration == 0.3
