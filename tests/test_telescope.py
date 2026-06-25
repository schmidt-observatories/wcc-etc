import astropy.units as u
import numpy as np

from wcc_etc.telescope import Telescope


def test_telescope_properties_unit_conversion_and_values():
    tel = Telescope(f_num=8.0, diameter_primary=2.0, jitter_sigma=5)

    # diameter should be converted to meters
    assert tel.diameter_primary == 2.0 * u.m

    # jitter sigma should be converted to milliarcsec
    assert tel.jitter_sigma == 5 * u.mas

    # surface is pi*(D/2)^2
    expected_surface = np.pi * (0.5 * (2.0 * u.m)) ** 2
    assert tel.surface == expected_surface

    # focal length = diameter * f_num
    assert tel.focal_len == tel.diameter_primary * tel.f_num


def test_from_config_creates_telescope():
    cfg = {"f_num": 5.6, "diameter_primary": 1.2, "jitter_sigma": 2}
    tel = Telescope.from_config(cfg)
    assert tel.f_num == 5.6
    assert tel.diameter_primary == 1.2 * u.m
