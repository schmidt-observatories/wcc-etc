import warnings

import numpy as np
import pytest

import wcc_etc
from wcc_etc.psfsim import ImageSimulator, AiryPSF, saturation_mask_from_image_e


def _sim(mag):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_helper_matches_simulate_image_mask():
    # The extracted helper must reproduce simulate_image's saturation_mask exactly.
    sim = _sim(12)
    imsim = ImageSimulator(sim, npix=128, oversample=11)
    res = imsim.simulate(time=60, psf=AiryPSF(), add_noise=False)
    expected = res.saturation_mask
    got = saturation_mask_from_image_e(sim.sensor, res.image_e)
    assert np.array_equal(got, expected)
