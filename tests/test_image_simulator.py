import numpy as np
import pytest
from wcc_etc.psfsim import SimulatedImage, AiryPSF, FitsImg


def _make_simimg():
    image_e = np.array([[100.0, 200.0], [300.0, 400.0]])
    clean = image_e.copy()
    sat = np.array([[False, False], [False, True]])
    return SimulatedImage(image_e=image_e, image_clean=clean, saturation_mask=sat,
                          gain=2.0, bias_level=100.0, npix=2,
                          pixel_scale_mas=20.0, psf=AiryPSF())


def test_simulated_image_to_adu():
    s = _make_simimg()
    adu = s.to_adu()
    assert np.allclose(adu, s.image_e / 2.0 + 100.0)


def test_simulated_image_to_fitsimg():
    s = _make_simimg()
    f = s.to_fitsimg()
    assert isinstance(f, FitsImg)
    assert np.allclose(f.data, s.image_e)
