import numpy as np
from wcc_etc.psfsim import AiryPSF, CustomPSF, DefocusPSF
from wcc_etc import DEFOCUS_1WAVE_PATH


def test_airy_cache_key_is_constant():
    assert AiryPSF().cache_key() == AiryPSF().cache_key()
    assert AiryPSF().cache_key() == ("AiryPSF",)


def test_resampled_cache_key_stable_per_object():
    data = np.ones((9, 9))
    psf = CustomPSF(data, src_um_per_pix=4.0)
    assert psf.cache_key() == psf.cache_key()              # stable across calls
    assert psf.cache_key()[0] == "CustomPSF"
    assert psf.cache_key()[1] == 4.0


def test_defocus_cache_key_uses_path():
    psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
    assert psf.cache_key()[0] == "DefocusPSF"
    assert psf.cache_key()[-1] == DEFOCUS_1WAVE_PATH
