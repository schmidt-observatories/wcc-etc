import numpy as np
from wcc_etc.psfsim import AiryPSF, CustomPSF, DefocusPSF
from wcc_etc import DEFOCUS_1WAVE_PATH


def test_airy_cache_key_is_constant():
    assert AiryPSF().cache_key() == AiryPSF().cache_key()
    assert AiryPSF().cache_key() == ("AiryPSF",)


def test_resampled_cache_key_stable_per_object():
    data = np.ones((9, 9))
    psf = CustomPSF(data, src_um_per_pix=4.0)
    key = psf.cache_key()
    assert psf.cache_key() == key          # stable across calls
    assert key[0] == "CustomPSF"
    assert key[1] == 4.0


def test_resampled_cache_key_differs_for_distinct_objects():
    data = np.ones((9, 9))
    p1 = CustomPSF(data.copy(), src_um_per_pix=4.0)
    p2 = CustomPSF(data.copy(), src_um_per_pix=4.0)
    assert p1.cache_key() != p2.cache_key()


def test_defocus_cache_key_uses_path():
    psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
    assert psf.cache_key()[0] == "DefocusPSF"
    assert psf.cache_key()[-1] == DEFOCUS_1WAVE_PATH
