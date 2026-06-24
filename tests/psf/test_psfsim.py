"""Tests for PSFSource cache_key stability and uniqueness."""

import numpy as np
from wcc_etc.psfsim import AiryPSF, CustomPSF, DefocusPSF
from wcc_etc import DEFOCUS_1WAVE_PATH


class TestPSFCacheKeys:
    def test_airy_cache_key_is_stable(self):
        assert AiryPSF().cache_key() == AiryPSF().cache_key()

    def test_airy_cache_key_value(self):
        assert AiryPSF().cache_key() == ("AiryPSF",)

    def test_custom_cache_key_is_stable(self):
        data = np.ones((9, 9))
        psf = CustomPSF(data, src_um_per_pix=4.0)
        key = psf.cache_key()
        assert psf.cache_key() == key

    def test_custom_cache_key_type(self):
        data = np.ones((9, 9))
        psf = CustomPSF(data, src_um_per_pix=4.0)
        assert psf.cache_key()[0] == "CustomPSF"

    def test_custom_cache_key_contains_scale(self):
        data = np.ones((9, 9))
        psf = CustomPSF(data, src_um_per_pix=4.0)
        assert psf.cache_key()[1] == 4.0

    def test_custom_cache_key_differs_for_distinct_objects(self):
        data = np.ones((9, 9))
        p1 = CustomPSF(data.copy(), src_um_per_pix=4.0)
        p2 = CustomPSF(data.copy(), src_um_per_pix=4.0)
        assert p1.cache_key() != p2.cache_key()

    def test_defocus_cache_key_type(self):
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
        assert psf.cache_key()[0] == "DefocusPSF"

    def test_defocus_cache_key_contains_path(self):
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
        assert psf.cache_key()[-1] == DEFOCUS_1WAVE_PATH
