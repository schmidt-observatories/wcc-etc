"""An sdss_* normalization band without an explicit magsys warns (SDSS mags are AB) (#102)."""

import warnings

import pytest

from wcc_etc.scene import get_scene_element


class TestSdssMagsysGuard:
    def test_sdss_band_without_magsys_warns(self):
        """Default magsys with an SDSS band warns that abmag is probably intended."""
        with pytest.warns(UserWarning, match="abmag"):
            get_scene_element("G5V", mag=18, bandpass="sdss_r")

    def test_sdss_band_default_resolves_to_vegamag(self):
        """The stored magsys is still 'vegamag' when nothing is passed."""
        with pytest.warns(UserWarning):
            el = get_scene_element("G5V", mag=18, bandpass="sdss_r")
        assert el.meta["magsys"] == "vegamag"

    def test_explicit_vegamag_does_not_warn(self):
        """An explicit magsys='vegamag' is a deliberate choice: no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_scene_element("G5V", mag=18, bandpass="sdss_r", magsys="vegamag")

    def test_explicit_abmag_does_not_warn(self):
        """magsys='abmag' with an SDSS band is the intended usage."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_scene_element("G5V", mag=18, bandpass="sdss_r", magsys="abmag")

    def test_non_sdss_band_default_does_not_warn(self):
        """Johnson bands keep the silent vegamag default."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            get_scene_element("G5V", mag=18, bandpass="johnson_v")

    def test_non_sdss_band_default_stores_vegamag(self):
        """None resolves to 'vegamag' in meta, as before."""
        el = get_scene_element("G5V", mag=18, bandpass="johnson_v")
        assert el.meta["magsys"] == "vegamag"
