"""Tests for SDSS bandpass support (sdss_u/g/r/i/z) loaded from local data."""

import pytest
from synphot import SpectralElement

from wcc_etc.io import resolve_bandpass
from wcc_etc.scene import get_scene_element


# Peak-throughput wavelength of each SDSS primed filter, in Angstrom, as read
# from the local .dat files (these distinguish one filter from another).
SDSS_WPEAK_A = {
    "sdss_u": 3526,
    "sdss_g": 4991,
    "sdss_r": 6747,
    "sdss_i": 7878,
    "sdss_z": 10427,
}


@pytest.mark.parametrize("name,expected_a", SDSS_WPEAK_A.items())
def test_resolve_bandpass_loads_sdss_filters(name, expected_a):
    bp = resolve_bandpass(name)
    assert isinstance(bp, SpectralElement)
    # Loaded from the local .dat (Angstrom); wpeak identifies the right filter.
    assert abs(bp.wpeak().to("Angstrom").value - expected_a) < 50


def test_resolve_bandpass_is_case_insensitive():
    assert (
        resolve_bandpass("SDSS_G").avgwave().to("Angstrom").value
        == resolve_bandpass("sdss_g").avgwave().to("Angstrom").value
    )


def test_resolve_bandpass_falls_back_to_synphot_builtin():
    # Non-SDSS names defer to synphot's built-in from_filter.
    bp = resolve_bandpass("johnson_v")
    assert isinstance(bp, SpectralElement)


def test_resolve_bandpass_passes_through_spectral_element():
    bp = resolve_bandpass("sdss_r")
    assert resolve_bandpass(bp) is bp


def test_scene_element_accepts_sdss_bandpass():
    elem = get_scene_element(
        {
            "spectrum": "blackbody",
            "mag": 20,
            "magsys": "abmag",
            "teff": 5800,
            "bandpass": "sdss_g",
        }
    )
    band = elem.band
    assert isinstance(band, SpectralElement)
    assert abs(band.avgwave().to("Angstrom").value - 4700) < 400
