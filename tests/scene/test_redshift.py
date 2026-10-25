"""Redshift and absolute magnitude on scene elements (slicersim conventions)."""

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import Planck18
from synphot import Empirical1D, Observation, SourceSpectrum
from synphot import units as su

from tests.helpers import make_scene
from wcc_etc.io import resolve_bandpass
from wcc_etc.scene import get_scene_element

Z = 0.9
BAND = resolve_bandpass("sdss_r")


def sn(**kw):
    """The bundled hsiao07 SN Ia template as a scene element."""
    return get_scene_element("hsiao07", bandpass="sdss_r", magsys="abmag", **kw)


def rest_frame_absmag(spectrum, redshift):
    """Rest-frame AB absolute magnitude of an observed spectrum, as slicersim's test."""
    lbda_rest = np.linspace(2000, 12000, 5000)
    flux_obs = spectrum(lbda_rest * (1 + redshift) * u.AA, flux_unit="flam").value
    flux_10pc = (
        flux_obs * (1 + redshift) * 10 ** (0.4 * Planck18.distmod(redshift).value)
    )
    rest = SourceSpectrum(
        Empirical1D, points=lbda_rest, lookup_table=flux_10pc * su.FLAM
    )
    return Observation(rest, BAND, force="extrap").effstim("abmag").value


class TestRedshift:
    def test_stretches_the_waveset(self):
        """redshift=z multiplies the template's wavelengths by 1+z."""
        assert np.isclose(
            sn(mag=20, redshift=Z).spectrum.waveset.max().value,
            sn(mag=20).spectrum.waveset.max().value * (1 + Z),
        )

    def test_default_is_rest_frame(self):
        """No redshift leaves the spectrum untouched (z=0)."""
        assert sn(mag=20).spectrum.z == 0.0

    def test_update_rebuilds_the_spectrum(self):
        """update(redshift=...) rebuilds the (redshifted) spectrum."""
        el = sn(mag=20, redshift=Z)
        el.update(redshift=0.5)
        assert np.isclose(
            el.spectrum.waveset.max().value,
            sn(mag=20).spectrum.waveset.max().value * 1.5,
        )

    def test_redshift_is_mutable(self):
        """redshift is listed among the mutable parameters."""
        assert "redshift" in sn(mag=20).mutable_parameters


class TestAbsoluteMagnitude:
    def test_recovers_rest_frame_absmag(self):
        """magabs=-19.3 at z=0.9 gives a spectrum whose rest-frame absmag is -19.3."""
        el = sn(mag=None, magabs=-19.3, redshift=Z)
        assert np.isclose(rest_frame_absmag(el.get_spectrum(), Z), -19.3, atol=1e-3)

    def test_mag_property_is_the_observed_mag(self):
        """element.mag is the observed magnitude derived from magabs."""
        el = sn(mag=None, magabs=-19.3, redshift=Z)
        assert np.isclose(
            Observation(el.get_spectrum(), BAND, force="extrap").effstim("abmag").value,
            el.mag.value,
        )

    def test_mag_overrides_magabs(self):
        """An observed mag wins over magabs."""
        assert sn(mag=20, magabs=-19.3, redshift=Z).mag.value == 20

    def test_vegamag_differs_from_abmag(self):
        """magabs is read in the element's magsys: AB and Vega give different fluxes.

        The observed magnitude *number* is the same in both systems (the
        zero-point offset cancels between the rest and observed frames), so
        the flux is what tells them apart."""
        kw = dict(mag=None, magabs=-19.3, redshift=Z, bandpass="johnson_v")
        ab = get_scene_element("hsiao07", magsys="abmag", **kw)
        vega = get_scene_element("hsiao07", magsys="vegamag", **kw)
        assert not np.isclose(
            ab.get_spectrum()(6000 * u.AA).value, vega.get_spectrum()(6000 * u.AA).value
        )

    def test_update_magabs_invalidates_cache(self):
        """Changing magabs misses the normalised-spectrum cache."""
        el = sn(mag=None, magabs=-19.3, redshift=Z)
        first = el.get_spectrum()
        el.update(magabs=-20)
        assert el.get_spectrum() is not first

    def test_spectrum_is_cached(self):
        """Repeated get_spectrum with unchanged magabs returns the same object."""
        el = sn(mag=None, magabs=-19.3, redshift=Z)
        assert el.get_spectrum() is el.get_spectrum()

    def test_scene_source_with_magabs(self):
        """make_scene passes magabs/redshift through to the source."""
        scene = make_scene(
            name="hsiao07",
            mag=None,
            bandpass="sdss_r",
            magsys="abmag",
            magabs=-19.3,
            redshift=Z,
        )
        assert scene.source.meta["magabs"] == -19.3
