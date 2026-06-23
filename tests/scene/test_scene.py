"""Tests for Scene, get_scene_from_file, and broadcast_mapping."""

import numpy as np
import astropy.units as u
from synphot import units as su
from wcc_etc.scene import Scene, SceneElement, broadcast_mapping, get_scene_from_file
import pytest


class TestBroadcastMapping:
    def test_scalar_to_many(self):
        out = broadcast_mapping(5, 3)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        assert np.all(out == 5)

    def test_1d_array(self):
        out = broadcast_mapping([1, 2, 3], 3)
        assert out.shape == (3,)
        assert np.all(out == np.array([1, 2, 3]))

    def test_2d_broadcast(self):
        out = broadcast_mapping([[1.0, 2.0]], 4)
        assert out.shape == (4, 2)
        assert np.all(out[0] == np.array([1.0, 2.0]))


class TestSceneClass:
    @pytest.fixture
    def scene(self):
        src = SceneElement(spectrum=None, mag=20, magsys="abmag")
        host = SceneElement(spectrum=None, mag=18, magsys="abmag")
        bkg = SceneElement(
            spectrum=None, mag=23, magsys="abmag", surface_brightness=True
        )
        return Scene(source=src, host=host, background=bkg)

    def test_has_element_flags(self, scene):
        assert scene.has_source()
        assert scene.has_host()
        assert scene.has_background()

    def test_get_elements_returns_all_keys(self, scene):
        elems = scene.get_elements(as_dict=True)
        assert set(elems.keys()) == {"source", "host", "background"}

    def test_get_mag_point_sources(self, scene):
        mags = scene.get_mag(which=["source", "host"], as_dict=True)
        assert mags["source"].value == 20
        assert mags["host"].value == 18

    def test_get_mag_background_with_area(self, scene):
        mags = scene.get_mag(area=4.0, as_dict=True)
        assert mags["background"] == (23.0 - 2.5 * np.log10(4.0)) * u.ABmag

    def test_update_source_magnitude(self, scene):
        updated = scene.update(source__mag=21)
        assert "source__mag" in updated
        assert scene.source.get_mag().value == 21


class TestSceneElementMag:
    def test_point_source_mag_unit(self):
        se = SceneElement(spectrum=None, mag=20, magsys="abmag")
        magq = se.get_mag()
        assert magq.value == 20
        assert magq.unit.is_equivalent(u.ABmag)

    def test_surface_brightness_mag_with_float_area(self):
        se_sb = SceneElement(
            spectrum=None, mag=22.0, magsys="abmag", surface_brightness=True
        )
        mag_area = se_sb.get_mag(area=4.0)
        expected = (22.0 - 2.5 * np.log10(4.0)) * u.ABmag
        assert mag_area == expected

    def test_surface_brightness_mag_with_quantity_area(self):
        se_sb = SceneElement(
            spectrum=None, mag=22.0, magsys="abmag", surface_brightness=True
        )
        mag_area = se_sb.get_mag(area=(2.0 * u.arcsec**2))
        expected = (22.0 - 2.5 * np.log10(2.0)) * u.ABmag
        assert mag_area == expected


class TestGetSceneFromFile:
    def test_preserves_absolute_flux(self, tmp_path):
        specfile = tmp_path / "source.csv"
        specfile.write_text("wavelength,flux\n4000,1e-16\n5000,2e-16\n6000,3e-16\n")
        scene = get_scene_from_file(
            str(specfile),
            mag=None,
            wave_column="wavelength",
            flux_column="flux",
            names=True,
            background=None,
        )
        sp = scene.source.get_spectrum(apply_mag=False)
        flam = sp(np.array([4000.0, 5000.0, 6000.0]) * u.AA, flux_unit=su.FLAM).value
        assert np.allclose(flam, [1e-16, 2e-16, 3e-16])


class TestSceneUpdate:
    def test_scene_update_rebuilds_source_spectrum(self):
        import wcc_etc
        from synphot import units as su

        scene = wcc_etc.get_scene(
            name="blackbody",
            mag=15,
            teff=5777,
            bandpass="johnson_r",
            background="zodi",
            background_prop={"bandpass": "johnson_r", "mag": 22.5},
        )

        def _ratio(sp, l1, l2):
            return (
                sp(np.array([l1, l2]) * u.AA, flux_unit=su.FLAM).value[0]
                / sp(np.array([l1, l2]) * u.AA, flux_unit=su.FLAM).value[1]
            )

        before = _ratio(scene.source.get_spectrum(apply_mag=False), 4500, 7500)
        scene.update(source__teff=3000)
        after = _ratio(scene.source.get_spectrum(apply_mag=False), 4500, 7500)
        assert not np.isclose(after, before, rtol=1e-3)
