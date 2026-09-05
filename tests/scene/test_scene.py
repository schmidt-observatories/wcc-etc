"""Tests for Scene, get_scene_from_file, and broadcast_mapping."""

import astropy.units as u
import numpy as np
import pytest
from synphot import units as su

from tests.helpers import make_scene
from wcc_etc.scene import (
    Scene,
    SceneElement,
    broadcast_mapping,
    get_scene,
    get_scene_from_file,
)


class TestBroadcastMapping:
    def test_broadcast_scalar_to_many_is_ndarray(self):
        out = broadcast_mapping(5, 3)
        assert isinstance(out, np.ndarray)

    def test_broadcast_scalar_to_many_shape(self):
        out = broadcast_mapping(5, 3)
        assert out.shape == (3,)

    def test_broadcast_scalar_to_many_values(self):
        out = broadcast_mapping(5, 3)
        assert np.all(out == 5)

    def test_broadcast_1d_shape(self):
        out = broadcast_mapping([1, 2, 3], 3)
        assert out.shape == (3,)

    def test_broadcast_1d_values(self):
        out = broadcast_mapping([1, 2, 3], 3)
        assert np.all(out == np.array([1, 2, 3]))

    def test_broadcast_2d_shape(self):
        out = broadcast_mapping([[1.0, 2.0]], 4)
        assert out.shape == (4, 2)

    def test_broadcast_2d_values(self):
        out = broadcast_mapping([[1.0, 2.0]], 4)
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

    def test_has_source(self, scene):
        assert scene.has_source()

    def test_has_host(self, scene):
        assert scene.has_host()

    def test_has_background(self, scene):
        assert scene.has_background()

    def test_get_elements_returns_all_keys(self, scene):
        elems = scene.get_elements(as_dict=True)
        assert set(elems.keys()) == {"source", "host", "background"}

    def test_get_mag_source(self, scene):
        mags = scene.get_mag(which=["source", "host"], as_dict=True)
        assert mags["source"].value == 20

    def test_get_mag_host(self, scene):
        mags = scene.get_mag(which=["source", "host"], as_dict=True)
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


class TestGetScenePropDefaults:
    """host_prop / background_prop passed as None must behave like {} (#66)."""

    def test_none_host_prop_builds_the_host(self):
        scene = get_scene("G5IV", mag=20, host="G2V", host_prop=None, background=None)
        assert scene.host is not None

    def test_none_background_prop_builds_the_background(self):
        scene = get_scene("G5IV", mag=20, background="zodi", background_prop=None)
        assert scene.background is not None


class TestGetSceneFromFileDefaults:
    """The documented default call must work (#66)."""

    def test_default_call_returns_a_scene(self, tmp_path):
        specfile = tmp_path / "source.txt"
        specfile.write_text("4000 1e-16\n5000 2e-16\n6000 3e-16\n")
        scene = get_scene_from_file(str(specfile), mag=20)
        assert isinstance(scene, Scene)

    def test_default_call_builds_the_zodi_background(self, tmp_path):
        specfile = tmp_path / "source.txt"
        specfile.write_text("4000 1e-16\n5000 2e-16\n6000 3e-16\n")
        scene = get_scene_from_file(str(specfile), mag=20)
        assert scene.background is not None


class TestSceneUpdate:
    def test_scene_update_rebuilds_source_spectrum(self):
        from synphot import units as su

        import wcc_etc

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


class TestSpectrumSamplingControls:
    """get_spectrum / show accept an explicit wavelength grid and flux unit, so
    callers can plot F_lambda over the optical without resampling by hand."""

    def test_wave_sets_the_sampling_grid(self):
        """as_array sampling honors an explicit wavelength grid."""
        scene = make_scene(name="blackbody", teff=5500)
        grid = np.arange(4000.0, 8000.0, 100.0)
        wave, _flux = scene.source.get_spectrum(as_array=True, wave=grid)
        assert wave.size == grid.size

    def test_wave_defaults_to_the_native_waveset(self):
        """Without wave= the spectrum's own waveset is used."""
        scene = make_scene(name="blackbody", teff=5500)
        wave, _flux = scene.source.get_spectrum(as_array=True)
        assert wave.max().value > 20000.0

    def test_flux_unit_converts_the_flux(self):
        """flux_unit='flam' returns erg/s/cm^2/A rather than PHOTLAM."""
        scene = make_scene(name="blackbody", teff=5500)
        _wave, flux = scene.source.get_spectrum(as_array=True, flux_unit="flam")
        assert flux.unit.to_string() == "FLAM"

    def test_flux_unit_defaults_to_photlam(self):
        """Without flux_unit the historic PHOTLAM sampling is unchanged."""
        scene = make_scene(name="blackbody", teff=5500)
        _wave, flux = scene.source.get_spectrum(as_array=True)
        assert flux.unit.to_string() == "PHOTLAM"

    def test_powerlaw_flam_follows_the_alpha_exponent(self):
        """Sampled in FLAM, a powerlaw source obeys F_lambda ~ lambda^alpha."""
        alpha = -1.0
        scene = make_scene(name="powerlaw", alpha=alpha)
        grid = np.array([4500.0, 7500.0])
        _wave, flux = scene.source.get_spectrum(
            as_array=True, wave=grid, flux_unit="flam"
        )
        ratio = float(flux[0] / flux[1])
        assert ratio == pytest.approx((4500.0 / 7500.0) ** alpha, rel=1e-3)

    def test_show_labels_the_axis_with_the_flux_unit(self):
        """The y label names the unit actually plotted."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scene = make_scene(name="blackbody", teff=5500)
        _fig, ax = plt.subplots()
        scene.source.show(ax=ax, flux_unit="flam")
        assert ax.get_ylabel() == "Flux [FLAM]"

    def test_show_forwards_the_label_to_the_line(self):
        """label= reaches ax.plot so overlaid spectra can carry a legend."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scene = make_scene(name="blackbody", teff=5500)
        _fig, ax = plt.subplots()
        scene.source.show(ax=ax, label="5500 K")
        assert ax.lines[0].get_label() == "5500 K"
