from wcc_etc.io import expand_path
import numpy as np
import numpy as np
import astropy.units as u
from synphot import SpectralElement, Observation, units as su
from wcc_etc.scene import broadcast_mapping, SceneElement, Scene


def _observed_abmag(spectrum, band_name="johnson_v"):
    band = SpectralElement.from_filter(band_name)
    return Observation(spectrum, band, force="extrap").effstim(u.ABmag).value


def test_name_and_config():
    """ """
    baseconfig = {"mag": 20,
                  "bandpass": "johnson_v"}
    config1 = {"spectrum": expand_path('astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits')}
    config2 = {"spectrum": "uk_55"}
    config3 = {"spectrum": "G5IV"}

    source1 = SceneElement.from_config(config1 | baseconfig)
    lbda1, spec1 = source1.get_spectrum(as_array=True)

    source2 = SceneElement.from_config(config2 | baseconfig)
    lbda2, spec2 = source2.get_spectrum(as_array=True)

    source3 = SceneElement.from_config(config3 | baseconfig)
    lbda3, spec3 = source3.get_spectrum(as_array=True)

    assert np.all(spec1 == spec2)
    assert np.all(spec1 == spec3)
    assert np.all(lbda1 == lbda2)
    assert np.all(lbda1 == lbda3)




def test_blackbody_source_roundtrips_magnitude():
    se = SceneElement.from_config({"spectrum": "blackbody", "teff": 5777,
                                   "mag": 15, "bandpass": "johnson_v"})
    sp = se.get_spectrum()  # magnitude-normalized
    assert abs(_observed_abmag(sp) - 15) < 0.01


def test_blackbody_shape_matches_get_blackbody_flux():
    from wcc_etc.wcc_etc import get_blackbody_flux
    se = SceneElement.from_config({"spectrum": "blackbody", "teff": 5777,
                                   "mag": 15, "bandpass": "johnson_v"})
    sp = se.get_spectrum()
    w = np.array([4000.0, 6000.0, 8000.0])
    flam = sp(w * u.AA, flux_unit=su.FLAM).value
    ref = np.asarray(get_blackbody_flux(w, 5777, 15))
    # the normalization differs (AB vs Vega) but the blackbody *shape* must match
    assert np.allclose(flam / flam[0], ref / ref[0], rtol=1e-3)


def test_flat_source_is_constant_fnu_and_roundtrips_mag():
    se = SceneElement.from_config({"spectrum": "flat", "mag": 18,
                                   "bandpass": "johnson_v"})
    sp = se.get_spectrum()
    w = np.array([4000.0, 6000.0, 8000.0]) * u.AA
    fnu = sp(w, flux_unit=u.Jy).value
    assert np.allclose(fnu, fnu[0], rtol=1e-3)
    assert abs(_observed_abmag(sp) - 18) < 0.01


def test_powerlaw_source_slope():
    alpha = -1.0
    se = SceneElement.from_config({"spectrum": "powerlaw", "alpha": alpha,
                                   "mag": 18, "bandpass": "johnson_v"})
    sp = se.get_spectrum()
    w1, w2 = 4000.0, 8000.0
    f1 = sp(w1 * u.AA, flux_unit=su.FLAM).value
    f2 = sp(w2 * u.AA, flux_unit=su.FLAM).value
    # F_lambda proportional to lambda**alpha
    assert np.isclose(f1 / f2, (w1 / w2) ** alpha, rtol=1e-3)


def test_emission_line_recovers_absolute_flux():
    flux = 1e-15
    se = SceneElement.from_config({"spectrum": "emission",
                                   "lines": [{"wave": 6563, "flux": flux, "fwhm": 3}],
                                   "mag": None})
    sp = se.get_spectrum()  # mag is None -> no normalization
    w = np.arange(6500, 6630, 0.05) * u.AA
    flam = sp(w, flux_unit=su.FLAM).value  # erg/s/cm^2/A
    integral = np.trapz(flam, w.value)     # erg/s/cm^2
    assert np.isclose(integral, flux, rtol=1e-2)
    # line centroid sits at the requested wavelength
    centroid = np.trapz(flam * w.value, w.value) / integral
    assert abs(centroid - 6563) < 0.5


def test_emission_lines_sum():
    se = SceneElement.from_config({"spectrum": "emission",
                                   "lines": [{"wave": 6563, "flux": 1e-15, "fwhm": 3},
                                             {"wave": 6583, "flux": 4e-16, "fwhm": 3}],
                                   "mag": None})
    sp = se.get_spectrum()
    w = np.arange(6400, 6700, 0.05) * u.AA
    flam = sp(w, flux_unit=su.FLAM).value
    integral = np.trapz(flam, w.value)
    assert np.isclose(integral, 1.4e-15, rtol=1e-2)


def test_get_spectrum_skips_normalization_when_mag_is_none():
    se = SceneElement.from_config({"spectrum": "blackbody", "teff": 5777,
                                   "mag": None, "bandpass": "johnson_v"})
    raw = se.get_spectrum(apply_mag=False)
    out = se.get_spectrum(apply_mag=True)  # mag is None -> should be a no-op
    w = np.array([5000.0, 6000.0]) * u.AA
    assert np.allclose(raw(w, flux_unit=su.FLAM).value,
                       out(w, flux_unit=su.FLAM).value)


def test_explicit_none_mag_does_not_warn(recwarn):
    se = SceneElement.from_config({"spectrum": "emission",
                                   "lines": [{"wave": 6563, "flux": 1e-15, "fwhm": 3}],
                                   "mag": None})
    se.get_mag()
    assert not any("not mag in self.meta" in str(w.message) for w in recwarn.list)


def test_broadcast_mapping_scalar_and_array():
    # scalar to many
    v = 5
    out = broadcast_mapping(v, 3)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.all(out == 5)

    # 1D array broadcast
    v2 = [1, 2, 3]
    out2 = broadcast_mapping(v2, 3)
    assert out2.shape == (3,)
    assert np.all(out2 == np.array([1, 2, 3]))

    # 2D array broadcasting to ntargets x len
    v3 = [[1.0, 2.0]]
    out3 = broadcast_mapping(v3, 4)
    assert out3.shape == (4, 2)
    assert np.all(out3[0] == np.array([1.0, 2.0]))


def test_sceneelement_get_mag_surface_brightness_and_units():
    # point source (not surface brightness)
    se = SceneElement(spectrum=None, mag=20)
    magq = se.get_mag()
    assert magq.value == 20
    assert magq.unit.is_equivalent(u.ABmag)

    # surface brightness: mag per arcsec^2
    se_sb = SceneElement(spectrum=None, mag=22.0, surface_brightness=True)
    # area as float (arcsec^2)
    area = 4.0
    mag_area = se_sb.get_mag(area=area)
    expected = (22.0 - 2.5 * np.log10(area)) * u.ABmag
    assert mag_area == expected

    # area as astropy quantity
    mag_area2 = se_sb.get_mag(area=(2.0 * u.arcsec**2))
    expected2 = (22.0 - 2.5 * np.log10(2.0)) * u.ABmag
    assert mag_area2 == expected2


def test_scene_get_elements_get_mag_and_update():
    src = SceneElement(spectrum=None, mag=20)
    host = SceneElement(spectrum=None, mag=18)
    bkg = SceneElement(spectrum=None, mag=23, surface_brightness=True)

    scene = Scene(source=src, host=host, background=bkg)

    assert scene.has_source()
    assert scene.has_host()
    assert scene.has_background()

    elems = scene.get_elements(as_dict=True)
    assert set(elems.keys()) == {"source", "host", "background"}

    # get_mag without area: request only source and host (background needs area)
    mags = scene.get_mag(which=["source", "host"], as_dict=True)
    assert mags["source"].value == 20
    assert mags["host"].value == 18

    # get_mag with area affects background only
    mags_area = scene.get_mag(area=4.0, as_dict=True)
    # background adjusted
    assert mags_area["background"] == (23.0 - 2.5 * np.log10(4.0)) * u.ABmag

    # update source magnitude using scene.update
    updated = scene.update(source__mag=21)
    assert "source__mag" in updated
    assert scene.source.get_mag().value == 21
