import numpy as np
import astropy.units as u
from wcc_etc.scene import broadcast_mapping, SceneElement, Scene


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
