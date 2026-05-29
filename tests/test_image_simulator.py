import numpy as np
import pytest
import astropy.units as u
import wcc_etc
from wcc_etc.psfsim import SimulatedImage, AiryPSF, FitsImg, ImageSimulator, DefocusPSF, DEFOCUS_2WAVE_PATH


def _make_simimg():
    image_e = np.array([[100.0, 200.0], [300.0, 400.0]])
    clean = image_e.copy()
    sat = np.array([[False, False], [False, True]])
    return SimulatedImage(image_e=image_e, image_clean=clean, saturation_mask=sat,
                          gain=2.0, bias_level=100.0, npix=2,
                          pixel_scale_mas=20.0, psf=AiryPSF())


def test_simulated_image_to_adu():
    s = _make_simimg()
    adu = s.to_adu()
    assert np.allclose(adu, s.image_e / 2.0 + 100.0)


def test_simulated_image_to_fitsimg():
    s = _make_simimg()
    f = s.to_fitsimg()
    assert isinstance(f, FitsImg)
    assert np.allclose(f.data, s.image_e)


def _scene(mag=15):
    return wcc_etc.get_scene(
        name='G5V', mag=mag, host=None, background="zodi",
        bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})


def test_image_simulator_clean_flux_conservation():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=128)
    res = imsim.simulate(time=10, add_noise=False)
    assert res.image_clean.shape == (128, 128)

    sim = imsim.sim
    t = 10 * u.second
    cr = sim.get_countrates(units="e/s", as_dict=True)
    ee = sim.psf_profile["ee_at_aper"]
    npp = sim.psf_profile["num_psf_pixels"].value
    source_e = (cr["source"] / ee * t).to(u.electron).value
    bkg = (cr["background"] * t / npp).to(u.electron).value
    dark = (sim.sensor.dark_current * t).to(u.electron / u.pix).value
    expected = source_e + (bkg + dark) * 128 * 128
    assert res.image_clean.sum() == pytest.approx(expected, rel=0.02)


def test_image_simulator_source_scales_with_time():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=128)
    peak10 = imsim.simulate(time=10, add_noise=False).image_clean.max()
    peak100 = imsim.simulate(time=100, add_noise=False).image_clean.max()
    assert peak100 > peak10


def test_image_simulator_noise_is_seed_reproducible():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=64)
    a = imsim.simulate(time=10, add_noise=True, seed=1).image_e
    b = imsim.simulate(time=10, add_noise=True, seed=1).image_e
    c = imsim.simulate(time=10, add_noise=True, seed=2).image_e
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_image_simulator_read_noise_in_blank_corner():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(20), npix=128)
    res = imsim.simulate(time=1, add_noise=True, seed=0)
    corner = res.image_e[:16, :16]
    rn = imsim.sim.sensor.read_noise.to(u.electron / u.pix).value
    assert np.std(corner) == pytest.approx(rn, rel=0.2)


def test_image_simulator_saturation_flips_with_brightness():
    faint = ImageSimulator.from_sensor_and_scene('sony:r', _scene(20), npix=64)
    bright = ImageSimulator.from_sensor_and_scene('sony:r', _scene(6), npix=64)
    assert not faint.simulate(time=1, add_noise=False).saturation_mask.any()
    assert bright.simulate(time=100, add_noise=False).saturation_mask.any()


def test_image_simulator_sensor_pixel_scales_differ():
    sony = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=64)
    hwk = ImageSimulator.from_sensor_and_scene('qcmos:r', _scene(15), npix=64)
    rs = sony.simulate(time=10, add_noise=False)
    rh = hwk.simulate(time=10, add_noise=False)
    assert rs.pixel_scale_mas != pytest.approx(rh.pixel_scale_mas)


def test_image_simulator_runs_with_defocus_psf():
    imsim = ImageSimulator.from_sensor_and_scene('sony:r', _scene(15), npix=300)
    res = imsim.simulate(time=10, psf=DefocusPSF(DEFOCUS_2WAVE_PATH), add_noise=False)
    assert res.image_clean.shape == (300, 300)
    assert np.isfinite(res.image_clean).all()


def test_image_simulator_saturation_mask_with_noise_is_bool_array():
    bright = ImageSimulator.from_sensor_and_scene('sony:r', _scene(6), npix=64)
    res = bright.simulate(time=100, add_noise=True, seed=0)
    assert res.saturation_mask.dtype == bool
    assert res.saturation_mask.shape == (64, 64)
    assert res.saturation_mask.any()
