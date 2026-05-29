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


def test_public_exports_available():
    import wcc_etc
    for name in ["ImageSimulator", "AiryPSF", "DefocusPSF", "CustomPSF", "SimulatedImage"]:
        assert hasattr(wcc_etc, name), f"{name} not exported from wcc_etc"


from wcc_etc.psfsim import aperture_snr_radial, select_aperture


def _point_psf(npix=21):
    a = np.zeros((npix, npix)); a[npix // 2, npix // 2] = 1.0
    return a


def _gaussian_psf(npix=41, sigma=3.0):
    c = (npix - 1) / 2
    yy, xx = np.mgrid[0:npix, 0:npix]
    a = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma ** 2)))
    return a / a.sum()


def test_aperture_snr_radial_enclosed_monotonic_to_one():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    enc = prof["enclosed_fraction"]
    assert np.all(np.diff(enc) >= -1e-12)
    assert enc[-1] == pytest.approx(1.0)
    assert prof["n_pix"][0] == 1
    assert prof["n_pix"][-1] == 21 * 21
    assert np.all(np.diff(prof["r_mas"]) >= 0)


def test_aperture_snr_point_source_noiseless_is_sqrt_signal():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    assert np.allclose(prof["snr"], np.sqrt(1e4))


def test_aperture_snr_read_noise_penalizes_large_apertures():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=5.0)
    assert prof["snr"][0] > prof["snr"][-1]


def test_aperture_snr_noise_consistent_with_signal_and_snr():
    prof = aperture_snr_radial(_gaussian_psf(), plate_scale_mas=10.0,
                               source_e_total=1e5, diffuse_per_pix=2.0,
                               dark_per_pix=1.0, read_noise=5.0)
    assert np.allclose(prof["snr"], prof["signal_e"] / prof["noise_e"])


def test_select_aperture_optimize_picks_max_snr():
    prof = aperture_snr_radial(_gaussian_psf(), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=1.0,
                               dark_per_pix=1.0, read_noise=5.0)
    assert select_aperture(prof, optimize=True) == int(np.argmax(prof["snr"]))


def test_select_aperture_ee_frac():
    prof = aperture_snr_radial(_gaussian_psf(41, 3.0), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    idx = select_aperture(prof, ee_frac=0.9)
    assert prof["enclosed_fraction"][idx] >= 0.9


def test_select_aperture_fixed_radius():
    prof = aperture_snr_radial(_point_psf(41), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    idx = select_aperture(prof, r_aper_mas=35.0)
    assert prof["r_mas"][idx] <= 35.0


def test_select_aperture_requires_a_mode():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    with pytest.raises(ValueError):
        select_aperture(prof)
