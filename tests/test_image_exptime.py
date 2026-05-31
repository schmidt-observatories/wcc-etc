import numpy as np
import wcc_etc


def _sim(mag=15):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_image_snr_n_reads_one_matches_baseline():
    sim = _sim()
    base = sim.get_image_snr(time=60)["snr"]
    assert np.isclose(sim.get_image_snr(time=60, n_reads=1)["snr"], base)


def test_image_exptime_for_snr_roundtrips_fixed_aperture():
    sim = _sim()
    target = 80.0
    res = sim.get_image_exptime_for_snr(target, r_aper_mas=70)
    t = res["time_s"]
    got = sim.get_image_snr(time=t, r_aper_mas=70)["snr"]
    assert np.isclose(got, target, rtol=2e-3)


def test_image_exptime_matches_analytic_for_infocus_default_aperture():
    # in-focus Airy + default aperture: PSF-aware inverse ~ analytic inverse
    sim = _sim()
    t_img = sim.get_image_exptime_for_snr(50.0)["time_s"]
    t_ana = sim.get_exptime_for_snr(50.0).value
    assert np.isclose(t_img, t_ana, rtol=0.02)
