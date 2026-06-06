import numpy as np
import pytest
from wcc_etc.simulation import (
    calculate_bg_normalization_magnitude,
    Simulation,
)


def test_calculate_bg_normalization_magnitude_basic():
    # area = 1 arcsec^2 should give same magnitude
    bg = 20.0
    area = 1.0
    assert calculate_bg_normalization_magnitude(bg, area) == pytest.approx(bg)


def test_calculate_bg_normalization_magnitude_scaling():
    bg = 20.0
    area = 10.0
    expected = bg - 2.5 * np.log10(area)
    assert calculate_bg_normalization_magnitude(bg, area) == pytest.approx(expected)


def test_fullkey_to_element_and_key():
    # fully qualified
    element, key = Simulation._fullkey_to_element_and_key("sensor__gain")
    assert element == "sensor"
    assert key == "gain"

    # single key returns (None, key)
    element2, key2 = Simulation._fullkey_to_element_and_key("time")
    assert element2 is None
    assert key2 == "time"


def test_get_parameter_time_default_and_mutable_parameters():
    # create a Simulation with a custom time and check get_parameter
    sim = Simulation(telescope=None, sensor=None, scene=None, time=123, r_aper_mas=70)
    val = sim.get_parameter("time")
    assert isinstance(val, list)
    assert val[0] == 123

    # mutable parameters should contain the defaults
    mp = sim.mutable_parameters
    assert "time" in mp
    assert "r_aper_mas" in mp


def test_has_element_and_setting_attribute():
    sim = Simulation(telescope=None, sensor=None, scene=None)
    assert sim.has_element("telescope") is False
    # set a dummy attribute and test
    sim._telescope = object()
    assert sim.has_element("telescope") is True


import wcc_etc
import astropy.units as u


def _bright_sim(mag=20, sensor="sony:r"):
    scene = wcc_etc.get_scene(
        name='G5V', mag=mag, host=None, background="zodi",
        bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene(sensor, scene)


def test_psf_profile_has_peak_pixel_fraction():
    sim = _bright_sim()
    profile = sim.psf_profile
    assert "peak_pixel_fraction" in profile
    assert 0.0 < profile["peak_pixel_fraction"] <= 1.0


def test_get_peak_pixel_increases_with_time():
    sim = _bright_sim(mag=15)
    p10 = sim.get_peak_pixel(10, units="adu")
    p100 = sim.get_peak_pixel(100, units="adu")
    assert p100 > p10


def test_get_peak_pixel_adu_units_are_ct():
    sim = _bright_sim(mag=15)
    p = sim.get_peak_pixel(10, units="adu")
    assert p.unit == u.ct


def test_get_peak_pixel_includes_bias():
    sim = _bright_sim(mag=15)
    base = sim.get_peak_pixel(10, units="adu")
    sim.update(sensor__bias_level=100)
    biased = sim.get_peak_pixel(10, units="adu")
    assert biased.value == pytest.approx(base.value + 100, rel=1e-6)


def test_get_peak_pixel_electrons_excludes_bias():
    sim = _bright_sim(mag=15)
    e = sim.get_peak_pixel(10, units="e-")
    assert e.unit == u.electron
    assert e.value > 0


def test_get_peak_pixel_adu_matches_electron_conversion():
    sim = _bright_sim(mag=15)
    e = sim.get_peak_pixel(10, units="e-")
    adu = sim.get_peak_pixel(10, units="adu")
    expected = (e / sim.sensor.gain).to(u.ct) + sim.sensor.bias_level
    assert adu.value == pytest.approx(expected.value, rel=1e-9)


def test_get_peak_pixel_brighter_background_increases_value():
    sim = _bright_sim(mag=15)
    base = sim.get_peak_pixel(100, units="e-")
    sim.update(background__mag=18)  # lower mag = brighter background
    brighter = sim.get_peak_pixel(100, units="e-")
    assert brighter.value > base.value


def test_get_peak_pixel_accepts_array_time():
    sim = _bright_sim(mag=15)
    vals = sim.get_peak_pixel(np.array([10.0, 100.0]), units="adu")
    assert np.shape(vals) == (2,)
    assert vals[1] > vals[0]


def test_get_peak_pixel_excludes_host():
    # host elements are intentionally excluded from the saturation budget,
    # so adding a host must not change the peak-pixel value.
    scene_no_host = wcc_etc.get_scene(
        name='G5V', mag=15, host=None, background="zodi",
        bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})
    sim_no_host = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene_no_host)

    scene_host = wcc_etc.get_scene(
        name='G5V', mag=15, host='G5V', host_prop={"mag": 16, "bandpass": 'johnson_r'},
        background="zodi", bandpass='johnson_r',
        background_prop={"bandpass": 'johnson_r', "mag": 22.5})
    sim_host = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene_host)

    no_host = sim_no_host.get_peak_pixel(100, units="e-").value
    with_host = sim_host.get_peak_pixel(100, units="e-").value
    assert with_host == pytest.approx(no_host, rel=1e-9)


def test_is_saturated_flips_with_time():
    sim = _bright_sim(mag=8)  # bright star on a 16-bit sensor (adc_max=65535)
    assert sim.is_saturated(0.001) == False
    assert sim.is_saturated(1000) == True


def test_is_saturated_accepts_array_time():
    sim = _bright_sim(mag=8)
    result = sim.is_saturated(np.array([0.001, 1000.0]))
    assert np.shape(result) == (2,)
    assert bool(result[0]) is False
    assert bool(result[1]) is True


def test_get_image_snr_returns_expected_keys():
    sim = _bright_sim(16)
    out = sim.get_image_snr(time=60)
    assert set(out) >= {"snr", "signal_e", "noise_e", "enclosed_fraction", "r_aper_mas", "n_pix"}
    assert 0 < out["enclosed_fraction"] <= 1
    assert out["n_pix"] >= 1
    assert out["snr"] > 0


def test_get_image_snr_matches_get_snr_in_focus():
    sim = _bright_sim(16)
    for t in [30, 300]:
        for m in [16, 20]:
            sim.update(source__mag=m)
            etc = sim.get_snr(t)
            etc = float(etc.value) if hasattr(etc, "value") else float(etc)
            img = sim.get_image_snr(time=t)["snr"]
            assert img == pytest.approx(etc, rel=0.03)


def test_get_image_snr_ee_frac_aperture():
    sim = _bright_sim(16)
    out = sim.get_image_snr(time=60, ee_frac=0.9)
    assert out["enclosed_fraction"] >= 0.9


def test_get_image_snr_optimize_at_least_default():
    sim = _bright_sim(16)
    base = sim.get_image_snr(time=60)["snr"]
    opt = sim.get_image_snr(time=60, optimize=True)["snr"]
    assert opt >= base - 1e-9


def test_get_image_snr_defocus_lower_at_fixed_aperture():
    sim = _bright_sim(16)
    airy = sim.get_image_snr(time=60, r_aper_mas=70)["snr"]
    defo = sim.get_image_snr(time=60, r_aper_mas=70,
                             psf=wcc_etc.DefocusPSF(wcc_etc.DEFOCUS_2WAVE_PATH))["snr"]
    assert defo < airy


def test_get_image_snr_optimize_defocus_uses_larger_radius():
    sim = _bright_sim(16)
    r_airy = sim.get_image_snr(time=60, optimize=True)["r_aper_mas"]
    r_defo = sim.get_image_snr(time=60, optimize=True,
                               psf=wcc_etc.DefocusPSF(wcc_etc.DEFOCUS_2WAVE_PATH))["r_aper_mas"]
    assert r_defo > r_airy


def test_get_image_snr_runs_on_qcmos():
    sim = _bright_sim(16, sensor="qcmos:r")
    assert sim.get_image_snr(time=60)["snr"] > 0


def test_get_image_snr_no_background_runs():
    # scene with no background exercises the diffuse_per_pix == 0 path
    scene = wcc_etc.get_scene(name='G5V', mag=16, host=None, background=None,
                              bandpass='johnson_r')
    sim = wcc_etc.Simulation.from_sensor_and_scene('sony:r', scene)
    out = sim.get_image_snr(time=60)
    assert out['snr'] > 0
    assert out['n_pix'] >= 1


def test_image_render_bundle_cached(monkeypatch):
    import wcc_etc.psfsim as psfsim
    sim = _bright_sim(16)
    calls = {"n": 0}
    orig = psfsim.AiryPSF.render
    def counting_render(self, ctx):
        calls["n"] += 1
        return orig(self, ctx)
    monkeypatch.setattr(psfsim.AiryPSF, "render", counting_render)
    a = sim.get_image_snr(time=30)["snr"]
    b = sim.get_image_snr(time=60)["snr"]
    assert calls["n"] == 1                       # rendered once, reused
    assert len(sim._image_render_bundle_cache) == 1
    assert b > a                                 # longer exposure -> higher SNR


def test_update_invalidates_render_cache():
    sim = _bright_sim(16)
    snr1 = sim.get_image_snr(time=60)["snr"]
    assert len(sim._image_render_bundle_cache) == 1
    sim.update(source__mag=20)                   # fainter source
    assert len(sim._image_render_bundle_cache) == 0
    assert len(sim._psf_profile) == 0            # stale-PSF bug fix
    snr2 = sim.get_image_snr(time=60)["snr"]
    assert snr2 < snr1


def test_update_jitter_clears_caches():
    sim = _bright_sim(16)
    sim.get_image_snr(time=60)
    sim.update(jitter_sigma=50)
    assert len(sim._image_render_bundle_cache) == 0
    assert len(sim._psf_profile) == 0


def test_set_sensor_clears_render_cache():
    sim = _bright_sim(16)
    sim.get_image_snr(time=60)
    assert len(sim._image_render_bundle_cache) == 1
    sim.set_sensor(sim.sensor)            # re-setting must drop the render cache
    assert len(sim._image_render_bundle_cache) == 0


def test_set_telescope_clears_render_cache():
    sim = _bright_sim(16)
    sim.get_image_snr(time=60)
    assert len(sim._image_render_bundle_cache) == 1
    sim.set_telescope(sim.telescope)      # re-setting must drop the render cache
    assert len(sim._image_render_bundle_cache) == 0


def test_reset_clears_render_cache():
    sim = _bright_sim(16)
    sim.get_image_snr(time=60)
    assert len(sim._image_render_bundle_cache) == 1
    sim.reset()
    assert len(sim._image_render_bundle_cache) == 0


def test_render_cache_tracks_direct_telescope_jitter_change():
    sim = _bright_sim(16)
    snr1 = sim.get_image_snr(time=60)["snr"]
    sim.telescope.update(jitter_sigma=80)   # direct mutation, bypasses Simulation.update
    snr2 = sim.get_image_snr(time=60)["snr"]
    assert snr2 < snr1                       # more jitter -> lower fixed-aperture SNR, not a stale hit


def test_get_image_snr_array_time():
    sim = _bright_sim(16)
    times = np.array([30., 60., 120.])
    out = sim.get_image_snr(time=times)
    assert np.shape(out["snr"]) == (3,)
    assert np.shape(out["n_pix"]) == (3,)
    assert out["n_pix"].dtype.kind == "i"
    # monotonic increasing SNR with exposure time
    assert out["snr"][0] < out["snr"][1] < out["snr"][2]
    # each element matches the corresponding scalar call across all keys
    for i, t in enumerate(times):
        scalar = sim.get_image_snr(time=float(t))
        for key in ("snr", "signal_e", "noise_e", "enclosed_fraction", "r_aper_mas"):
            assert out[key][i] == pytest.approx(scalar[key], rel=1e-9)
        assert int(out["n_pix"][i]) == scalar["n_pix"]


def test_get_image_snr_array_time_optimize():
    sim = _bright_sim(16)
    times = np.array([30., 300., 3000.])
    out = sim.get_image_snr(time=times, optimize=True)
    assert np.shape(out["r_aper_mas"]) == (3,)
    # each element matches the corresponding scalar optimize call
    for i, t in enumerate(times):
        scalar = sim.get_image_snr(time=float(t), optimize=True)
        assert out["snr"][i] == pytest.approx(scalar["snr"], rel=1e-9)
        assert out["r_aper_mas"][i] == pytest.approx(scalar["r_aper_mas"], rel=1e-9)


def test_get_image_snr_scalar_still_dict_of_floats():
    sim = _bright_sim(16)
    out = sim.get_image_snr(time=60)
    assert isinstance(out["snr"], float)
    assert isinstance(out["n_pix"], int)
