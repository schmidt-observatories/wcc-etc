# tests/test_psf_aware_saturation.py
import warnings
import numpy as np
import astropy.units as u
import wcc_etc


def _scene(mag):
    return wcc_etc.get_scene(name='G2V', mag=mag, background='zodi',
                             bandpass='johnson_v',
                             background_prop={'bandpass': 'johnson_v', 'mag': 22.5})


def test_defocus_peak_fraction_far_below_airy():
    # zwo:bb2 = broadband + 2-wave defocus; zwo:r = in-focus
    sim_def = wcc_etc.Simulation.from_sensorfilter('zwo:bb2', _scene(15.0))
    b = sim_def._image_render_bundle(sim_def._default_psf, None, 128, 11)
    airy_b = sim_def._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
    assert b["psf_norm"].max() < 0.2 * airy_b["psf_norm"].max()  # defocus spreads the peak


def test_is_saturated_agrees_with_image_snr_flag():
    sim = wcc_etc.Simulation.from_sensorfilter('zwo:bb2', _scene(13.0))
    for t in [0.05, 0.1, 0.5, 2.0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flag_snr = bool(sim.get_image_snr(time=t)["saturated"])
            flag_sat = bool(sim.is_saturated(t))
        assert flag_sat == flag_snr, f"disagreement at t={t}"


def test_get_peak_pixel_array_time_linear():
    sim = wcc_etc.Simulation.from_sensorfilter('zwo:r', _scene(16.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = sim.get_peak_pixel(1.0, units="e-").to(u.electron).value
        p2 = sim.get_peak_pixel(np.array([1.0, 2.0]), units="e-").to(u.electron).value
    assert p2.shape == (2,)
    assert p2[0] == __import__("pytest").approx(p1, rel=1e-6)
    assert p2[1] == __import__("pytest").approx(2 * p1, rel=1e-6)
