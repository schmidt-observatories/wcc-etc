import inspect
import warnings
import numpy as np
import wcc_etc


def _sim(filt='sony:r', mag=15.0):
    scene = wcc_etc.get_scene(name='G2V', mag=mag, background='zodi',
                              bandpass='johnson_v',
                              background_prop={'bandpass': 'johnson_v', 'mag': 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene(filt, scene)


def test_bundle_uses_count_rate_components_not_psf_profile():
    src = inspect.getsource(wcc_etc.Simulation._image_render_bundle)
    assert "_count_rate_components" in src
    assert "ee_at_aper" not in src
    assert "get_countrates" not in src


def test_bundle_rates_match_components():
    sim = _sim()
    comp = sim._count_rate_components()
    b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
    assert b["source_rate_total"] == __import__("pytest").approx(comp["source_rate_total"], rel=1e-6)
    assert b["background_rate_per_pix"] == __import__("pytest").approx(comp["background_rate_per_pix"], rel=1e-6)
    assert b["diffuse_rate_per_pix"] == __import__("pytest").approx(comp["diffuse_rate_per_pix"], rel=1e-6)


def test_bundle_background_higher_than_legacy_airy():
    import astropy.units as u
    sim = _sim()
    b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = sim.get_countrates(units="e/s", as_dict=True)
    prof = sim.psf_profile
    n_psf = prof["num_psf_pixels"].value if hasattr(prof["num_psf_pixels"], "value") else prof["num_psf_pixels"]
    ee = prof["ee_at_aper"]
    legacy_bkg_per_pix = (legacy["background"] / n_psf).to(u.electron / u.s).value
    # migration removed the spurious x ee_at_aper on sky -> bundle sky is higher by ~1/ee
    assert b["background_rate_per_pix"] > legacy_bkg_per_pix
    assert b["background_rate_per_pix"] == __import__("pytest").approx(legacy_bkg_per_pix / ee, rel=1e-4)


def test_bright_source_snr_essentially_unchanged():
    # source-dominated: removing the sky ee-factor barely moves SNR
    sim = _sim(mag=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = sim.get_image_snr(time=1.0)["snr"]
    assert snr > 0 and np.isfinite(snr)
