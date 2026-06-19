import warnings
import numpy as np
import astropy.units as u
import wcc_etc


def _sim():
    scene = wcc_etc.get_scene(name='G2V', mag=15.0, background='zodi',
                              bandpass='johnson_v',
                              background_prop={'bandpass': 'johnson_v', 'mag': 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene('sony:r', scene)


def test_source_total_matches_pre_ee_countrate():
    sim = _sim()
    comp = sim._count_rate_components()
    # legacy in-aperture source / ee_at_aper == total source rate (PSF-independent)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = sim.get_countrates(units="e/s", as_dict=True)
    ee = sim.psf_profile["ee_at_aper"]
    legacy_total = (legacy["source"] / ee).to(u.electron / u.s).value
    assert comp["source_rate_total"] == __import__("pytest").approx(legacy_total, rel=1e-6)


def test_background_per_pix_removes_spurious_ee_factor():
    sim = _sim()
    comp = sim._count_rate_components()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = sim.get_countrates(units="e/s", as_dict=True)
    prof = sim.psf_profile
    n_psf = prof["num_psf_pixels"].value if hasattr(prof["num_psf_pixels"], "value") else prof["num_psf_pixels"]
    ee = prof["ee_at_aper"]
    legacy_bkg_per_pix = (legacy["background"] / n_psf).to(u.electron / u.s).value
    # new per-pixel sky == legacy / ee_at_aper  (the spurious factor removed)
    assert comp["background_rate_per_pix"] == __import__("pytest").approx(legacy_bkg_per_pix / ee, rel=1e-4)
    assert comp["background_rate_per_pix"] > 0
