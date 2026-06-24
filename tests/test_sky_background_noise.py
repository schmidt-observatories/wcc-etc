"""Regression tests guarding the sky/zodiacal background term in the SNR noise.

These exist because the PSF-aware SNR path (get_snr / get_image_snr /
get_image_exptime_for_snr) once dropped the per-pixel sky (background) rate from
its noise budget entirely, overstating SNR by up to ~2x in the sky-limited
regime. The tests below pin the fix against three *independent* checks:

  1. the closed-form CCD ("CCD equation") noise, computed by hand including sky;
  2. the Monte-Carlo scatter of noisy rendered images (ImageSimulator.simulate),
     which has always included the sky term;
  3. the qualitative requirement that a brighter sky must lower the SNR.

None of these re-uses the get_image_snr noise expression, so they would all fail
if the sky term were dropped again.
"""
import warnings

import numpy as np
import pytest
import astropy.units as u

import wcc_etc
from wcc_etc.psfsim import AiryPSF, ImageSimulator, psf_center

warnings.simplefilter("ignore")

NPIX, OVERSAMPLE = 128, 11   # match get_snr / get_image_snr defaults


def _sim(source_mag=24.0, sky_mag=18.0, sensor="sony:r"):
    """G5V point source on a (deliberately bright) zodi sky, AB mags."""
    scene = wcc_etc.get_scene(
        name="G5V", mag=source_mag, magsys="abmag", host=None,
        background="zodi", bandpass="johnson_r",
        background_prop={"bandpass": "johnson_r", "mag": sky_mag, "magsys": "abmag"})
    return wcc_etc.Simulation.from_sensor_and_scene(sensor, scene)


def _ccd_equation_noise(sim, time_s, n_pix):
    """Independent closed-form aperture noise (electrons): sqrt(S + (sky+dark+RN^2)*n_pix).

    S here is the *source* signal inside the aperture, reconstructed from the
    enclosed fraction; sky/dark/read are per-pixel. This is the standard CCD
    equation and is computed without touching aperture_snr_radial.
    """
    comps = sim._count_rate_components()
    res = sim.get_snr(time=time_s, r_aper_mas=70, warn=False)
    sky = comps["background_rate_per_pix"] + comps["diffuse_rate_per_pix"]
    dark = sim.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
    rn = sim.sensor.read_noise.to(u.electron / u.pix).value
    signal = res["signal_e"]
    noise = np.sqrt(signal + (sky * time_s + dark * time_s + rn ** 2) * n_pix)
    return signal, noise


# ---------------------------------------------------------------------------
# 1. Closed-form CCD equation (sky-dominated): the reported noise must match.
# ---------------------------------------------------------------------------
def test_get_snr_noise_matches_ccd_equation_with_sky():
    sim = _sim(source_mag=24.0, sky_mag=18.0)   # bright sky -> sky dominates
    t = 300.0
    res = sim.get_snr(time=t, r_aper_mas=70, warn=False)

    # Sanity: this configuration really is sky-dominated, so the bug (if present)
    # would be large -> the test is meaningful.
    comps = sim._count_rate_components()
    sky_e = comps["background_rate_per_pix"] * t * res["n_pix"]
    assert sky_e > res["signal_e"], "test scene is not sky-dominated; tighten it"

    signal, noise_ccd = _ccd_equation_noise(sim, t, res["n_pix"])
    assert res["signal_e"] == pytest.approx(signal, rel=1e-9)
    assert res["noise_e"] == pytest.approx(noise_ccd, rel=1e-6)
    assert res["snr"] == pytest.approx(signal / noise_ccd, rel=1e-6)


def test_get_snr_noise_matches_ccd_equation_faint_readnoise_limited():
    # Also guard the read-noise-limited corner (small sky contribution).
    sim = _sim(source_mag=25.4, sky_mag=22.5)
    t = 60.0
    res = sim.get_snr(time=t, r_aper_mas=70, warn=False)
    signal, noise_ccd = _ccd_equation_noise(sim, t, res["n_pix"])
    assert res["noise_e"] == pytest.approx(noise_ccd, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Monte-Carlo: scatter of noisy aperture sums must equal the reported noise.
#    ImageSimulator.simulate has always included the sky, so this is independent.
# ---------------------------------------------------------------------------
def test_get_snr_noise_matches_monte_carlo_image_scatter():
    sim = _sim(source_mag=24.0, sky_mag=19.0)
    t = 120.0
    res = sim.get_snr(time=t, r_aper_mas=70, warn=False, npix=NPIX, oversample=OVERSAMPLE)
    n_pix = res["n_pix"]

    # Reproduce get_snr's circular aperture: the n_pix pixels nearest the PSF
    # centroid (same convention as _radial_cumulative -> stable argsort of radius).
    imsim = ImageSimulator(sim, npix=NPIX, oversample=OVERSAMPLE)
    clean = imsim.simulate(time=t, psf=AiryPSF(), add_noise=False)
    xc, yc = psf_center(clean.image_clean)
    yy, xx = np.mgrid[0:NPIX, 0:NPIX]
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2).ravel()
    aper_idx = np.argsort(r, kind="stable")[:n_pix]

    # The expected variance of the raw aperture sum equals noise_e**2:
    #   Var = source_enclosed + (sky + dark + RN^2) * n_pix.
    n_frames = 600
    sums = np.empty(n_frames)
    for i in range(n_frames):
        img = imsim.simulate(time=t, psf=AiryPSF(), add_noise=True, seed=i)
        sums[i] = img.image_e.ravel()[aper_idx].sum()

    mc_noise = sums.std(ddof=1)
    # statistical error on an std from N samples ~ 1/sqrt(2N) ~ 2.9% here; allow 8%.
    assert mc_noise == pytest.approx(res["noise_e"], rel=0.08), (
        f"MC noise {mc_noise:.1f} e- vs reported {res['noise_e']:.1f} e- "
        f"(sky term missing would make reported ~{np.sqrt(res['signal_e']):.1f})")


# ---------------------------------------------------------------------------
# 3. Qualitative: a brighter sky must REDUCE the SNR (it previously did nothing).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sky_bright,sky_faint", [(18.0, 22.5), (19.0, 21.0)])
def test_brighter_sky_lowers_snr(sky_bright, sky_faint):
    t = 200.0
    snr_bright = _sim(24.0, sky_bright).get_snr(time=t, r_aper_mas=70, warn=False)["snr"]
    snr_faint = _sim(24.0, sky_faint).get_snr(time=t, r_aper_mas=70, warn=False)["snr"]
    assert snr_bright < snr_faint, (
        f"brighter sky (mag {sky_bright}) did not lower SNR "
        f"({snr_bright:.2f}) vs fainter sky (mag {sky_faint}): {snr_faint:.2f}")


# ---------------------------------------------------------------------------
# 4. get_image_exptime_for_snr must use the same sky-inclusive noise:
#    the time it returns must reproduce the target SNR through get_snr.
# ---------------------------------------------------------------------------
def test_exptime_for_snr_round_trips_with_sky():
    sim = _sim(source_mag=24.0, sky_mag=19.0)
    target = 15.0
    res = sim.get_image_exptime_for_snr(target, r_aper_mas=70, warn=False)
    chk = sim.get_snr(time=res["time_s"], r_aper_mas=70, warn=False)
    assert chk["snr"] == pytest.approx(target, rel=1e-3)
