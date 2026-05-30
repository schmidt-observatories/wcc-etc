"""Physics-based spot checks for the parametric source spectra.

Each test compares a built source against an *independent* analytic ground
truth (Planck's law, Wien's law, the F_nu/F_lambda relation, a Gaussian
profile, Pogson's magnitude scaling) rather than re-deriving the expected
value from synphot, so they actually validate the physics rather than the
implementation echoing itself.
"""
import warnings

import numpy as np
import pytest
import astropy.units as u
from astropy.constants import h, c, k_B
from scipy.integrate import trapezoid
from synphot import units as su

from wcc_etc.scene import SceneElement

# silence the deprecation/PSF chatter that some paths emit
warnings.simplefilter("ignore")


def _flam(spectrum, wave_AA):
    """F_lambda (erg/s/cm^2/A) of a synphot spectrum at wave (scalar or array, A)."""
    return spectrum(np.asarray(wave_AA) * u.AA, flux_unit=su.FLAM).value


def _planck_lambda(wave_AA, teff):
    """Analytic Planck B_lambda(T) up to a constant (energy per wavelength)."""
    lam = (np.asarray(wave_AA) * u.AA).to(u.m).value
    return 1.0 / lam ** 5 / (np.expm1((h.value * c.value) / (lam * k_B.value * teff)))


# ----------------------------------------------------------------------------
#  Blackbody
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("teff", [3500, 5777, 9000])
@pytest.mark.parametrize("pair", [(4000.0, 6000.0), (4500.0, 7500.0), (5000.0, 9000.0)])
def test_blackbody_matches_planck_function(teff, pair):
    bb = SceneElement.from_config({"spectrum": "blackbody", "teff": teff,
                                   "mag": 15, "bandpass": "johnson_v"})
    sp = bb.get_spectrum(apply_mag=False)  # shape only; normalization is arbitrary
    l1, l2 = pair
    model_ratio = _flam(sp, l1) / _flam(sp, l2)
    planck_ratio = _planck_lambda(l1, teff) / _planck_lambda(l2, teff)
    assert np.isclose(model_ratio, planck_ratio, rtol=1e-3)


@pytest.mark.parametrize("teff", [4000, 5777, 8000])
def test_blackbody_peak_follows_wien_law(teff):
    bb = SceneElement.from_config({"spectrum": "blackbody", "teff": teff,
                                   "mag": 15, "bandpass": "johnson_v"})
    sp = bb.get_spectrum(apply_mag=False)
    w = np.arange(2000.0, 30000.0, 2.0)
    flam = _flam(sp, w)
    lam_peak = w[np.argmax(flam)]
    # Wien displacement for the energy distribution: lambda_max * T = 2.8977719e7 A.K
    lam_expected = 2.8977719e7 / teff
    assert abs(lam_peak - lam_expected) < 10.0  # grid is 2 A


# ----------------------------------------------------------------------------
#  Flat
# ----------------------------------------------------------------------------
def test_flat_fnu_is_constant_fnu_and_lambda_minus_two_in_flam():
    fl = SceneElement.from_config({"spectrum": "flat", "mag": 18,
                                   "bandpass": "johnson_v"})
    sp = fl.get_spectrum(apply_mag=False)
    w = np.array([4000.0, 6000.0, 8000.0])
    fnu = sp(w * u.AA, flux_unit=u.Jy).value
    assert np.allclose(fnu, fnu[0], rtol=1e-3)               # flat in F_nu
    # F_lambda = F_nu * c / lambda^2  ->  ratio = (l2/l1)^2
    l1, l2 = 4000.0, 8000.0
    assert np.isclose(_flam(sp, l1) / _flam(sp, l2), (l2 / l1) ** 2, rtol=1e-3)


def test_flat_flam_is_constant_in_flam():
    ff = SceneElement.from_config({"spectrum": "flat", "flat_unit": "flam",
                                   "mag": 18, "bandpass": "johnson_v"})
    sp = ff.get_spectrum(apply_mag=False)
    flam = _flam(sp, np.array([4000.0, 6000.0, 8000.0]))
    assert np.allclose(flam, flam[0], rtol=1e-3)


# ----------------------------------------------------------------------------
#  Power law
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [-2.0, -1.0, 0.0, 1.5])
def test_powerlaw_flam_follows_power_law(alpha):
    pl = SceneElement.from_config({"spectrum": "powerlaw", "alpha": alpha,
                                   "mag": 18, "bandpass": "johnson_v"})
    sp = pl.get_spectrum(apply_mag=False)
    l1, l2 = 4500.0, 7500.0
    # F_lambda proportional to lambda**alpha
    assert np.isclose(_flam(sp, l1) / _flam(sp, l2), (l1 / l2) ** alpha, rtol=1e-3)


def test_powerlaw_alpha_zero_equals_flat_flam():
    pl = SceneElement.from_config({"spectrum": "powerlaw", "alpha": 0.0,
                                   "mag": 18, "bandpass": "johnson_v"})
    sp = pl.get_spectrum(apply_mag=False)
    flam = _flam(sp, np.array([4000.0, 6000.0, 8000.0]))
    assert np.allclose(flam, flam[0], rtol=1e-3)


# ----------------------------------------------------------------------------
#  Emission lines
# ----------------------------------------------------------------------------
def test_emission_gaussian_peak_amplitude():
    flux, fwhm = 1e-15, 3.0
    em = SceneElement.from_config({"spectrum": "emission",
                                   "lines": [{"wave": 6563, "flux": flux, "fwhm": fwhm}],
                                   "mag": None})
    sp = em.get_spectrum()
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # peak of a normalized Gaussian of integral `flux` is flux / (sigma*sqrt(2pi))
    expected_peak = flux / (sigma * np.sqrt(2.0 * np.pi))
    assert np.isclose(_flam(sp, 6563), expected_peak, rtol=1e-3)


def test_emission_gaussian_fwhm_matches_input():
    fwhm = 4.0
    em = SceneElement.from_config({"spectrum": "emission",
                                   "lines": [{"wave": 6563, "flux": 1e-15, "fwhm": fwhm}],
                                   "mag": None})
    sp = em.get_spectrum()
    w = np.arange(6540.0, 6586.0, 0.02)
    flam = _flam(sp, w)
    half = flam.max() / 2.0
    above = w[flam >= half]
    measured_fwhm = above[-1] - above[0]
    assert np.isclose(measured_fwhm, fwhm, atol=0.1)  # grid-limited


def test_emission_two_line_flux_ratio_is_preserved():
    em = SceneElement.from_config(
        {"spectrum": "emission",
         "lines": [{"wave": 6563, "flux": 1e-15, "fwhm": 3},    # Halpha
                   {"wave": 6583, "flux": 4e-16, "fwhm": 3}],   # NII
         "mag": None})
    sp = em.get_spectrum()
    # integrate each well-separated line independently and compare the ratio
    w1 = np.arange(6545.0, 6573.0, 0.02)
    w2 = np.arange(6573.0, 6601.0, 0.02)
    flux1 = trapezoid(_flam(sp, w1), w1)
    flux2 = trapezoid(_flam(sp, w2), w2)
    assert np.isclose(flux1 / flux2, 1e-15 / 4e-16, rtol=1e-2)


# ----------------------------------------------------------------------------
#  Integration level: the source actually drives the ETC
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name,extra", [("blackbody", {"teff": 5777}),
                                        ("flat", {}),
                                        ("powerlaw", {"alpha": -1.0})])
def test_source_countrate_follows_pogson_scaling(name, extra):
    import wcc_etc
    rates = []
    for mag in (15, 20):  # 5 mag fainter -> 100x less flux
        scene = wcc_etc.get_scene(name=name, mag=mag, bandpass="johnson_r",
                                  background="zodi",
                                  background_prop={"bandpass": "johnson_r", "mag": 22.5},
                                  **extra)
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        rates.append(sim.get_countrates(units="e/s")["source"].value)
    assert np.isclose(rates[0] / rates[1], 100.0, rtol=1e-2)
