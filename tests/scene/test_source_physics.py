"""Physics-based spot checks for parametric source spectra.

Each test validates against an independent analytic ground truth (Planck's law,
Wien's law, the F_nu/F_lambda relation, Gaussian profiles, Pogson scaling).
"""

import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c, h, k_B
from scipy.integrate import trapezoid
from synphot import units as su

from wcc_etc.scene import SceneElement

warnings.simplefilter("ignore")


def _flam(spectrum, wave_AA):
    return spectrum(np.asarray(wave_AA) * u.AA, flux_unit=su.FLAM).value


def _planck_lambda(wave_AA, teff):
    lam = (np.asarray(wave_AA) * u.AA).to(u.m).value
    return 1.0 / lam**5 / (np.expm1((h.value * c.value) / (lam * k_B.value * teff)))


class TestBlackbody:
    @pytest.mark.parametrize("teff", [3500, 5777, 9000])
    @pytest.mark.parametrize(
        "pair", [(4000.0, 6000.0), (4500.0, 7500.0), (5000.0, 9000.0)]
    )
    def test_matches_planck_function(self, teff, pair):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": teff, "mag": 15, "bandpass": "johnson_v"}
        )
        sp = bb.get_spectrum(apply_mag=False)
        l1, l2 = pair
        model_ratio = _flam(sp, l1) / _flam(sp, l2)
        planck_ratio = _planck_lambda(l1, teff) / _planck_lambda(l2, teff)
        assert np.isclose(model_ratio, planck_ratio, rtol=1e-3)

    @pytest.mark.parametrize("teff", [4000, 5777, 8000])
    def test_peak_follows_wien_law(self, teff):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": teff, "mag": 15, "bandpass": "johnson_v"}
        )
        sp = bb.get_spectrum(apply_mag=False)
        w = np.arange(2000.0, 30000.0, 2.0)
        lam_peak = w[np.argmax(_flam(sp, w))]
        lam_expected = 2.8977719e7 / teff
        assert abs(lam_peak - lam_expected) < 10.0


class TestFlat:
    def test_fnu_is_constant_and_flam_falls_as_lambda_squared(self):
        fl = SceneElement.from_config(
            {"spectrum": "flat", "mag": 18, "bandpass": "johnson_v"}
        )
        sp = fl.get_spectrum(apply_mag=False)
        w = np.array([4000.0, 6000.0, 8000.0])
        fnu = sp(w * u.AA, flux_unit=u.Jy).value
        assert np.allclose(fnu, fnu[0], rtol=1e-3)
        l1, l2 = 4000.0, 8000.0
        assert np.isclose(_flam(sp, l1) / _flam(sp, l2), (l2 / l1) ** 2, rtol=1e-3)

    def test_flat_flam_is_constant_in_flam(self):
        ff = SceneElement.from_config(
            {
                "spectrum": "flat",
                "flat_unit": "flam",
                "mag": 18,
                "bandpass": "johnson_v",
            }
        )
        sp = ff.get_spectrum(apply_mag=False)
        flam = _flam(sp, np.array([4000.0, 6000.0, 8000.0]))
        assert np.allclose(flam, flam[0], rtol=1e-3)


class TestPowerlaw:
    @pytest.mark.parametrize("alpha", [-2.0, -1.0, 0.0, 1.5])
    def test_flam_follows_power_law(self, alpha):
        pl = SceneElement.from_config(
            {"spectrum": "powerlaw", "alpha": alpha, "mag": 18, "bandpass": "johnson_v"}
        )
        sp = pl.get_spectrum(apply_mag=False)
        l1, l2 = 4500.0, 7500.0
        assert np.isclose(_flam(sp, l1) / _flam(sp, l2), (l1 / l2) ** alpha, rtol=1e-3)

    def test_alpha_zero_equals_flat_flam(self):
        pl = SceneElement.from_config(
            {"spectrum": "powerlaw", "alpha": 0.0, "mag": 18, "bandpass": "johnson_v"}
        )
        sp = pl.get_spectrum(apply_mag=False)
        flam = _flam(sp, np.array([4000.0, 6000.0, 8000.0]))
        assert np.allclose(flam, flam[0], rtol=1e-3)


class TestEmissionLines:
    def test_gaussian_peak_amplitude(self):
        flux, fwhm = 1e-15, 3.0
        em = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [{"wave": 6563, "flux": flux, "fwhm": fwhm}],
                "mag": None,
            }
        )
        sp = em.get_spectrum()
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_peak = flux / (sigma * np.sqrt(2.0 * np.pi))
        assert np.isclose(_flam(sp, 6563), expected_peak, rtol=1e-3)

    def test_fwhm_matches_input(self):
        fwhm = 4.0
        em = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [{"wave": 6563, "flux": 1e-15, "fwhm": fwhm}],
                "mag": None,
            }
        )
        sp = em.get_spectrum()
        w = np.arange(6540.0, 6586.0, 0.02)
        flam = _flam(sp, w)
        half = flam.max() / 2.0
        above = w[flam >= half]
        measured_fwhm = above[-1] - above[0]
        assert np.isclose(measured_fwhm, fwhm, atol=0.1)

    def test_two_line_flux_ratio_preserved(self):
        em = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [
                    {"wave": 6563, "flux": 1e-15, "fwhm": 3},
                    {"wave": 6583, "flux": 4e-16, "fwhm": 3},
                ],
                "mag": None,
            }
        )
        sp = em.get_spectrum()
        w1 = np.arange(6545.0, 6573.0, 0.02)
        w2 = np.arange(6573.0, 6601.0, 0.02)
        flux1 = trapezoid(_flam(sp, w1), w1)
        flux2 = trapezoid(_flam(sp, w2), w2)
        assert np.isclose(flux1 / flux2, 1e-15 / 4e-16, rtol=1e-2)


class TestSourceCountrate:
    @pytest.mark.parametrize(
        "name,extra",
        [
            ("blackbody", {"teff": 5777}),
            ("flat", {}),
            ("powerlaw", {"alpha": -1.0}),
        ],
    )
    def test_countrate_follows_pogson_scaling(self, name, extra):
        import warnings

        import wcc_etc

        rates = []
        for mag in (15, 20):
            scene = wcc_etc.get_scene(
                name=name,
                mag=mag,
                bandpass="johnson_r",
                background="zodi",
                background_prop={"bandpass": "johnson_r", "mag": 22.5},
                **extra,
            )
            sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rates.append(sim.get_countrates(units="e/s")["source"].value)
        assert np.isclose(rates[0] / rates[1], 100.0, rtol=1e-2)
