"""Tests for SceneElement: parametric spectra, magnitude systems, and mutability."""

import numpy as np
import astropy.units as u
from scipy.integrate import trapezoid
from synphot import SpectralElement, Observation, SourceSpectrum, units as su
from wcc_etc.io import expand_path
from wcc_etc.scene import SceneElement
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _observed_abmag(spectrum, band_name="johnson_v"):
    band = SpectralElement.from_filter(band_name)
    return Observation(spectrum, band, force="extrap").effstim(u.ABmag).value


def _flam_ratio(spectrum, l1, l2):
    f = spectrum(np.array([l1, l2]) * u.AA, flux_unit=su.FLAM).value
    return f[0] / f[1]


def _planck_ratio(l1, l2, teff):
    from astropy.constants import h, c, k_B

    def b(lam_AA):
        lam = (lam_AA * u.AA).to(u.m).value
        return 1.0 / lam**5 / np.expm1((h.value * c.value) / (lam * k_B.value * teff))

    return b(l1) / b(l2)


def _band_ab_vega_offset(band_name):
    vega = SourceSpectrum.from_vega()
    band = SpectralElement.from_filter(band_name)
    obs = Observation(vega, band, force="extrap")
    return obs.effstim(u.ABmag).value - obs.effstim(su.VEGAMAG, vegaspec=vega).value


def _inband_flam(scene_element, band_name):
    band = SpectralElement.from_filter(band_name)
    return (
        Observation(scene_element.get_spectrum(), band, force="extrap")
        .effstim(su.FLAM)
        .value
    )


def _pickles_elements():
    base = {"mag": 20, "magsys": "abmag", "bandpass": "johnson_v"}
    s1 = SceneElement.from_config(
        {
            "spectrum": expand_path(
                "astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits"
            )
        }
        | base
    )
    s2 = SceneElement.from_config({"spectrum": "uk_55"} | base)
    s3 = SceneElement.from_config({"spectrum": "G5IV"} | base)
    l1, f1 = s1.get_spectrum(as_array=True)
    l2, f2 = s2.get_spectrum(as_array=True)
    l3, f3 = s3.get_spectrum(as_array=True)
    return l1, f1, l2, f2, l3, f3


# ---------------------------------------------------------------------------
# Spectrum type tests
# ---------------------------------------------------------------------------


class TestPicklesSpectrum:
    def test_alias_uk55_flux_matches_direct(self):
        l1, f1, l2, f2, l3, f3 = _pickles_elements()
        assert np.all(f1 == f2)

    def test_alias_g5iv_flux_matches_direct(self):
        l1, f1, l2, f2, l3, f3 = _pickles_elements()
        assert np.all(f1 == f3)

    def test_alias_uk55_wavelength_matches_direct(self):
        l1, f1, l2, f2, l3, f3 = _pickles_elements()
        assert np.all(l1 == l2)

    def test_alias_g5iv_wavelength_matches_direct(self):
        l1, f1, l2, f2, l3, f3 = _pickles_elements()
        assert np.all(l1 == l3)


class TestBlackbodySource:
    def test_roundtrips_abmag_magnitude(self):
        se = SceneElement.from_config(
            {
                "spectrum": "blackbody",
                "teff": 5777,
                "mag": 15,
                "magsys": "abmag",
                "bandpass": "johnson_v",
            }
        )
        assert abs(_observed_abmag(se.get_spectrum()) - 15) < 0.01

    def test_shape_matches_get_blackbody_flux(self):
        from wcc_etc.wcc_etc import get_blackbody_flux

        se = SceneElement.from_config(
            {
                "spectrum": "blackbody",
                "teff": 5777,
                "mag": 15,
                "magsys": "abmag",
                "bandpass": "johnson_v",
            }
        )
        sp = se.get_spectrum()
        w = np.array([4000.0, 6000.0, 8000.0])
        flam = sp(w * u.AA, flux_unit=su.FLAM).value
        ref = np.asarray(get_blackbody_flux(w, 5777, 15))
        assert np.allclose(flam / flam[0], ref / ref[0], rtol=1e-3)

    def test_blackbody_initial_teff_shape_correct(self):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": 5777, "mag": 15, "bandpass": "johnson_v"}
        )
        assert np.isclose(
            _flam_ratio(bb.get_spectrum(apply_mag=False), 4500, 7500),
            _planck_ratio(4500, 7500, 5777),
            rtol=1e-3,
        )

    def test_blackbody_update_teff_changes_shape(self):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": 5777, "mag": 15, "bandpass": "johnson_v"}
        )
        bb.update(teff=3000)
        assert np.isclose(
            _flam_ratio(bb.get_spectrum(apply_mag=False), 4500, 7500),
            _planck_ratio(4500, 7500, 3000),
            rtol=1e-3,
        )


class TestFlatSource:
    def test_flat_fnu_is_constant(self):
        se = SceneElement.from_config(
            {"spectrum": "flat", "mag": 18, "magsys": "abmag", "bandpass": "johnson_v"}
        )
        sp = se.get_spectrum()
        w = np.array([4000.0, 6000.0, 8000.0]) * u.AA
        fnu = sp(w, flux_unit=u.Jy).value
        assert np.allclose(fnu, fnu[0], rtol=1e-3)

    def test_flat_roundtrips_abmag(self):
        se = SceneElement.from_config(
            {"spectrum": "flat", "mag": 18, "magsys": "abmag", "bandpass": "johnson_v"}
        )
        sp = se.get_spectrum()
        assert abs(_observed_abmag(sp) - 18) < 0.01

    def test_flat_fnu_flam_is_not_flat(self):
        fl = SceneElement.from_config(
            {"spectrum": "flat", "flat_unit": "fnu", "mag": 15, "bandpass": "johnson_v"}
        )
        sp = fl.get_spectrum(apply_mag=False)
        assert (
            sp(4000 * u.AA, flux_unit=su.FLAM).value
            > sp(8000 * u.AA, flux_unit=su.FLAM).value
        )

    def test_flat_update_to_flam_is_flat(self):
        fl = SceneElement.from_config(
            {"spectrum": "flat", "flat_unit": "fnu", "mag": 15, "bandpass": "johnson_v"}
        )
        fl.update(flat_unit="flam")
        sp = fl.get_spectrum(apply_mag=False)
        flam = sp(np.array([4000.0, 6000.0, 8000.0]) * u.AA, flux_unit=su.FLAM).value
        assert np.allclose(flam, flam[0], rtol=1e-3)


class TestPowerlawSource:
    def test_slope(self):
        alpha = -1.0
        se = SceneElement.from_config(
            {
                "spectrum": "powerlaw",
                "alpha": alpha,
                "mag": 18,
                "magsys": "abmag",
                "bandpass": "johnson_v",
            }
        )
        sp = se.get_spectrum()
        w1, w2 = 4000.0, 8000.0
        f1 = sp(w1 * u.AA, flux_unit=su.FLAM).value
        f2 = sp(w2 * u.AA, flux_unit=su.FLAM).value
        assert np.isclose(f1 / f2, (w1 / w2) ** alpha, rtol=1e-3)

    def test_update_alpha_rebuilds_spectrum(self):
        pl = SceneElement.from_config(
            {"spectrum": "powerlaw", "alpha": -1.0, "mag": 15, "bandpass": "johnson_v"}
        )
        pl.update(alpha=2.0)
        assert np.isclose(
            _flam_ratio(pl.get_spectrum(apply_mag=False), 4500, 7500),
            (4500 / 7500) ** 2.0,
            rtol=1e-3,
        )

    def test_update_lambda_ref_rebuilds_spectrum(self):
        pl = SceneElement.from_config(
            {
                "spectrum": "powerlaw",
                "alpha": -1.0,
                "lambda_ref": 5500,
                "mag": 15,
                "bandpass": "johnson_v",
            }
        )
        sp = pl.get_spectrum(apply_mag=False)
        assert np.isclose(sp(5500 * u.AA, flux_unit=su.FLAM).value, 1.0, rtol=1e-6)

    def test_update_lambda_ref_moves_reference(self):
        pl = SceneElement.from_config(
            {
                "spectrum": "powerlaw",
                "alpha": -1.0,
                "lambda_ref": 5500,
                "mag": 15,
                "bandpass": "johnson_v",
            }
        )
        pl.update(lambda_ref=6000)
        sp = pl.get_spectrum(apply_mag=False)
        assert np.isclose(sp(6000 * u.AA, flux_unit=su.FLAM).value, 1.0, rtol=1e-6)


class TestEmissionSource:
    def test_recovers_absolute_flux(self):
        flux = 1e-15
        se = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [{"wave": 6563, "flux": flux, "fwhm": 3}],
                "mag": None,
            }
        )
        sp = se.get_spectrum()
        w = np.arange(6500, 6630, 0.05) * u.AA
        flam = sp(w, flux_unit=su.FLAM).value
        integral = trapezoid(flam, w.value)
        assert np.isclose(integral, flux, rtol=1e-2)

    def test_recovers_centroid(self):
        flux = 1e-15
        se = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [{"wave": 6563, "flux": flux, "fwhm": 3}],
                "mag": None,
            }
        )
        sp = se.get_spectrum()
        w = np.arange(6500, 6630, 0.05) * u.AA
        flam = sp(w, flux_unit=su.FLAM).value
        integral = trapezoid(flam, w.value)
        centroid = trapezoid(flam * w.value, w.value) / integral
        assert abs(centroid - 6563) < 0.5

    def test_multiple_lines_sum(self):
        se = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [
                    {"wave": 6563, "flux": 1e-15, "fwhm": 3},
                    {"wave": 6583, "flux": 4e-16, "fwhm": 3},
                ],
                "mag": None,
            }
        )
        sp = se.get_spectrum()
        w = np.arange(6400, 6700, 0.05) * u.AA
        flam = sp(w, flux_unit=su.FLAM).value
        integral = trapezoid(flam, w.value)
        assert np.isclose(integral, 1.4e-15, rtol=1e-2)

    def test_update_lines_rebuilds_spectrum(self):
        em = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [{"wave": 6563, "flux": 1e-15, "fwhm": 3}],
                "mag": None,
            }
        )
        p0 = em.get_spectrum()(6563 * u.AA, flux_unit=su.FLAM).value
        em.update(lines=[{"wave": 6563, "flux": 5e-15, "fwhm": 3}])
        p1 = em.get_spectrum()(6563 * u.AA, flux_unit=su.FLAM).value
        assert np.isclose(p1 / p0, 5.0, rtol=1e-3)


class TestFileSource:
    def test_reads_wavelength_and_flux_columns(self, tmp_path):
        specfile = tmp_path / "source.dat"
        np.savetxt(specfile, [[6000.0, 3e-16], [4000.0, 1e-16], [5000.0, 2e-16]])
        se = SceneElement.from_config(
            {"spectrum": "file", "source_file": str(specfile), "mag": None}
        )
        sp = se.get_spectrum(apply_mag=False)
        flam = sp(np.array([4000.0, 5000.0, 6000.0]) * u.AA, flux_unit=su.FLAM).value
        assert np.allclose(flam, [1e-16, 2e-16, 3e-16])

    def test_reads_named_csv_columns(self, tmp_path):
        specfile = tmp_path / "source.csv"
        specfile.write_text("wave_nm,flam\n400,1e-16\n500,2e-16\n600,3e-16\n")
        se = SceneElement.from_config(
            {
                "spectrum": "file",
                "source_file": str(specfile),
                "wave_column": "wave_nm",
                "flux_column": "flam",
                "wave_unit": "nm",
                "flux_unit": "FLAM",
                "mag": None,
            }
        )
        sp = se.get_spectrum(apply_mag=False)
        flam = sp(np.array([4000.0, 5000.0, 6000.0]) * u.AA, flux_unit=su.FLAM).value
        assert np.allclose(flam, [1e-16, 2e-16, 3e-16])

    def test_roundtrips_magnitude(self, tmp_path):
        specfile = tmp_path / "source.dat"
        np.savetxt(
            specfile,
            [[4000.0, 1e-16], [5000.0, 2e-16], [6000.0, 3e-16], [7000.0, 2e-16]],
        )
        se = SceneElement.from_config(
            {
                "spectrum": "file",
                "source_file": str(specfile),
                "mag": 18,
                "bandpass": "johnson_v",
            }
        )
        assert abs(_observed_abmag(se.get_spectrum()) - 18) < 0.01

    def test_file_source_initial_flux(self, tmp_path):
        f1 = tmp_path / "s1.dat"
        f2 = tmp_path / "s2.dat"
        np.savetxt(f1, [[4000.0, 1e-16], [5000.0, 2e-16], [6000.0, 3e-16]])
        np.savetxt(f2, [[4000.0, 3e-16], [5000.0, 2e-16], [6000.0, 1e-16]])
        se = SceneElement.from_config(
            {"spectrum": "file", "source_file": str(f1), "mag": None}
        )
        before = se.get_spectrum(apply_mag=False)(4000 * u.AA, flux_unit=su.FLAM).value
        assert np.isclose(before, 1e-16)

    def test_file_source_update_changes_flux(self, tmp_path):
        f1 = tmp_path / "s1.dat"
        f2 = tmp_path / "s2.dat"
        np.savetxt(f1, [[4000.0, 1e-16], [5000.0, 2e-16], [6000.0, 3e-16]])
        np.savetxt(f2, [[4000.0, 3e-16], [5000.0, 2e-16], [6000.0, 1e-16]])
        se = SceneElement.from_config(
            {"spectrum": "file", "source_file": str(f1), "mag": None}
        )
        se.update(source_file=str(f2))
        after = se.get_spectrum(apply_mag=False)(4000 * u.AA, flux_unit=su.FLAM).value
        assert np.isclose(after, 3e-16)


# ---------------------------------------------------------------------------
# Magnitude system tests
# ---------------------------------------------------------------------------


class TestMagnitudeSystem:
    def test_resolve_abmag_lowercase(self):
        from wcc_etc.scene import _resolve_magsys

        assert _resolve_magsys("abmag") == u.ABmag

    def test_resolve_abmag_uppercase(self):
        from wcc_etc.scene import _resolve_magsys

        assert _resolve_magsys("ABMAG") == u.ABmag

    def test_resolve_abmag_mixed_case(self):
        from wcc_etc.scene import _resolve_magsys

        assert _resolve_magsys("AbMag") == u.ABmag

    def test_resolve_vegamag_lowercase(self):
        from wcc_etc.scene import _resolve_magsys

        assert _resolve_magsys("vegamag") == su.VEGAMAG

    def test_resolve_vegamag_uppercase(self):
        from wcc_etc.scene import _resolve_magsys

        assert _resolve_magsys("VEGAMAG") == su.VEGAMAG

    def test_resolve_passes_through_abmag_unit(self):
        from wcc_etc.scene import _resolve_magsys

        assert _resolve_magsys(u.ABmag) == u.ABmag

    def test_resolve_raises_on_vega(self):
        from wcc_etc.scene import _resolve_magsys

        with pytest.raises(ValueError):
            _resolve_magsys("vega")

    def test_resolve_raises_on_ab(self):
        from wcc_etc.scene import _resolve_magsys

        with pytest.raises(ValueError):
            _resolve_magsys("AB")

    def test_resolve_raises_on_nonsense(self):
        from wcc_etc.scene import _resolve_magsys

        with pytest.raises(ValueError):
            _resolve_magsys("nonsense")

    def test_vegamag_normalization_differs_from_ab(self):
        band_name = "johnson_r"
        se_ab = SceneElement.from_config(
            {"spectrum": "flat", "mag": 15, "magsys": "abmag", "bandpass": band_name}
        )
        se_vg = SceneElement.from_config(
            {"spectrum": "flat", "mag": 15, "magsys": "vegamag", "bandpass": band_name}
        )
        ratio = _inband_flam(se_vg, band_name) / _inband_flam(se_ab, band_name)
        expected = 10 ** (-0.4 * _band_ab_vega_offset(band_name))
        assert np.isclose(ratio, expected, rtol=1e-6)

    def test_ab_vega_offset_is_band_dependent(self):
        off_v = _band_ab_vega_offset("johnson_v")
        assert abs(off_v) < 0.05

    def test_ab_vega_k_band_offset_large(self):
        off_k = _band_ab_vega_offset("johnson_k")
        assert off_k > 1.5

    def test_vegamag_roundtrips(self):
        se = SceneElement.from_config(
            {
                "spectrum": "flat",
                "mag": 14.0,
                "magsys": "vegamag",
                "bandpass": "johnson_r",
            }
        )
        vega = SourceSpectrum.from_vega()
        band = SpectralElement.from_filter("johnson_r")
        obs = Observation(se.get_spectrum(), band, force="extrap")
        assert abs(obs.effstim(su.VEGAMAG, vegaspec=vega).value - 14.0) < 0.01

    def test_default_magsys_sceneelement_is_vegamag(self):
        se = SceneElement.from_config(
            {"spectrum": "flat", "mag": 15, "bandpass": "johnson_r"}
        )
        assert se.mag.unit == su.VEGAMAG

    def test_default_magsys_get_scene_source_is_vegamag(self):
        import wcc_etc

        scene = wcc_etc.get_scene(name="G5V", mag=15, background="zodi")
        assert scene.source.mag.unit == su.VEGAMAG

    def test_default_magsys_get_scene_background_is_vegamag(self):
        import wcc_etc

        scene = wcc_etc.get_scene(name="G5V", mag=15, background="zodi")
        assert scene.background.mag.unit == su.VEGAMAG

    def test_none_mag_skips_normalization(self):
        se = SceneElement.from_config(
            {
                "spectrum": "blackbody",
                "teff": 5777,
                "mag": None,
                "bandpass": "johnson_v",
            }
        )
        raw = se.get_spectrum(apply_mag=False)
        out = se.get_spectrum(apply_mag=True)
        w = np.array([5000.0, 6000.0]) * u.AA
        assert np.allclose(
            raw(w, flux_unit=su.FLAM).value, out(w, flux_unit=su.FLAM).value
        )

    def test_explicit_none_mag_does_not_warn(self, recwarn):
        se = SceneElement.from_config(
            {
                "spectrum": "emission",
                "lines": [{"wave": 6563, "flux": 1e-15, "fwhm": 3}],
                "mag": None,
            }
        )
        se.get_mag()
        assert not any("not mag in self.meta" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# Mutable-parameter tests
# ---------------------------------------------------------------------------


class TestMutableParameters:
    def test_blackbody_has_teff_mutable(self):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": 5777, "mag": 15, "bandpass": "johnson_v"}
        )
        assert "teff" in bb.mutable_parameters

    def test_blackbody_lacks_alpha_mutable(self):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": 5777, "mag": 15, "bandpass": "johnson_v"}
        )
        assert "alpha" not in bb.mutable_parameters

    def test_powerlaw_has_alpha_and_lambda_ref_mutable(self):
        pl = SceneElement.from_config(
            {"spectrum": "powerlaw", "alpha": -1.0, "mag": 15, "bandpass": "johnson_v"}
        )
        assert {"alpha", "lambda_ref"} <= set(pl.mutable_parameters)

    def test_blackbody_spectrum_type_not_mutable(self):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": 5777, "mag": 15, "bandpass": "johnson_v"}
        )
        assert "spectrum" not in bb.mutable_parameters

    def test_powerlaw_spectrum_type_not_mutable(self):
        pl = SceneElement.from_config(
            {"spectrum": "powerlaw", "alpha": -1.0, "mag": 15, "bandpass": "johnson_v"}
        )
        assert "spectrum" not in pl.mutable_parameters

    def test_cannot_update_spectrum_type_warns(self, recwarn):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": 5777, "mag": 15, "bandpass": "johnson_v"}
        )
        bb.update(spectrum="powerlaw")
        assert any("not a mutable parameter" in str(w.message) for w in recwarn.list)

    def test_cannot_update_spectrum_type_unchanged(self, recwarn):
        bb = SceneElement.from_config(
            {"spectrum": "blackbody", "teff": 5777, "mag": 15, "bandpass": "johnson_v"}
        )
        before = _flam_ratio(bb.get_spectrum(apply_mag=False), 4500, 7500)
        bb.update(spectrum="powerlaw")
        after = _flam_ratio(bb.get_spectrum(apply_mag=False), 4500, 7500)
        assert np.isclose(after, before, rtol=1e-6)
