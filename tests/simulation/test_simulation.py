"""Tests for the Simulation class: init, parameters, peak pixel, saturation flag,
get_image_snr, render cache, and magnitude sweeps."""

import warnings

import astropy.units as u
import numpy as np
import pytest

import wcc_etc
from tests.helpers import make_scene, make_simulation
from wcc_etc.simulation import Simulation, calculate_bg_normalization_magnitude

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestBgNormalization:
    def test_unit_area_is_identity(self):
        assert calculate_bg_normalization_magnitude(20.0, 1.0) == pytest.approx(20.0)

    def test_larger_area_is_brighter(self):
        bg, area = 20.0, 10.0
        assert calculate_bg_normalization_magnitude(bg, area) == pytest.approx(
            bg - 2.5 * np.log10(area)
        )


class TestSimulationInit:
    def test_fullkey_to_element_is_sensor(self):
        element, key = Simulation._fullkey_to_element_and_key("sensor__gain")
        assert element == "sensor"

    def test_fullkey_to_key_is_gain(self):
        element, key = Simulation._fullkey_to_element_and_key("sensor__gain")
        assert key == "gain"

    def test_fullkey_simple_element_is_none(self):
        element, key = Simulation._fullkey_to_element_and_key("time")
        assert element is None

    def test_fullkey_simple_key_is_time(self):
        element, key = Simulation._fullkey_to_element_and_key("time")
        assert key == "time"

    def test_get_parameter_returns_list(self):
        sim = Simulation(
            telescope=None, sensor=None, scene=None, time=123, r_aper_mas=70
        )
        assert isinstance(sim.get_parameter("time"), list)

    def test_get_parameter_value(self):
        sim = Simulation(
            telescope=None, sensor=None, scene=None, time=123, r_aper_mas=70
        )
        assert sim.get_parameter("time")[0] == 123

    def test_mutable_parameters_contains_time(self):
        sim = Simulation(
            telescope=None, sensor=None, scene=None, time=123, r_aper_mas=70
        )
        assert "time" in sim.mutable_parameters

    def test_mutable_parameters_contains_r_aper_mas(self):
        sim = Simulation(
            telescope=None, sensor=None, scene=None, time=123, r_aper_mas=70
        )
        assert "r_aper_mas" in sim.mutable_parameters

    def test_has_element_false_when_none(self):
        sim = Simulation(telescope=None, sensor=None, scene=None)
        assert sim.has_element("telescope") is False

    def test_has_element_true_when_set(self):
        sim = Simulation(telescope=None, sensor=None, scene=None)
        sim._telescope = object()
        assert sim.has_element("telescope") is True


# ---------------------------------------------------------------------------
# Peak-pixel tests
# ---------------------------------------------------------------------------


class TestPeakPixel:
    @pytest.fixture
    def sim(self):
        return make_simulation(mag=15)

    def test_increases_with_time(self, sim):
        assert sim.get_peak_pixel(100, units="adu") > sim.get_peak_pixel(
            10, units="adu"
        )

    def test_adu_units_are_ct(self, sim):
        assert sim.get_peak_pixel(10, units="adu").unit == u.ct

    def test_includes_bias(self, sim):
        base = sim.get_peak_pixel(10, units="adu")
        sim.update(sensor__bias_level=100)
        biased = sim.get_peak_pixel(10, units="adu")
        assert biased.value == pytest.approx(base.value + 100, rel=1e-6)

    def test_electrons_unit_is_electron(self, sim):
        e = sim.get_peak_pixel(10, units="e-")
        assert e.unit == u.electron

    def test_electrons_value_is_positive(self, sim):
        e = sim.get_peak_pixel(10, units="e-")
        assert e.value > 0

    def test_adu_matches_electron_conversion(self, sim):
        e = sim.get_peak_pixel(10, units="e-")
        adu = sim.get_peak_pixel(10, units="adu")
        expected = (e / sim.sensor.gain).to(u.ct) + sim.sensor.bias_level
        assert adu.value == pytest.approx(expected.value, rel=1e-9)

    def test_brighter_background_increases_value(self, sim):
        base = sim.get_peak_pixel(100, units="e-")
        sim.update(background__mag=18)
        assert sim.get_peak_pixel(100, units="e-").value > base.value

    def test_array_time_shape(self, sim):
        vals = sim.get_peak_pixel(np.array([10.0, 100.0]), units="adu")
        assert np.shape(vals) == (2,)

    def test_array_time_monotonic(self, sim):
        vals = sim.get_peak_pixel(np.array([10.0, 100.0]), units="adu")
        assert vals[1] > vals[0]

    def test_excludes_host(self):
        scene_no_host = wcc_etc.get_scene(
            name="G5V",
            mag=15,
            host=None,
            background="zodi",
            bandpass="johnson_r",
            background_prop={"bandpass": "johnson_r", "mag": 22.5},
        )
        scene_host = wcc_etc.get_scene(
            name="G5V",
            mag=15,
            host="G5V",
            host_prop={"mag": 16, "bandpass": "johnson_r"},
            background="zodi",
            bandpass="johnson_r",
            background_prop={"bandpass": "johnson_r", "mag": 22.5},
        )
        sim_no = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene_no_host)
        sim_with = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene_host)
        assert sim_with.get_peak_pixel(100, units="e-").value == pytest.approx(
            sim_no.get_peak_pixel(100, units="e-").value, rel=1e-9
        )

    def test_psf_profile_does_not_have_peak_pixel_fraction(self, sim):
        assert "peak_pixel_fraction" not in sim.psf_profile


class TestIsSaturated:
    @pytest.fixture
    def bright_sim(self):
        return make_simulation(mag=8)

    def test_not_saturated_at_short_time(self, bright_sim):
        assert not bright_sim.is_saturated(0.001)

    def test_saturated_at_long_time(self, bright_sim):
        assert bright_sim.is_saturated(1000)

    def test_is_saturated_array_shape(self, bright_sim):
        result = bright_sim.is_saturated(np.array([0.001, 1000.0]))
        assert np.shape(result) == (2,)

    def test_is_saturated_array_first_false(self, bright_sim):
        result = bright_sim.is_saturated(np.array([0.001, 1000.0]))
        assert bool(result[0]) is False

    def test_is_saturated_array_second_true(self, bright_sim):
        result = bright_sim.is_saturated(np.array([0.001, 1000.0]))
        assert bool(result[1]) is True


# ---------------------------------------------------------------------------
# get_image_snr
# ---------------------------------------------------------------------------


class TestGetImageSnr:
    @pytest.fixture
    def sim(self):
        return make_simulation(mag=16)

    def test_returns_expected_keys(self, sim):
        out = sim.get_image_snr(time=60)
        assert set(out) >= {
            "snr",
            "signal_e",
            "noise_e",
            "enclosed_fraction",
            "r_aper_mas",
            "n_pix",
        }

    def test_enclosed_fraction_in_unit_interval(self, sim):
        out = sim.get_image_snr(time=60)
        assert 0 < out["enclosed_fraction"] <= 1

    def test_n_pix_is_positive(self, sim):
        out = sim.get_image_snr(time=60)
        assert out["n_pix"] >= 1

    def test_snr_is_positive(self, sim):
        out = sim.get_image_snr(time=60)
        assert out["snr"] > 0

    def test_snr_is_float(self, sim):
        out = sim.get_image_snr(time=60)
        assert isinstance(out["snr"], float)

    def test_n_pix_is_int(self, sim):
        out = sim.get_image_snr(time=60)
        assert isinstance(out["n_pix"], int)

    def test_matches_snr_airy_in_focus(self, sim):
        for t in [30, 300]:
            with pytest.warns(DeprecationWarning):
                etc = float(sim.get_snr_airy(t).value)
            img = sim.get_image_snr(time=t)["snr"]
            assert img == pytest.approx(etc, rel=0.03)

    def test_ee_frac_aperture(self, sim):
        out = sim.get_image_snr(time=60, ee_frac=0.9)
        assert out["enclosed_fraction"] >= 0.9

    def test_optimize_at_least_default(self, sim):
        base = sim.get_image_snr(time=60)["snr"]
        opt = sim.get_image_snr(time=60, optimize=True)["snr"]
        assert opt >= base - 1e-9

    def test_defocus_lower_at_fixed_aperture(self, sim):
        airy = sim.get_image_snr(time=60, r_aper_mas=70)["snr"]
        defo = sim.get_image_snr(
            time=60, r_aper_mas=70, psf=wcc_etc.DefocusPSF(wcc_etc.DEFOCUS_2WAVE_PATH)
        )["snr"]
        assert defo < airy

    def test_optimize_defocus_uses_larger_radius(self, sim):
        r_airy = sim.get_image_snr(time=60, optimize=True)["r_aper_mas"]
        r_defo = sim.get_image_snr(
            time=60, optimize=True, psf=wcc_etc.DefocusPSF(wcc_etc.DEFOCUS_2WAVE_PATH)
        )["r_aper_mas"]
        assert r_defo > r_airy

    def test_runs_on_qcmos(self):
        sim = make_simulation(mag=16, sensor="qcmos:r")
        assert sim.get_image_snr(time=60)["snr"] > 0

    def test_no_background_snr_is_positive(self):
        scene = wcc_etc.get_scene(
            name="G5V", mag=16, host=None, background=None, bandpass="johnson_r"
        )
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        assert sim.get_image_snr(time=60)["snr"] > 0

    def test_no_background_n_pix_is_positive(self):
        scene = wcc_etc.get_scene(
            name="G5V", mag=16, host=None, background=None, bandpass="johnson_r"
        )
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        assert sim.get_image_snr(time=60)["n_pix"] >= 1

    def test_array_time_snr_shape(self, sim):
        times = np.array([30.0, 60.0, 120.0])
        out = sim.get_image_snr(time=times)
        assert np.shape(out["snr"]) == (3,)

    def test_array_time_snr_is_monotonic(self, sim):
        times = np.array([30.0, 60.0, 120.0])
        out = sim.get_image_snr(time=times)
        assert out["snr"][0] < out["snr"][1] < out["snr"][2]

    def test_array_time_key_values_match_scalar(self, sim):
        times = np.array([30.0, 60.0, 120.0])
        out = sim.get_image_snr(time=times)
        for i, t in enumerate(times):
            scalar = sim.get_image_snr(time=float(t))
            for key in (
                "snr",
                "signal_e",
                "noise_e",
                "enclosed_fraction",
                "r_aper_mas",
            ):
                assert out[key][i] == pytest.approx(scalar[key], rel=1e-9)

    def test_array_time_n_pix_matches_scalar(self, sim):
        times = np.array([30.0, 60.0, 120.0])
        out = sim.get_image_snr(time=times)
        for i, t in enumerate(times):
            scalar = sim.get_image_snr(time=float(t))
            assert int(out["n_pix"][i]) == scalar["n_pix"]

    def test_array_time_optimize_shape(self, sim):
        times = np.array([30.0, 300.0, 3000.0])
        out = sim.get_image_snr(time=times, optimize=True)
        assert np.shape(out["r_aper_mas"]) == (3,)

    def test_array_time_optimize_matches_scalar(self, sim):
        times = np.array([30.0, 300.0, 3000.0])
        out = sim.get_image_snr(time=times, optimize=True)
        for i, t in enumerate(times):
            scalar = sim.get_image_snr(time=float(t), optimize=True)
            assert out["snr"][i] == pytest.approx(scalar["snr"], rel=1e-9)

    def test_get_snr_delegates_scalar(self, sim):
        assert sim.get_snr(60)["snr"] == pytest.approx(
            sim.get_image_snr(time=60)["snr"], rel=1e-12
        )

    def test_get_snr_delegates_with_n_reads(self, sim):
        assert sim.get_snr(60, n_reads=3)["snr"] == pytest.approx(
            sim.get_image_snr(time=60, n_reads=3)["snr"], rel=1e-12
        )

    def test_get_snr_array_time(self, sim):
        out = sim.get_snr(np.array([30.0, 60.0, 120.0]))
        assert out["snr"][0] < out["snr"][1] < out["snr"][2]

    def test_snr_airy_raises_deprecation_warning(self, sim):
        with pytest.warns(DeprecationWarning):
            sim.get_snr_airy(60)

    def test_snr_airy_matches_signal_variance(self, sim):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            airy = sim.get_snr_airy(60)
            signal, variance = sim.get_signal_and_variance(60)
        assert float(airy.value) == pytest.approx(
            float((signal / np.sqrt(variance)).value), rel=1e-12
        )


# ---------------------------------------------------------------------------
# Render cache
# ---------------------------------------------------------------------------


class TestRenderCache:
    @pytest.fixture
    def sim(self):
        return make_simulation(mag=16)

    def test_render_bundle_cached_calls_render_once(self, sim, monkeypatch):
        import wcc_etc.psfsim as psfsim

        calls = {"n": 0}
        orig = psfsim.AiryPSF.render

        def counting_render(self_psf, ctx):
            calls["n"] += 1
            return orig(self_psf, ctx)

        monkeypatch.setattr(psfsim.AiryPSF, "render", counting_render)
        sim.get_image_snr(time=30)
        sim.get_image_snr(time=60)
        assert calls["n"] == 1

    def test_render_bundle_cache_has_one_entry(self, sim, monkeypatch):
        import wcc_etc.psfsim as psfsim

        orig = psfsim.AiryPSF.render

        def passthrough_render(self_psf, ctx):
            return orig(self_psf, ctx)

        monkeypatch.setattr(psfsim.AiryPSF, "render", passthrough_render)
        sim.get_image_snr(time=30)
        sim.get_image_snr(time=60)
        assert len(sim._image_render_bundle_cache) == 1

    def test_render_bundle_second_call_gives_higher_snr(self, sim):
        a = sim.get_image_snr(time=30)["snr"]
        b = sim.get_image_snr(time=60)["snr"]
        assert b > a

    def test_update_clears_render_cache(self, sim):
        sim.get_image_snr(time=60)
        sim.update(source__mag=20)
        assert len(sim._image_render_bundle_cache) == 0

    def test_update_clears_psf_profile(self, sim):
        sim.get_image_snr(time=60)
        sim.update(source__mag=20)
        assert len(sim._psf_profile) == 0

    def test_update_changes_snr(self, sim):
        snr1 = sim.get_image_snr(time=60)["snr"]
        sim.update(source__mag=20)
        assert sim.get_image_snr(time=60)["snr"] < snr1

    def test_update_jitter_clears_render_cache(self, sim):
        sim.get_image_snr(time=60)
        sim.update(jitter_sigma=50)
        assert len(sim._image_render_bundle_cache) == 0

    def test_update_jitter_clears_psf_profile(self, sim):
        sim.get_image_snr(time=60)
        sim.update(jitter_sigma=50)
        assert len(sim._psf_profile) == 0

    def test_set_sensor_clears_cache(self, sim):
        sim.get_image_snr(time=60)
        sim.set_sensor(sim.sensor)
        assert len(sim._image_render_bundle_cache) == 0

    def test_set_telescope_clears_cache(self, sim):
        sim.get_image_snr(time=60)
        sim.set_telescope(sim.telescope)
        assert len(sim._image_render_bundle_cache) == 0

    def test_reset_clears_cache(self, sim):
        sim.get_image_snr(time=60)
        sim.reset()
        assert len(sim._image_render_bundle_cache) == 0

    def test_direct_telescope_jitter_change_invalidates(self, sim):
        snr1 = sim.get_image_snr(time=60)["snr"]
        sim.telescope.update(jitter_sigma=80)
        assert sim.get_image_snr(time=60)["snr"] < snr1

    def test_set_scene_clears_render_cache(self, sim):
        sim.get_snr(90)
        faint = make_scene(mag=22)
        sim.set_scene(faint)
        assert len(sim._image_render_bundle_cache) == 0

    def test_set_scene_clears_psf_profile(self, sim):
        sim.get_snr(90)
        faint = make_scene(mag=22)
        sim.set_scene(faint)
        assert len(sim._psf_profile) == 0

    def test_set_scene_changes_snr(self, sim):
        snr1 = sim.get_snr(90)["snr"]
        faint = make_scene(mag=22)
        sim.set_scene(faint)
        assert sim.get_snr(90)["snr"] < snr1


# ---------------------------------------------------------------------------
# Mags sweep
# ---------------------------------------------------------------------------


class TestMagsSweep:
    @pytest.fixture
    def sim(self):
        return make_simulation(mag=20)

    def test_mags_scalar_key_values_match_rebuild(self, sim):
        rebuilt = make_simulation(mag=25).get_image_snr(time=60)
        swept = sim.get_image_snr(time=60, mags=25)
        for key in ("snr", "signal_e", "noise_e", "enclosed_fraction", "r_aper_mas"):
            assert swept[key] == pytest.approx(rebuilt[key], rel=1e-12)

    def test_mags_scalar_n_pix_matches_rebuild(self, sim):
        rebuilt = make_simulation(mag=25).get_image_snr(time=60)
        swept = sim.get_image_snr(time=60, mags=25)
        assert swept["n_pix"] == rebuilt["n_pix"]

    def test_mags_scalar_snr_is_float(self, sim):
        swept = sim.get_image_snr(time=60, mags=25)
        assert isinstance(swept["snr"], float)

    def test_at_reference_mag_is_noop(self, sim):
        base = sim.get_image_snr(time=60)
        same = sim.get_image_snr(time=60, mags=20)
        assert same["snr"] == pytest.approx(base["snr"], rel=1e-12)

    def test_requires_set_magnitude(self):
        scene = wcc_etc.get_scene(
            name="G5V",
            mag=None,
            host=None,
            background="zodi",
            bandpass="johnson_r",
            background_prop={"bandpass": "johnson_r", "mag": 22.5},
        )
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        with pytest.raises(ValueError, match="set magnitude"):
            sim.get_image_snr(time=60, mags=20)

    def test_mags_array_shape(self, sim):
        mags = np.array([10.0, 15.0, 20.0, 25.0, 28.0])
        out = sim.get_image_snr(time=60, mags=mags)
        assert np.shape(out["snr"]) == (5,)

    def test_mags_array_snr_is_decreasing(self, sim):
        mags = np.array([10.0, 15.0, 20.0, 25.0, 28.0])
        out = sim.get_image_snr(time=60, mags=mags)
        assert out["snr"][0] > out["snr"][-1]

    def test_mags_array_key_values_match_rebuild(self, sim):
        mags = np.array([10.0, 15.0, 20.0, 25.0, 28.0])
        out = sim.get_image_snr(time=60, mags=mags)
        for i, m in enumerate(mags):
            rebuilt = make_simulation(float(m)).get_image_snr(time=60)
            for key in (
                "snr",
                "signal_e",
                "noise_e",
                "enclosed_fraction",
                "r_aper_mas",
            ):
                assert out[key][i] == pytest.approx(rebuilt[key], rel=1e-12)

    def test_mags_array_n_pix_matches_rebuild(self, sim):
        mags = np.array([10.0, 15.0, 20.0, 25.0, 28.0])
        out = sim.get_image_snr(time=60, mags=mags)
        for i, m in enumerate(mags):
            rebuilt = make_simulation(float(m)).get_image_snr(time=60)
            assert int(out["n_pix"][i]) == rebuilt["n_pix"]

    def test_mags_array_length_one_shape(self, sim):
        out = sim.get_image_snr(time=60, mags=[20.0])
        assert np.shape(out["snr"]) == (1,)

    def test_mags_array_length_one_value(self, sim):
        out = sim.get_image_snr(time=60, mags=[20.0])
        scalar = sim.get_image_snr(time=60)
        assert out["snr"][0] == pytest.approx(scalar["snr"], rel=1e-12)

    def test_optimize_aperture_non_increasing_with_mag(self):
        sim = make_simulation(mag=18)
        mags = np.array([14.0, 18.0, 22.0, 26.0])
        out = sim.get_image_snr(time=60, mags=mags, optimize=True)
        assert np.all(np.diff(out["r_aper_mas"]) <= 1e-9)

    def test_time_and_mags_both_arrays_raises(self, sim):
        with pytest.raises(ValueError, match="both be arrays"):
            sim.get_image_snr(time=np.array([30.0, 60.0]), mags=np.array([18.0, 20.0]))

    def test_without_source_raises(self):
        scene = wcc_etc.get_scene(
            name="G5V",
            mag=20,
            host=None,
            background="zodi",
            bandpass="johnson_r",
            background_prop={"bandpass": "johnson_r", "mag": 22.5},
        )
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        sim.scene._source = None
        with pytest.raises(ValueError, match="requires a scene with a source"):
            sim.get_image_snr(time=60, mags=20)

    def test_image_exptime_roundtrips(self):
        sim = make_simulation(mag=16)
        target = 20.0
        res = sim.get_image_exptime_for_snr(target)
        assert sim.get_image_snr(time=res["time_s"])["snr"] == pytest.approx(
            target, rel=0.02
        )
