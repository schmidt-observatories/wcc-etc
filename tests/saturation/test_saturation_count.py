"""Tests for saturation counting in get_snr, get_image_snr, and exptime methods."""

import warnings
import numpy as np
import pytest
import astropy.units as u
import wcc_etc
from wcc_etc.psfsim import ImageSimulator, AiryPSF, saturation_mask_from_image_e


def _sim(mag):
    scene = wcc_etc.get_scene(
        name="G5V",
        mag=mag,
        background="zodi",
        bandpass="johnson_r",
        background_prop={"bandpass": "johnson_r", "mag": 22.5},
    )
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


class TestSaturationMask:
    def test_helper_matches_simulate_image_mask(self):
        sim = _sim(12)
        imsim = ImageSimulator(sim, npix=128, oversample=11)
        res = imsim.simulate(time=60, psf=AiryPSF(), add_noise=False)
        expected = res.saturation_mask
        got = saturation_mask_from_image_e(sim.sensor, res.image_e)
        assert np.array_equal(got, expected)

    def test_helper_thresholds_adc_clip(self):
        sensor = _sim(12).sensor
        gain = sensor.gain.to(u.electron / u.ct).value
        adc_max = sensor.adc_max.to(u.ct).value
        well_depth = sensor.meta.get("well_depth")
        adc_threshold_e = adc_max * gain

        lower_limit = (
            min(adc_threshold_e, well_depth)
            if well_depth is not None
            else adc_threshold_e
        )
        clearly_below = lower_limit / 2.0
        clearly_above_adc = adc_threshold_e + 1.0
        at_adc = adc_threshold_e
        image_e = np.array(
            [[clearly_below, at_adc, clearly_above_adc, clearly_above_adc + 100.0]]
        )

        adc_mask = (image_e / gain) >= adc_max
        if well_depth is not None:
            well_mask = image_e >= well_depth
            expected = adc_mask | well_mask
        else:
            expected = adc_mask

        assert not adc_mask[0, 0]
        assert adc_mask[0, 1]
        got = saturation_mask_from_image_e(sensor, image_e)
        assert np.array_equal(got, expected)


class TestGetSnrSaturation:
    def test_faint_no_saturation_no_warning(self):
        sim = _sim(25.4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sim.get_snr(time=60)
        assert res["n_saturated"] == 0
        assert res["saturated"] is False

    def test_bright_counts_and_warns(self):
        sim = _sim(12)
        with pytest.warns(UserWarning, match="saturat"):
            res = sim.get_snr(time=60)
        assert res["n_saturated"] > 0
        assert res["saturated"] is True

    def test_warn_false_silences_but_keeps_count(self):
        sim = _sim(12)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sim.get_snr(time=60, warn=False)
        assert res["n_saturated"] > 0
        assert res["saturated"] is True

    def test_array_time_returns_count_array(self):
        sim = _sim(12)
        res = sim.get_snr(time=[30, 60, 120], warn=False)
        assert res["n_saturated"].shape == (3,)
        assert res["n_saturated"].dtype.kind == "i"
        assert res["saturated"].dtype == bool
        assert np.all(np.diff(res["n_saturated"]) >= 0)

    def test_values_unchanged_regression(self):
        sim = _sim(25.4)
        snr = sim.get_snr(time=60)["snr"]
        assert np.isclose(snr, sim.get_image_snr(time=60, warn=False)["snr"])

    def test_count_superset_of_is_saturated(self):
        sim = _sim(12)
        res = sim.get_snr(time=60, warn=False)
        assert bool(sim.is_saturated(60))
        assert res["n_saturated"] > 0

    def test_count_matches_rendered_image(self):
        sim = _sim(12)
        res = sim.get_snr(time=60, n_reads=1, warn=False)
        img = ImageSimulator(sim, npix=128, oversample=11).simulate(
            time=60, psf=AiryPSF(), add_noise=False
        )
        assert res["n_saturated"] == int(img.saturation_mask.sum())


class TestExptimeSaturation:
    def test_reports_count_keys(self):
        sim = _sim(25.4)
        res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
        assert isinstance(res["n_saturated"], int)
        assert res["saturated"] is False
        assert res["n_saturated"] == 0

    def test_count_consistent_with_get_image_snr(self):
        sim = _sim(12)
        res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
        chk = sim.get_image_snr(time=res["time_s"], r_aper_mas=70, warn=False)
        assert res["n_saturated"] == chk["n_saturated"]
        assert res["saturated"] == chk["saturated"]

    def test_warns_when_solved_time_saturates(self):
        sim = _sim(12)
        with pytest.warns(UserWarning, match="saturat"):
            sim.get_image_exptime_for_snr(1e5, r_aper_mas=70)

    def test_get_snr_airy_warns_on_saturation_and_keeps_type(self):
        sim = _sim(12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = sim.get_snr_airy(60)
        assert isinstance(val, u.Quantity)
        assert any(
            issubclass(w.category, UserWarning) and "saturat" in str(w.message)
            for w in caught
        )

    def test_get_snr_airy_warn_false_no_saturation_warning(self):
        sim = _sim(12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.get_snr_airy(60, warn=False)
        assert not any(
            issubclass(w.category, UserWarning) and "saturat" in str(w.message)
            for w in caught
        )

    def test_get_exptime_for_snr_warns_when_solved_time_saturates(self):
        sim = _sim(12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = sim.get_exptime_for_snr(1e5)
        assert isinstance(t, u.Quantity)
        assert any(
            issubclass(w.category, UserWarning) and "saturat" in str(w.message)
            for w in caught
        )
