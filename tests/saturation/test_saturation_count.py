"""Tests for saturation counting in get_snr, get_image_snr, and exptime methods."""

import warnings
import numpy as np
import pytest
import astropy.units as u
from wcc_etc.psfsim import ImageSimulator, AiryPSF, saturation_mask_from_image_e
from tests.helpers import make_simulation


def _adc_clip_setup():
    sensor = make_simulation(mag=12).sensor
    gain = sensor.gain.to(u.electron / u.ct).value
    adc_max = sensor.adc_max.to(u.ct).value
    well_depth = sensor.meta.get("well_depth")
    adc_threshold_e = adc_max * gain
    lower_limit = (
        min(adc_threshold_e, well_depth) if well_depth is not None else adc_threshold_e
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
    return sensor, image_e, adc_mask, expected


class TestSaturationMask:
    def test_helper_matches_simulate_image_mask(self):
        sim = make_simulation(mag=12)
        imsim = ImageSimulator(sim, npix=128, oversample=11)
        res = imsim.simulate(time=60, psf=AiryPSF(), add_noise=False)
        expected = res.saturation_mask
        got = saturation_mask_from_image_e(sim.sensor, res.image_e)
        assert np.array_equal(got, expected)

    def test_below_threshold_not_saturated(self):
        sensor, image_e, adc_mask, expected = _adc_clip_setup()
        assert not adc_mask[0, 0]

    def test_at_threshold_is_saturated(self):
        sensor, image_e, adc_mask, expected = _adc_clip_setup()
        assert adc_mask[0, 1]

    def test_saturation_mask_matches_expected(self):
        sensor, image_e, adc_mask, expected = _adc_clip_setup()
        got = saturation_mask_from_image_e(sensor, image_e)
        assert np.array_equal(got, expected)


class TestGetSnrSaturation:
    def test_faint_n_saturated_is_zero(self):
        sim = make_simulation(mag=25.4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sim.get_snr(time=60)
        assert res["n_saturated"] == 0

    def test_faint_saturated_flag_is_false(self):
        sim = make_simulation(mag=25.4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sim.get_snr(time=60)
        assert res["saturated"] is False

    def test_bright_n_saturated_is_positive(self):
        sim = make_simulation(mag=12)
        with pytest.warns(UserWarning, match="saturat"):
            res = sim.get_snr(time=60)
        assert res["n_saturated"] > 0

    def test_bright_saturated_flag_is_true(self):
        sim = make_simulation(mag=12)
        with pytest.warns(UserWarning, match="saturat"):
            res = sim.get_snr(time=60)
        assert res["saturated"] is True

    def test_warn_false_keeps_n_saturated(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sim.get_snr(time=60, warn=False)
        assert res["n_saturated"] > 0

    def test_warn_false_keeps_saturated_flag(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = sim.get_snr(time=60, warn=False)
        assert res["saturated"] is True

    def test_array_time_count_shape(self):
        sim = make_simulation(mag=12)
        res = sim.get_snr(time=[30, 60, 120], warn=False)
        assert res["n_saturated"].shape == (3,)

    def test_array_time_count_dtype(self):
        sim = make_simulation(mag=12)
        res = sim.get_snr(time=[30, 60, 120], warn=False)
        assert res["n_saturated"].dtype.kind == "i"

    def test_array_time_saturated_dtype(self):
        sim = make_simulation(mag=12)
        res = sim.get_snr(time=[30, 60, 120], warn=False)
        assert res["saturated"].dtype == bool

    def test_array_time_count_monotonic(self):
        sim = make_simulation(mag=12)
        res = sim.get_snr(time=[30, 60, 120], warn=False)
        assert np.all(np.diff(res["n_saturated"]) >= 0)

    def test_values_unchanged_regression(self):
        sim = make_simulation(mag=25.4)
        snr = sim.get_snr(time=60)["snr"]
        assert np.isclose(snr, sim.get_image_snr(time=60, warn=False)["snr"])

    def test_is_saturated_agrees_with_count(self):
        sim = make_simulation(mag=12)
        assert bool(sim.is_saturated(60))

    def test_count_is_positive_when_saturated(self):
        sim = make_simulation(mag=12)
        res = sim.get_snr(time=60, warn=False)
        assert res["n_saturated"] > 0

    def test_count_matches_rendered_image(self):
        sim = make_simulation(mag=12)
        res = sim.get_snr(time=60, n_reads=1, warn=False)
        img = ImageSimulator(sim, npix=128, oversample=11).simulate(
            time=60, psf=AiryPSF(), add_noise=False
        )
        assert res["n_saturated"] == int(img.saturation_mask.sum())


class TestExptimeSaturation:
    def test_reports_n_saturated_is_int(self):
        sim = make_simulation(mag=25.4)
        res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
        assert isinstance(res["n_saturated"], int)

    def test_reports_saturated_is_false(self):
        sim = make_simulation(mag=25.4)
        res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
        assert res["saturated"] is False

    def test_reports_n_saturated_is_zero(self):
        sim = make_simulation(mag=25.4)
        res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
        assert res["n_saturated"] == 0

    def test_exptime_n_saturated_matches_snr(self):
        sim = make_simulation(mag=12)
        res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
        chk = sim.get_image_snr(time=res["time_s"], r_aper_mas=70, warn=False)
        assert res["n_saturated"] == chk["n_saturated"]

    def test_exptime_saturated_flag_matches_snr(self):
        sim = make_simulation(mag=12)
        res = sim.get_image_exptime_for_snr(50.0, r_aper_mas=70, warn=False)
        chk = sim.get_image_snr(time=res["time_s"], r_aper_mas=70, warn=False)
        assert res["saturated"] == chk["saturated"]

    def test_warns_when_solved_time_saturates(self):
        sim = make_simulation(mag=12)
        with pytest.warns(UserWarning, match="saturat"):
            sim.get_image_exptime_for_snr(1e5, r_aper_mas=70)

    def test_get_snr_airy_returns_quantity(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            val = sim.get_snr_airy(60)
        assert isinstance(val, u.Quantity)

    def test_get_snr_airy_warns_on_saturation(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.get_snr_airy(60)
        assert any(
            issubclass(w.category, UserWarning) and "saturat" in str(w.message)
            for w in caught
        )

    def test_get_snr_airy_warn_false_no_saturation_warning(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.get_snr_airy(60, warn=False)
        assert not any(
            issubclass(w.category, UserWarning) and "saturat" in str(w.message)
            for w in caught
        )

    def test_get_exptime_returns_quantity(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            t = sim.get_exptime_for_snr(1e5)
        assert isinstance(t, u.Quantity)

    def test_get_exptime_warns_on_saturation(self):
        sim = make_simulation(mag=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.get_exptime_for_snr(1e5)
        assert any(
            issubclass(w.category, UserWarning) and "saturat" in str(w.message)
            for w in caught
        )
