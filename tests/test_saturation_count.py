import warnings

import numpy as np
import pytest

import wcc_etc
from wcc_etc.psfsim import ImageSimulator, AiryPSF, saturation_mask_from_image_e


def _sim(mag):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_helper_matches_simulate_image_mask():
    # The extracted helper must reproduce simulate_image's saturation_mask exactly.
    sim = _sim(12)
    imsim = ImageSimulator(sim, npix=128, oversample=11)
    res = imsim.simulate(time=60, psf=AiryPSF(), add_noise=False)
    expected = res.saturation_mask
    got = saturation_mask_from_image_e(sim.sensor, res.image_e)
    assert np.array_equal(got, expected)


def test_helper_thresholds_adc_clip():
    # Verify the helper correctly flags pixels at/above each saturation limit and
    # leaves pixels below both limits unset.  Direct logic test — no simulate().
    from astropy import units as u

    sensor = _sim(12).sensor

    gain = sensor.gain.to(u.electron / u.ct).value       # e/ct
    adc_max = sensor.adc_max.to(u.ct).value               # ct
    well_depth = sensor.meta.get("well_depth")             # electrons (or None)

    # ADC threshold in electrons: (image_e / gain) >= adc_max
    adc_threshold_e = adc_max * gain

    # For the "clearly below both limits" pixel, sit below whichever limit is
    # lower (well_depth on this sensor is ~16 ke, adc_threshold ~17 ke).
    lower_limit = min(adc_threshold_e, well_depth) if well_depth is not None else adc_threshold_e
    clearly_below = lower_limit / 2.0

    # Pixel clearly above the ADC clip (also above any well depth)
    clearly_above_adc = adc_threshold_e + 1.0

    # Pixel exactly at the ADC threshold
    at_adc = adc_threshold_e

    # Four representative pixels (row vector for easy reading):
    #   [clearly_below, at_adc, clearly_above_adc, clearly_above_adc + 100]
    image_e = np.array([[clearly_below, at_adc, clearly_above_adc, clearly_above_adc + 100.0]])

    # Hand-computed expected mask via the same formula as the helper:
    adc_mask = (image_e / gain) >= adc_max
    if well_depth is not None:
        well_mask = image_e >= well_depth
        expected = adc_mask | well_mask
    else:
        expected = adc_mask

    # Sanity: the "clearly below" pixel must not be flagged by either limit alone
    assert not adc_mask[0, 0], f"clearly_below ({clearly_below:.1f} e) unexpectedly clips ADC"
    if well_depth is not None:
        assert not well_mask[0, 0], f"clearly_below ({clearly_below:.1f} e) unexpectedly clips well"
    # Sanity: the "at ADC threshold" pixel must be flagged by ADC
    assert adc_mask[0, 1], f"at_adc ({at_adc:.1f} e) should clip ADC"

    got = saturation_mask_from_image_e(sensor, image_e)
    assert np.array_equal(got, expected), (
        f"adc_threshold={adc_threshold_e:.2f} e  gain={gain} e/ct  "
        f"adc_max={adc_max} ct  well_depth={well_depth}\n"
        f"image_e={image_e}\nexpected={expected}\ngot={got}"
    )


def test_get_snr_faint_no_saturation_no_warning():
    sim = _sim(25.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning -> error
        res = sim.get_snr(time=60)
    assert res["n_saturated"] == 0
    assert res["saturated"] is False


def test_get_snr_bright_counts_and_warns():
    sim = _sim(12)                               # bright: saturates a 60-s frame
    with pytest.warns(UserWarning, match="saturat"):
        res = sim.get_snr(time=60)
    assert res["n_saturated"] > 0
    assert res["saturated"] is True


def test_get_snr_warn_false_silences_but_keeps_count():
    sim = _sim(12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sim.get_snr(time=60, warn=False)
    assert res["n_saturated"] > 0
    assert res["saturated"] is True


def test_get_snr_array_time_returns_count_array():
    sim = _sim(12)
    res = sim.get_snr(time=[30, 60, 120], warn=False)
    assert res["n_saturated"].shape == (3,)
    assert res["n_saturated"].dtype.kind == "i"
    assert res["saturated"].dtype == bool
    # more exposure -> at least as many saturated pixels (monotone, per-frame)
    assert np.all(np.diff(res["n_saturated"]) >= 0)


def test_get_snr_values_unchanged_regression():
    # adding the count must not change the SNR itself
    sim = _sim(25.4)
    snr = sim.get_snr(time=60)["snr"]
    assert np.isclose(snr, sim.get_image_snr(time=60, warn=False)["snr"])


def test_get_snr_count_superset_of_is_saturated():
    # is_saturated (ADC clip only) implies n_saturated > 0 (ADC clip OR full well).
    # Not iff: n_saturated can exceed 0 via full well alone. At mag 12 both hold.
    sim = _sim(12)
    res = sim.get_snr(time=60, warn=False)
    assert bool(sim.is_saturated(60))            # mag 12 clips the ADC at 60 s
    assert res["n_saturated"] > 0                 # ... so the image mask must agree


def test_get_snr_count_matches_rendered_image():
    # n_saturated must equal the saturated-pixel count of the actual rendered
    # image (ImageSimulator.simulate), on the same grid, at n_reads=1 (tf == time).
    from wcc_etc.psfsim import ImageSimulator, AiryPSF
    sim = _sim(12)
    res = sim.get_snr(time=60, n_reads=1, warn=False)        # defaults: npix=128, oversample=11
    img = ImageSimulator(sim, npix=128, oversample=11).simulate(
        time=60, psf=AiryPSF(), add_noise=False)
    assert res["n_saturated"] == int(img.saturation_mask.sum())
