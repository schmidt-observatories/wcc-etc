import pytest

import wcc_etc


def test_snrs_25p4_mag_60s():
    """
    Verify SNR for a few ZWO and qCMOS sensors around a 25.4 AB-mag star in 60s.

    Allow a small tolerance.
    """
    scene = wcc_etc.get_scene(
        name="G5V",
        mag=25.4,
        magsys="abmag",
        host=None,
        background="zodi",
        bandpass="johnson_r",
        background_prop={"bandpass": "johnson_r", "mag": 22.5, "magsys": "abmag"},
    )

    # Define the sensor and filter combination
    sensor_and_filter = "sony:r"
    simu = wcc_etc.Simulation.from_sensor_and_scene(sensor_and_filter, scene)

    snr_val = simu.get_snr(time=60)["snr"]

    print(f"SNR for 25.4 AB-mag star in 60s with {sensor_and_filter}: {snr_val:.2f}")

    # AB-mag SNR (the scene pins magsys="abmag", so this is unaffected by the
    # vegamag default). Expected ~4.95: the noise budget now includes the sky
    # shot noise and the PSF is rendered at the pivot wavelength (not peak
    # transmission). The independent closed-form/Monte-Carlo cross-checks of the
    # sky term live in test_sky_background_noise.py.
    assert snr_val == pytest.approx(4.95, abs=0.1)
