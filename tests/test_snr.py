import pytest
import wcc_etc
import numpy as np


def test_snrs_25p4_mag_60s():
    """
    Verify SNR for a few ZWO and qCMOS sensors around a 25.4 AB-mag star in 60s.

    Allow a small tolerance.
    """
    scene = wcc_etc.get_scene(name='G5V', 
                          mag=25.4, 
                          host=None, 
                          background="zodi",
                          bandpass='johnson_r',
                          background_prop={"bandpass": 'johnson_r', "mag": 22.5})

    # Define the sensor and filter combination
    sensor_and_filter = 'sony:r'
    simu = wcc_etc.Simulation.from_sensor_and_scene(sensor_and_filter, scene)

    snr_val = simu.get_snr(time = 60)["snr"]

    print(f"SNR for 25.4 AB-mag star in 60s with {sensor_and_filter}: {snr_val:.2f}")

    assert snr_val == pytest.approx(5.1, abs=0.1)