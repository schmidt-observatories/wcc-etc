import pytest
from astropy import units as u
import wcc_etc
from wcc_etc import io as wcc_io
import numpy as np


def test_snrs_25p4_mag_60s():
    """
    Verify SNR for a few ZWO and qCMOS sensors around a 25.4 AB-mag star in 60s.

    Allow a small tolerance.
    """
    # build source using repository bundled files (same as notebook)
    source_config = {
        "spectrum": wcc_io.expand_path('astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits'),
        "background": wcc_io.expand_path('astr_obj_models/galaxies/brown/ngc_2537_spec.fits'),
        "bg_surface_brightness": 22.5,
    }

    source = wcc_etc.Source.from_config(source_config)

    ###########################################
    ###########################################
    # Testing ZWO, r
    # create simulation for target sensor/telescope combo
    sim = wcc_etc.Simulation.from_sensorname_and_source("zwo:r", source)
    test_sim = wcc_etc.Simulation( telescope=sim.telescope, sensor=sim.sensor,
                                   source=sim.source, mag=25.4, time=60 * u.s,)
    snr = test_sim.get_snr()
    snr_val = float(snr)
    assert pytest.approx(5.5, abs=0.2) == snr_val
    ###########################################

    ###########################################
    ###########################################
    # Testing ZWO, g
    # create simulation for target sensor/telescope combo
    sim = wcc_etc.Simulation.from_sensorname_and_source("zwo:g", source)
    test_sim = wcc_etc.Simulation( telescope=sim.telescope, sensor=sim.sensor,
                                   source=sim.source, mag=25.4, time=60 * u.s,)
    snr = test_sim.get_snr()
    snr_val = float(snr)
    assert pytest.approx(9.2, abs=0.2) == snr_val
    ###########################################

    ###########################################
    ###########################################
    # Testing ZWO, bb
    # create simulation for target sensor/telescope combo
    sim = wcc_etc.Simulation.from_sensorname_and_source("zwo:bb", source)
    test_sim = wcc_etc.Simulation( telescope=sim.telescope, sensor=sim.sensor,
                                   source=sim.source, mag=25.4, time=60 * u.s,)
    snr = test_sim.get_snr()
    snr_val = float(snr)
    assert pytest.approx(13.4, abs=0.2) == snr_val 
    ###########################################

    ###########################################
    ###########################################
    # Testing hwk, bb
    # create simulation for target sensor/telescope combo
    sim = wcc_etc.Simulation.from_sensorname_and_source("qcmos:bb", source)
    test_sim = wcc_etc.Simulation( telescope=sim.telescope, sensor=sim.sensor,
                                   source=sim.source, mag=25.4, time=60 * u.s,)
    snr = test_sim.get_snr()
    snr_val = float(snr)
    assert pytest.approx(18.4, abs=0.2) == snr_val 
    ###########################################

    ###########################################
    ###########################################
    # Testing hwk, u
    # create simulation for target sensor/telescope combo
    sim = wcc_etc.Simulation.from_sensorname_and_source("qcmos:u", source)
    test_sim = wcc_etc.Simulation( telescope=sim.telescope, sensor=sim.sensor,
                                   source=sim.source, mag=25.4, time=60 * u.s,)
    snr = test_sim.get_snr()
    snr_val = float(snr)
    assert pytest.approx(3.6, abs=0.2) == snr_val 
    ###########################################