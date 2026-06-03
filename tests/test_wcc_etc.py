import warnings

import numpy as np
import pytest


def test_get_wcc_snr_and_simulation_blackbody():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # module is deprecated
        from wcc_etc.wcc_etc import get_wcc_snr_and_simulation
        snr, sim = get_wcc_snr_and_simulation(mag=15, texp=60,
                                              source_type="blackbody", teff=5777)
    assert np.isfinite(snr)
    assert snr > 0


def test_get_wcc_snr_and_simulation_file_source(tmp_path):
    specfile = tmp_path / "source.dat"
    wave = np.linspace(3000.0, 11000.0, 40)
    flux = np.full_like(wave, 1e-16)
    np.savetxt(specfile, np.column_stack([wave, flux]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # module is deprecated
        from wcc_etc.wcc_etc import get_wcc_snr_and_simulation
        snr, sim = get_wcc_snr_and_simulation(mag=15, texp=60,
                                              source_type="file",
                                              source_file=str(specfile))
    assert np.isfinite(snr)
    assert snr > 0
    assert sim.scene.source.meta["spectrum"] == "file"
