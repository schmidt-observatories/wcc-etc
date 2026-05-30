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
