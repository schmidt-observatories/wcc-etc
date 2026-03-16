from wcc_etc.io import expand_path
from wcc_etc.scene import SceneElement
import numpy as np


def test_name_and_config():
    """ """
    baseconfig = {"mag": 20,
                  "bandpass": "johnson_v"}
    config1 = {"spectrum": wcc_io.expand_path('astr_obj_models/stars/pickles_models/dat_uvk/pickles_uk_55.fits')} 
                  
    config2 = {"spectrum": "uk_55"}
    config3 = {"spectrum": "G5IV"}

    source1 = SceneElement.from_config(config1 | baseconfig)
    lbda1, spec1 = source1.get_spectrum(as_array=True)

    source2 = SceneElement.from_config(config2 | baseconfig)
    lbda2, spec2 = source2.get_spectrum(as_array=True)

    source3 = SceneElement.from_config(config3 | baseconfig)
    lbda3, spec3 = source3.get_spectrum(as_array=True)

    assert np.all(spec1 == spec2)
    assert np.all(spec1 == spec3)
    assert np.all(lbda1 == lbda2)
    assert np.all(lbda1 == lbda3)
