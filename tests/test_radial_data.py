import numpy as np
import pandas as pd

from wcc_etc.radial_data import calc_ee


def test_calc_ee_runs_and_is_normalized():
    # exercises the trapezoid integration path (regression guard for the
    # numpy 2.x removal of np.trapz)
    r = np.linspace(0, 10, 200)
    profile = np.exp(-(r / 2.0) ** 2)
    df = pd.DataFrame({"r": r, "mean": profile})

    K = np.array([1.0, 2.0, 3.0])
    ee = calc_ee(df, hwhm=2.0, K=K, endpoint=3.0, verbose=False)
    assert ee.shape == (3,)
    assert np.all(np.diff(ee) >= 0)     # grows with aperture
    assert np.isclose(ee[-1], 1.0)      # K == endpoint -> fully enclosed

    # scalar-K branch also integrates and normalizes
    ee_scalar = calc_ee(df, hwhm=2.0, K=1.0, endpoint=3.0, verbose=False)
    assert 0.0 < float(ee_scalar) < 1.0
