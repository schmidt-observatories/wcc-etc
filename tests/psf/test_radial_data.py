"""Tests for calc_ee in radial_data."""

import numpy as np
import pandas as pd
from wcc_etc.radial_data import calc_ee


class TestCalcEE:
    def _profile_df(self):
        r = np.linspace(0, 10, 200)
        profile = np.exp(-((r / 2.0) ** 2))
        return pd.DataFrame({"r": r, "mean": profile}), r

    def test_runs_and_is_normalized(self):
        df, _ = self._profile_df()
        K = np.array([1.0, 2.0, 3.0])
        ee = calc_ee(df, hwhm=2.0, K=K, endpoint=3.0, verbose=False)
        assert ee.shape == (3,)
        assert np.all(np.diff(ee) >= 0)
        assert np.isclose(ee[-1], 1.0)

    def test_scalar_k_branch(self):
        df, _ = self._profile_df()
        ee_scalar = calc_ee(df, hwhm=2.0, K=1.0, endpoint=3.0, verbose=False)
        assert 0.0 < float(ee_scalar) < 1.0
