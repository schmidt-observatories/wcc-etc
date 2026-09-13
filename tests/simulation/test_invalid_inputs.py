"""Invalid scientific inputs raise instead of returning plausible values (#86, R10)."""

import numpy as np
import pytest

from wcc_etc.psfsim import aperture_time_for_snr, select_aperture, solve_time_for_snr


def _gauss(n=31, sigma=2.0):
    yy, xx = np.mgrid[0:n, 0:n]
    a = np.exp(-(((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / (2 * sigma**2)))
    return a / a.sum()


class TestTargetSnr:
    @pytest.mark.parametrize("snr", [-5.0, 0.0, np.nan, np.inf])
    def test_solver_rejects_nonpositive_or_nonfinite(self, snr):
        with pytest.raises(ValueError):
            solve_time_for_snr(snr, 10.0, 5.0, 100.0)

    @pytest.mark.parametrize("snr", [-5.0, np.nan])
    def test_image_exptime_rejects(self, sim, snr):
        with pytest.raises(ValueError):
            sim.get_image_exptime_for_snr(snr)

    def test_unreachable_target_raises_not_inf(self):
        with pytest.raises(ValueError, match="unreachable"):
            aperture_time_for_snr(
                _gauss(), 10.0, 0.0, 0.5, 0.1, 3.0, snr=10.0, ee_frac=0.5
            )


class TestNReads:
    @pytest.mark.parametrize("n_reads", [1.9, 0.5, np.nan])
    def test_nonintegral_rejected(self, sim, n_reads):
        with pytest.raises(ValueError):
            sim.get_snr(60, n_reads=n_reads)

    def test_integral_float_accepted(self, sim):
        assert sim.get_snr(60, n_reads=2.0)["snr"] > 0


class TestExposureTime:
    @pytest.mark.parametrize("time", [-1.0, np.nan, np.inf, [10.0, -1.0]])
    def test_negative_or_nonfinite_rejected(self, sim, time):
        with pytest.raises(ValueError):
            sim.get_snr(time)

    def test_zero_time_gives_zero_snr(self, sim):
        assert sim.get_snr(0.0)["snr"] == 0.0


class TestApertureRequest:
    def _prof(self):
        from wcc_etc.psfsim import aperture_snr_radial

        return aperture_snr_radial(_gauss(), 10.0, 1e4, 1.0, 0.1, 3.0)

    @pytest.mark.parametrize("r", [0.0, -10.0, np.nan, np.inf, 1e6])
    def test_radius_out_of_range_raises(self, r):
        with pytest.raises(ValueError):
            select_aperture(self._prof(), r_aper_mas=r)

    @pytest.mark.parametrize("f", [0.0, -0.1, 1.5, np.nan])
    def test_ee_frac_out_of_range_raises(self, f):
        with pytest.raises(ValueError):
            select_aperture(self._prof(), ee_frac=f)

    def test_radius_beyond_grid_raises_through_simulation(self, sim):
        with pytest.raises(ValueError, match="npix"):
            sim.get_snr(60, r_aper_mas=1e6)
