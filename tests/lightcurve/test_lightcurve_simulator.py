"""Tests for LightCurveSimulator and LightCurve."""

import numpy as np
import pytest
from wcc_etc.lightcurve import FluxModel, LightCurve, LightCurveSimulator


class _ConstDip(FluxModel):
    def relative_flux(self, time):
        time = np.asarray(time, dtype=float)
        f = np.ones_like(time)
        f[len(f) // 4 : 3 * len(f) // 4] = 0.99
        return f


class _FakeSim:
    def __init__(self, snr):
        self._snr = snr
        self.calls = []

    def get_image_snr(self, **kw):
        self.calls.append(kw)
        return {"snr": self._snr}


class TestLightCurveSimulator:
    def test_sigma_is_inverse_snr_and_called_once_with_nreads1(self):
        sim = _FakeSim(snr=200.0)
        lc = LightCurveSimulator(sim, _ConstDip()).simulate(
            time=np.linspace(0, 1, 50), exptime=30.0, seed=0
        )
        assert lc.flux_err == pytest.approx(1 / 200.0)
        assert len(sim.calls) == 1
        assert sim.calls[0]["n_reads"] == 1
        assert sim.calls[0]["time"] == 30.0
        assert lc.snr == pytest.approx(200.0)
        assert lc.exptime == 30.0

    def test_clean_matches_model_and_seed_is_reproducible(self):
        model = _ConstDip()
        t = np.linspace(0, 1, 200)
        lc1 = LightCurveSimulator(_FakeSim(100.0), model).simulate(t, 30.0, seed=42)
        lc2 = LightCurveSimulator(_FakeSim(100.0), model).simulate(t, 30.0, seed=42)
        np.testing.assert_array_equal(lc1.flux_clean, model.relative_flux(t))
        np.testing.assert_array_equal(lc1.flux, lc2.flux)

    def test_noise_statistics_match_sigma(self):
        t = np.linspace(0, 1, 5000)
        lc = LightCurveSimulator(_FakeSim(50.0), _ConstDip()).simulate(t, 30.0, seed=1)
        resid = lc.flux - lc.flux_clean
        assert np.std(resid) == pytest.approx(1 / 50.0, rel=0.1)
        assert isinstance(lc, LightCurve)

    def test_repr_is_informative(self):
        lc = LightCurveSimulator(_FakeSim(150.0), _ConstDip()).simulate(
            np.linspace(0, 1, 40), exptime=30.0, seed=0
        )
        r = repr(lc)
        assert r.startswith("LightCurve(")
        assert "n=40" in r
        assert "snr=150.0" in r
