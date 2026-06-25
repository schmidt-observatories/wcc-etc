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
    def _simulate_200(self):
        sim = _FakeSim(snr=200.0)
        lc = LightCurveSimulator(sim, _ConstDip()).simulate(
            time=np.linspace(0, 1, 50), exptime=30.0, seed=0
        )
        return sim, lc

    def test_flux_err_is_inverse_snr(self):
        sim, lc = self._simulate_200()
        assert lc.flux_err == pytest.approx(1 / 200.0)

    def test_get_image_snr_called_once(self):
        sim, lc = self._simulate_200()
        assert len(sim.calls) == 1

    def test_get_image_snr_called_with_n_reads_one(self):
        sim, lc = self._simulate_200()
        assert sim.calls[0]["n_reads"] == 1

    def test_get_image_snr_called_with_exptime(self):
        sim, lc = self._simulate_200()
        assert sim.calls[0]["time"] == 30.0

    def test_lc_snr_attribute(self):
        sim, lc = self._simulate_200()
        assert lc.snr == pytest.approx(200.0)

    def test_lc_exptime_attribute(self):
        sim, lc = self._simulate_200()
        assert lc.exptime == 30.0

    def test_flux_clean_matches_model(self):
        model = _ConstDip()
        t = np.linspace(0, 1, 200)
        lc1 = LightCurveSimulator(_FakeSim(100.0), model).simulate(t, 30.0, seed=42)
        np.testing.assert_array_equal(lc1.flux_clean, model.relative_flux(t))

    def test_same_seed_gives_same_flux(self):
        model = _ConstDip()
        t = np.linspace(0, 1, 200)
        lc1 = LightCurveSimulator(_FakeSim(100.0), model).simulate(t, 30.0, seed=42)
        lc2 = LightCurveSimulator(_FakeSim(100.0), model).simulate(t, 30.0, seed=42)
        np.testing.assert_array_equal(lc1.flux, lc2.flux)

    def test_noise_statistics_match_sigma(self):
        t = np.linspace(0, 1, 5000)
        lc = LightCurveSimulator(_FakeSim(50.0), _ConstDip()).simulate(t, 30.0, seed=1)
        resid = lc.flux - lc.flux_clean
        assert np.std(resid) == pytest.approx(1 / 50.0, rel=0.1)

    def test_simulate_returns_lightcurve_instance(self):
        t = np.linspace(0, 1, 5000)
        lc = LightCurveSimulator(_FakeSim(50.0), _ConstDip()).simulate(t, 30.0, seed=1)
        assert isinstance(lc, LightCurve)

    def test_repr_starts_with_lightcurve(self):
        lc = LightCurveSimulator(_FakeSim(150.0), _ConstDip()).simulate(
            np.linspace(0, 1, 40), exptime=30.0, seed=0
        )
        assert repr(lc).startswith("LightCurve(")

    def test_repr_contains_n(self):
        lc = LightCurveSimulator(_FakeSim(150.0), _ConstDip()).simulate(
            np.linspace(0, 1, 40), exptime=30.0, seed=0
        )
        assert "n=40" in repr(lc)

    def test_repr_contains_snr(self):
        lc = LightCurveSimulator(_FakeSim(150.0), _ConstDip()).simulate(
            np.linspace(0, 1, 40), exptime=30.0, seed=0
        )
        assert "snr=150.0" in repr(lc)
