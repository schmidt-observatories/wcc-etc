"""Tests for LightCurveSimulator and LightCurve.

BATMAN_EXP300_F was generated offline with batman 2.5.1 using exp_time=300 s
and supersample_factor=1001, and is frozen here as plain numbers so nothing
GPL-licensed is needed to run the suite.
"""

import numpy as np
import pytest

from wcc_etc.lightcurve import (
    FluxModel,
    LightCurve,
    LightCurveSimulator,
    TransitModel,
    supersample_factor,
)

# batman 2.5.1: t0=0, per=1, rp=0.1, a=15, inc=87, ecc=0, w=90,
# limb_dark="quadratic", u=(0.1, 0.3), exp_time=300 s, supersample_factor=1001
BATMAN_EXP300_T = np.linspace(-0.012, 0.012, 13)
BATMAN_EXP300_F = np.array(
    [
        1.0,
        1.0,
        0.9987877958430657,
        0.9943466268616378,
        0.9908914325987047,
        0.9901756398030463,
        0.9900489441023405,
        0.9901756398030465,
        0.9908914325987045,
        0.9943466268616378,
        0.9987877958430657,
        1.0,
        1.0,
    ]
)


class _ConstDip(FluxModel):
    """Box dip of depth 0.01 over 0.25 < t < 0.75 (in the time units used)."""

    def relative_flux(self, time):
        time = np.asarray(time, dtype=float)
        return np.where((time > 0.25) & (time < 0.75), 0.99, 1.0)


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
        lc1 = LightCurveSimulator(_FakeSim(100.0), model).simulate(
            t, 30.0, seed=42, supersample=1
        )
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


class TestSupersampleFactor:
    def test_short_exposure_is_not_oversampled(self):
        """An exposure at or below the 10 s sub-sample target needs one sample."""
        assert supersample_factor(10.0) == 1

    def test_long_exposure_scales_with_exptime(self):
        """A 300 s exposure is split into 10 s sub-samples."""
        assert supersample_factor(300.0) == 31

    def test_factor_is_odd(self):
        """The count is forced odd so one sub-sample lands on the midpoint."""
        assert supersample_factor(40.0) % 2 == 1


class TestExposureIntegration:
    """`time` holds mid-exposure timestamps; the model is averaged over the
    exposure window centred on each one."""

    def _transit_sim(self):
        model = TransitModel(
            t0=0.0,
            per=1.0,
            rp=0.1,
            a=15.0,
            inc=87.0,
            ecc=0.0,
            w=90.0,
            limb_dark="quadratic",
            u=(0.1, 0.3),
        )
        return LightCurveSimulator(_FakeSim(1e9), model)

    def test_default_supersample_follows_the_rule(self):
        """With no override, the sub-sample count comes from supersample_factor."""
        lc = self._simulate_box(exptime=300.0)
        assert lc.supersample == supersample_factor(300.0)

    def _simulate_box(self, exptime, supersample=None):
        return LightCurveSimulator(_FakeSim(1e9), _ConstDip()).simulate(
            np.linspace(0, 1, 201), exptime, seed=0, supersample=supersample
        )

    def test_supersample_one_reproduces_instantaneous_model(self):
        """supersample=1 restores point sampling of the model."""
        model = _ConstDip()
        t = np.linspace(0, 1, 201)
        lc = self._simulate_box(exptime=300.0, supersample=1)
        np.testing.assert_array_equal(lc.flux_clean, model.relative_flux(t))

    def test_long_exposure_smears_ingress(self):
        """An exposure straddling ingress lands between in- and out-of-transit."""
        pytest.importorskip("jaxoplanet")
        # t = -0.00757 d is the first contact for this geometry.
        t = np.array([-0.00757])
        lc = self._transit_sim().simulate(t, 600.0, seed=0)
        assert 0.99 < lc.flux_clean[0] < 1.0

    def test_ingress_is_less_steep_than_instantaneous(self):
        """Averaging over a long exposure flattens the steepest ingress slope."""
        pytest.importorskip("jaxoplanet")
        t = np.linspace(-0.009, -0.005, 201)
        sim = self._transit_sim()
        smeared = sim.simulate(t, 600.0, seed=0).flux_clean
        instant = sim.simulate(t, 600.0, seed=0, supersample=1).flux_clean
        assert np.abs(np.gradient(smeared)).max() < np.abs(np.gradient(instant)).max()

    def test_converges_to_instantaneous_for_short_exposures(self):
        """As exptime -> 0 the integrated model returns to the point sample."""
        pytest.importorskip("jaxoplanet")
        t = np.linspace(-0.012, 0.012, 41)
        sim = self._transit_sim()
        short = sim.simulate(t, 1.0, seed=0).flux_clean
        instant = sim.simulate(t, 1.0, seed=0, supersample=1).flux_clean
        assert np.max(np.abs(short - instant)) < 1e-6

    def test_matches_batman_finite_exposure_reference(self):
        """The integrated model reproduces batman run with exp_time=300 s."""
        pytest.importorskip("jaxoplanet")
        lc = self._transit_sim().simulate(BATMAN_EXP300_T, 300.0, seed=0)
        np.testing.assert_allclose(lc.flux_clean, BATMAN_EXP300_F, atol=1e-5)

    def test_instantaneous_sampling_misses_the_smearing(self):
        """The bug this guards: point sampling is off the finite-exposure curve."""
        pytest.importorskip("jaxoplanet")
        lc = self._transit_sim().simulate(BATMAN_EXP300_T, 300.0, seed=0, supersample=1)
        assert np.max(np.abs(lc.flux_clean - BATMAN_EXP300_F)) > 1e-4

    def test_zero_supersample_rejected(self):
        """supersample must be at least 1."""
        with pytest.raises(ValueError):
            self._simulate_box(exptime=300.0, supersample=0)


class TestConstantSigmaApproximation:
    """flux_err is the baseline sigma, applied in and out of transit alike.
    Documented in the lightcurve module docstring; locked in here."""

    def test_flux_err_is_a_scalar(self):
        lc = LightCurveSimulator(_FakeSim(200.0), _ConstDip()).simulate(
            np.linspace(0, 1, 201), 30.0, seed=0
        )
        assert np.isscalar(lc.flux_err)

    def test_in_transit_scatter_matches_out_of_transit_scatter(self):
        t = np.linspace(0, 1, 20001)
        lc = LightCurveSimulator(_FakeSim(50.0), _ConstDip()).simulate(t, 30.0, seed=3)
        resid = lc.flux - lc.flux_clean
        in_tr = (t > 0.25) & (t < 0.75)
        assert np.std(resid[in_tr]) == pytest.approx(np.std(resid[~in_tr]), rel=0.05)
