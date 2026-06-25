"""Tests for FluxModel base class and TransitModel."""

import numpy as np
import pytest

from wcc_etc.lightcurve import FluxModel, TransitModel


class TestFluxModel:
    def test_base_is_abstract(self):
        with pytest.raises(NotImplementedError):
            FluxModel().relative_flux(np.linspace(0, 1, 5))


class TestTransitModel:
    def test_matches_batman_directly(self):
        batman = pytest.importorskip("batman")
        t = np.linspace(-0.1, 0.1, 200)
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
        got = model.relative_flux(t)

        params = batman.TransitParams()
        params.t0, params.per, params.rp, params.a = 0.0, 1.0, 0.1, 15.0
        params.inc, params.ecc, params.w = 87.0, 0.0, 90.0
        params.limb_dark, params.u = "quadratic", [0.1, 0.3]
        expected = batman.TransitModel(params, t).light_curve(params)

        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    def _transit_flux(self):
        pytest.importorskip("batman")
        t = np.linspace(-0.25, 0.25, 400)
        model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=90.0, u=(0.0, 0.0))
        return model.relative_flux(t)

    def test_out_of_transit_is_unity(self):
        flux = self._transit_flux()
        assert flux[0] == pytest.approx(1.0, abs=1e-6)

    def test_dip_is_present(self):
        flux = self._transit_flux()
        assert flux.min() < 1.0

    def test_dip_depth_matches_rp_squared(self):
        flux = self._transit_flux()
        assert flux.min() == pytest.approx(1 - 0.1**2, abs=2e-3)
