"""Tests for FluxModel base class and TransitModel.

The models themselves live in `lazuli_transit` and are re-exported by
`wcc_etc.lightcurve`; the exhaustive physics suite lives with that package.
These tests cover what the ETC depends on: that the re-exported model
produces the right transit shape and rejects bad limb-darkening input.

QUAD_CIRC_F was generated offline with batman 2.5.1 and is frozen here as
plain numbers, so nothing GPL-licensed is needed to run the suite.
"""

import numpy as np
import pytest

from wcc_etc.lightcurve import FluxModel, TransitModel

# batman 2.5.1: t0=0, per=1, rp=0.1, a=15, inc=87, ecc=0, w=90,
# limb_dark="quadratic", u=(0.1, 0.3)
QUAD_CIRC_T = np.array(
    [-0.012, -0.009, -0.006, -0.003, 0.0, 0.003, 0.006, 0.009, 0.012]
)
QUAD_CIRC_F = np.array([
    1.0, 1.0, 0.9940213954705622,
    0.990300785471927, 0.9900207560814324, 0.990300785471927,
    0.9940213954705622, 1.0, 1.0,
])


class TestFluxModel:
    def test_base_is_abstract(self):
        with pytest.raises(NotImplementedError):
            FluxModel().relative_flux(np.linspace(0, 1, 5))


class TestTransitModel:
    def test_matches_batman_reference_curve(self):
        """The re-exported model reproduces the frozen batman reference."""
        pytest.importorskip("jaxoplanet")
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
        got = model.relative_flux(QUAD_CIRC_T)

        np.testing.assert_allclose(got, QUAD_CIRC_F, rtol=0, atol=1e-7)

    def transit_flux(self):
        pytest.importorskip("jaxoplanet")
        t = np.linspace(-0.25, 0.25, 400)
        model = TransitModel(
            t0=0.0, per=1.0, rp=0.1, a=15.0, inc=90.0, limb_dark="uniform", u=()
        )
        return model.relative_flux(t)

    def test_out_of_transit_is_unity(self):
        flux = self.transit_flux()
        assert flux[0] == pytest.approx(1.0, abs=1e-6)

    def test_dip_is_present(self):
        flux = self.transit_flux()
        assert flux.min() < 1.0

    def test_dip_depth_matches_rp_squared(self):
        flux = self.transit_flux()
        assert flux.min() == pytest.approx(1 - 0.1**2, abs=2e-3)

    def test_unsupported_limb_dark_law_raises(self):
        """jaxoplanet does polynomial laws only; anything else errors clearly."""
        pytest.importorskip("jaxoplanet")
        model = TransitModel(limb_dark="nonlinear", u=(0.5, 0.1, 0.1, -0.1))
        with pytest.raises(ValueError, match="polynomial"):
            model.relative_flux(np.linspace(-0.01, 0.01, 5))
