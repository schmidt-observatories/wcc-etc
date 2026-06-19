import numpy as np
import pytest

from lazuli_transit import FluxModel, TransitModel


def test_fluxmodel_base_is_abstract():
    with pytest.raises(NotImplementedError):
        FluxModel().relative_flux(np.linspace(0, 1, 5))


def test_transit_relative_flux_matches_batman_directly():
    batman = pytest.importorskip("batman")
    t = np.linspace(-0.1, 0.1, 200)
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0,
                         ecc=0.0, w=90.0, limb_dark="quadratic", u=(0.1, 0.3))
    got = model.relative_flux(t)

    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = 0.0, 1.0, 0.1, 15.0
    params.inc, params.ecc, params.w = 87.0, 0.0, 90.0
    params.limb_dark, params.u = "quadratic", [0.1, 0.3]
    expected = batman.TransitModel(params, t).light_curve(params)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
