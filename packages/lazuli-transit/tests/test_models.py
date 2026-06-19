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


import pandas as pd

RJUP_RSUN = 0.10276268506540175
AU_RSUN = 215.03215567054764


def _df():
    return pd.DataFrame([
        dict(pl_name="WASP-12 b", pl_orbper=1.0914, pl_ratror=0.117,
             pl_ratdor=3.04, pl_orbincl=83.3, pl_tranmid=2456305.46,
             pl_orbeccen=0.0, pl_orblper=90.0, pl_radj=1.9, pl_orbsmax=0.0234,
             st_rad=1.66),
        dict(pl_name="HD 209458 b", pl_orbper=3.5247, pl_ratror=np.nan,
             pl_ratdor=np.nan, pl_orbincl=86.7, pl_tranmid=2451370.0,
             pl_orbeccen=np.nan, pl_orblper=np.nan, pl_radj=1.38,
             pl_orbsmax=0.0475, st_rad=1.19),
    ])


def test_from_planet_uses_ratio_columns_directly():
    m = TransitModel.from_planet("WASP-12 b", df=_df())
    assert m.per == pytest.approx(1.0914)
    assert m.rp == pytest.approx(0.117)
    assert m.a == pytest.approx(3.04)
    assert m.inc == pytest.approx(83.3)
    assert m.t0 == pytest.approx(2456305.46)


def test_from_planet_rp_and_a_fallbacks():
    m = TransitModel.from_planet("HD 209458 b", df=_df())
    assert m.rp == pytest.approx(1.38 * RJUP_RSUN / 1.19)
    assert m.a == pytest.approx(0.0475 * AU_RSUN / 1.19)


def test_from_planet_nan_orbital_defaults():
    m = TransitModel.from_planet("HD 209458 b", df=_df())
    assert m.ecc == pytest.approx(0.0)
    assert m.w == pytest.approx(90.0)


def test_from_planet_name_is_case_and_space_insensitive():
    m = TransitModel.from_planet("wasp-12b", df=_df())
    assert m.per == pytest.approx(1.0914)


def test_from_planet_unknown_raises():
    with pytest.raises(ValueError, match="not found"):
        TransitModel.from_planet("Kepler-999 z", df=_df())


def test_from_planet_missing_period_raises():
    df = _df()
    df.loc[df["pl_name"] == "WASP-12 b", "pl_orbper"] = np.nan
    with pytest.raises(ValueError, match="pl_orbper"):
        TransitModel.from_planet("WASP-12 b", df=df)
