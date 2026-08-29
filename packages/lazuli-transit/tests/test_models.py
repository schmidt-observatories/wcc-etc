"""Tests for FluxModel and the jaxoplanet-backed TransitModel.

The reference light curves below were generated offline with batman 2.5.1
(GPLv3) and are frozen here as plain numbers, so the test suite pins the
transit physics without depending on batman at run time. jaxoplanet
reproduces them to <1e-8; the tests allow 1e-7.
"""

import sys

import numpy as np
import pandas as pd
import pytest
from lazuli_transit import FluxModel, TransitModel

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

# batman 2.5.1: as above but ecc=0.3, w=45
QUAD_ECC_T = QUAD_CIRC_T
QUAD_ECC_F = np.array([
    1.0, 1.0, 0.99175589838116,
    0.9896562609977289, 0.9894350670330849, 0.9896724773346288,
    0.992153457953354, 1.0, 1.0,
])

# batman 2.5.1: t0=0, per=3.5, rp=0.08, a=10, inc=88.5, limb_dark="linear", u=(0.4,)
LINEAR_T = np.array([-0.14, -0.1, -0.05, 0.0, 0.05, 0.1, 0.14])
LINEAR_F = np.array([
    1.0, 1.0, 0.9947802968381505,
    0.9927234687661699, 0.9947802968381505, 1.0,
    1.0,
])

# batman 2.5.1: t0=0, per=1, rp=0.1, a=15, inc=87, limb_dark="uniform", u=()
UNIFORM_T = QUAD_CIRC_T
UNIFORM_F = np.array([
    1.0, 1.0, 0.9930277748778754,
    0.99, 0.99, 0.99,
    0.9930277748778754, 1.0, 1.0,
])

REF_ATOL = 1e-7


def test_fluxmodel_base_is_abstract():
    with pytest.raises(NotImplementedError):
        FluxModel().relative_flux(np.linspace(0, 1, 5))


def test_transit_circular_matches_batman_reference():
    """Circular quadratic-limb-darkened curve reproduces the batman reference."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0, ecc=0.0,
                         w=90.0, limb_dark="quadratic", u=(0.1, 0.3))
    got = model.relative_flux(QUAD_CIRC_T)
    np.testing.assert_allclose(got, QUAD_CIRC_F, rtol=0, atol=REF_ATOL)


def test_transit_eccentric_matches_batman_reference():
    """Eccentric orbit (e=0.3, w=45 deg) reproduces the batman reference."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0, ecc=0.3,
                         w=45.0, limb_dark="quadratic", u=(0.1, 0.3))
    got = model.relative_flux(QUAD_ECC_T)
    np.testing.assert_allclose(got, QUAD_ECC_F, rtol=0, atol=REF_ATOL)


def test_transit_linear_limb_dark_matches_batman_reference():
    """Linear limb-darkening law reproduces the batman reference."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(t0=0.0, per=3.5, rp=0.08, a=10.0, inc=88.5,
                         limb_dark="linear", u=(0.4,))
    got = model.relative_flux(LINEAR_T)
    np.testing.assert_allclose(got, LINEAR_F, rtol=0, atol=REF_ATOL)


def test_transit_uniform_limb_dark_matches_batman_reference():
    """Uniform (no limb darkening) law reproduces the batman reference."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0,
                         limb_dark="uniform", u=())
    got = model.relative_flux(UNIFORM_T)
    np.testing.assert_allclose(got, UNIFORM_F, rtol=0, atol=REF_ATOL)


def test_transit_out_of_transit_is_unity():
    """Flux well outside the transit window is exactly 1."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=90.0, u=(0.1, 0.3))
    assert model.relative_flux(np.array([-0.25]))[0] == pytest.approx(1.0, abs=1e-12)


def test_transit_uniform_depth_equals_rp_squared():
    """A central uniform-disk transit is exactly rp**2 deep (analytic)."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=90.0,
                         limb_dark="uniform", u=())
    flux = model.relative_flux(np.linspace(-0.02, 0.02, 101))
    assert flux.min() == pytest.approx(1 - 0.1**2, abs=1e-6)


def test_transit_limb_darkening_deepens_the_floor():
    """Limb darkening makes the central transit deeper than a uniform disk."""
    pytest.importorskip("jaxoplanet")
    t = np.linspace(-0.02, 0.02, 101)
    kw = dict(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=90.0)
    uniform = TransitModel(limb_dark="uniform", u=(), **kw).relative_flux(t).min()
    quad = (
        TransitModel(limb_dark="quadratic", u=(0.1, 0.3), **kw).relative_flux(t).min()
    )
    assert quad < uniform


def test_transit_is_symmetric_about_t0_for_circular_orbit():
    """A circular-orbit transit is mirror-symmetric around t0."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(t0=0.0, per=1.0, rp=0.1, a=15.0, inc=87.0, u=(0.1, 0.3))
    offsets = np.linspace(0.001, 0.012, 12)
    np.testing.assert_allclose(
        model.relative_flux(offsets), model.relative_flux(-offsets),
        rtol=0, atol=1e-12)


def test_transit_deeper_planet_gives_deeper_minimum():
    """Increasing rp deepens the transit floor."""
    pytest.importorskip("jaxoplanet")
    t = np.linspace(-0.02, 0.02, 101)
    kw = dict(t0=0.0, per=1.0, a=15.0, inc=90.0, u=(0.1, 0.3))
    small = TransitModel(rp=0.05, **kw).relative_flux(t).min()
    big = TransitModel(rp=0.15, **kw).relative_flux(t).min()
    assert big < small


def test_transit_preserves_input_shape():
    """relative_flux returns an array shaped like its time argument."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel()
    t = np.linspace(-0.02, 0.02, 24)
    assert model.relative_flux(t).shape == t.shape


def test_transit_non_polynomial_limb_dark_raises():
    """A non-polynomial limb-darkening law is rejected with a clear error."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(limb_dark="nonlinear", u=(0.5, 0.1, 0.1, -0.1))
    with pytest.raises(ValueError, match="polynomial"):
        model.relative_flux(np.linspace(-0.01, 0.01, 5))


def test_transit_wrong_coefficient_count_raises():
    """A limb-darkening law given the wrong number of coefficients is rejected."""
    pytest.importorskip("jaxoplanet")
    model = TransitModel(limb_dark="quadratic", u=(0.3,))
    with pytest.raises(ValueError, match="coefficient"):
        model.relative_flux(np.linspace(-0.01, 0.01, 5))


def test_transit_missing_jaxoplanet_raises_hint(monkeypatch):
    """Without jaxoplanet installed, the error names the package to install."""
    for name in list(sys.modules):
        if name == "jaxoplanet" or name.startswith("jaxoplanet."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "jaxoplanet", None)
    with pytest.raises(ImportError, match="jaxoplanet"):
        TransitModel().relative_flux(np.linspace(-0.01, 0.01, 5))


RJUP_RSUN = 0.10276268506540175
AU_RSUN = 215.03215567054764


def _df():
    return pd.DataFrame(
        [
            dict(
                pl_name="WASP-12 b",
                tran_flag=1,
                pl_orbper=1.0914,
                pl_ratror=0.117,
                pl_ratdor=3.04,
                pl_orbincl=83.3,
                pl_tranmid=2456305.46,
                pl_orbeccen=0.0,
                pl_orblper=90.0,
                pl_radj=1.9,
                pl_orbsmax=0.0234,
                st_rad=1.66,
            ),
            dict(
                pl_name="HD 209458 b",
                tran_flag=1,
                pl_orbper=3.5247,
                pl_ratror=np.nan,
                pl_ratdor=np.nan,
                pl_orbincl=86.7,
                pl_tranmid=2451370.0,
                pl_orbeccen=np.nan,
                pl_orblper=np.nan,
                pl_radj=1.38,
                pl_orbsmax=0.0475,
                st_rad=1.19,
            ),
        ]
    )


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


def test_from_planet_non_transiting_raises():
    df = _df()
    df.loc[df["pl_name"] == "WASP-12 b", "tran_flag"] = 0
    with pytest.raises(ValueError, match="not flagged as transiting"):
        TransitModel.from_planet("WASP-12 b", df=df)


def test_from_planet_non_transiting_override():
    df = _df()
    df.loc[df["pl_name"] == "WASP-12 b", "tran_flag"] = 0
    m = TransitModel.from_planet("WASP-12 b", df=df, require_transit=False)
    assert m.per == pytest.approx(1.0914)


def test_from_planet_skips_check_when_tran_flag_absent():
    df = _df().drop(columns=["tran_flag"])
    m = TransitModel.from_planet("WASP-12 b", df=df)
    assert m.per == pytest.approx(1.0914)
