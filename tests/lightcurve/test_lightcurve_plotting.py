"""Tests for plot_lightcurve_mpl and plot_lightcurve_bokeh."""

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
from wcc_etc.lightcurve import FluxModel, LightCurveSimulator
from wcc_etc.plotting import plot_lightcurve_bokeh, plot_lightcurve_mpl


class _Flat(FluxModel):
    def relative_flux(self, time):
        return np.ones_like(np.asarray(time, dtype=float))


class _FakeSim:
    def get_image_snr(self, **kw):
        return {"snr": 100.0}


def _lc():
    t = np.linspace(0, 1, 100)
    return LightCurveSimulator(_FakeSim(), _Flat()).simulate(t, 30.0, seed=0)


class TestLightCurvePlotting:
    def test_mpl_has_data(self):
        fig, ax = plot_lightcurve_mpl(_lc())
        assert ax.has_data()

    def test_mpl_draws_lines(self):
        fig, ax = plot_lightcurve_mpl(_lc())
        assert len(ax.lines) >= 1

    def test_mpl_draws_collections(self):
        fig, ax = plot_lightcurve_mpl(_lc())
        assert len(ax.collections) >= 1

    def test_mpl_show_model_only_has_lines(self):
        fig, ax = plot_lightcurve_mpl(_lc(), show_noise=False, show_model=True)
        assert len(ax.lines) >= 1

    def test_mpl_show_model_only_has_no_collections(self):
        fig, ax = plot_lightcurve_mpl(_lc(), show_noise=False, show_model=True)
        assert len(ax.collections) == 0

    def test_mpl_accepts_raw_arrays(self):
        lc = _lc()
        fig, ax = plot_lightcurve_mpl(
            time=lc.time, flux=lc.flux, flux_clean=lc.flux_clean, flux_err=lc.flux_err
        )
        assert ax.has_data()

    def test_mpl_both_off_no_legend_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, ax = plot_lightcurve_mpl(_lc(), show_noise=False, show_model=False)
        assert ax.get_legend() is None

    def test_bokeh_components_returns_tuple(self):
        out = plot_lightcurve_bokeh(_lc(), return_="components")
        assert isinstance(out, tuple) and len(out) == 2

    def test_bokeh_components_div_contains_html(self):
        out = plot_lightcurve_bokeh(_lc(), return_="components")
        assert "<div" in out[1]

    def test_bokeh_both_off_does_not_raise(self):
        out = plot_lightcurve_bokeh(
            _lc(), show_noise=False, show_model=False, return_="components"
        )
        assert isinstance(out, tuple) and len(out) == 2

    def test_bokeh_raw_array_flux_err_does_not_raise(self):
        lc = _lc()
        err = np.full_like(lc.flux, lc.flux_err)
        out = plot_lightcurve_bokeh(
            time=lc.time,
            flux=lc.flux,
            flux_clean=lc.flux_clean,
            flux_err=err,
            return_="components",
        )
        assert isinstance(out, tuple) and len(out) == 2
