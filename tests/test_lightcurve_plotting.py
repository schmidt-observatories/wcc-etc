import numpy as np
import matplotlib
matplotlib.use("Agg")

from wcc_etc.lightcurve import FluxModel, LightCurveSimulator
from wcc_etc.plotting import plot_lightcurve_mpl, plot_lightcurve_bokeh


class _Dip(FluxModel):
    def relative_flux(self, time):
        return np.ones_like(np.asarray(time, dtype=float))


class _FakeSim:
    def get_image_snr(self, **kw):
        return {"snr": 100.0}


def _lc():
    t = np.linspace(0, 1, 100)
    return LightCurveSimulator(_FakeSim(), _Dip()).simulate(t, 30.0, seed=0)


def test_mpl_returns_fig_ax_and_draws_both():
    fig, ax = plot_lightcurve_mpl(_lc())
    assert ax.has_data()
    assert len(ax.lines) >= 1          # model line
    assert len(ax.collections) >= 1    # errorbar points

def test_mpl_show_model_only_has_no_errorbar_collection():
    fig, ax = plot_lightcurve_mpl(_lc(), show_noise=False, show_model=True)
    assert len(ax.lines) >= 1
    assert len(ax.collections) == 0

def test_mpl_accepts_raw_arrays():
    lc = _lc()
    fig, ax = plot_lightcurve_mpl(time=lc.time, flux=lc.flux,
                                  flux_clean=lc.flux_clean, flux_err=lc.flux_err)
    assert ax.has_data()

def test_bokeh_components_returns_script_div():
    out = plot_lightcurve_bokeh(_lc(), return_="components")
    assert isinstance(out, tuple) and len(out) == 2
    assert "<div" in out[1]
