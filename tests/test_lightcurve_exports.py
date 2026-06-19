import wcc_etc


def test_lightcurve_names_exported():
    for name in ["FluxModel", "TransitModel", "LightCurveSimulator",
                 "LightCurve", "plot_lightcurve_mpl", "plot_lightcurve_bokeh"]:
        assert hasattr(wcc_etc, name), f"missing export: {name}"
        assert name in wcc_etc.__all__
