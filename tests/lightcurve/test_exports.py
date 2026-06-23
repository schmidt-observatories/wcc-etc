"""Tests that lightcurve and exoarchive names are exported from wcc_etc."""

import wcc_etc


class TestLightcurveExports:
    def test_lightcurve_names_exported(self):
        for name in [
            "FluxModel",
            "TransitModel",
            "LightCurveSimulator",
            "LightCurve",
            "plot_lightcurve_mpl",
            "plot_lightcurve_bokeh",
        ]:
            assert hasattr(wcc_etc, name), f"missing export: {name}"
            assert name in wcc_etc.__all__


class TestExoarchiveExports:
    def test_exoarchive_functions_exported(self):
        assert hasattr(wcc_etc, "download_exoplanet_archive")
        assert hasattr(wcc_etc, "load_exoplanet_archive")
        assert "download_exoplanet_archive" in wcc_etc.__all__
        assert "load_exoplanet_archive" in wcc_etc.__all__
