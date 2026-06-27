"""Tests that lightcurve and exoarchive names are exported from wcc_etc."""

import wcc_etc


class TestLightcurveExports:
    def test_lightcurve_names_are_attributes(self):
        for name in [
            "FluxModel",
            "TransitModel",
            "LightCurveSimulator",
            "LightCurve",
            "plot_lightcurve_mpl",
            "plot_lightcurve_bokeh",
        ]:
            assert hasattr(wcc_etc, name), f"missing attribute: {name}"

    def test_lightcurve_names_in_all(self):
        for name in [
            "FluxModel",
            "TransitModel",
            "LightCurveSimulator",
            "LightCurve",
            "plot_lightcurve_mpl",
            "plot_lightcurve_bokeh",
        ]:
            assert name in wcc_etc.__all__, f"missing from __all__: {name}"


class TestExoarchiveExports:
    def test_download_exoplanet_archive_is_attribute(self):
        assert hasattr(wcc_etc, "download_exoplanet_archive")

    def test_load_exoplanet_archive_is_attribute(self):
        assert hasattr(wcc_etc, "load_exoplanet_archive")

    def test_download_exoplanet_archive_in_all(self):
        assert "download_exoplanet_archive" in wcc_etc.__all__

    def test_load_exoplanet_archive_in_all(self):
        assert "load_exoplanet_archive" in wcc_etc.__all__
