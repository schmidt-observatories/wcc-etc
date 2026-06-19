import wcc_etc


def test_exoarchive_functions_exported():
    assert hasattr(wcc_etc, "download_exoplanet_archive")
    assert hasattr(wcc_etc, "load_exoplanet_archive")
    assert "download_exoplanet_archive" in wcc_etc.__all__
    assert "load_exoplanet_archive" in wcc_etc.__all__
