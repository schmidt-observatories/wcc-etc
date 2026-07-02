import lazuli_transit


def test_top_level_exports():
    for name in (
        "FluxModel",
        "TransitModel",
        "download_exoplanet_archive",
        "load_exoplanet_archive",
        "default_archive_path",
        "ARCHIVE_COLUMNS",
    ):
        assert hasattr(lazuli_transit, name), name
        assert name in lazuli_transit.__all__
