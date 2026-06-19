import pandas as pd
import pytest

from lazuli_transit.archive import (
    ARCHIVE_COLUMNS,
    default_archive_path,
    load_exoplanet_archive,
)


def test_default_path_is_under_home_and_expanded():
    p = default_archive_path()
    assert p.name == "exoplanet_archive_pscomppars.csv"
    assert "~" not in str(p)
    assert ".lazuli_transit" in str(p)


def test_archive_columns_contains_required_fields():
    for col in ("pl_name", "pl_orbper", "pl_ratror", "pl_ratdor", "st_rad"):
        assert col in ARCHIVE_COLUMNS


def test_load_reads_existing_csv(tmp_path):
    csv = tmp_path / "arch.csv"
    pd.DataFrame({"pl_name": ["WASP-12 b"], "pl_orbper": [1.09]}).to_csv(csv, index=False)
    df = load_exoplanet_archive(path=csv)
    assert list(df["pl_name"]) == ["WASP-12 b"]


def test_load_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_exoplanet_archive"):
        load_exoplanet_archive(path=tmp_path / "nope.csv")
