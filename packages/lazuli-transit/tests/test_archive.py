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


def test_archive_columns_contains_stellar_and_transit_fields():
    for col in ("tran_flag", "st_teff", "sy_gaiamag", "sy_vmag", "sy_tmag",
                "sy_jmag", "sy_hmag", "sy_kmag"):
        assert col in ARCHIVE_COLUMNS


def test_load_reads_existing_csv(tmp_path):
    csv = tmp_path / "arch.csv"
    pd.DataFrame({"pl_name": ["WASP-12 b"], "pl_orbper": [1.09]}).to_csv(csv, index=False)
    df = load_exoplanet_archive(path=csv)
    assert list(df["pl_name"]) == ["WASP-12 b"]


def test_load_missing_file_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_exoplanet_archive"):
        load_exoplanet_archive(path=tmp_path / "nope.csv")


import sys

import lazuli_transit.archive as arch


def test_download_returns_cache_when_present_without_network(tmp_path, monkeypatch):
    csv = tmp_path / "arch.csv"
    pd.DataFrame({"pl_name": ["WASP-12 b"]}).to_csv(csv, index=False)

    def _boom():
        raise AssertionError("network/astroquery must not be touched on cache hit")

    monkeypatch.setattr(arch, "_query_pscomppars", _boom)
    df = arch.download_exoplanet_archive(path=csv, refresh=False)
    assert list(df["pl_name"]) == ["WASP-12 b"]


def test_download_writes_cache_using_injected_query(tmp_path, monkeypatch):
    csv = tmp_path / "sub" / "arch.csv"  # parent dir does not exist yet
    fake = pd.DataFrame({c: [0] for c in arch.ARCHIVE_COLUMNS})
    fake["pl_name"] = ["WASP-12 b"]
    monkeypatch.setattr(arch, "_query_pscomppars", lambda: fake)

    df = arch.download_exoplanet_archive(path=csv, refresh=True)
    assert csv.exists()
    assert list(df["pl_name"]) == ["WASP-12 b"]
    assert list(arch.load_exoplanet_archive(path=csv)["pl_name"]) == ["WASP-12 b"]


def test_query_raises_hint_without_astroquery(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "astroquery.ipac.nexsci.nasa_exoplanet_archive", None
    )
    with pytest.raises(ImportError, match=r"pip install lazuli-transit\[archive\]"):
        arch._query_pscomppars()
