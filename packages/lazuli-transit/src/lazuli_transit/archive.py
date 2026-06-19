"""Download and load NASA Exoplanet Archive planet parameters.

`download_exoplanet_archive` fetches the PSCompPars table (one complete row per
confirmed planet) to a local CSV cache; `load_exoplanet_archive` reads it.
`TransitModel.from_planet` consumes the result to build transit models for real
planets.
"""
from pathlib import Path

import pandas as pd

#: Identifier + transit columns we select and persist.
#:
#: Brightness/Teff columns chosen for high completeness (~95% populated in
#: PSCompPars): Gaia G, Johnson V, TESS T, and 2MASS J/H/K. The Sloan ugriz
#: bands that the WCC filters map to are only ~half-populated, so they are not
#: included by default. ``tran_flag`` (1 = transiting, 0 = not) is fully
#: populated and drives TransitModel.from_planet's transit check.
ARCHIVE_COLUMNS = [
    "pl_name", "hostname", "pl_orbper", "pl_ratror", "pl_ratdor",
    "pl_orbincl", "pl_tranmid", "pl_orbeccen", "pl_orblper",
    "pl_radj", "pl_orbsmax", "tran_flag",
    "st_rad", "st_teff",
    "sy_gaiamag", "sy_vmag", "sy_tmag", "sy_jmag", "sy_hmag", "sy_kmag",
]

_ASTROQUERY_HINT = (
    "download_exoplanet_archive requires 'astroquery'. "
    "Install it with: pip install lazuli-transit[archive]"
)


def default_archive_path():
    """Default cache location for the downloaded PSCompPars CSV."""
    return Path("~/.lazuli_transit/exoplanet_archive_pscomppars.csv").expanduser()


def load_exoplanet_archive(path=None):
    """Read the cached PSCompPars CSV into a DataFrame.

    Does not hit the network. Raises FileNotFoundError (pointing at
    download_exoplanet_archive) if the cache file does not exist.
    """
    path = Path(path) if path is not None else default_archive_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No archive cache at {path}. Run download_exoplanet_archive() first."
        )
    return pd.read_csv(path)


def _query_pscomppars():
    """Fetch the PSCompPars table via astroquery as a pandas DataFrame.

    Isolated (and monkeypatchable) so download_exoplanet_archive's caching
    logic can be tested without the network.
    """
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import (
            NasaExoplanetArchive,
        )
    except ImportError as exc:  # pragma: no cover - exercised via hint test
        raise ImportError(_ASTROQUERY_HINT) from exc

    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars", select=",".join(ARCHIVE_COLUMNS)
    )
    return table.to_pandas()


def download_exoplanet_archive(path=None, refresh=False):
    """Download the PSCompPars table to a local CSV cache and return it.

    If the cache file exists and ``refresh`` is False, load it from disk
    instead of querying the archive. Otherwise query astroquery, write the CSV
    (creating parent directories), and return the DataFrame.
    """
    path = Path(path) if path is not None else default_archive_path()
    if path.exists() and not refresh:
        return load_exoplanet_archive(path=path)

    df = _query_pscomppars()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
