"""Download and load NASA Exoplanet Archive planet parameters.

`download_exoplanet_archive` fetches the PSCompPars table (one complete row per
confirmed planet) to a local CSV cache; `load_exoplanet_archive` reads it.
`TransitModel.from_planet` consumes the result to build transit models for real
planets.
"""
from pathlib import Path

import pandas as pd

#: Identifier + transit columns we select and persist.
ARCHIVE_COLUMNS = [
    "pl_name", "hostname", "pl_orbper", "pl_ratror", "pl_ratdor",
    "pl_orbincl", "pl_tranmid", "pl_orbeccen", "pl_orblper",
    "pl_radj", "pl_orbsmax", "st_rad",
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
