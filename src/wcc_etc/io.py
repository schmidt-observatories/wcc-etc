import os
import tomllib
import numpy as np
from importlib.resources import files

PACKAGE_PATH = str(files("wcc_etc.data")._paths[0])    #: Path to data & config files.

__all__ = ["read_config", "get_sensor_config"]

SENSORS = {"zwo": {"bb": "Lazuli_WCC_kepler_20251010_eol_zwo",
                      "u": "Lazuli_WCC_u_20250912",
                      "r": "Lazuli_WCC_r_20250907_EOL",
                      "z": None,
                      "g": "Lazuli_WCC_g_20250907_EOL",
                      "i": None,
                      "r_defocus": None,
                      "bb_defocus": None,
                      "halpha": None,
                      "nii": None,
                      "oiii": None,
                      "heii": None,
                  },
          "qcmos": {"u": None,
                    "bb": "Lazuli_WCC_kepler_20251010_eol_qCMOS",
                    "r": "Lazuli_WCC_r_20251008_EOL_qCMOS",
                    "g": None,
                    "z": None,
                    "i": None,
                   }
          }

# shortcut to simplify usage.
_KIND_NAMES = {shortcut:"zwo" for shortcut in ["sony", "imx", "imx455"]}

def read_config(filename, source="config"):
    """Read a single configuration file.

    - If the input filename does not specifically include a path, it will be
      looked for in the default :data:`PACKAGE_PATH` directory.
    - Currently, only `.toml` configuration files are supported.

    Parameters
    ----------
    filename : str or list
        Filename of the configuration file. If no extension is provided,
        `.toml` is assumed. `filename="this"` is equivalent to `filename="this.toml".
    source: str
        provide the directory where the file is supposed to be stored, e.g. source="config".
        This is used only if the input filename is not a fullpath and this function
        has to look for the fullpath using expand_path.

    Returns
    -------
    dict
        Configuration as a nested dictionary.

    Raises
    ------
    NotImplementedError
        If the configuration file extension is not supported.
    """
    # dict structure
    if type(filename) is dict:
        return filename
    
    # make sure you get the fullpath
    filename = expand_path(filename, source=source, test_extension=False)

    # parse the extension to know how to read it.
    _, extension = os.path.splitext(filename)
    if extension is None or len(extension) == 0:
        raise ValueError(f"no extension associated to given filename {fname=}. It cannot be loaded")
        
    if extension.lower() == ".toml":
        config = tomllib.load( open(filename, "rb") )
    # other supported extensions here: e.g. parquet, csv etc.
    else:
        raise NotImplementedError(f"Unknown configuration extension {extension=}.")
    
    return config

def get_sensor_config(kind, band, **kwargs):
    """ """
    # trick to allow nicknames like 'sony' in place of 'zwo'
    kind = _KIND_NAMES.get(kind, kind) 
    kind_sensors = SENSORS.get(kind)
    if band not in kind_sensors:
        raise ValueError(f"{kind_sensors} sensor do not have {band} band.")
    else:
        throughput_filter = kind_sensors.get(band)
        if throughput_filter is None:
            raise NotImplementedError("{kind_sensors} {band} sensor exists but no throghputcurve implemented yet.")

    # Build the config file
    config = read_config("lazuli")
    config |= read_config(kind)
    config["sensor"]["path_total_throughput"] = os.path.join("throughput", throughput_filter,
                                                  f"{throughput_filter}_throughput.csv")
    return config | kwargs


def expand_path(filename, source=None, test_extension=False):
    """Get the full file path, including the config path if necessary.

    If the input filename does not specifically include a path, it will be
    looked for in the default :data:`PACKAGE_PATH` directory.

    Parameters
    ----------
    filename : str
        File name.
    source: str, None
        provide the directory where the file is supposed to be stored, e.g. source="config"
        if given, the file will be looked for inside PACKAGE_PATH/{source}.
        if None, it will be inside PACKAGE_PATH/
    Returns
    -------
    str
        Filename including the default path if needed.
    """

    if os.path.isfile(filename):  # filename includes a path
        fname = filename
    else:                          # use PACKAGE_PATH as default
        if source is not None:
            fname = os.path.join(PACKAGE_PATH, source, filename)
        else:
            fname = os.path.join(PACKAGE_PATH, filename)

        _, extension = os.path.splitext(fname)
        if extension is None or len(extension) == 0:
            fname = f"{fname}.toml"
        elif test_extension and (extension not in [".toml", ".csv", ".parquet"]): # specify here list of accepted extensions.
            raise NotImplementedError(f"Unknown configuration extension {extension=}.")

    return fname
