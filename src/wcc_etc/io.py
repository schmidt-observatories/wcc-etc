import os
import pandas as pd
import tomllib
import numpy as np
from importlib.resources import files

PACKAGE_PATH = str(files("wcc_etc.data")._paths[0])    #: Path to data & config files.
PICKLES_DIR = os.path.join(PACKAGE_PATH, "astr_obj_models","stars","pickles_models")
PICKLES_MAPPING = os.path.join(PICKLES_DIR, "pickles_mapping.csv")

__all__ = ["read_config", "get_sensor_config"]

SENSORS = {"zwo": {"bb": "wcc_imx_bb_throughput.csv",#"Lazuli_WCC_kepler_20251010_eol_zwo",
                   "u":  "wcc_imx_u_throughput.csv", #"Lazuli_WCC_u_20250912",
                   "g":  "wcc_imx_g_throughput.csv", #"Lazuli_WCC_g_20250907_EOL",
                   "r":  "wcc_imx_r_throughput.csv", #"Lazuli_WCC_r_20250907_EOL",
                   "i":  "wcc_imx_i_throughput.csv", # None,
                   "z":  "wcc_imx_z_throughput.csv", # None,
                   "r_defocus": None,
                   "bb_defocus": None,
                   "halpha": None,
                   "nii": None,
                   "oiii": None,
                   "heii": None,
                  },
          "qcmos": {"bb": "wcc_hwk_bb_throughput.csv",#"Lazuli_WCC_kepler_20251010_eol_qCMOS",
                     "u": "wcc_hwk_u_throughput.csv",
                     "g": "wcc_hwk_g_throughput.csv",#"Lazuli_WCC_g_20250907_EOL_qCMOS",
                     "r": "wcc_hwk_r_throughput.csv",#"Lazuli_WCC_r_20251008_EOL_qCMOS",
                     "i": "wcc_hwk_i_throughput.csv",
                     "z": "wcc_hwk_z_throughput.csv",# None,
                  }
          }

# shortcut to simplify usage.
_KIND_NAMES = {shortcut:"zwo" for shortcut in ["sony", "imx", "imx455"]}

def get_pickles_mapping():
    """Get the pickles mapping DataFrame."""
    df = pd.read_csv(PICKLES_MAPPING, sep='\s+')
    return df

def get_pickles_spectrum_filename(spt,fullpath=True):
    """Get the pickles spectrum for a given spectral type. The following are available:
        'O5V', 'O9V', 'B0V', 'B1V', 'B3V', 'B5-7V', 'B8V', 'A0V', 'A2V',
        'A3V', 'A5V', 'F0V', 'F2V', 'F5V', 'F8V', 'G0V', 'G2V', 'G5V',
        'G8V', 'K0V', 'K2V', 'K5V', 'K7V', 'M0V', 'M2V', 'M4V', 'M5V',
        'B2IV', 'B6IV', 'A0IV', 'A4-7IV', 'F0-2IV', 'F5IV', 'F8IV', 'G0IV',
        'G2IV', 'G5IV', 'G8IV', 'K0IV', 'K1IV', 'K3IV', 'O8III', 'B1-2III',
        'B5III', 'B9III', 'A0III', 'A5III', 'F0III', 'F5III', 'G0III',
        'G5III', 'G8III', 'K0III', 'K3III', 'K5III', 'M0III', 'M5III',
        'M10III', 'B2II', 'B5II', 'F0II', 'F2II', 'G5II', 'K0-1II',
        'K3-4II', 'M3II', 'B0I', 'B5I', 'B8I', 'A0I', 'F0I', 'F5I', 'F8I',
        'G0I', 'G5I', 'G8I', 'K2I', 'K4I', 'M2I'
    """
    df = get_pickles_mapping()
    filename = df[df['spt'].values == spt]['filename'].values[0] + '.fits'
    if not filename:
        raise ValueError(f"No spectrum found for {spt}. Available SPT are {df['spt'].values}.")
    if fullpath:
        filename = os.path.join(PICKLES_DIR,'dat_uvk', filename)
    return filename


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
    """
    Get sensor configuration for a specific band.
    """
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
    # Old
    #config["sensor"]["path_total_throughput"] = os.path.join("throughput", throughput_filter,
    #                                              f"{throughput_filter}_throughput.csv")
    config["sensor"]["path_total_throughput"] = os.path.join("throughput", "20260304_wcc_throughputs",
                                                             throughput_filter)
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
