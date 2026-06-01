import os
import warnings
import pandas as pd
import tomllib
import pandas

from importlib.resources import files


from glob import glob

PACKAGE_PATH = str(files("wcc_etc.data")._paths[0])    #: Path to data & config files.
_PICKLES_DIR = os.path.join(PACKAGE_PATH, "astr_obj_models", "stars","pickles_models")
PICKLES_MAPPING = pd.read_csv( os.path.join(_PICKLES_DIR, "pickles_mapping.csv") , sep=r'\s+')

# Generate the name database
_list_of_astropath = glob(PACKAGE_PATH + "*/astrophysics/**", recursive=True) + \
                     glob(PACKAGE_PATH + "*/astr_obj_models/**", recursive=True)
                     
ASTROFILE_DF = pandas.DataFrame({"basename": [os.path.basename(entry_) for entry_ in _list_of_astropath],
                                "fullpath": _list_of_astropath})


__all__ = ["read_config", "get_sensor_config"]

SENSORS = {"zwo": {"bb":     "wcc_imx_bb_throughput.csv",
                   "u":      "wcc_imx_u_throughput.csv",
                   "g":      "wcc_imx_g_throughput.csv",
                   "r":      "wcc_imx_r_throughput.csv",
                   "i":      "wcc_imx_i_throughput.csv",
                   "z":      "wcc_imx_z_throughput.csv",
                   "r+1":    "wcc_imx_r_throughput.csv",
                   "r-1":    "wcc_imx_r_throughput.csv",
                   "bb2":    "wcc_imx_bb_throughput.csv",
                   "halpha": None,
                   "nii":    None,
                   "oiii":   None,
                   "heii":   None,
                   "hbeta":  None,
                  },
          "qcmos": {"bb": "wcc_hwk_bb_throughput.csv",#"Lazuli_WCC_kepler_20251010_eol_qCMOS",
                  "u":  "wcc_hwk_u_throughput.csv",
                  "g":  "wcc_hwk_g_throughput.csv",#"Lazuli_WCC_g_20250907_EOL_qCMOS",
                  "r":  "wcc_hwk_r_throughput.csv",#"Lazuli_WCC_r_20251008_EOL_qCMOS",
                  "i":  "wcc_hwk_i_throughput.csv",
                  "z":  "wcc_hwk_z_throughput.csv",# None,
                  }
          }

sensor_info = {
        "1":  {"center": (-1248.5, 437.0),  "sensorfilter": 'qcmos:bb',   "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "BB"},
        "2":  {"center": (-746.5, 437.0),   "sensorfilter": 'qcmos:z',    "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "z"},
        "3":  {"center": (-414.5, 437.0),   "sensorfilter": 'qcmos:bb',   "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "BB"},
        "4":  {"center": (-78.0, 437.0),    "sensorfilter": 'qcmos:g',    "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "g"},
        "5":  {"center": (256.0, 437.0),    "sensorfilter": 'qcmos:bb',   "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "BB"},
        "6":  {"center": (590.5, 437.0),    "sensorfilter": 'qcmos:u',    "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "u"},
        "7":  {"center": (924.5, 437.0),    "sensorfilter": 'qcmos:bb',   "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "BB"},
        "8":  {"center": (1258.5, 437.0),   "sensorfilter": 'qcmos:i',    "sensor": "hwk4123", "focus_level": "0wave", "filter_label": "i"},
        "9":  {"center": (-1244.5, 12.0),   "sensorfilter": 'zwo:nii',    "sensor": "imx455" , "focus_level": "0wave", "filter_label": "N-II"},
        "10": {"center": (-886.5, 12.0),    "sensorfilter": 'zwo:halpha', "sensor": "imx455" , "focus_level": "0wave", "filter_label": "H-alpha"},
        "11": {"center": (-533.5, 12.0),    "sensorfilter": 'zwo:hbeta',  "sensor": "imx455" , "focus_level": "0wave", "filter_label": "H-Beta"},
        "12": {"center": (-176.0, 12.0),    "sensorfilter": 'zwo:bb2',    "sensor": "imx455" , "focus_level": "2wave", "filter_label": "BB +2w"},
        "13": {"center": (183.5, 12.0),     "sensorfilter": 'zwo:i',      "sensor": "imx455" , "focus_level": "0wave", "filter_label": "i"},
        "14": {"center": (542.5, 12.0),     "sensorfilter": 'zwo:g',      "sensor": "imx455" , "focus_level": "0wave", "filter_label": "g"},
        "15": {"center": (896.5, 12.0),     "sensorfilter": 'zwo:r',      "sensor": "imx455" , "focus_level": "0wave", "filter_label": "r"},
        "16": {"center": (1254.0, 12.0),    "sensorfilter": 'zwo:r+1',    "sensor": "imx455" , "focus_level": "1wave", "filter_label": "r +1w"},
        "17": {"center": (-1254.0, -399.5), "sensorfilter": 'zwo:bb',     "sensor": "imx455" , "focus_level": "0wave", "filter_label": "BB"},
        "18": {"center": (-891.5, -399.5),  "sensorfilter": 'zwo:u',      "sensor": "imx455" , "focus_level": "0wave", "filter_label": "u"},
        "19": {"center": (-532.5, -399.5),  "sensorfilter": 'zwo:heii',   "sensor": "imx455" , "focus_level": "0wave", "filter_label": "He II"},
        "20": {"center": (-174.5, -399.5),  "sensorfilter": 'zwo:z',      "sensor": "imx455" , "focus_level": "0wave", "filter_label": "z"},
        "21": {"center": (184.5, -399.5),   "sensorfilter": 'zwo:oiii',   "sensor": "imx455" , "focus_level": "0wave", "filter_label": "O-III"},
        "22": {"center": (541.5, -399.5),   "sensorfilter": 'zwo:bb',     "sensor": "imx455" , "focus_level": "0wave", "filter_label": "BB"},
        "23": {"center": (901.0, -399.5),   "sensorfilter": 'zwo:r-1',    "sensor": "imx455" , "focus_level": "1wave", "filter_label": "r -1w"},
    }

_SENSORFILTER_FOCUS: dict = {}
for _sf_entry in sensor_info.values():
    _sf_key = _sf_entry["sensorfilter"]
    if _sf_key not in _SENSORFILTER_FOCUS:
        _SENSORFILTER_FOCUS[_sf_key] = _sf_entry["focus_level"]

# shortcut to simplify usage.
_KIND_NAMES = {shortcut:"zwo" for shortcut in ["sony", "imx", "imx455"]}

def get_any_astro_name(name, retry=True):
    """
    Search for an astronomical object spectrum file by name or spectral type.

    Parameters
    ----------
    name : str
        The name of the astronomical object or its spectral type.
    retry : bool, optional
        Whether to attempt a more precise search if the initial search fails. Default is True.

    Returns
    -------
    str or None
        The full path to the matching spectrum file, or None if no match is found.
    """
    # check if spectral type given.
    pickle_entry = PICKLES_MAPPING[PICKLES_MAPPING["spt"] == name]
    # if so, then get its true name.
    if len(pickle_entry)==1:
        name = pickle_entry.iloc[0]["filename"] + ".fits"
    elif len(pickle_entry)>1:
        raise ValueError(f"multiple entry for {name=}")
    
    astrofile = ASTROFILE_DF[ASTROFILE_DF["basename"].str.contains(name)]
    # nothing matches...
    if len(astrofile) == 0:
        return None
    if len(astrofile) == 1:
        return astrofile["fullpath"].iloc[0]
        
    # there are several entries matching, let's clean name
    new_name = name+"."
    astrofile = ASTROFILE_DF[ASTROFILE_DF["basename"].str.contains(new_name)]
    if len(astrofile) == 1:
        return astrofile["fullpath"].iloc[0]

    warnings.warn(f"cannot parse {name=}")
    return None

def get_pickles_spectrum_filename(spectral_type, fullpath=True):
    """
    Get the Pickles spectrum filename for a given spectral type.

    Parameters
    ----------
    spectral_type : str
        The spectral type of the star. Available types include:
        'O5V', 'O9V', 'B0V', 'B1V', 'B3V', 'B5-7V', 'B8V', 'A0V', 'A2V',
        'A3V', 'A5V', 'F0V', 'F2V', 'F5V', 'F8V', 'G0V', 'G2V', 'G5V',
        'G8V', 'K0V', 'K2V', 'K5V', 'K7V', 'M0V', 'M2V', 'M4V', 'M5V',
        'B2IV', 'B6IV', 'A0IV', 'A4-7IV', 'F0-2IV', 'F5IV', 'F8IV', 'G0IV',
        'G2IV', 'G5IV', 'G8IV', 'K0IV', 'K1IV', 'K3IV', 'O8III', 'B1-2III',
        'B5III', 'B9III', 'A0III', 'A5III', 'F0III', 'F5III', 'G0III',
        'G5III', 'G8III', 'K0III', 'K3III', 'K5III', 'M0III', 'M5III',
        'M10III', 'B2II', 'B5II', 'F0II', 'F2II', 'G5II', 'K0-1II',
        'K3-4II', 'M3II', 'B0I', 'B5I', 'B8I', 'A0I', 'F0I', 'F5I', 'F8I',
        'G0I', 'G5I', 'G8I', 'K2I', 'K4I', 'M2I'.
    fullpath : bool, optional
        Whether to return the full path (True) or just the basename (False). Default is True.

    Returns
    -------
    str
        Path to the spectrum file.

    Raises
    ------
    ValueError
        If no spectrum is found for the given spectral type.
    """
    filename = PICKLES_MAPPING[PICKLES_MAPPING['spt'].values == spectral_type]['filename'].values[0] + '.fits'
    
    if not filename:
        raise ValueError(f"No spectrum found for {spectral_type=}. Available SPT are {PICKLES_MAPPING['spt'].values}.")
    
    if fullpath:
        filename = os.path.join(_PICKLES_DIR, 'dat_uvk', filename)
        
    return filename

def read_config(filename, source="config"):
    """
    Read a single configuration file.

    - If the input filename does not specifically include a path, it will be
      looked for in the default `PACKAGE_PATH` directory.
    - Currently, only `.toml` configuration files are supported.

    Parameters
    ----------
    filename : str or dict
        Filename of the configuration file or a dictionary (returned as is).
        If no extension is provided, `.toml` is assumed.
    source : str, optional
        The directory where the file is supposed to be stored (e.g., "config").
        Used if the filename is not a full path. Default is "config".

    Returns
    -------
    dict
        Configuration as a nested dictionary.

    Raises
    ------
    ValueError
        If no extension is associated with the given filename.
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
        raise ValueError(f"no extension associated to given filename {filename=}. It cannot be loaded")
        
    if extension.lower() == ".toml":
        config = tomllib.load( open(filename, "rb") )
    # other supported extensions here: e.g. parquet, csv etc.
    else:
        raise NotImplementedError(f"Unknown configuration extension {extension=}.")
    
    return config

def get_sensor_config(kind, band, **kwargs):
    """
    Get sensor configuration for a specific detector kind and band.

    Parameters
    ----------
    kind : str
        The kind of sensor (e.g., 'zwo', 'qcmos', 'sony', 'imx').
    band : str
        The observation band (e.g., 'bb', 'u', 'g', 'r', 'i', 'z').
    **kwargs
        Additional keyword arguments to override or add to the configuration.

    Returns
    -------
    dict
        The combined configuration dictionary.

    Raises
    ------
    ValueError
        If the specified band is not available for the sensor kind.
    NotImplementedError
        If the throughput curve for the specified band is not yet implemented.
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
    """
    Get the full file path, including the package path if necessary.

    If the input filename does not specifically include a path, it will be
    looked for in the default `PACKAGE_PATH` directory.

    Parameters
    ----------
    filename : str
        The file name or path.
    source : str, optional
        The subdirectory inside `PACKAGE_PATH` to look into (e.g., "config").
        If None, it will be looked for directly in `PACKAGE_PATH`.
    test_extension : bool, optional
        Whether to validate if the extension is supported. Default is False.

    Returns
    -------
    str
        Filename including the default path if needed.

    Raises
    ------
    NotImplementedError
        If `test_extension` is True and the extension is not supported.
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

