import os
import tomllib
import numpy as np
from importlib.resources import files

PACKAGE_PATH = files("wcc_etc.data")     #: Path to data & config files.

__all__ = ["read_config"]

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
    filename = expand_path(filename, source=source)

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

def expand_path(filename, source=None):
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

    if os.path.dirname(filename):  # filename includes a path
        fname = filename
    else:                          # use PACKAGE_PATH as default
        if source is not None:
            fname = PACKAGE_PATH.joinpath(source).joinpath(filename)
        else:
            fname = PACKAGE_PATH.joinpath(filename)

        _, extension = os.path.splitext(fname)
        if extension is None or len(extension) == 0:
            fname = f"{fname}.toml"
        elif extension not in [".toml"]: # specify here list of accepted extensions.
            raise NotImplementedError(f"Unknown configuration extension {extension=}.")

    return fname
