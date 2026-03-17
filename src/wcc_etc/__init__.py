import importlib.metadata
from pathlib import Path
__version__ = importlib.metadata.version(__package__ or "wcc_etc")

__all__ = [ "__version__"]

from .wcc_etc import WCCETC, get_blackbody_flux, get_wcc_snr_and_simulation

from .io import read_config, get_sensor_config, get_pickles_spectrum_filename
from .simulation import Simulation
from .sensor import Sensor
from .scene import *
#from .airy import *
#from .psfsim import *
#from .radial_data import *
