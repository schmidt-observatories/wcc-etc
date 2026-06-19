import importlib.metadata
from pathlib import Path
__version__ = importlib.metadata.version(__package__ or "wcc_etc")

__all__ = [
    "__version__",
    "plot_image_mpl", "plot_image_bokeh",
    "plot_image_row_mpl", "plot_image_row_bokeh",
    "plot_radial_mpl", "plot_radial_bokeh",
    "plot_encircled_energy_mpl", "plot_encircled_energy_bokeh",
    "set_wcc_style", "WCC_STYLE",
    "FluxModel", "TransitModel", "LightCurveSimulator", "LightCurve",
    "plot_lightcurve_mpl", "plot_lightcurve_bokeh",
    "download_exoplanet_archive", "load_exoplanet_archive",
]

from .wcc_etc import *
from .io import read_config, get_sensor_config, get_pickles_spectrum_filename
from .simulation import Simulation
from .psfsim import (
    ImageSimulator, SimulatedImage,
    AiryPSF, DefocusPSF, CustomPSF,
    DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH,
)
from .sensor import Sensor
from .scene import *
from .astro import *
from .plotting import (
    plot_image_mpl, plot_image_bokeh,
    plot_image_row_mpl, plot_image_row_bokeh,
    plot_radial_mpl, plot_radial_bokeh,
    plot_encircled_energy_mpl, plot_encircled_energy_bokeh,
    set_wcc_style, WCC_STYLE,
    plot_lightcurve_mpl, plot_lightcurve_bokeh,
)
from .lightcurve import FluxModel, TransitModel, LightCurveSimulator, LightCurve
from lazuli_transit import download_exoplanet_archive, load_exoplanet_archive
#from .airy import *
#from .psfsim import *
#from .radial_data import *
