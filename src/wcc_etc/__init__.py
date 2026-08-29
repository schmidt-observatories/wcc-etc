import importlib.metadata
from pathlib import Path

__version__ = importlib.metadata.version(__package__ or "wcc_etc")

__all__ = [
    "__version__",
    "plot_image_mpl",
    "plot_image_bokeh",
    "plot_image_row_mpl",
    "plot_image_row_bokeh",
    "plot_radial_mpl",
    "plot_radial_bokeh",
    "plot_encircled_energy_mpl",
    "plot_encircled_energy_bokeh",
    "set_wcc_style",
    "WCC_STYLE",
    "FluxModel",
    "TransitModel",
    "LightCurveSimulator",
    "LightCurve",
    "plot_lightcurve_mpl",
    "plot_lightcurve_bokeh",
    "download_exoplanet_archive",
    "load_exoplanet_archive",
]

from lazuli_transit import download_exoplanet_archive, load_exoplanet_archive

# Bind the `airy` and `psfsim` submodules as package attributes, so
# `wcc_etc.airy` / `wcc_etc.psfsim` work after a bare `import wcc_etc`.
from . import airy, psfsim  # noqa: F401  (re-export as package attributes)
from .astro import get_moon_magnitude
from .io import get_pickles_spectrum_filename, get_sensor_config, read_config
from .lightcurve import FluxModel, LightCurve, LightCurveSimulator, TransitModel
from .plotting import (
    WCC_STYLE,
    plot_encircled_energy_bokeh,
    plot_encircled_energy_mpl,
    plot_image_bokeh,
    plot_image_mpl,
    plot_image_row_bokeh,
    plot_image_row_mpl,
    plot_lightcurve_bokeh,
    plot_lightcurve_mpl,
    plot_radial_bokeh,
    plot_radial_mpl,
    set_wcc_style,
)
from .psfsim import (
    DEFOCUS_1WAVE_PATH,
    DEFOCUS_2WAVE_PATH,
    AiryPSF,
    CustomPSF,
    DefocusPSF,
    ImageSimulator,
    SimulatedImage,
)
from .scene import Scene, get_scene, get_scene_element, get_scene_from_file
from .sensor import Sensor
from .simulation import Simulation
