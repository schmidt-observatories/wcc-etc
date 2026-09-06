import importlib.metadata

__version__ = importlib.metadata.version(__package__ or "wcc_etc")

__all__ = [
    "__version__",
    # scene / instrument / simulation — the Quick Start surface
    "Scene",
    "Sensor",
    "Simulation",
    "Telescope",
    "get_scene",
    "get_scene_element",
    "get_scene_from_file",
    # PSF
    "AiryPSF",
    "CustomPSF",
    "DefocusPSF",
    "PolychromaticPSF",
    "ImageSimulator",
    "SimulatedImage",
    "DEFOCUS_1WAVE_PATH",
    "DEFOCUS_2WAVE_PATH",
    "effective_wavelength",
    "photon_weighted_subbands",
    # config / astro helpers
    "get_moon_magnitude",
    "get_pickles_spectrum_filename",
    "get_sensor_config",
    "read_config",
    # lightcurve
    "FluxModel",
    "TransitModel",
    "LightCurveSimulator",
    "LightCurve",
    "download_exoplanet_archive",
    "load_exoplanet_archive",
    # plotting
    "plot_image_mpl",
    "plot_image_bokeh",
    "plot_image_row_mpl",
    "plot_image_row_bokeh",
    "plot_bandpass_mpl",
    "plot_radial_mpl",
    "plot_radial_bokeh",
    "plot_encircled_energy_mpl",
    "plot_encircled_energy_bokeh",
    "plot_lightcurve_mpl",
    "plot_lightcurve_bokeh",
    "set_wcc_style",
    "WCC_STYLE",
]

from lazuli_transit import download_exoplanet_archive, load_exoplanet_archive

# Bind the `airy`, `psfsim` and `spectral` submodules as package attributes,
# so `wcc_etc.airy` / `wcc_etc.psfsim` / `wcc_etc.spectral` work after a bare
# `import wcc_etc`.
from . import airy, extended, psfsim, spectral  # noqa: F401  (re-export as package attributes)
from .astro import get_moon_magnitude
from .io import get_pickles_spectrum_filename, get_sensor_config, read_config
from .lightcurve import FluxModel, LightCurve, LightCurveSimulator, TransitModel
from .plotting import (
    WCC_STYLE,
    plot_bandpass_mpl,
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
    PolychromaticPSF,
    SimulatedImage,
)
from .scene import Scene, get_scene, get_scene_element, get_scene_from_file
from .sensor import Sensor
from .telescope import Telescope
from .simulation import Simulation
from .spectral import effective_wavelength, photon_weighted_subbands
