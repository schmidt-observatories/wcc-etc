import numpy as np
from astropy import units as u
from synphot import SpectralElement, Observation
from copy import deepcopy
import logging

from .telescope import Telescope
from .sensor import Sensor
from .scene import Scene
from .meta import _MetaHolder_




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_bg_normalization_magnitude(bg_surface_brightness, psf_area):
    """
    Convert the Background Surface Brightness into the total magnitude given the PSF area (in arcseconds squared)
    The area needs to be in square arcseconds since this the typical definition of Surface Brightness is in units
    of magnitudes per arcseconds^2
    :return: None
    """
    bg_magnitude = bg_surface_brightness - 2.5 * np.log10(psf_area)
    return bg_magnitude

class Simulation(_MetaHolder_):
    """ """

    _mutable_parameters = ["mag", "bandpass", "skymag", "skybandpass",
                          "time", "r_aper_mas"]
    def __init__(self, 
                 telescope,
                 sensor, 
                 scene=None, 
                 mag=20, 
                 bandpass="sensor",
                 skymag=24, 
                 skybandpass="johnson_v",
                 time=90, 
                 r_aper_mas=70, 
                 meta={}
                 ):
        """ 
        Simulation object

        Parameters:
        -----------
        telescope: Telescope
            telescope to be used for this simulation
        sensor: Sensor
            sensor to be used for this simulation
        scene: Scene
            scene to be used for this simulation. This could be set later.
        mag: float, 
            magnitude of the scene
        bandpass: str, 
            bandpass filter to use
        skymag: float
            sky background magnitude
        skybandpass: str, 
            sky background bandpass filter
        time: float, array
            exposure time(s) in seconds
        r_aper_mas: float
            radius of the aperture in milliarcseconds
        meta: dict
            additional parameters to store in the meta dictionary (optional)

        Returns
        -------
        """
        self._telescope = telescope
        self._sensor = sensor
        self.set_scene(scene)

        input_parameters = {key:value for key,value in locals().items()
                             if key not in ["self", "telescope", "sensor", "scene", "meta"]
                                and value is not None}

        super().__init__(meta | input_parameters)
        
    # ============= #
    #  properties   #
    # ============= #
    @classmethod
    def from_config(cls, config):
        """
        Initialize from a configuration directory.
        """
        # Telescope
        config_telescope = config.get("telescope", None)
        if config_telescope is not None:
            telescope = Telescope.from_config(config_telescope)
        else:
            telescope = None
            
        # Sensor
        config_sensor = config.get("sensor", None)
        if config_sensor is not None:
            sensor = Sensor.from_config(config_sensor)
        else:
            sensor = None

        # Scene
        config_scene = config.get("scene", None)
        if config_scene is not None:
            scene = Scene.from_config(config_scene)
        else:
            scene = None

        return cls(telescope=telescope, sensor=sensor, scene=scene)

    @classmethod
    def from_sensorname_and_scene(cls, name, scene):
        """
        Initialize using a sensor name and a scene
        """
        from .io import get_sensor_config

        # accept strings like 'sony:bb' or just 'r' (default kind -> 'sony')
        if isinstance(name, str) and ":" in name:
            kind, band = name.split(":", 1)
        else:
            # treat the whole name as band and default to 'sony' kind
            kind = "sony"
            band = name

        config = get_sensor_config(kind, band)

        this = cls.from_config(config) # this has no scene
        this.set_scene(scene)
        return this
    
    # ================ #
    #   methods        #
    # ================ #
    def set_scene(self, scene_or_config):
        """ """
        if isinstance(scene_or_config, dict):
            scene = Scene.from_config(scene_or_config)
        else:
            scene = scene_or_config

        self._scene = scene

        # this should move inside scene eventually
        self._h_spec_observation = None
        self._h_bkgd_observation = None

    def set_sensor(self, sensor_or_config):
        """ """
        if isinstance(sensor_or_config, dict):
            sensor = Sensor.from_config(sensor_or_config)
        else:
            sensor = sensor_or_config

        self._sensor = sensor
        self._psf_profile = {}  # reset the psf profile
        # these following entries might depend on sensor for the bandpass
        self._h_spec_observation = None
        self._h_bkgd_observation = None

    def set_telescope(self, telescope_or_config):
        """ """
        if isinstance(telescope_or_config, dict):
            telescope = Telescope.from_config(telescope_or_config)
        else:
            telescope = telescope_or_config
        
        self._telescope = telescope
        self._psf_profile = {} # reset the psf profile


    # ------- #
    #  GETTER #
    # ------- #
    @staticmethod
    def _get_spectrum_observation(spectrum, abmag, bandpass):
        """ """
        spec_at_mag = spectrum.normalize(abmag * u.ABmag, bandpass, force='extrap')
        return Observation(spec_at_mag, bandpass, force='extrap')

    def get_countrates(self, scene=None, band=None, units="adu/s", as_dict=True):
        """ get the countrates in """
        if units not in ["adu/s", "e/s", "e-/s"]:
            raise ValueError(f"unknown countrate units. Should be 'adu/s' or 'e/s'. {units=} given")

        
        # INFO: self.psf_profile is computed automatically if needed.
        if band is None:
            band = self.sensor.bandpass

        if scene is None:
            scene = self.scene
            
        
        # these are the countrate in e/s
        if np.any( scene.call_down("mag_is_surface_brightness")):
            area = self.psf_profile['psf_area'].value # area in arcsec**2
        else:
            area = None
            
        scene_observations = scene.get_observation(band=band, area=area, as_dict=True)

        # loop over the scene elements and get the countrate for each.
        countrates = {}
        for element, observation in scene_observations.items():
            # - scene
            count_rate_total = observation.countrate(area=self.telescope.surface) * u.electron/u.ct 
            count_rate = count_rate_total * self.psf_profile["ee_at_aper"]
            
            if units in ["adu/s"]: # in [] enables short cut.
                count_rate /= self.sensor.gain  #   ADU/s

            countrates[element] = count_rate

        if as_dict:
            return countrates
        
        return list(countrates.values())


    def get_signal_and_variance(self, time=None, units="e-"):
        """ """
        if time is None:
            time = self.meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
            
        # make sure the time is in the current units.
        elif not isinstance(time, u.Quantity):
           time = time*u.second

        # get the count rates in {adu,e-}/s
        count_rate, sky_count_rate = self.get_countrates(units="e/s")
        logging.info(f"Scene count rate: {count_rate:.2f} e/s")
        logging.info(f"Background count rate: {sky_count_rate:.2f} e/s")

        # poisson noise is at the electron level, not adu.
        scene_signal = count_rate * time # e-
        sky_signal = sky_count_rate * time # e-
        dark_signal = self.sensor.dark_current * time # e-
        detector_variance = (dark_signal + self.sensor.read_noise**2) * self.psf_profile["num_psf_pixels"]
        logging.info(f"signal: {scene_signal:.2f} e-")
        logging.info(f"sky signal: {sky_signal:.2f} e-")
        logging.info(f"dark signal: {dark_signal:.2f} e-")
        logging.info(f"detector variance: {detector_variance:.2f} e-^2.")

        # total noise
        total_variance = (scene_signal + sky_signal + detector_variance) * u.electron # variance is in e-**2
        
        # units
        if units.lower() == "adu":
            scene_signal /= self.sensor.gain
            total_variance /= self.sensor.gain**2
        elif units not in ["e", "e-", "electron"]:
            raise ValueError(f"unknown units {units=}. adu or electron/e- expected.")
            
        return scene_signal, total_variance
    
    def get_snr(self, time=None):
        """ Get the signal to noise ration for a given exposure """
        # scenes of noise
        signal, variance = self.get_signal_and_variance(time) # units doesn't matter
        return signal / np.sqrt(variance)
    
    # -------------- #
    #  Internal      #
    # -------------- #
    def _parse_bandpass(self, bandpass):
        """ """
        if bandpass == "sensor":
            return self.sensor.bandpass
        
        return SpectralElement.from_filter(bandpass)
        
    def compute_psf_profile(self):
        """ """
        from .airy import get_airy_and_ee_curve
        
        wavelength = self.sensor.wavelength.to("m")
        r_psf_mas, psf1d, ee, ee_at_aper = get_airy_and_ee_curve(wavelength, 
                                                                 r_aper_mas = self.meta["r_aper_mas"], # no default allower 
                                                                 jitter_sigma_mas = self.telescope.jitter_sigma.to("mas"), 
                                                                 fnum=self.telescope.f_num, 
                                                                 D=self.telescope.diameter_primary.value,
                                                                 pixel_size=self.sensor.pixel_size.value,
                                                                 verbose=False)

        # compute the number of pixels associated to the PSF
        plate_scale = self.sensor.get_plate_scale(self.telescope) # in arcsec/pix
        num_pixels_at_r = self.meta["r_aper_mas"]*u.arcsec/(plate_scale * 1000) # pix
        num_psf_pixels = (np.pi * num_pixels_at_r**2) # in pixels**2

        # area of the psf in angular units
        psf_area = num_psf_pixels * plate_scale **2 # in arcsec**2

        logging.info(f"PSF profile computed:")
        logging.info(f"EE={ee_at_aper:.2f} at {self.meta['r_aper_mas']} mas aperture")
        logging.info(f"num_psf_pixels={num_psf_pixels:.1f} pixels")
        logging.info(f"PSF area={psf_area:.2f} arcsec^2")

        return {"wavelength": wavelength,
                "r_psf_mas": r_psf_mas,
                "psf1d": psf1d,
                "ee": ee,
                "ee_at_aper": ee_at_aper,
                "num_psf_pixels": num_psf_pixels,
                "psf_area": psf_area
                }

    def has_element(self, which):
        """ """
        return getattr(self, which) is not None
        
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def scene(self):
        """ """
        return self._scene

    @property
    def telescope(self):
        """ """
        return self._telescope
    
    @property
    def sensor(self):
        """ """
        return self._sensor

    @property
    def meta(self):
        """ generic parameters """
        return self._meta | {element: getattr(self, element).meta
                for element in ["telescope", "sensor", "scene"]
                if self.has_element(element)}

    # ---------- #
    # cashed     #
    # ---------- #
    @property
    def psf_profile(self):
        """ """
        if not hasattr(self,"_psf_profile") or self._psf_profile is None or len(self._psf_profile) == 0 : # like {}
            self._psf_profile = self.compute_psf_profile()
            
        return self._psf_profile
