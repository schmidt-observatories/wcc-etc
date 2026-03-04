import numpy as np
from astropy import units as u
from synphot import SpectralElement, Observation

from copy import deepcopy

from .telescope import Telescope
from .sensor import Sensor
from .source import Source


class Simulation():
    """ """
    def __init__(self, telescope, sensor, 
                 source=None, 
                 #
                  mag=20, bandpass="sensor",
                  skymag=24, skybandpass="johnson_v",
                  time=90, 
                  r_aper_mas=40, 
                  meta={}
                  ):
        """ """
        self._telescope = telescope
        self._sensor = sensor
        self.set_source(source)

        input_parameters = {key:value for key,value in locals().items()
                             if key not in ["self", "telescope", "sensor", "source", "meta"] and value is not None}
        
        self._meta = deepcopy(meta) | input_parameters
        self._meta_in = deepcopy(self._meta)
        
    # ============= #
    #  properties   #
    # ============= #
    @classmethod
    def from_config(cls, config):
        """ """
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

        # Source
        config_source = config.get("source", None)
        if config_source is not None:
            source = Source.from_config(config_source)
        else:
            source = None

        return cls(telescope=telescope, sensor=sensor, source=source)


    @classmethod
    def from_sensorname_and_source(cls, name, source):
        """ """
        from .io import get_sensor_config
        config = get_sensor_config("sony", "bb")

        this = cls.from_config(config) # this has no source
        this.set_source(source)
        return this
    
    # ================ #
    #   methods        #
    # ================ #
    def set_source(self, source_or_config):
        """ """
        if isinstance(source_or_config, dict):
            source = Source.from_config(source_or_config)
        else:
            source = source_or_config
        
        self._source = source
        
        # this should move inside source eventually
        self._h_spec_observation = None
        self._h_sky_observation = None

    def set_sensor(self, sensor_or_config):
        """ """
        if isinstance(sensor_or_config, dict):
            sensor = Sensor.from_config(sensor_or_config)
        else:
            sensor = sensor_or_config
        
        self._sensor = source
        self._psf_profile = {} # reset the psf profile
        # these following entry 'might' depend on sensor for the bandpass
        self._h_spec_observation = None
        self._h_sky_observation = None 

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

    def get_countrates(self, units="adu/s"):
        """ get the countrates in """
        if units not in ["adu/s", "e/s", "e-/s"]:
            raise ValueError(f"unknown countrate units. Should be 'adu/s' or 'e/s'. {units=} given")

        
        # INFO: self.psf_profile is computed automatically if needed.
        
        # these are the countrate in e/s
        # - source
        count_rate_total = self._spec_observation.countrate(area=self.telescope.surface) * u.electron/u.ct 
        count_rate = count_rate_total * self.psf_profile["ee_at_aper"]

        # - background
        if self.source.has_background():
            sky_counts_total = self._background_observation.countrate(area=self.telescope.surface)* u.electron/u.ct # e/s
            sky_count_rate = sky_counts_total * self.psf_profile["ee_at_aper"]
        else:
            sky_count_rate = 0
            
        if units in ["adu/s"]: # in [] enables short cut.
            count_rate /= self.sensor.gain  #   ADU/s
            sky_count_rate /= self.sensor.gain  #   ADU/s
            
        return count_rate, sky_count_rate


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

        # poisson noise is at the electron level, not adu.
        source_signal = count_rate * time # e-
        sky_signal = sky_count_rate * time # e-
        dark_signal = self.sensor.dark_current * time # e-
        detector_variance = (dark_signal + self.sensor.read_noise**2) * self.psf_profile["num_psf_pixels"]
        
        # total noise
        total_variance = (source_signal + sky_signal + detector_variance) * u.electron # variance is in e-**2
        
        # units
        if units.lower() == "adu":
            source_signal /= self.sensor.gain
            total_variance /= self.sensor.gain**2
        elif units not in ["e", "e-", "electron"]:
            raise ValueError(f"unknown units {units=}. adu or electron/e- expected.")
            
        return source_signal, total_variance
    
    def get_snr(self, time=None):
        """ """
        # sources of noise
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
        num_pixels_at_r = self.meta["r_aper_mas"]/(plate_scale * 1000) # pix
        num_psf_pixels = (np.pi * num_pixels_at_r**2).value * u.pix # in pixels

        # area of the psf in angular units
        psf_area = num_psf_pixels * plate_scale **2
        
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
    def source(self):
        """ """
        return self._source

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
                for element in ["telescope", "sensor", "source"]
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

    @property
    def _spec_observation(self):
        """ """
        if not hasattr(self, "_h_spec_observation") or self._h_spec_observation is None:
            
                
            self._h_spec_observation = self._get_spectrum_observation( spectrum = self.source.spectrum, 
                                                                       abmag = self.meta["mag"],
                                                                       bandpass = self._parse_bandpass(self.meta["bandpass"])
                                                                     )
        return self._h_spec_observation

    @property
    def _background_observation(self):
        """ """
        if not hasattr(self, "_h_bkgd_observation") or self._h_bkgd_observation is None:
            self._h_bkgd_observation = self._get_spectrum_observation( spectrum = self.source.background, 
                                                                       abmag = self.meta["skymag"],
                                                                       bandpass = self._parse_bandpass(self.meta["skybandpass"])
                                                                     )
        return self._h_bkgd_observation       
