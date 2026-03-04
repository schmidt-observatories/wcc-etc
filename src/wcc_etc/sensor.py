
import numpy as np
from astropy import units as u
from copy import deepcopy

from .utils import parse_element, parse_and_interpolate

class Sensor():
    """ """
    def __init__(self, bandpass, 
                 pixel_size,                  
                 read_noise,
                 dark_current, 
                 gain,
                 area,
                 temperature=None,
                 qe= 1, # part of the total throughput for now.
                meta={}):
        """ """

        init_parameters = {key: value for key, value in locals().items()
                            if key not in ["self", "bandpass", "meta"] and value is not None}
        
        # overwrite meta with manually given ones.
        self.set_bandpass(bandpass)
        self._meta = deepcopy(meta) | init_parameters
        self._meta_in = deepcopy(self._meta)
        
        

    @classmethod
    def from_name(cls, name):
        """ """
        sensor, band = name.split(":")
        return cls.from_kind_and_band(sensor, band)

    @classmethod
    def from_config(cls, config_or_name):
        """ """
        if isinstance(config_or_name, str):
            return cls.from_name(config_or_name)

        config = deepcopy(config_or_name)

        # read the total throughput allowing 2 formating
        throughput = config.get("throughput", config.get("path_total_throughput"))
        bandpass = parse_element(throughput)
        
        # basic sensor information:
        pixel_size = config.get("pixel_size")
        sensor_area = config.get("sensor_area") * u.mm**2
        
        # optional
        sensor_temp = config.get("sensor_temp", None)
        if sensor_temp is not None:
            sensor_temp *= u.Celsius
        
        
        # gain
        gain_setting = config.get('gain_setting', None)        
        if gain_setting is not None:
            gain = parse_and_interpolate(config.get("path_gain_curve"), gain_setting)
            read_noise = parse_and_interpolate(config.get("path_read_noise"), gain_setting)
            dark_current = parse_and_interpolate(config.get("path_dark_current"), sensor_temp.to("Celsius").value) # careful temperature here.
            well_depth = parse_and_interpolate(config.get("path_well_depth"), gain_setting)
            
        else: # no default allowed here, must be provided.
            gain = config.get("gain")
            read_noise = config.get("read_noise")
            dark_current = config.get("dark_current")
            well_depth = config.get("well_depth")        

        
        return cls(bandpass=bandpass, 
                     pixel_size=pixel_size,                  
                     read_noise=read_noise,
                     dark_current=dark_current, 
                     gain=gain,
                     area=sensor_area,
                     temperature=sensor_temp,
                     qe=1, # forced qe=1 as included in total throughput
                    meta=config)
    
    @classmethod
    def from_kind_and_band(cls, kind, band):
        """ loads the instance given the detector kind and filter

        Parameters
        ----------
        kind: str
            kind of sensor of the WCC: 
            - 'zwo' ('sony', 'imx', 'imx455' accepted)
            - 'qcmos'
        band: str
            name of the band associated to the sensor.
            e.g. bb, u, r, z etc.
        Returns
        -------
        instance
        """
        # grabs the configuration associated to this sensor
        config = get_sensor_config(kind, band)
        return cls.from_config(config)

    # ================ #
    #  Methods         #
    # ================ #
    def set_bandpass(self, bandpass):
        """ """
        self._bandpass = parse_element(bandpass)
    
    def get_plate_scale(self, telescope):
        """ """
        # why 206265
        return (self.pixel_size.to("m/pix") / telescope.diameter_primary.to("m") / telescope.f_num * 206265) # arcsec/pix
        
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def bandpass(self):
        """ """
        return self._bandpass
        
    @property
    def wavelength(self):
        """ """
        # store in memory as a bit slow
        if not hasattr(self, "_wavelength") or self._wavelength is None:
            self._wavelength = self.bandpass.wpeak().to(u.nm)
            
        return self._wavelength 

    @property
    def area(self):
        """ """
        return self.meta["area"]
        
    @property
    def gain(self):
        """ """
        return self.meta["gain"] * (u.electron / u.ct)
        
    @property
    def dark_current(self):
        """ """
        return self.meta["dark_current"] * (u.electron / (u.s * u.pix))

    @property
    def read_noise(self):
        """ """
        return self.meta["read_noise"] * np.sqrt(1.0 * u.electron / u.pix)
        
    @property
    def pixel_size(self):
        """ """
        return self.meta["pixel_size"] * u.um/u.pix

    @property
    def meta(self):
        """ """
        return self._meta
