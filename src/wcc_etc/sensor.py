
from astropy import units as u
from copy import deepcopy

from .utils import parse_element, parse_and_interpolate

from .meta import _MetaHolder_
class Sensor(_MetaHolder_):
    """
    A class representing the sensor (detector) properties.

    Attributes
    ----------
    bandpass : SpectralElement
        The total throughput (filter + sensor QE + telescope optics).
    wavelength : Quantity
        The peak wavelength of the bandpass.
    area : Quantity
        The total area of the sensor.
    gain : Quantity
        The sensor gain (e-/ADU).
    dark_current : Quantity
        The dark current (e-/s/pix).
    read_noise : Quantity
        The read noise (e-/pix).
    pixel_size : Quantity
        The pixel size (microns/pix).
    """

    _mutable_parameters = ["bandpass", "bandpass_name",
                            "pixel_size", "read_noise", "dark_current",
                            "gain", "area", "temperature"]
    
    def __init__(self, bandpass, 
                 pixel_size,                  
                 read_noise,
                 dark_current, 
                 gain,
                 area,
                 temperature=None,
                 qe= 1, # part of the total throughput for now.
                 well_depth=None,
                meta={}):
        """
        Initialize the sensor.

        Parameters
        ----------
        bandpass : SpectralElement or str
            The bandpass element or its specification.
        pixel_size : float or Quantity
            The pixel size (microns if float).
        read_noise : float or Quantity
            The read noise (e- if float).
        dark_current : float or Quantity
            The dark current (e-/s if float).
        gain : float or Quantity
            The gain (e-/adu if float).
        area : float or Quantity
            The sensor area (mm^2 if float).
        temperature : float or Quantity, optional
            The sensor temperature. Default is None.
        qe : float, optional
            Quantum Efficiency (usually included in bandpass). Default is 1.
        well_depth : float or Quantity, optional
            The full well depth. Default is None.
        meta : dict, optional
            Additional metadata. Default is {}.
        """

        init_parameters = {key: value for key, value in locals().items()
                            if key not in ["self", "bandpass", "meta"] and value is not None
                            and not key.startswith("__")}
        
        # overwrite meta with manually given ones.
        self.set_bandpass(bandpass)
        super().__init__(meta | init_parameters)
        
    @classmethod
    def from_name(cls, name):
        """
        Create a Sensor instance from a name string 'kind:band'.

        Parameters
        ----------
        name : str
            Sensor specification, e.g., 'zwo:r'.

        Returns
        -------
        Sensor
        """
        sensor, band = name.split(":")
        return cls.from_kind_and_band(sensor, band)

    @classmethod
    def from_config(cls, config_or_name):
        """
        Create a Sensor instance from a configuration or name.

        Parameters
        ----------
        config_or_name : dict or str
            Configuration dictionary or name string.

        Returns
        -------
        Sensor
        """
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
            read_noise = parse_and_interpolate(config.get("path_read_noise"), gain_setting) * 2 # multiply by 2 to allow for unmodelled noise sources
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
                     well_depth=well_depth,
                     qe=1, # forced qe=1 as included in total throughput
                    meta=config)
    
    @classmethod
    def from_kind_and_band(cls, kind, band):
        """
        Load the instance given the detector kind and filter.

        Parameters
        ----------
        kind : str
            Kind of sensor of the WCC: 
            - 'zwo' ('sony', 'imx', 'imx455' accepted)
            - 'qcmos'
        band : str
            Name of the band associated to the sensor (e.g., 'bb', 'u', 'r', 'z').

        Returns
        -------
        Sensor
        """
        from .io import get_sensor_config
        # grabs the configuration associated to this sensor
        config = get_sensor_config(kind, band)
        return cls.from_config(config["sensor"])

    # ================ #
    #  Methods         #
    # ================ #
    def set_bandpass(self, bandpass):
        """
        Set the sensor bandpass.

        Parameters
        ----------
        bandpass : str or SpectralElement
        """
        self._bandpass = parse_element(bandpass)
    
    def get_plate_scale(self, telescope):
        """
        Calculate the plate scale in arcsec/pix.

        Parameters
        ----------
        telescope : Telescope
            The telescope object.

        Returns
        -------
        Quantity
            The plate scale in arcsec/pix.
        """
        # why 206265
        return (self.pixel_size.to("m/pix") / telescope.diameter_primary.to("m") / telescope.f_num * 206265*u.arcsec) # arcsec/pix
        
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def bandpass(self):
        """
        The sensor bandpass (SpectralElement).
        """
        return self._bandpass
        
    @property
    def wavelength(self):
        """
        The peak wavelength of the bandpass.
        """
        # store in memory as a bit slow
        if not hasattr(self, "_wavelength") or self._wavelength is None:
            self._wavelength = self.bandpass.wpeak().to(u.nm)
            
        return self._wavelength 

    @property
    def area(self):
        """
        The sensor area.
        """
        return self.meta["area"]
        
    @property
    def gain(self):
        """
        The sensor gain (e-/ct).
        """
        return self.meta["gain"] * (u.electron / u.ct)
        
    @property
    def dark_current(self):
        """
        The dark current (e-/s/pix).
        """
        return self.meta["dark_current"] * (u.electron / (u.s * u.pix))

    @property
    def read_noise(self):
        """
        The read noise (e-/pix).
        """
        return self.meta["read_noise"] * u.electron / u.pix
        
    @property
    def pixel_size(self):
        """
        The pixel size (um/pix).
        """
        return self.meta["pixel_size"] * u.um/u.pix

