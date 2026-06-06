import numpy as np
from astropy import units as u
from synphot import SpectralElement, Observation

import warnings

from .io import get_sensor_config, _SENSORFILTER_FOCUS, _SENSORFILTER_IMPLEMENTED
from .telescope import Telescope
from .sensor import Sensor
from .scene import Scene
from .meta import _MetaHolder_
from .utils import list_of_quantity_to_array


#import logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

def calculate_bg_normalization_magnitude(bg_surface_brightness, psf_area):
    """
    Convert the Background Surface Brightness into the total magnitude given the PSF area.

    The area needs to be in square arcseconds since the typical definition of
    Surface Brightness is in units of magnitudes per arcseconds^2.

    Parameters
    ----------
    bg_surface_brightness : float
        The background surface brightness in mag/arcsec^2.
    psf_area : float
        The area of the PSF in arcsec^2.

    Returns
    -------
    float
        The total background magnitude.
    """
    bg_magnitude = bg_surface_brightness - 2.5 * np.log10(psf_area)
    return bg_magnitude


def _psf_from_focus_level(focus_level):
    """
    Create a PSF object from a focus level string.

    Parameters
    ----------
    focus_level : str
        The focus level, one of "0wave", "1wave", or "2wave".

    Returns
    -------
    AiryPSF or DefocusPSF
        The PSF object corresponding to the focus level.

    Raises
    ------
    ValueError
        If focus_level is not one of the expected values.
    """
    from .psfsim import AiryPSF, DefocusPSF, DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH
    if focus_level == "0wave":
        return AiryPSF()
    if focus_level == "1wave":
        return DefocusPSF(DEFOCUS_1WAVE_PATH)
    if focus_level == "2wave":
        return DefocusPSF(DEFOCUS_2WAVE_PATH)
    raise ValueError(f"Unknown focus_level {focus_level!r}. Expected '0wave', '1wave', or '2wave'.")


class Simulation(_MetaHolder_):
    """
    A class to manage and run image exposure simulations.

    Attributes
    ----------
    telescope : Telescope
        The telescope used for the simulation.
    sensor : Sensor
        The sensor used for the simulation.
    scene : Scene
        The scene being observed.
    psf_profile : dict
        Calculated PSF profile parameters.
    """
    _mutable_parameters = ["time", "r_aper_mas", "n_reads"]

    def __init__(self,
                 telescope,
                 sensor,
                 scene=None,
                 time=90,
                 r_aper_mas=70,
                 n_reads=1,
                 meta={}
                 ):
        """ 
        Initialize a Simulation object.

        Parameters
        ----------
        telescope : Telescope
            Telescope to be used for this simulation.
        sensor : Sensor
            Sensor to be used for this simulation.
        scene : Scene, optional
            Scene to be used for this simulation. Default is None.
        time : float or array_like, optional
            Exposure time(s) in seconds. Default is 90.
        r_aper_mas : float, optional
            Radius of the aperture in milliarcseconds. Default is 70.
        n_reads : int, optional
            Number of coadded frames (read noise is incurred per frame; saturation
            is evaluated per-frame). Default is 1.
        meta : dict, optional
            Additional parameters to store in the meta dictionary. Default is {}.
        """
        self._telescope = telescope
        self._sensor = sensor
        self.set_scene(scene)
        self._default_psf = None  # set by from_sensorfilter only
        self._psf_profile = {}
        self._image_render_bundle_cache = {}

        input_parameters = {key:value for key,value in locals().items()
                             if key not in ["self", "telescope", "sensor", "scene", "meta"]
                                and value is not None and not key.startswith("__")}
        non_attr_meta  = {key: value for key, value in meta.items() if key not in ["telescope", "sensor", "scene"]}

        super().__init__(non_attr_meta | input_parameters)
        
    # ============= #
    #  properties   #
    # ============= #
    @classmethod
    def from_config(cls, config):
        """
        Initialize a Simulation from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing 'telescope', 'sensor', and 'scene' keys.

        Returns
        -------
        Simulation
            The initialized Simulation object.
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
    def from_sensor(cls, sensor, scene=None):
        """
        Naming simplification of from_sensor_and_scene().

        Parameters
        ----------
        sensor : str, list, tuple, or Sensor
            Sensor specification.
        scene : Scene, optional
            Scene object.

        Returns
        -------
        Simulation
        """
        return cls.from_sensor_and_scene(sensor, scene=scene)
    
    @classmethod
    def from_sensor_and_scene(cls, sensor, scene):
        """
        Create a Simulation from a sensor specification and a scene.

        Parameters
        ----------
        sensor : str, list, tuple, or Sensor
            Sensor specification. If str, format 'kind:band'.
        scene : Scene
            The scene to observe.

        Returns
        -------
        Simulation
        """
        if type(sensor) is str or type(sensor) in [list, tuple]:
            if type(sensor) is str:
                kind, band = sensor.split(":", 1)
            else:
                kind, band = sensor

            config = get_sensor_config(kind, band)
            this = cls.from_config(config) # this has no scene
            
        elif isinstance(sensor, Sensor):
            this = cls(sensor=sensor, telescope=sensor.telescope)

        else:
            raise ValueError(f"I cannot parse the input {sensor=}")

        this.set_scene(scene)
        return this

    @classmethod
    def from_sensorfilter(cls, sensorfilter, scene):
        """
        Create a Simulation from a sensorfilter label, auto-selecting the PSF.

        The label must be a key in sensor_info (e.g. 'zwo:r', 'zwo:r+1',
        'qcmos:bb'). The PSF matching the sensor's focus_level is stored as
        _default_psf; get_image_snr uses it when no psf= argument is given.

        Parameters
        ----------
        sensorfilter : str
            Label from sensor_info, format 'kind:band' (e.g. 'zwo:r',
            'zwo:r+1', 'qcmos:bb'). Only canonical labels are accepted;
            nicknames like 'sony:r' are not resolved here (use
            from_sensor_and_scene for those).
        scene : Scene

        Returns
        -------
        Simulation
        """
        if sensorfilter not in _SENSORFILTER_FOCUS:
            known = sorted(_SENSORFILTER_FOCUS)
            raise ValueError(
                f"Unknown sensorfilter {sensorfilter!r}. "
                f"Known labels: {known}"
            )
        if not _SENSORFILTER_IMPLEMENTED[sensorfilter]:
            raise NotImplementedError(
                f"Sensorfilter {sensorfilter!r} is not yet implemented "
                f"(no throughput curve available)."
            )
        focus_level = _SENSORFILTER_FOCUS[sensorfilter]
        kind, band = sensorfilter.split(":", 1)
        config = get_sensor_config(kind, band)
        this = cls.from_config(config)
        this.set_scene(scene)
        this._default_psf = _psf_from_focus_level(focus_level)
        return this

    # ================ #
    #   methods        #
    # ================ #
    def set_scene(self, scene_or_config):
        """
        Set the scene for the simulation.

        Parameters
        ----------
        scene_or_config : Scene or dict
            The Scene object or its configuration.
        """
        if isinstance(scene_or_config, dict):
            scene = Scene.from_config(scene_or_config)
        else:
            scene = scene_or_config

        self._scene = scene

        # this should move inside scene eventually
        self._h_spec_observation = None
        self._h_bkgd_observation = None
        # scene drives the count rates / PSF profile: drop derived caches
        self._psf_profile = {}
        self._image_render_bundle_cache = {}

    def set_sensor(self, sensor_or_config):
        """
        Set the sensor for the simulation.

        Parameters
        ----------
        sensor_or_config : Sensor or dict
            The Sensor object or its configuration.
        """
        if isinstance(sensor_or_config, dict):
            sensor = Sensor.from_config(sensor_or_config)
        else:
            sensor = sensor_or_config

        self._sensor = sensor
        self._psf_profile = {}  # reset the psf profile
        self._image_render_bundle_cache = {}
        # these following entries might depend on sensor for the bandpass
        self._h_spec_observation = None
        self._h_bkgd_observation = None

    def set_telescope(self, telescope_or_config):
        """
        Set the telescope for the simulation.

        Parameters
        ----------
        telescope_or_config : Telescope or dict
            The Telescope object or its configuration.
        """
        if isinstance(telescope_or_config, dict):
            telescope = Telescope.from_config(telescope_or_config)
        else:
            telescope = telescope_or_config
        
        self._telescope = telescope
        self._psf_profile = {} # reset the psf profile
        self._image_render_bundle_cache = {}

    # ------- #
    # update  #
    # ------- #
    @staticmethod
    def _fullkey_to_element_and_key(fullkey):
        """
        Split a double-underscore key into element and attribute key.

        Parameters
        ----------
        fullkey : str
            Key like 'sensor__gain'.

        Returns
        -------
        tuple
            (element, key)
        """
        element, *keys = fullkey.split("__")
        if len(keys) == 0:
            key = element
            element = None
        else:
            key = "__".join(keys) # trick to allow scene__background__mag => scene, background__mag
                
        return element, key

    def _fullkey_to_value(self, fullkey):
        """
        Get value from a double-underscore key.

        Parameters
        ----------
        fullkey : str
            Key like 'sensor__gain'.

        Returns
        -------
        object
            The value of the parameter.
        """
        *origin, baseparam = fullkey.split("__")
        origin = "__".join(origin)
        if len(origin)==0:
            return self.meta.get(baseparam)
        else:
            return eval(f"self.{origin.replace('__', '.')}").meta.get(baseparam)     
    
    def reset(self):
        """
        Reset the simulation and its components.
        """
        super().reset() # this resets the meta
        for element in [self.telescope, self.sensor, self.scene]:
            if element is not None:
                element.reset()
        self._psf_profile = {}
        self._image_render_bundle_cache = {}
            
                
    def update(self, **kwargs):
        """
        Update simulation parameters.

        The keyword should in principle have the following structure
        `scene__source__mag = 21` to change the `self.scene.source.mag` 
        parameter to 21.

        To simplify the use, if the name has no ambiguity, you 
        can skip the first structure elements. 
        For instance, only `scene__` has a `source__mag`. So `source__mag=21`
        will automatically be associated with `scene__source__mag=21`. 
        Same would work, for instance for `dark_current`: only `detector__` 
        has a `dark_current` mutable_parameter.
        However `mag=21` is not clear enough, as `scene__source__mag`, 
        `scene__background__mag` or `scene__host__mag` exist. 

        Parameters
        ----------
        **kwargs
            Parameters to update. Can use double-underscore for sub-elements.
        """        
        update_this = {}
        to_update = {"telescope": {},
                     "sensor": {},
                     "scene": {}
                    }
        for key, value in kwargs.items():
            key = key.replace(".", "__") # generi trick, sensor.gain == sensor__gain.
            
            # is that a fully defined name like sensor__dark_current ?
            if np.any([key.startswith(f"{element}__")
                       for element in ["telescope", "sensor", "scene"]]):
                # yes ? easy then
                element, down_key = self._fullkey_to_element_and_key(key)
                to_update[element][down_key] = value
                continue
    
            # see if it is missing a key
            fetch_key = [fullkey for fullkey in self.mutable_parameters
                          if fullkey.endswith(key)]
            if len(fetch_key) == 1:
                # ok, well defined, easy:
                element, down_key = self._fullkey_to_element_and_key(fetch_key[0])
                if element is None:
                    # means not a mutable parameter of a sub-element:
                    update_this[down_key] = value
                else:
                    # it is a sub-element
                    to_update[element][down_key] = value
                continue
                
            if len(fetch_key)>1:
                # not well defined, more than one entry exist for they key
                warnings.warn(f"several entries found matiching {key=} : {fetch_key}. Please clarify. *{key=} ignored*")
                continue
    
            # if we are here, it means fetch_key didn't match.
            warnings.warn(f"no entries found matiching {key=}. *{key=} ignored*")
    
        self.update_telescope(**to_update["telescope"])
        self.update_sensor(**to_update["sensor"])
        self.update_scene(**to_update["scene"])
    
        self._meta |= update_this
        # any parameter change can affect the PSF/count rates: drop caches
        self._psf_profile = {}
        self._image_render_bundle_cache = {}

    def update_telescope(self, **kwargs):
        """
        Update telescope parameters.

        Parameters
        ----------
        **kwargs
            Telescope parameters.
        """
        shortcuts = {"jitter": "jitter_sigma", 
                    "diameter": "diameter_primary"}
        to_update = {shortcuts.get(key, key): value for key, value in kwargs.items()}
        self.telescope.update(**to_update)
    
    def update_scene(self, **kwargs):
        """
        Update scene parameters.

        Parameters
        ----------
        **kwargs
            Scene parameters.
        """
        self.scene.update(**kwargs)
    
    def update_sensor(self, **kwargs):
        """
        Update sensor parameters.

        Parameters
        ----------
        **kwargs
            Sensor parameters.
        """
        self.sensor.update(**kwargs)

    def get_parameter(self, name, as_dict=False):
        """
        Get simulation or sub-element parameters.

        Parameters
        ----------
        name : str or list of str
            Parameter name(s).
        as_dict : bool, optional
            Whether to return as a dictionary. Default is False.

        Returns
        -------
        list or dict
            The requested parameters.
        """
        values = []
        names = np.atleast_1d(name)
        for name_ in names:
            name_ = name_.replace(".","__") # accept this generic way
            fetch_key = [fullkey for fullkey in self.mutable_parameters
                                if fullkey.endswith(name_)]
            if len(fetch_key) == 1:
                values.append( self._fullkey_to_value(fetch_key[0]) ) 
            else:
                if len(fetch_key)==0:
                    warnings.warn(f"not matching found for {name_=}")
                else:
                    warnings.warn(f"several matching found for {name_=} ; {fetch_key}")
                                      
                values.append(None)
    
        if as_dict:
            return dict(zip(names, values))
        
        return values
    
      
    # ------- #
    #  GETTER #
    # ------- #
    @staticmethod
    def _get_spectrum_observation(spectrum, abmag, bandpass):
        """
        Normalize a spectrum and return an observation.

        Parameters
        ----------
        spectrum : SourceSpectrum
            The spectrum to normalize.
        abmag : float
            The AB magnitude to normalize to.
        bandpass : SpectralElement
            The bandpass filter.

        Returns
        -------
        Observation
        """
        spec_at_mag = spectrum.normalize(abmag * u.ABmag, bandpass, force='extrap')
        return Observation(spec_at_mag, bandpass, force='extrap')

    def get_countrates(self, scene=None, band=None, units="adu/s", as_dict=True):
        """
        Get the countrates for each element in the scene.

        Parameters
        ----------
        scene : Scene, optional
            The scene to use. Defaults to self.scene.
        band : SpectralElement, optional
            The bandpass to use. Defaults to sensor bandpass.
        units : str, optional
            Units of the countrate ('adu/s', 'e/s', 'e-/s'). Default is "adu/s".
        as_dict : bool, optional
            Whether to return as a dictionary. Default is True.

        Returns
        -------
        dict or ndarray
            The countrates.
        """
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

        return list_of_quantity_to_array(countrates.values())


    def get_signal_and_variance(self, time=None, units="e-", n_reads=None):
        """
        Get the signal and total variance for a given exposure time.

        Parameters
        ----------
        time : float or Quantity, optional
            Exposure time. Defaults to self.meta['time'].
        units : str, optional
            Units of the signal ('e-', 'adu'). Default is "e-".
        n_reads : int, optional
            Number of coadded frames; the read-noise variance is incurred
            n_reads times. Defaults to self.meta['n_reads'] (or 1).

        Returns
        -------
        tuple
            (source_signal, total_variance)
        """
        if time is None:
            time = self._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
            
        # make sure the time is in the current units.
        elif not isinstance(time, u.Quantity):
           time = time*u.second

        n_reads = self._resolve_n_reads(n_reads)

        # get the count rates in {adu,e-}/s
        count_rates = self.get_countrates(units="e/s", as_dict=True) # this is an array
        
        #logging.info(f"Scene count rate: {count_rates:.2f} e/s")
        #logging.info(f"Background count rate: {sky_count_rate:.2f} e/s")

        source_signal = count_rates["source"] * time # e-
        
        # poisson noise is at the electron level, not adu.
        # poisson noise comes from the full scene source.
        all_countrates = list_of_quantity_to_array( count_rates.values() )
        scene_signal = np.nansum( all_countrates ) * time # e-
        
        dark_signal = self.sensor.dark_current * time # e-/pix
        
        # u.electron/u.pixel as variance, so unit square
        detector_variance = (dark_signal * u.electron/u.pixel
                             + n_reads * self.sensor.read_noise**2) * self.psf_profile["num_psf_pixels"] # e-**2
        #logging.info(f"signal: {scene_signal:.2f} e-")
#        logging.info(f"sky signal: {sky_signal:.2f} e-") part of the scene signal?
        #logging.info(f"dark signal: {dark_signal:.2f} e-")
        #logging.info(f"detector variance: {detector_variance:.2f} e-^2.")

        # total noise
        # * u.electron as photon noise ; already there in etector_variance
        total_variance = (scene_signal * u.electron + detector_variance)  # variance in  e-**2
        
        # units
        if units.lower() == "adu":
            scene_signal /= self.sensor.gain
            total_variance /= self.sensor.gain**2
        elif units not in ["e", "e-", "electron"]:
            raise ValueError(f"unknown units {units=}. adu or electron/e- expected.")
            
        return source_signal, total_variance

    def get_peak_pixel(self, time=None, units="adu", n_reads=None):
        """
        Get the brightest-pixel value for a given exposure time.

        The peak pixel combines the source PSF peak, the per-pixel sky
        background, the per-pixel dark current, and (for ADU) the additive bias
        level. Used to test ADC-clip saturation against ``sensor.adc_max``.
        Saturation is per-frame: with ``n_reads`` coadded frames spanning the
        total ``time``, the peak is evaluated for a single ``time / n_reads`` frame.

        Parameters
        ----------
        time : float or Quantity or array_like, optional
            Exposure time(s) in seconds. Defaults to self.meta['time'].
        units : str, optional
            'adu' (default, includes bias) or 'e-'/'e'/'electron' (excludes bias).
        n_reads : int, optional
            Number of coadded frames; the per-frame integration is time/n_reads.
            Defaults to self.meta['n_reads'] (or 1).

        Returns
        -------
        Quantity
            The peak-pixel value, in ADU (u.ct) or electrons.
        """
        if time is None:
            time = self._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second
        n_reads = self._resolve_n_reads(n_reads)
        time = time / n_reads          # per-frame integration; saturation is per-frame

        profile = self.psf_profile
        peak_fraction = profile["peak_pixel_fraction"]
        ee_at_aper = profile["ee_at_aper"]
        num_psf_pixels = profile["num_psf_pixels"]
        n_pix = num_psf_pixels.value if isinstance(num_psf_pixels, u.Quantity) else num_psf_pixels

        if ee_at_aper == 0:
            raise ValueError("ee_at_aper is zero; aperture radius is degenerate.")

        # count rates within the aperture, in electron/s
        count_rates = self.get_countrates(units="e/s", as_dict=True)

        # source: recover total flux (divide out aperture EE), take peak fraction
        source_peak = (count_rates["source"] / ee_at_aper * peak_fraction * time).to(u.electron)

        # sky background per pixel (uniform across the aperture); 0 if absent.
        # Only the background contributes here; host elements are excluded by
        # design (the approved saturation budget is source + background + dark).
        if "background" in count_rates:
            bkg_peak = (count_rates["background"] * time / n_pix).to(u.electron)
        else:
            bkg_peak = 0 * u.electron

        # dark current per pixel (dark_current is electron/(s*pix))
        dark_peak = (self.sensor.dark_current * time).to(u.electron / u.pix).value * u.electron

        peak_e = source_peak + bkg_peak + dark_peak  # electrons in the brightest pixel

        if units in ["e", "e-", "electron"]:
            return peak_e

        if units.lower() == "adu":
            return (peak_e / self.sensor.gain).to(u.ct) + self.sensor.bias_level

        raise ValueError(f"unknown units {units=}. 'adu' or electron/'e-' expected.")

    def is_saturated(self, time=None, n_reads=None):
        """
        Whether the brightest pixel reaches the ADC full scale (ADU clip).

        Parameters
        ----------
        time : float or Quantity or array_like, optional
            Exposure time(s) in seconds. Defaults to self.meta['time'].
        n_reads : int, optional
            Number of coadded reads. The total time is split into n_reads
            frames; saturation is evaluated on the per-frame time t/n_reads.
            Defaults to self.meta['n_reads'] or 1.

        Returns
        -------
        bool or ndarray of bool
            True where the peak pixel (in ADU) >= sensor.adc_max.
        """
        peak_adu = self.get_peak_pixel(time, units="adu", n_reads=n_reads)
        return peak_adu >= self.sensor.adc_max

    def _image_render_bundle(self, psf, jitter_sigma_mas, npix, oversample):
        """
        Cached, time-independent inputs for the PSF-aware SNR/exptime path.

        Returns a dict with the rendered normalized PSF, the plate scale (mas),
        and the source/diffuse count *rates* (electrons / s). Memoized on
        render-affecting state; cleared by update()/set_sensor()/set_telescope().
        """
        from .psfsim import ImageSimulator

        if jitter_sigma_mas is None:
            jitter_sigma_mas = self.telescope.jitter_sigma.to("mas").value

        cache = getattr(self, "_image_render_bundle_cache", None)
        if cache is None:
            cache = self._image_render_bundle_cache = {}
        key = (psf.cache_key(), jitter_sigma_mas, int(npix), int(oversample))
        if key in cache:
            return cache[key]

        imsim = ImageSimulator(self, npix=npix, oversample=oversample)
        ctx = imsim._context(jitter_sigma_mas=jitter_sigma_mas)
        psf_norm = psf.render(ctx)

        profile = self.psf_profile
        ee_at_aper = profile["ee_at_aper"]
        if ee_at_aper == 0:
            raise ValueError("ee_at_aper is zero; aperture radius is degenerate.")
        num_psf_pixels = profile["num_psf_pixels"]
        n_psf = num_psf_pixels.value if isinstance(num_psf_pixels, u.Quantity) else num_psf_pixels

        count_rates = self.get_countrates(units="e/s", as_dict=True)
        source_rate_total = (count_rates["source"] / ee_at_aper).to(u.electron / u.s).value
        diffuse_rate_per_pix = 0.0
        for name, rate in count_rates.items():
            if name == "source":
                continue
            diffuse_rate_per_pix += (rate / n_psf).to(u.electron / u.s).value

        bundle = {"psf_norm": psf_norm,
                  "plate_scale_mas": ctx.plate_scale_mas,
                  "source_rate_total": source_rate_total,
                  "diffuse_rate_per_pix": diffuse_rate_per_pix}
        cache[key] = bundle
        return bundle

    def get_image_snr(self, time=None, psf=None, r_aper_mas=None, ee_frac=None,
                      optimize=False, jitter_sigma_mas=None, n_reads=None,
                      npix=128, oversample=11):
        """
        PSF-aware aperture signal-to-noise ratio.

        Unlike get_snr_airy (the analytic Airy-disk approximation), this renders
        the given PSF (default AiryPSF) on the detector grid and computes the SNR
        for a circular aperture. get_snr now delegates to this method; the
        in-focus default-aperture case reproduces get_snr_airy to within ~1%.
        Aperture precedence: optimize > r_aper_mas > ee_frac; if none is
        given, the Simulation's r_aper_mas is used.

        Parameters
        ----------
        time : float, array_like, or Quantity, optional
            Exposure time(s) in seconds (bare floats are interpreted as
            seconds). Defaults to meta['time']. A scalar yields a dict of
            Python float/int; an array yields a dict of equal-length ndarrays.
        psf : PSFSource, optional
            PSF model. If None, uses _default_psf (set by from_sensorfilter)
            when available, otherwise falls back to AiryPSF().
        r_aper_mas : float, optional
            Fixed aperture radius (mas).
        ee_frac : float, optional
            Aperture enclosing this fraction of the PSF.
        optimize : bool, optional
            If True, use the radius that maximizes SNR.
        jitter_sigma_mas : float, optional
            Override the telescope jitter (mas).
        npix, oversample : int, optional
            Render grid size and oversampling. `npix` must be large enough to
            contain the PSF; the default (128) contains the bundled defocus PSFs
            for the current sensors, but very small pixels or stronger defocus
            may need a larger value.

        Returns
        -------
        dict
            'snr', 'signal_e', 'noise_e', 'enclosed_fraction', 'r_aper_mas',
            'n_pix'. Values are Python float/int for scalar `time`, or
            ndarrays (n_pix as int) for array `time`.
        """
        from .psfsim import AiryPSF, aperture_snr_radial, select_aperture

        if time is None:
            time = self._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second

        n_reads = self._resolve_n_reads(n_reads)
        if psf is None:
            psf = self._default_psf if self._default_psf is not None else AiryPSF()

        b = self._image_render_bundle(psf, jitter_sigma_mas, npix, oversample)

        # default to the ETC aperture if no mode was requested
        if not optimize and r_aper_mas is None and ee_frac is None:
            r_aper_mas = self._meta.get("r_aper_mas")

        read_noise = self.sensor.read_noise.to(u.electron / u.pix).value * np.sqrt(n_reads)
        dark_rate_per_pix = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value

        def _snr_at(t_sec):
            source_e_total = b["source_rate_total"] * t_sec
            diffuse_per_pix = b["diffuse_rate_per_pix"] * t_sec
            dark_per_pix = dark_rate_per_pix * t_sec
            prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                                       source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
            idx = select_aperture(prof, r_aper_mas=r_aper_mas, ee_frac=ee_frac, optimize=optimize)
            return {"snr": float(prof["snr"][idx]),
                    "signal_e": float(prof["signal_e"][idx]),
                    "noise_e": float(prof["noise_e"][idx]),
                    "enclosed_fraction": float(prof["enclosed_fraction"][idx]),
                    "r_aper_mas": float(prof["r_mas"][idx]),
                    "n_pix": int(prof["n_pix"][idx])}

        if time.isscalar:
            return _snr_at(time.to(u.second).value)

        results = [_snr_at(t) for t in time.to(u.second).value]
        out = {k: np.array([r[k] for r in results])
               for k in ("snr", "signal_e", "noise_e", "enclosed_fraction", "r_aper_mas")}
        out["n_pix"] = np.array([r["n_pix"] for r in results], dtype=int)
        return out

    def get_image_exptime_for_snr(self, snr, psf=None, r_aper_mas=None,
                                  ee_frac=None, optimize=False,
                                  jitter_sigma_mas=None, n_reads=None,
                                  npix=128, oversample=11):
        """
        Exposure time (s) to reach a target SNR on the PSF-aware path.

        Inverse of get_image_snr. Aperture precedence: optimize > r_aper_mas >
        ee_frac; if none is given the Simulation's r_aper_mas is used. Note
        optimize here picks the radius that reaches the target SNR *fastest*
        (minimum time), the inverse of get_image_snr's max-SNR optimize. Returns
        a dict {'time_s', 'snr', 'r_aper_mas', 'enclosed_fraction', 'n_pix'}.

        Parameters
        ----------
        psf : PSFSource, optional
            PSF model. If None, uses _default_psf (set by from_sensorfilter)
            when available, otherwise falls back to AiryPSF().
        """
        from .psfsim import AiryPSF, aperture_time_for_snr

        n_reads = self._resolve_n_reads(n_reads)
        if psf is None:
            psf = self._default_psf if self._default_psf is not None else AiryPSF()

        b = self._image_render_bundle(psf, jitter_sigma_mas, npix, oversample)

        dark_rate_per_pix = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        read_noise = self.sensor.read_noise.to(u.electron / u.pix).value

        if not optimize and r_aper_mas is None and ee_frac is None:
            r_aper_mas = self._meta.get("r_aper_mas")

        return aperture_time_for_snr(b["psf_norm"], b["plate_scale_mas"],
                                     b["source_rate_total"], b["diffuse_rate_per_pix"],
                                     dark_rate_per_pix, read_noise,
                                     n_reads=n_reads, snr=snr,
                                     r_aper_mas=r_aper_mas, ee_frac=ee_frac,
                                     optimize=optimize)

    def get_snr(self, time=None, psf=None, r_aper_mas=None, ee_frac=None,
                optimize=False, jitter_sigma_mas=None, n_reads=None,
                npix=128, oversample=11):
        """
        Signal-to-noise ratio via the 2D image simulation (PSF-aware default).

        Delegates to get_image_snr; see it for parameter details and the
        returned dict. `time` may be a scalar or an array (returns a dict of
        arrays). For the legacy analytic Airy approximation use get_snr_airy
        (deprecated).

        Returns
        -------
        dict
            Same as get_image_snr: 'snr', 'signal_e', 'noise_e',
            'enclosed_fraction', 'r_aper_mas', 'n_pix' (scalar values for scalar
            `time`, ndarrays for array `time`).
        """
        return self.get_image_snr(
            time=time, psf=psf, r_aper_mas=r_aper_mas, ee_frac=ee_frac,
            optimize=optimize, jitter_sigma_mas=jitter_sigma_mas,
            n_reads=n_reads, npix=npix, oversample=oversample)

    def get_snr_airy(self, time=None, n_reads=None):
        """
        DEPRECATED analytic Airy-disk SNR approximation.

        Use get_snr, which computes the SNR via the 2D image simulation.

        Parameters
        ----------
        time : float or Quantity, optional
            Exposure time.
        n_reads : int, optional
            Number of reads. Defaults to meta['n_reads'] if set, else 1.

        Returns
        -------
        Quantity
            The SNR.
        """
        warnings.warn(
            "get_snr_airy (analytic Airy approximation) is deprecated; "
            "use get_snr, which now uses the 2D image simulation.",
            DeprecationWarning, stacklevel=2)
        signal, variance = self.get_signal_and_variance(time, n_reads=n_reads)
        return signal / np.sqrt(variance)

    def get_exptime_for_snr(self, snr, n_reads=None):
        """
        Exposure time (seconds, Quantity) to reach a target SNR on the analytic
        (Airy) path. Inverse of get_snr_airy. Returns inf*u.s if the source rate is 0.
        """
        from .psfsim import solve_time_for_snr
        A, B, C = self._snr_coefficients(n_reads=n_reads)
        return solve_time_for_snr(snr, A, B, C) * u.second

    # -------------- #
    #  Internal      #
    # -------------- #
    def _resolve_n_reads(self, n_reads):
        """n_reads from the argument, else meta['n_reads'], else 1. Must be >= 1."""
        if n_reads is None:
            n_reads = self._meta.get("n_reads", 1)
        n_reads = int(n_reads)
        if n_reads < 1:
            raise ValueError(f"n_reads must be >= 1, got {n_reads}")
        return n_reads

    def _snr_coefficients(self, n_reads=None):
        """
        (A, B, C) floats for SNR(t) = A*t / sqrt(B*t + C), electrons & seconds.
        A = source rate; B = total scene rate + n_pix*dark; C = n_pix*N*RN**2.
        """
        n_reads = self._resolve_n_reads(n_reads)
        count_rates = self.get_countrates(units="e/s", as_dict=True)
        A = count_rates["source"].to(u.electron / u.s).value
        all_rates = float(np.nansum([r.to(u.electron / u.s).value
                                     for r in count_rates.values()]))
        n_pix = self.psf_profile["num_psf_pixels"]
        n_pix = n_pix.value if isinstance(n_pix, u.Quantity) else float(n_pix)
        dark = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        rn = self.sensor.read_noise.to(u.electron / u.pix).value
        B = all_rates + n_pix * dark
        C = n_pix * n_reads * rn ** 2
        return A, B, C

    def _parse_bandpass(self, bandpass):
        """
        Parse a bandpass name or object.

        Parameters
        ----------
        bandpass : str or SpectralElement
            The bandpass. If 'sensor', returns the sensor's bandpass.

        Returns
        -------
        SpectralElement
        """
        if bandpass == "sensor":
            return self.sensor.bandpass
        
        return SpectralElement.from_filter(bandpass)
        
    def compute_psf_profile(self):
        """
        Compute the PSF profile and associated metrics.

        Returns
        -------
        dict
            Dictionary containing 'wavelength', 'r_psf_mas', 'psf1d', 'ee',
            'ee_at_aper', 'num_psf_pixels', 'psf_area', and
            'peak_pixel_fraction'.
        """
        from .airy import get_airy_and_ee_curve, render_detector_psf
        
        wavelength = self.sensor.wavelength.to("m")
        r_psf_mas, psf1d, ee, ee_at_aper = get_airy_and_ee_curve(wavelength, 
                                                                 r_aper_mas = self._meta["r_aper_mas"], # no default allower 
                                                                 jitter_sigma_mas = self.telescope.jitter_sigma.to("mas"), 
                                                                 fnum=self.telescope.f_num, 
                                                                 D=self.telescope.diameter_primary.value,
                                                                 pixel_size=self.sensor.pixel_size.value,
                                                                 verbose=False)

        # compute the number of pixels associated to the PSF
        plate_scale = self.sensor.get_plate_scale(self.telescope) # in arcsec/pix
        num_pixels_at_r = self._meta["r_aper_mas"]*u.arcsec/(plate_scale * 1000) # pix
        num_psf_pixels = (np.pi * num_pixels_at_r**2) # in pixels**2

        # area of the psf in angular units
        psf_area = num_psf_pixels * plate_scale **2 # in arcsec**2

        
        #logging.info(f"PSF profile computed:")
        #logging.info(f"EE={ee_at_aper:.2f} at {self.meta['r_aper_mas']} mas aperture")
        #logging.info(f"num_psf_pixels={num_psf_pixels:.1f} pixels")
        #logging.info(f"PSF area={psf_area:.2f} arcsec^2")

        # brightest-pixel energy fraction on the detector grid
        psf_detector, _ = render_detector_psf(
            wavelength=wavelength,
            fnum=self.telescope.f_num,
            D=self.telescope.diameter_primary.value,
            pixel_size=self.sensor.pixel_size.value,
            jitter_sigma_mas=self.telescope.jitter_sigma.to("mas").value,
            verbose=False)
        peak_pixel_fraction = float(psf_detector.max())

        return {"wavelength": wavelength,
                "r_psf_mas": r_psf_mas,
                "psf1d": psf1d,
                "ee": ee,
                "ee_at_aper": ee_at_aper,
                "num_psf_pixels": num_psf_pixels,
                "psf_area": psf_area,
                "peak_pixel_fraction": peak_pixel_fraction
                }

    def has_element(self, which):
        """
        Check if the simulation has a specific component.

        Parameters
        ----------
        which : str
            Component name ('telescope', 'sensor', 'scene').

        Returns
        -------
        bool
        """
        return getattr(self, which) is not None
        
    # ================= #
    #   Properties      #
    # ================= #
    @property
    def scene(self):
        """
        The Scene object.
        """
        return self._scene

    @property
    def telescope(self):
        """
        The Telescope object.
        """
        return self._telescope
    
    @property
    def sensor(self):
        """
        The Sensor object.
        """
        return self._sensor

    @property
    def meta(self):
        """
        Combined metadata from simulation and components.
        """
        return self._meta | {element_name: element.meta
                                 for element_name in ["telescope", "sensor", "scene"]
                                 if (element := getattr(self, element_name)) is not None
                                 }

    @property
    def mutable_parameters(self):
        """
        List of all mutable parameters (including sub-elements).
        """
        return self._mutable_parameters +  [f"{element_name}__{k}"
                                                for element_name in ["telescope", "sensor", "scene"]
                                                if (element := getattr(self, element_name)) is not None
                                                for k in element.mutable_parameters  
                                            ]
    # ---------- #
    # cashed     #
    # ---------- #
    @property
    def psf_profile(self):
        """
        The calculated PSF profile.
        """
        if not hasattr(self,"_psf_profile") or self._psf_profile is None or len(self._psf_profile) == 0 : # like {}
            self._psf_profile = self.compute_psf_profile()
            
        return self._psf_profile

