import warnings
import os
import numpy as np

from astropy import units as u
from synphot import units, SourceSpectrum, SpectralElement, Observation

from .meta import _MetaHolder_


# ONGOING WORK

# ============= #
#   Source      #
# ============= # 
def broadcast_mapping(value, ntargets):
    """Broadcast a value to a given number of targets."""
    value = np.atleast_1d(value)
    if np.ndim(value)>1:
        # squeeze drop useless dimensions.
        broadcasted_values = np.broadcast_to(value, (ntargets, value.shape[-1]) )
    else:
        broadcasted_values = np.broadcast_to(value, ntargets)
        
    return broadcasted_values


# ============= #
#   Source      #
# ============= # 

class SceneElement(_MetaHolder_):
    """ """
    _mutable_parameters = ["spectrum", "mag", "magsys", "bandpass", "surface_brightness"]
    
    def __init__(self, spectrum, mag, 
                 magsys="ABmag", bandpass="johnson_v", 
                 surface_brightness=False,
                 meta={}):
        """ """
        input_parameters = {key:value for key,value in locals().items()
                             if key not in ["self", "meta"]
                                and value is not None}
        
        super().__init__(meta | input_parameters)
        self.set_spectrum(spectrum)
        
    @classmethod
    def from_config(cls, config):
        """ """
        # make sure these keys exist
        input_kwargs = {key:config.get(key) for key in ["spectrum", "mag"]}
        input_kwargs |= {key:config.get(key) for key in ["magsys", "bandpass", "surface_brightness"]
                        if key in config} # else default as given by __init__
        
        return cls(meta=config, **input_kwargs)


    # =========== #
    #  methods    #
    # =========== #
    def set_spectrum(self, spec_or_file):
        """ generic setter for any source component """
        
        # make sure you have a spectrum.
        if type(spec_or_file) in [str]:
            if not os.path.isfile(spec_or_file):
                # may that is a spectral type:
                from .io import get_any_astro_name
                spec_or_file = get_any_astro_name(spec_or_file)
            
            spectrum = SourceSpectrum.from_file(spec_or_file)
            
        # the or None enables to switch of the host by setting it to None            
        elif isinstance(spec_or_file, SourceSpectrum) or spec_or_file is None:
            spectrum = spec_or_file

        self._spectrum = spectrum

    # ------- #
    # GETTER  #
    # ------- # 
    def get_mag(self, area=None):
        """ returns the actual magnitude accounting for the area if mag is a surface brightness """
        if self.mag_is_surface_brightness:
            # must work in float (not astropy unit) because of mag and np.log.
            ## area should be in arcsec**2
            if "astropy.units" in str(type(area)): 
                area_arcsec2 = area.to("arcsec**2").value
            else:
                # if float, assuming it is in arcsec2
                area_arcsec2 = area
                
            ## take the value of mag 
            mag_per_arcsec2 = self.mag.value 
            mag_unit = self.mag.unit
            mag = (mag_per_arcsec2 - 2.5 * np.log10(area_arcsec2)) * mag_unit
        else:
            mag = self.mag
            
        return mag
            
    def get_spectrum(self, apply_mag=True, as_array=False, area=None):
        """ """
        if isinstance(self.spectrum, SourceSpectrum):
            if apply_mag:
                mag = self.get_mag(area=area)
                spectrum = self.spectrum.normalize(mag, band=self.band)
            else:
                spectrum = self.spectrum
        else:
            warnings.warn(f"cannot get the spectrum of the stored {self.spectrum=}")
            return None

        if as_array:
            spectrum = spectrum._get_arrays(None)

        return spectrum
        
    def get_observation(self, band=None, area=None):
        """ """
        if band is None:
            band = self.band

        spectrum = self.get_spectrum(apply_mag=True, as_array=False, area=area)
        if spectrum is None:
            return None
            
        return Observation(spectrum, band, force='extrap')
        
    def show(self, ax=None, apply_mag=True, **kwargs):
        """
        Show the spectrum or host flux.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7,3))
        else:
            fig = ax.figure

        # data to show
        lbda, flux = self.get_spectrum(as_array=True, apply_mag=apply_mag)

        # plot
        ax.plot(lbda, flux, **kwargs)
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        return fig
        
    # ------------ #
    #  internal    #
    # ------------ # 
    def _parse_mag_(self):
        """ """
         # magnitude
        mag = self.meta.get("mag", None)
        if mag is None:
            warnings.warn("not mag in self.meta")
            
        else:
            # make sure mag has the correct units.
            magsys = self.meta.get("magsys", "ABmag")
            # make sure it is an astropy units.
            if not hasattr(magsys, "is_equivalent"):
                magsys = getattr(u, magsys)
            #         
            mag = mag * magsys

        return mag

    def _parse_band_(self):
        """ """
        # band or bandpass accepted
        band = self.meta.get("band", self.meta.get("bandpass"))
        if type(band) is str:
            band = SpectralElement.from_filter(band)
        elif not isinstance(band, SpectralElement):
            raise ValueError(f"{band=} meta is neither a str nor a SpectralElement.")
        
        return band

    # ============= #
    #  Properties   #
    # ============= # 
    @property
    def spectrum(self):
        """ """
        return self._spectrum
        
    @property
    def mag(self):
        """ """
        return self._parse_mag_()

    @property
    def band(self):
        """ """
        return self._parse_band_()

    @property
    def mag_is_surface_brightness(self):
        """ """
        return self.meta.get("surface_brightness", False)

class Scene(_MetaHolder_):
    """ """
    # ZODI TO BE IMPLEMENTED
    # self.background = self.config['zodi']['zodi_mag_r']

    _mutable_parameters = []
    
    def __init__(self, source=None,
                     host=None,
                     background=None,
                     meta={}
                ):
        """
        Source object

        INPUT:
            spectrum: SourceSpectrum object
            host: SourceSpectrum object (optional)
            background: float, background surface brightness (optional)
        """
        self._source = source
        self._host = host
        self._background = background

        super().__init__(meta)
        
    @classmethod
    def from_config(cls, config):
        """ """
        # Source
        if (source_config := config.get("source", {})):
            source = SceneElement.from_config(source_config)
        else:
            source = None

        # host
        if (host_config := config.get("host", {})):
            host = SceneElement.from_config(host_config)
        else:
            host = None        
            
        # background
        if (background_config := config.get("background", {})):
            background = SceneElement.from_config(background_config)
        else:
            background = None

        # And build it.
        return cls(source=source,
                   host=host,
                   background=background,
                   meta=config)

    # ============= #
    #   methods     #
    # ============= #
    def reset(self):
        """ """
        # reset each element
        _ = self.call_down("reset", which="all")
        
    def update(self, reset=False, **kwargs):
        """ """
        update_source = {}
        update_host = {}
        update_background = {}
        updated_key = []
        for key, value in kwargs.items():
            # does not matter
            if value is None:
                continue

            if key.startswith("source__"):
                update_source[key.replace("source__", "")] = value

            elif key.startswith("host__"):
                update_host[key.replace("host__", "")] = value

            elif key.startswith("background__"):
                update_background[key.replace("background__", "")] = value
            else:
                warnings.warn("cannot parse {key=} ; should start by 'source__' etc.")
                continue
            updated_key.append(key)
            
        if self.has_source():
            self.source.update(**update_source)
        if self.has_host():
            self.host.update(**update_host)
        if self.has_background():
            self.background.update(**update_background)
        
        return updated_key
        
    def get_elements(self, which="*", as_dict=False):
        """ """
        if which in ["*", "all"]:
            which = self.element_names
        else:
            which = np.atleast_1d(which)

        values = [getattr(self, which_) for which_ in self.element_names if which_ in which]
        if as_dict:
            return dict(zip(which, values))
            
        return values

    def get_mag(self, area=None, which="*", as_dict=False):
        """ """
        return self.call_down("get_mag", area=area, which=which, as_dict=as_dict)

    def get_spectrum(self, area=None, apply_mag=True, which="*", as_dict=False):
        """ """
        return self.call_down("get_spectrum", area=area, apply_mag=apply_mag, 
                              which=which, as_dict=as_dict)
    
    def get_observation(self, band=None, area=None, which="*", as_dict=False):
        """ """
        return self.call_down("get_observation", band=band, area=area, which=which, as_dict=as_dict)
        
    def has_source(self):
        """ """
        return self._source is not None

    def has_host(self):
        """ """
        return self._host is not None

    def has_background(self):
        """ """
        return self._background is not None
    
    # -------- #
    #  GETTER  #
    # -------- #
    
    # ----------- #
    #  Internal   #
    # ----------- #
    def call_down(self, what, mapargs=None, allow_call=True, which="*", as_dict=False, **kwargs):
        """ Call a method on each target in the collection. """
        # applied to target.simulation
        elements = self.get_elements(which=which, as_dict=as_dict)
        if as_dict:
            element_names = elements.keys()
            elements = elements.values()
        
        if mapargs is not None:
            mapargs = broadcast_mapping(mapargs, elements)
            values = [getattr(element, what)(maparg_, **kwargs)
                        for maparg_, element in zip(mapargs, elements)]
        
        else:
            values = [attr if not (callable(attr := getattr(element, what)) and allow_call) else attr(**kwargs) 
                       for element in elements]

        if as_dict:
            return dict(zip(element_names, values))
            
        return values
        
    # -------- #
    # PLOTTER  #
    # -------- #
    @property
    def source(self):
        """ """
        return self._source

    @property
    def host(self):
        """ """
        return self._host

    @property
    def background(self):
        """ """
        return self._background

    @property
    def element_names(self):
        """ """
        return ["source", "host", "background"]
