import warnings
from astropy import units as u
from synphot import units, SourceSpectrum, SpectralElement, Observation

from .meta import _MetaHolder_


# ONGOING WORK

# ============= #
#   Source      #
# ============= # 

class SceneElement(_MetaHolder_):
    """ """
    _mutable_parameters = ["spectrum", "mag", "magsys", "bandpass"]
    _accepted_specific_origins = ["surface_brightness"]
    
    def __init__(self, spectrum, mag, magsys="ABmag", bandpass="johnson_v", meta={}):
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
        minimal_kwargs = {key:config.get(key) for key in ["spectrum", "mag"]}
        
        return cls(meta=config, **minimal_kwargs)
        
    # =========== #
    #  methods    #
    # =========== #
    def set_spectrum(self, spec_or_file):
        """ generic setter for any source component """
        
        # make sure you have a spectrum.
        if type(spec_or_file) in [str]:
            if spec_or_file in self._accepted_specific_origins:
                spectrum = spec_or_file
            else:
                spectrum = SourceSpectrum.from_file(spec_or_file)

        # the or None enables to switch of the host by setting it to None            
        elif isinstance(spec_or_file, SourceSpectrum) or None:
            spectrum = spec_or_file

        self._spectrum = spectrum

    # ------- #
    # GETTER  #
    # ------- # 
    def get_mag(self, area=None):
        """ returns the actual magnitude accounting for the area if mag is a surface brightness """
        if self.meta.get("surface_brighness", False):
            return self.mag
        else:
            return self.mag - 2.5 * np.log10(area)
            
    def get_spectrum(self, apply_mag=True, as_array=False, area=None):
        """ """
        if isinstance(self.spectrum, SourceSpectrum):
            if apply_mag:
                spectrum = self.spectrum.normalize(self.get_mag(area), band=self.band)
            else:
                spectrum = self.spectrum
        else:
            warnings.warn(f"cannot get the spectrum of the stored {self.spectrum=}")
            return None

        if as_array:
            spectrum = spectrum._get_arrays(None)

        return spectrum
        
    def get_observation(self, band=None):
        """ """
        if band is None:
            band = self.band

        spectrum = self.get_spectrum(apply_mag=True, as_array=False)
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

        #if self.meta.get("surface_brightness", False):
        #    mag = mag/(u.arcsec**2)
            
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
            source = SourceElement.from_config(source_config)
        else:
            source = None

        # host
        if (host_config := config.get("host", {})):
            host = SourceElement.from_config(host_config)
        else:
            host = None        
            
        # background
        if (background_config := config.get("background", {})):
            background = SourceElement.from_config(background_config)
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
