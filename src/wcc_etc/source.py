from astropy import units as u
from synphot import units, SourceSpectrum

from .meta import _MetaHolder_

# => Loggin not used.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= #
#   Source      #
# ============= # 

class SourceElement(_MetaHolder_):
    """ """
    _mutable_parameters = ["origin", "mag", "magsys", "bandpass"]
    _accepted_specific_origins = ["surface_brightness"]
    
    def __init__(self, origin, mag, magsys="ABmag", bandpass="johnson_v", meta={}):
        """ """
        input_parameters = {key:value for key,value in locals().items()
                             if key not in ["self", "meta"]
                                and value is not None}
        super().__init__(meta | input_parameters)

    # =========== #
    #  methods    #
    # =========== #
    def set_spectrum(self, spec_or_file, apply_normalization=True):
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

        # normalized given meta info (if any)
        # -> this enters if this_prop is not {} nor None
        if apply_normalization:
            if mag is not None:
                spectrum = spectrum.normalize(self.mag, band=self.band)
            else:
                warnings.warn(f"apply_normalization requested during setting but no mag given for {which=}")
            
        self._spectrum = spectrum

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
        # band
        band = self.meta.get("band")
        if type(band) is str:
            band = SpectralElement.from_filter(bandpass)
        elif not isinstance(band, SpectralElement):
            raise ValueError(f"{band=} from {which=} meta is neither a str nor a SpectralElement.")
        
        return band

    # ============= #
    #  Properties   #
    # ============= # 
    @property
    def mag(self):
        """ """
        return self._parse_mag_()
    
    @property
    def band(self):
        """ """
        return self._parse_band_()
    
    
class Source(_MetaHolder_):
    """ """
    # ZODI TO BE IMPLEMENTED
    # self.background = self.config['zodi']['zodi_mag_r']

    _mutable_parameters = []
    
    def __init__(self, spectrum,
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
        self.set_spectrum(spectrum)
        self.set_host(host)
        self.set_background(background)

        super().__init__(meta)
        
    @classmethod
    def from_file(cls, spec_file, host_file=None, background_file=None,
                      meta={}):
        """
        Create a Source object from a file.

        Parameters:
        ----------
        spec_file: str, 
            path to the source spectrum file
        host_file: str, None
            path to the host spectrum (if any)
        bkgd_file: str, None
            path to the diffuse background spectrum (if any)
        """
        spectrum = SourceSpectrum.from_file(spec_file)
        if host_file is not None:
            host = SourceSpectrum.from_file(bkgd_file)
        else:
            host = None
            
        if background_file is not None:
            from .io import read_config
            print('read background surface brightness from config')
            config = read_config("astro")
            background = config.get("background", None)
            logging.info(f"Background surface brightness: {background}")
        else:
            background = None

        return cls(spectrum=spectrum, host=host, background=background)

    @classmethod
    def from_config(cls, config):
        """ """
        spec_or_file = config.get("spectrum", {})
        host_or_file = config.get("host", {})
        background_or_file = config.get("background", {})
        
        return cls(spectrum=spec_or_file,
                   host=host_or_file,
                   background=background_or_file,
                   meta=config)

    # ============= #
    #   methods     #
    # ============= #
    
    # Setter
    def _set_any_spectrum_(self, spec_of_file, which, apply_normalization=True):
        """ generic setter for any source component 
        """
        if which not in ["spectrum", "host", "background"]:
            raise ValueError(f"cannot set {which=} spectrum. 'spectrum', 'host', or 'background' available")

        # make sure you have a spectrum.
        if type(spec_of_file) in [str]:
            spectrum = SourceSpectrum.from_file(spec_of_file)
            
        # the or None enables to switch of the host by setting it to None            
        elif isinstance(spec_of_file, SourceSpectrum) or None:
            spectrum = spec_of_file

        # normalized given meta info (if any)
        # -> this enters if this_prop is not {} nor None
        if apply_normalization:
            mag, band = self.get_mag_and_band(which)
            if mag is not None:
                spectrum = spectrum.normalize(mag, band=band)
            else:
                warnings.warn(f"apply_normalization requested during setting but no mag given for {which=}")
            
        setattr(self, f"_{which}", spectrum)
        
    def set_spectrum(self, spectrum_or_file, apply_normalization=True):
        """
        Set the spectrum of the source.

        INPUT:
            spectrum: SourceSpectrum object or path to a spectrum file
        """
        self._set_any_spectrum_(spectrum_or_file, "spectrum",
                                apply_normalization=apply_normalization)

    def set_host(self, spectrum_or_file, apply_normalization=True):
        """
        Set the host spectrum of the source.

        INPUT:
            spectrum: SourceSpectrum object or path to a spectrum file
        """
        self._set_any_spectrum_(spectrum_or_file, "host",
                                apply_normalization=apply_normalization)

    def set_background(self, spectrum_or_file, apply_normalization=True):
        """
        Set the host surface brightness of the source.

        INPUT:
            background: float, background surface brightness
        """
        self._set_any_spectrum_(spectrum_or_file, "background",
                                apply_normalization=apply_normalization)

    def has_host(self):
        """ """
        return self._host is not None

    def has_background(self):
        """ """
        return self._background is not None
    
    # -------- #
    #  GETTER  #
    # -------- #
    def get_flux(self, which="source"):
        """ """
        if which in ["source", "spectrum"]:
            return self.spectrum._get_arrays(None)
        
        elif which == "host":
            return self.host._get_arrays(None)
        
        else:
            raise ValueError(f"only source and host known. {which=} given")
        
    def get_observations(self, which, at_mag=True):
        """ """
        if which in ["*", "all"]:
            which = ["spectrum", "host", "background"]
        else:
            which = np.atleast_1d(which)

        observations = []
        for which_ in which:
            spec = getattr(self, which)
            if spec is None:
                observation = None
            else:
                band = self._parse_which_band_(which)
                observation = Observation(spec, band, force='extrap')
                
            observations.append(observation)
            
        return observations

    def get_mag_and_band(self, which):
        """ """
        which_prop = self.meta.get(which, None)
        mag = self._parse_which_mag_(which_prop)
        band = self._parse_which_band_(which_prop)
        return mag, band

    # ----------- #
    #  Internal   #
    # ----------- #
        
    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, which="source",  ax=None, **kwargs):
        """
        Show the spectrum or host flux.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7,3))
        else:
            fig = ax.figure

        # data to show
        lbda, flux = self.get_flux(which)

        # plot
        ax.plot(lbda, flux, **kwargs)
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        return fig
        
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def spectrum(self):
        """ spectrum of the source """
        return self._spectrum

    @property
    def host(self):
        """ spectrum of the host """
        return self._host

    @property
    def background(self):
        """ background surface brightness of the source """
        return self._background

    @property
    def meta(self):
        """ """
        return self._meta
