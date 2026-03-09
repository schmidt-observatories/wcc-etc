
from synphot import units, SourceSpectrum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= #
#   Source      #
# ============= # 

class Source():
    """ """
    # ZODI TO BE IMPLEMENTED
    # self.bg_surface_brightness = self.config['zodi']['zodi_mag_r']

    def __init__(self, spectrum, background=None, bg_surface_brightness=22.5):
        """
        Source object

        INPUT:
            spectrum: SourceSpectrum object
            background: SourceSpectrum object (optional)
            bg_surface_brightness: float, background surface brightness (optional)
        """
        self.set_spectrum(spectrum)
        self.set_background(background)
        self.set_bg_surface_brightness(bg_surface_brightness)

        self._meta = {}
        
    @classmethod
    def from_file(cls, filepath, bkgd_file=None, bg_surface_brightness_file=None):
        """
        Create a Source object from a file.

        INPUT:
            filepath: path to the source spectrum file
            bkgd_file: path to the background spectrum file (optional)
        """
        spectrum = SourceSpectrum.from_file(filepath)
        if bkgd_file is not None:
            background = SourceSpectrum.from_file(bkgd_file)
        else:
            background = None
        if bg_surface_brightness_file is not None:
            from .io import read_config
            print('read background surface brightness from config')
            config = read_config("astro")
            bg_surface_brightness = config.get("bg_surface_brightness", None)
            logging.info(f"Background surface brightness: {bg_surface_brightness}")
        else:
            bg_surface_brightness = None

        return cls(spectrum=spectrum, background=background, bg_surface_brightness=bg_surface_brightness)

    @classmethod
    def from_config(cls, config):
        """ """
        spec_or_file = config.get("spectrum")
        background_or_file = config.get("background", None)
        bg_surface_brightness_or_file = config.get("bg_surface_brightness", None)
        return cls(spectrum=spec_or_file, background=background_or_file, bg_surface_brightness=bg_surface_brightness_or_file)

    # ============= #
    #   methods     #
    # ============= #
    def set_spectrum(self, spectrum_or_file):
        """
        Set the spectrum of the source.

        INPUT:
            spectrum: SourceSpectrum object or path to a spectrum file
        """
        if type(spectrum_or_file) in [str]:
            spectrum = SourceSpectrum.from_file(spectrum_or_file)
            
        # the or None enables to switch of the background by setting it to None            
        elif isinstance(spectrum_or_file, SourceSpectrum) or None:
            spectrum = spectrum_or_file

        self._spectrum = spectrum

    def set_background(self, spectrum_or_file):
        """
        Set the background spectrum of the source.

        INPUT:
            spectrum: SourceSpectrum object or path to a spectrum file
        """
        if type(spectrum_or_file) in [str]:
            spectrum = SourceSpectrum.from_file(spectrum_or_file)
            
        # the or None enables to switch of the background by setting it to None
        elif isinstance(spectrum_or_file, SourceSpectrum) or None:
            spectrum = spectrum_or_file

        self._background = spectrum

    def set_bg_surface_brightness(self, bg_surface_brightness):
        """
        Set the background surface brightness of the source.

        INPUT:
            bg_surface_brightness: float, background surface brightness
        """
        #if type(bg_surface_brightness_or_file) in [str]:
        #    from .io import read_config
        #    config = read_config("astro")
        #    bg_surface_brightness = config.get("bg_surface_brightness", None)

        #elif isinstance(bg_surface_brightness_or_file, (int, float)) or None:
        self._bg_surface_brightness = bg_surface_brightness

    def has_background(self):
        """ """
        return self._background is not None

    def has_bg_surface_brightness(self):
        """ """
        return self._bg_surface_brightness is not None

    # -------- #
    #  GETTER  #
    # -------- #
    def get_flux(self, which="source"):
        """ """
        if which in ["source", "spectrum"]:
            return self.spectrum._get_arrays(None)
        
        elif which == "background":
            return self.background._get_arrays(None)
        
        else:
            raise ValueError(f"only source and background known. {which=} given")

    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, which="source",  ax=None, **kwargs):
        """
        Show the spectrum or background flux.
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
    def background(self):
        """ spectrum of the background """
        return self._background

    @property
    def bg_surface_brightness(self):
        """ background surface brightness of the source """
        return self._bg_surface_brightness

    @property
    def meta(self):
        """ """
        return self._meta
