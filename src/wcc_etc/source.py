
from synphot import units, SourceSpectrum

# ============= #
#   Source      #
# ============= # 

class Source():
    """ """
    # ZODI TO BE IMPLEMENTED
    # self.bg_surface_brightness = self.config['zodi']['zodi_mag_r']

    
    def __init__(self, spectrum, background=None):
        """ """
        self.set_spectrum(spectrum)
        self.set_background(background)

        self._meta = {}
        
    @classmethod
    def from_file(cls, filepath, bkgd_file=None):
        """ """
        spectrum = SourceSpectrum.from_file(filepath)
        if bkgd_file is not None:
            background = SourceSpectrum.from_file(bkgd_file)
        else:
            background = None
            
        return cls(spectrum=spectrum, background=background)

    @classmethod
    def from_config(cls, config):
        """ """
        spec_or_file = config.get("spectrum")
        background_or_file = config.get("background", None)
        return cls(spectrum=spec_or_file, background=background_or_file)
        
    # ============= #
    #   methods     #
    # ============= #
    def set_spectrum(self, spectrum_or_file):
        """ """
        if type(spectrum_or_file) in [str]:
            spectrum = SourceSpectrum.from_file(spectrum_or_file)
            
        # the or None enables to switch of the background by setting it to None            
        elif isinstance(spectrum_or_file, SourceSpectrum) or None:
            spectrum = spectrum_or_file

        self._spectrum = spectrum

    def set_background(self, spectrum_or_file):
        """ """
        if type(spectrum_or_file) in [str]:
            spectrum = SourceSpectrum.from_file(spectrum_or_file)
            
        # the or None enables to switch of the background by setting it to None
        elif isinstance(spectrum_or_file, SourceSpectrum) or None:
            spectrum = spectrum_or_file

        self._background = spectrum

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
        
        elif which == "background":
            return self.background._get_arrays(None)
        
        else:
            raise ValueError(f"only source and background known. {which=} given")

    # -------- #
    # PLOTTER  #
    # -------- #
    def show(self, which="source",  ax=None, **kwargs):
        """ """
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
    def meta(self):
        """ """
        return self._meta
