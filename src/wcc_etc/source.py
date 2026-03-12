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

class Source(_MetaHolder_):
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
    
    
