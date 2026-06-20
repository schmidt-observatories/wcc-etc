import warnings
import os
import numpy as np

from astropy import units as u
from synphot import SourceSpectrum, SpectralElement, Observation, units as su
from synphot.models import (BlackBodyNorm1D, ConstFlux1D, Empirical1D,
                            PowerLawFlux1D, GaussianFlux1D)

from .meta import _MetaHolder_

__all__ = ["get_scene", "get_scene_from_file", "get_scene_element", "Scene"]

# Magnitude-system resolution: 'abmag' lives in astropy.units, 'vegamag' in
# synphot.units. getattr(u, ...) cannot see VEGAMAG, so resolve explicitly.
_MAGSYS = {"abmag": u.ABmag, "vegamag": su.VEGAMAG}


def _resolve_magsys(magsys):
    """Resolve a magnitude-system spec to an astropy/synphot unit.

    Parameters
    ----------
    magsys : str or Unit
        Case-insensitive ``'abmag'`` or ``'vegamag'``, or a unit object
        (anything exposing ``is_equivalent``), which is returned unchanged.

    Returns
    -------
    Unit
        ``astropy.units.ABmag`` or ``synphot.units.VEGAMAG``.

    Raises
    ------
    ValueError
        If ``magsys`` is an unrecognized string.
    """
    if hasattr(magsys, "is_equivalent"):
        return magsys
    try:
        return _MAGSYS[str(magsys).lower()]
    except KeyError:
        raise ValueError(
            f"unknown magsys {magsys!r}; expected one of {sorted(_MAGSYS)} "
            "or an astropy/synphot magnitude unit"
        )


_VEGA = None


def _get_vega():
    """Lazily load and cache the Vega reference spectrum for VEGAMAG work."""
    global _VEGA
    if _VEGA is None:
        _VEGA = SourceSpectrum.from_vega()
    return _VEGA


# ===================== #
#  Parametric spectra   #
# ===================== #
# Reserved spectrum names that build a synphot SourceSpectrum from parameters
# carried in a SceneElement's meta (see SceneElement.set_spectrum).

def _build_blackbody(meta):
    """Blackbody spectrum. Requires meta['teff'] in K."""
    teff = meta.get("teff")
    if teff is None:
        raise ValueError("blackbody source requires 'teff' (in K)")
    return SourceSpectrum(BlackBodyNorm1D, temperature=teff)


def _build_flat(meta):
    """Flat spectrum. meta['flat_unit'] is 'fnu' (default, AB-flat) or 'flam'.

    The absolute amplitude is arbitrary because the spectrum is normalized to
    a magnitude downstream; only the flat-in-F_nu vs flat-in-F_lambda shape
    matters here.
    """
    flat_unit = str(meta.get("flat_unit", "fnu")).lower()
    if flat_unit == "fnu":
        amplitude = 1 * u.Jy
    elif flat_unit == "flam":
        amplitude = 1 * su.FLAM
    else:
        raise ValueError(f"unknown flat_unit={flat_unit!r}; use 'fnu' or 'flam'")
    return SourceSpectrum(ConstFlux1D, amplitude=amplitude)


def _build_powerlaw(meta):
    """Power-law spectrum: F_lambda proportional to (lambda/lambda_ref)**alpha.

    Requires meta['alpha']; meta['lambda_ref'] defaults to 5500 A. Note synphot's
    PowerLawFlux1D uses (x/x_0)**(-alpha), so the synphot alpha is negated to make
    the public 'alpha' the F_lambda exponent.
    """
    alpha = meta.get("alpha")
    if alpha is None:
        raise ValueError("powerlaw source requires 'alpha'")
    lambda_ref = meta.get("lambda_ref", 5500.0)
    return SourceSpectrum(PowerLawFlux1D, amplitude=1 * su.FLAM,
                          x_0=lambda_ref * u.AA, alpha=-alpha)


def _build_emission(meta):
    """Sum of Gaussian emission lines with absolute integrated flux.

    Requires meta['lines'], a list of dicts with keys 'wave' (A), 'flux'
    (integrated line flux in erg/s/cm^2) and optional 'fwhm' (A, default 2).
    Intended to be used with mag=None so the absolute flux is preserved.
    """
    lines = meta.get("lines")
    if not lines:
        raise ValueError("emission source requires a non-empty 'lines' list")
    spectrum = None
    for line in lines:
        gauss = SourceSpectrum(GaussianFlux1D,
                               total_flux=line["flux"] * u.erg / u.s / u.cm ** 2,
                               mean=line["wave"] * u.AA,
                               fwhm=line.get("fwhm", 2.0) * u.AA)
        spectrum = gauss if spectrum is None else spectrum + gauss
    return spectrum


def _get_unit(unit):
    """Resolve astropy or synphot unit names."""
    if hasattr(unit, "is_equivalent"):
        return unit
    if hasattr(su, str(unit)):
        return getattr(su, str(unit))
    return u.Unit(unit)


def _get_file_column(data, column, label):
    if data.dtype.names is not None:
        if isinstance(column, str):
            return data[column]
        return data[data.dtype.names[int(column)]]

    if isinstance(column, str):
        raise ValueError(f"{label}_column={column!r} requires names=True or a named table")
    return np.asarray(data)[:, int(column)]


def _build_file(meta):
    """Empirical spectrum read from wavelength and flux columns in a text file.

    Requires ``source_file`` (or ``file``/``filename``). By default, column 0 is
    wavelength in Angstrom and column 1 is flux in FLAM. CSV delimiters are
    inferred from a ``.csv`` extension; otherwise whitespace-separated input is
    assumed. Pass string column names with ``names=True`` for named tables.
    """
    filename = meta.get("source_file", meta.get("file", meta.get("filename")))
    if filename is None:
        raise ValueError("file source requires 'source_file' (or 'file'/'filename')")
    delimiter = meta.get("delimiter")
    if delimiter is None and str(filename).lower().endswith(".csv"):
        delimiter = ","

    wave_column = meta.get("wave_column", meta.get("wavelength_column", 0))
    flux_column = meta.get("flux_column", 1)
    names = meta.get("names")
    if names is None:
        names = isinstance(wave_column, str) or isinstance(flux_column, str)
    if names is False:
        names = None

    data = np.genfromtxt(filename, delimiter=delimiter, names=names,
                         comments=meta.get("comments", "#"), dtype=float)
    wave = np.atleast_1d(_get_file_column(data, wave_column, "wave")).astype(float)
    flux = np.atleast_1d(_get_file_column(data, flux_column, "flux")).astype(float)

    if wave.size != flux.size:
        raise ValueError("file source wavelength and flux columns must have the same length")
    finite = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[finite]
    flux = flux[finite]
    if wave.size < 2:
        raise ValueError("file source requires at least two finite wavelength/flux rows")

    order = np.argsort(wave)
    wave = wave[order] * _get_unit(meta.get("wave_unit", "AA"))
    flux = flux[order] * _get_unit(meta.get("flux_unit", "FLAM"))
    return SourceSpectrum(Empirical1D, points=wave, lookup_table=flux)


_SPECTRUM_BUILDERS = {"blackbody": _build_blackbody,
                      "flat": _build_flat,
                      "powerlaw": _build_powerlaw,
                      "emission": _build_emission,
                      "file": _build_file}

# Shape-defining parameters each parametric spectrum accepts. These become
# updatable for a source of that type (see SceneElement.mutable_parameters);
# changing one rebuilds the spectrum. The spectrum *type* itself is not here,
# so it cannot be changed after creation.
_SPECTRUM_PARAMS = {"blackbody": ["teff"],
                    "flat": ["flat_unit"],
                    "powerlaw": ["alpha", "lambda_ref"],
                    "emission": ["lines"],
                    "file": ["source_file", "file", "filename",
                             "wave_column", "wavelength_column", "flux_column",
                             "wave_unit", "flux_unit", "delimiter", "names",
                             "comments"]}

# Top level

def get_scene(name, mag, 
              host = None, host_prop={},
              background = "zodi", background_prop={},
              **kwargs):
    """
    Get a Scene object with a source, an optional host, and an optional background.

    Parameters
    ----------
    name : str
        Name of the source spectrum.
    mag : float
        Magnitude of the source.
    host : str, optional
        Name of the host spectrum. Default is None.
    host_prop : dict, optional
        Properties for the host spectrum. Default is {}.
    background : str, optional
        Name of the background spectrum. Default is "zodi".
    background_prop : dict, optional
        Properties for the background spectrum. Default is {}.
    **kwargs
        Additional keyword arguments passed to get_scene_element for the source.

    Returns
    -------
    Scene
        The initialized Scene object.
    """
    source = get_scene_element(name, mag=mag, **kwargs)
    if host is None and not host_prop:
        host = None
    else:
        host = get_scene_element(host, **host_prop)
        
    if background is None and not background_prop:
        background = None
    else:
        background = get_scene_element(background, **background_prop)

    return Scene(source=source, host=host, background=background)


def get_scene_from_file(source_file, mag=None,
                        wave_column=0, flux_column=1,
                        wave_unit="AA", flux_unit="FLAM",
                        host=None, host_prop=None,
                        background="zodi", background_prop=None,
                        **kwargs):
    """
    Build a Scene whose source spectrum is read from a wavelength/flux file.

    Parameters
    ----------
    source_file : str
        Path to a text file containing wavelength and flux columns.
    mag : float or None, optional
        Magnitude used to normalize the source. If None, the file's absolute
        flux is preserved.
    wave_column, flux_column : int or str, optional
        Wavelength and flux columns. String columns require a named table or
        ``names=True``.
    wave_unit, flux_unit : str or Unit, optional
        Units for the wavelength and flux columns. Defaults are Angstrom and
        FLAM.
    **kwargs
        Additional keyword arguments passed to get_scene. Useful file-reading
        options include ``delimiter``, ``names``, and ``comments``.

    Returns
    -------
    Scene
        A scene with a file-backed source spectrum.
    """
    return get_scene("file", mag=mag,
                     source_file=source_file,
                     wave_column=wave_column,
                     flux_column=flux_column,
                     wave_unit=wave_unit,
                     flux_unit=flux_unit,
                     host=host, host_prop=host_prop,
                     background=background,
                     background_prop=background_prop,
                     **kwargs)


def get_scene_element(element=None, **kwargs):
    """
    Generic top level function to instanciate a scene element.

    Parameters
    ----------
    element : str, dict, or None, optional
        Flexible variable to help instantiating a scene element.
        - If str: name of a specific pre-defined entry, like 'zodi'.
        - If dict: configuration that will supersede default config. It overrides kwargs entries.
        - If None: not used, all rely on kwargs.
        Example equivalent calls:
        - get_scene_element("G5IV", mag=20)
        - get_scene_element({"name": "G5IV", "mag": 20})
        - get_scene_element(name="G5IV", mag=20)
    **kwargs
        Configuration parameters used as SceneElement.from_config(kwargs).

    Returns
    -------
    SceneElement
        The loaded scene element.
    """
    if type(element) is str:
        name = kwargs["name"] = element
    elif type(element) is dict:
        kwargs |= element

    name = kwargs.get("name", kwargs.get("spectrum", None))
    # build default configuration given names.
    ## Zodi
    if name is not None and name in ["zodi", "zodiacal", "background"]:
        default_config = {"mag": 22.5, "surface_brightness": True, "bandpass": "johnson_v"}
    ## anything else
    else:
        default_config = {"mag": 21, "surface_brightness": False, "bandpass": "johnson_v"}

    return SceneElement.from_config((default_config | kwargs))
    


# ============= #
#   Source      #
# ============= # 
def broadcast_mapping(value, ntargets):
    """
    Broadcast a value to a given number of targets.

    Parameters
    ----------
    value : array_like
        Value(s) to be broadcasted.
    ntargets : int
        Number of targets to broadcast to.

    Returns
    -------
    ndarray
        The broadcasted values.
    """
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
    """
    A class representing a single element of a scene (source, host, or background).

    Attributes
    ----------
    spectrum : SourceSpectrum
        The spectrum of the element.
    mag : Quantity
        The magnitude of the element.
    band : SpectralElement
        The bandpass used for magnitude normalization.
    mag_is_surface_brightness : bool
        Whether the magnitude is defined per unit area.
    """
    # NOTE: "spectrum" is deliberately absent — the source *type* is fixed at
    # creation. Type-specific shape parameters are added in mutable_parameters.
    _mutable_parameters = ["mag", "magsys", "bandpass", "surface_brightness"]
    
    def __init__(self, spectrum, mag, 
                 magsys="ABmag", bandpass="johnson_v", 
                 surface_brightness=False,
                 meta={}):
        """
        Initialize a SceneElement.

        Parameters
        ----------
        spectrum : str or SourceSpectrum
            The spectrum or path to a spectrum file.
        mag : float
            The magnitude of the element.
        magsys : str, optional
            The magnitude system (e.g., 'ABmag'). Default is "ABmag".
        bandpass : str or SpectralElement, optional
            The bandpass filter. Default is "johnson_v".
        surface_brightness : bool, optional
            Whether the magnitude is surface brightness (mag/arcsec^2). Default is False.
        meta : dict, optional
            Additional metadata. Default is {}.
        """
        input_parameters = {key:value for key,value in locals().items()
                             if key not in ["self", "meta"]
                                and value is not None and not key.startswith("__")
                           }
        
        super().__init__(meta | input_parameters)
        self.set_spectrum(spectrum)
        
    @classmethod
    def from_config(cls, config):
        """
        Create a SceneElement from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing 'spectrum' (or 'name'), 'mag', and optionally
            'magsys', 'bandpass', 'surface_brightness'.

        Returns
        -------
        SceneElement
            The initialized SceneElement.
        """
        # make sure these keys exist
        if "name" in config:
            config["spectrum"] = config.pop("name")
            
        input_kwargs = {key:config.get(key) for key in ["spectrum", "mag"]}
        input_kwargs |= {key:config.get(key) for key in ["magsys", "bandpass", "surface_brightness"]
                        if key in config} # else default as given by __init__
        
        return cls(meta=config, **input_kwargs)


    # =========== #
    #  methods    #
    # =========== #
    def set_spectrum(self, spec_or_file):
        """
        Generic setter for any source component.

        Parameters
        ----------
        spec_or_file : str or SourceSpectrum or None
            The spectrum, a path to a spectrum file, or a spectral type name.
        """
        
        # make sure you have a spectrum.
        if type(spec_or_file) in [str]:
            if spec_or_file.lower() in _SPECTRUM_BUILDERS:
                # parametric spectrum (blackbody, flat, powerlaw, emission)
                # built from parameters carried in self.meta.
                spectrum = _SPECTRUM_BUILDERS[spec_or_file.lower()](self.meta)
            else:
                if not os.path.isfile(spec_or_file):
                    # may that is a spectral type:
                    from .io import get_any_astro_name
                    spec_or_file = get_any_astro_name(spec_or_file)

                spectrum = SourceSpectrum.from_file(spec_or_file)

        # the or None enables to switch of the host by setting it to None            
        elif isinstance(spec_or_file, SourceSpectrum) or spec_or_file is None:
            spectrum = spec_or_file

        self._spectrum = spectrum

    def update(self, reset=False, **kwargs):
        """
        Update mutable parameters, rebuilding the spectrum if a shape parameter
        changed.

        The spectrum *type* is fixed at creation, so ``spectrum`` is not mutable.
        Updating a type-specific shape parameter (e.g. ``teff``, ``alpha``,
        ``lines``) rebuilds the underlying synphot spectrum; ``mag`` / ``bandpass``
        / etc. take effect at ``get_spectrum`` time and need no rebuild.
        """
        updated = super().update(reset=reset, **kwargs)
        name = self.meta.get("spectrum")
        shape_params = _SPECTRUM_PARAMS.get(name.lower(), []) if isinstance(name, str) else []
        if any(key in shape_params for key in updated):
            self.set_spectrum(name)
        return updated

    # ------- #
    # GETTER  #
    # ------- #
    def get_mag(self, area=None):
        """
        Return the actual magnitude, accounting for the area if mag is a surface brightness.

        Parameters
        ----------
        area : float or Quantity, optional
            The area in arcsec^2. Required if mag_is_surface_brightness is True.

        Returns
        -------
        Quantity
            The magnitude.
        """
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
        """
        Get the spectrum of the element.

        Parameters
        ----------
        apply_mag : bool, optional
            Whether to normalize the spectrum to the stored magnitude. Default is True.
        as_array : bool, optional
            Whether to return the spectrum as a wavelength/flux array pair. Default is False.
        area : float or Quantity, optional
            The area for surface brightness normalization. Default is None.

        Returns
        -------
        SourceSpectrum or tuple
            The spectrum or (wavelength, flux) tuple if as_array is True.
        """
        if isinstance(self.spectrum, SourceSpectrum):
            mag = self.get_mag(area=area) if apply_mag else None
            if apply_mag and mag is not None:
                if mag.unit == su.VEGAMAG:
                    spectrum = self.spectrum.normalize(mag, band=self.band,
                                                       vegaspec=_get_vega())
                else:
                    spectrum = self.spectrum.normalize(mag, band=self.band)
            else:
                # mag is None -> spectrum already carries absolute flux
                # (e.g. emission-line sources); pass it through unchanged.
                spectrum = self.spectrum
        else:
            warnings.warn(f"cannot get the spectrum of the stored {self.spectrum=}")
            return None

        if as_array:
            spectrum = spectrum._get_arrays(None)

        return spectrum
        
    def get_observation(self, band=None, area=None):
        """
        Get a synphot Observation of the element.

        Parameters
        ----------
        band : SpectralElement, optional
            The bandpass filter. Defaults to the element's stored band.
        area : float or Quantity, optional
            The area for surface brightness normalization. Default is None.

        Returns
        -------
        Observation or None
            The synphot Observation.
        """
        if band is None:
            band = self.band

        spectrum = self.get_spectrum(apply_mag=True, as_array=False, area=area)
        if spectrum is None:
            return None
            
        return Observation(spectrum, band, force='extrap')
        
    def show(self, ax=None, apply_mag=True, **kwargs):
        """
        Show the spectrum or host flux.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        apply_mag : bool, optional
            Whether to apply magnitude normalization. Default is True.
        **kwargs
            Keyword arguments passed to ax.plot.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
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
        """
        Parse the magnitude from metadata.

        Returns
        -------
        Quantity
            The parsed magnitude with units.
        """
         # magnitude
        mag = self.meta.get("mag", None)
        if mag is None:
            # explicit None is a valid "no normalization" sentinel (e.g.
            # absolute-flux emission sources); only warn if it is truly missing.
            if "mag" not in self.meta:
                warnings.warn("not mag in self.meta")

        else:
            # make sure mag has the correct units.
            magsys = _resolve_magsys(self.meta.get("magsys", "ABmag"))
            mag = mag * magsys

        return mag

    def _parse_band_(self):
        """
        Parse the bandpass from metadata.

        Returns
        -------
        SpectralElement
            The parsed bandpass.

        Raises
        ------
        ValueError
            If the band metadata is neither a string nor a SpectralElement.
        """
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
    def mutable_parameters(self):
        """
        Parameters that can be updated. Always the common normalization
        parameters, plus the shape parameters specific to this source's
        spectrum type (e.g. ``teff`` for a blackbody). The spectrum *type*
        itself is fixed at creation and is not included.
        """
        params = list(self._mutable_parameters)
        name = self.meta.get("spectrum")
        if isinstance(name, str):
            params += _SPECTRUM_PARAMS.get(name.lower(), [])
        return params

    @property
    def spectrum(self):
        """
        The source spectrum.
        """
        return self._spectrum
        
    @property
    def mag(self):
        """
        The magnitude as an astropy Quantity.
        """
        return self._parse_mag_()

    @property
    def band(self):
        """
        The bandpass as a synphot SpectralElement.
        """
        return self._parse_band_()

    @property
    def mag_is_surface_brightness(self):
        """
        Boolean indicating if the magnitude is a surface brightness.
        """
        return self.meta.get("surface_brightness", False)

class Scene(_MetaHolder_):
    """
    A collection of SceneElements (source, host, background) representing a full observation scene.

    Attributes
    ----------
    source : SceneElement
        The main source of interest.
    host : SceneElement
        The host galaxy or environment.
    background : SceneElement
        The sky background.
    """
    # ZODI TO BE IMPLEMENTED
    # self.background = self.config['zodi']['zodi_mag_r']

    _mutable_parameters = []
    
    def __init__(self, source=None,
                     host=None,
                     background=None,
                     meta={}
                ):
        """
        Initialize a Scene object.

        Parameters
        ----------
        source : SceneElement, optional
            The source element. Default is None.
        host : SceneElement, optional
            The host element. Default is None.
        background : SceneElement, optional
            The background element. Default is None.
        meta : dict, optional
            Additional metadata. Default is {}.
        """
        self._source = source
        self._host = host
        self._background = background

        super().__init__(meta)
        
    @classmethod
    def from_config(cls, config):
        """
        Create a Scene from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing 'source', 'host', and 'background' keys,
            each being a configuration for SceneElement.from_config.

        Returns
        -------
        Scene
            The initialized Scene.
        """
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
        """
        Reset each element in the scene.
        """
        # reset each element
        _ = self.call_down("reset", which="all")
        
    def update(self, reset=False, **kwargs):
        """
        Update scene elements using double-underscore prefixed keywords.

        Parameters
        ----------
        reset : bool, optional
            Not currently used. Default is False.
        **kwargs
            Keywords like 'source__mag=20' or 'background__mag=22'.

        Returns
        -------
        list
            List of updated keys.
        """
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
        """
        Get specified scene elements.

        Parameters
        ----------
        which : str or list, optional
            Which elements to get ('*', 'all', 'source', 'host', 'background'). Default is "*".
        as_dict : bool, optional
            Whether to return elements as a dictionary. Default is False.

        Returns
        -------
        list or dict
            The requested scene elements.
        """
        if which in ["*", "all"]:
            which = [element_ for element_ in self.element_names if getattr(self, element_) is not None]
        else:
            which = np.atleast_1d(which)

        values = [element_obj for which_ in self.element_names if which_ in which and (element_obj:=getattr(self, which_)) is not None]
        if as_dict:
            return dict(zip(which, values))
            
        return values

    def get_mag(self, area=None, which="*", as_dict=False):
        """
        Get magnitude for specified scene elements.

        Parameters
        ----------
        area : float or Quantity, optional
            The area for surface brightness normalization. Default is None.
        which : str or list, optional
            Which elements to get magnitudes for. Default is "*".
        as_dict : bool, optional
            Whether to return as a dictionary. Default is False.

        Returns
        -------
        list or dict
            The magnitudes.
        """
        return self.call_down("get_mag", area=area, which=which, as_dict=as_dict)

    def get_spectrum(self, area=None, apply_mag=True, which="*", as_dict=False):
        """
        Get spectrum for specified scene elements.

        Parameters
        ----------
        area : float or Quantity, optional
            The area for surface brightness normalization. Default is None.
        apply_mag : bool, optional
            Whether to normalize the spectrum. Default is True.
        which : str or list, optional
            Which elements to get spectra for. Default is "*".
        as_dict : bool, optional
            Whether to return as a dictionary. Default is False.

        Returns
        -------
        list or dict
            The spectra.
        """
        return self.call_down("get_spectrum", area=area, apply_mag=apply_mag, 
                              which=which, as_dict=as_dict)
    
    def get_observation(self, band=None, area=None, which="*", as_dict=False):
        """
        Get observation for specified scene elements.

        Parameters
        ----------
        band : SpectralElement, optional
            The bandpass filter. Default is None.
        area : float or Quantity, optional
            The area for surface brightness normalization. Default is None.
        which : str or list, optional
            Which elements to get observations for. Default is "*".
        as_dict : bool, optional
            Whether to return as a dictionary. Default is False.

        Returns
        -------
        list or dict
            The observations.
        """
        return self.call_down("get_observation", band=band, area=area, which=which, as_dict=as_dict)
        
    def has_source(self):
        """
        Check if the scene has a source element.

        Returns
        -------
        bool
        """
        return self._source is not None

    def has_host(self):
        """
        Check if the scene has a host element.

        Returns
        -------
        bool
        """
        return self._host is not None

    def has_background(self):
        """
        Check if the scene has a background element.

        Returns
        -------
        bool
        """
        return self._background is not None
    
    # -------- #
    #  GETTER  #
    # -------- #
    
    # ----------- #
    #  Internal   #
    # ----------- #
    def call_down(self, what, mapargs=None, allow_call=True, which="*", as_dict=False, **kwargs):
        """
        Call a method on each target element in the collection.

        Parameters
        ----------
        what : str
            The name of the attribute or method to access on each element.
        mapargs : array_like, optional
            Positional arguments to be mapped to each element call. Default is None.
        allow_call : bool, optional
            If True and the attribute is callable, call it. Default is True.
        which : str or list, optional
            Which elements to include. Default is "*".
        as_dict : bool, optional
            Whether to return results as a dictionary. Default is False.
        **kwargs
            Keyword arguments passed to the method call.

        Returns
        -------
        list or dict
            The results from each element.
        """
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
        """
        The source SceneElement.
        """
        return self._source

    @property
    def host(self):
        """
        The host SceneElement.
        """
        return self._host

    @property
    def background(self):
        """
        The background SceneElement.
        """
        return self._background

    @property
    def element_names(self):
        """
        Names of the potential elements in a scene.
        """
        return ["source", "host", "background"]

    @property
    def meta(self):
        """
        Combined metadata for the scene and its elements.
        """
        return self._meta | {element_name: element.meta
                                 for element_name in self.element_names
                                 if (element := getattr(self,element_name)) is not None
                            }
    @property
    def mutable_parameters(self):
        """
        List of parameters that can be updated.
        """
        return self._mutable_parameters +  [f"{element_name}__{k}"
                                                for element_name in self.element_names
                                                if (element := getattr(self,element_name)) is not None
                                                for k in element.mutable_parameters  
                                            ]
