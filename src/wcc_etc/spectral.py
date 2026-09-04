"""Source-weighted spectral quantities that set the wavelength(s) a PSF is rendered at.

The diffraction scale is linear in wavelength, so the wavelength used to render a
PSF has to reflect the *source* as well as the bandpass. The pivot wavelength
(``Sensor.wavelength``) is a property of the throughput curve alone: it is the
same 582 nm for an O5V and an M5V star observed through the WCC broad band, even
though the photons those two stars actually deliver through that band average
541 nm and 716 nm respectively. This module supplies the photon-weighted
alternatives:

- :func:`effective_wavelength` -- one representative wavelength,
  ``<lambda> = int(lambda S T dlambda) / int(S T dlambda)``,
  with ``S`` the source photon flux density and ``T`` the total throughput.
- :func:`photon_weighted_subbands` -- ``n_sub`` (wavelength, weight) samples for a
  polychromatic PSF coadd (see :class:`wcc_etc.psfsim.PolychromaticPSF`).

All integrals use :func:`scipy.integrate.trapezoid`; ``numpy.trapz`` is gone in
numpy 2.x, which is what CI runs.
"""

import astropy.units as u
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

__all__ = [
    "photon_weights",
    "effective_wavelength",
    "photon_weighted_subbands",
]

#: Photon flux density unit synphot spectra are sampled in for weighting.
PHOTON_FLUX_UNIT = "photlam"


def build_waveset(bandpass, spectrum=None):
    """Union of the bandpass and (in-band) spectrum wavesets, in Angstrom.

    The bandpass sets the range: throughput is zero outside it, so nothing the
    spectrum does out there matters. Inside it, the spectrum's own samples are
    folded in so that emission lines and other sharp features are not skipped by
    a coarse filter grid.

    Parameters
    ----------
    bandpass : SpectralElement
        Total throughput (filter + QE + optics).
    spectrum : SourceSpectrum, optional
        Source spectrum. Its waveset is used only where it overlaps the bandpass.

    Returns
    -------
    Quantity
        Sorted, deduplicated wavelength grid in Angstrom.
    """
    if bandpass.waveset is None:
        raise ValueError("bandpass has no waveset; cannot build a weighting grid.")
    wave = np.asarray(bandpass.waveset.to_value(u.AA), dtype=float)
    if spectrum is not None and spectrum.waveset is not None:
        src = np.asarray(spectrum.waveset.to_value(u.AA), dtype=float)
        src = src[(src >= wave.min()) & (src <= wave.max())]
        if src.size:
            wave = np.union1d(wave, src)
    if wave.size < 2:
        raise ValueError(
            f"weighting grid needs at least 2 wavelengths, got {wave.size}."
        )
    return wave * u.AA


def photon_weights(bandpass, spectrum=None, wave=None):
    """Photon weight ``S(lambda) T(lambda)`` on a common wavelength grid.

    The weights are returned unnormalized -- every consumer here divides by its
    own integral, so the absolute scale (and hence the source magnitude) drops
    out. That is why callers can pass an un-normalized spectrum.

    Parameters
    ----------
    bandpass : SpectralElement
        Total throughput.
    spectrum : SourceSpectrum, optional
        Source spectrum. ``None`` means "flat in photons", which reduces the
        weighting to the throughput curve alone.
    wave : Quantity or array_like, optional
        Wavelength grid (Angstrom if unitless). Defaults to
        :func:`build_waveset`.

    Returns
    -------
    wave : Quantity
        The wavelength grid.
    weight : ndarray
        ``S(lambda) * T(lambda)``, clipped at zero.

    Raises
    ------
    ValueError
        If the bandpass and spectrum have no wavelength where the product is
        positive.
    """
    if wave is None:
        wave = build_waveset(bandpass, spectrum)
    elif not isinstance(wave, u.Quantity):
        wave = np.asarray(wave, dtype=float) * u.AA

    throughput = np.asarray(bandpass(wave).value, dtype=float)
    if spectrum is None:
        flux = np.ones_like(throughput)
    else:
        flux = np.asarray(spectrum(wave, flux_unit=PHOTON_FLUX_UNIT).value, dtype=float)

    weight = np.clip(throughput * flux, 0.0, None)
    weight[~np.isfinite(weight)] = 0.0
    if not np.any(weight > 0):
        raise ValueError(
            "bandpass x spectrum has no positive photon flux on the weighting "
            "grid; check that the throughput and the source overlap in wavelength."
        )
    return wave, weight


def effective_wavelength(bandpass, spectrum=None, wave=None):
    """Photon-weighted effective wavelength of a bandpass times a source SED.

    ``lambda_eff = int(lambda S T dlambda) / int(S T dlambda)``, with ``S`` the
    source photon flux density and ``T`` the total throughput. This is the
    wavelength to render a monochromatic PSF at: unlike the pivot wavelength it
    moves with the source colour, and the Airy scale is linear in wavelength.

    Parameters
    ----------
    bandpass : SpectralElement
        Total throughput.
    spectrum : SourceSpectrum, optional
        Source spectrum. ``None`` gives the throughput-weighted average
        wavelength (``avgwave``), *not* the pivot wavelength.
    wave : Quantity or array_like, optional
        Wavelength grid. Defaults to :func:`build_waveset`.

    Returns
    -------
    Quantity
        The effective wavelength, in nm.

    Examples
    --------
    Through the WCC broad band the same filter gives very different effective
    wavelengths for an O5V and an M5V source (541 nm vs 716 nm), while the pivot
    wavelength is 582 nm for both.
    """
    wave, weight = photon_weights(bandpass, spectrum, wave=wave)
    lam = wave.to_value(u.AA)
    denominator = trapezoid(weight, lam)
    if denominator <= 0:
        raise ValueError("photon weight integrates to zero; cannot form a mean.")
    return (trapezoid(lam * weight, lam) / denominator * u.AA).to(u.nm)


def photon_weighted_subbands(bandpass, spectrum=None, n_sub=7, wave=None):
    """Split a band into ``n_sub`` sub-bands of equal photon weight.

    Each sub-band contributes one render wavelength (its photon-weighted
    centroid) and one coadd weight (its share of the in-band photons). The
    sub-band *edges* are placed at equal quantiles of the cumulative photon
    weight, so samples cluster where the source actually delivers light rather
    than being spread evenly over a filter's dead tails. The weights therefore
    come out close to ``1/n_sub`` each, which also means no render is wasted on a
    negligible sub-band.

    ``n_sub=1`` reduces exactly to :func:`effective_wavelength` with weight 1.

    Parameters
    ----------
    bandpass : SpectralElement
        Total throughput.
    spectrum : SourceSpectrum, optional
        Source spectrum; ``None`` weights by throughput alone.
    n_sub : int, optional
        Number of sub-bands. 5-9 is enough for a broad optical band; the cost of
        a polychromatic PSF is linear in this. Default 7.
    wave : Quantity or array_like, optional
        Wavelength grid. Defaults to :func:`build_waveset`.

    Returns
    -------
    wavelengths : Quantity
        ``n_sub`` increasing sub-band centroids, in nm.
    weights : ndarray
        ``n_sub`` coadd weights summing to 1.
    """
    n_sub = int(n_sub)
    if n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub}.")

    wave, weight = photon_weights(bandpass, spectrum, wave=wave)
    lam = wave.to_value(u.AA)

    # Cumulative photon weight and cumulative first moment, so that both the
    # sub-band weight and its centroid follow from interpolation at the edges.
    cum_weight = cumulative_trapezoid(weight, lam, initial=0.0)
    cum_moment = cumulative_trapezoid(lam * weight, lam, initial=0.0)
    total = cum_weight[-1]
    if total <= 0:
        raise ValueError("photon weight integrates to zero; cannot form sub-bands.")

    edges = np.interp(np.linspace(0.0, 1.0, n_sub + 1), cum_weight / total, lam)
    edges[0], edges[-1] = lam[0], lam[-1]  # keep the full support, tails included

    per_band_weight = np.diff(np.interp(edges, lam, cum_weight))
    per_band_moment = np.diff(np.interp(edges, lam, cum_moment))
    if np.any(per_band_weight <= 0):
        raise ValueError(
            f"n_sub={n_sub} produced an empty sub-band; use a smaller n_sub."
        )

    centroids = per_band_moment / per_band_weight
    return (centroids * u.AA).to(u.nm), per_band_weight / total
