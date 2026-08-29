"""Legacy module-level ETC helpers, kept for backward compatibility.

The `WCCETC` class that used to live here was a parallel implementation of the
ETC, superseded by `Simulation` + `get_scene`; it was never constructed
anywhere and has been removed. What remains are the standalone helpers, most
importantly `get_wcc_snr_and_simulation`, which is itself a thin wrapper over
the modern stack.
"""


import astropy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import synphot
from astropy import units as u
from synphot import SourceSpectrum, units
from synphot.models import BlackBodyNorm1D

from .scene import get_scene
from .simulation import Simulation


def phot_error(star_ADU, n_pix, n_b, sky_ADU, dark, read, gain=1.0):
    """
    Photometric error

    INPUT:
        star_ADU - ADU counts from the star
        n_pix - number of pixels in the aperture
        n_b - number of background pixels
        sky_ADU - ADU counts from the sky
        dark_ADU - ADU counts from the dark current
        read_ADU - ADU counts from the read noise
        gain - gain of the detector (default is 1.0) in e/ADU

    OUTPUT:
        noise - calculated noise in ADU counts
    """
    noise = (
        np.sqrt(
            gain * star_ADU
            + n_pix
            * (
                (1.0 + n_pix / n_b)
                * (gain * sky_ADU + dark + read**2.0 + (gain * 0.289) ** 2.0)
            )
        )
        / gain
    )
    return noise


def calculate_bg_normalization_magnitude(bg_surface_brightness, psf_area):
    """
    Convert the Background Surface Brightness into the total magnitude given the PSF area (in arcseconds squared)
    The area needs to be in square arcseconds since this the typical definition of Surface Brightness is in units
    of magnitudes per arcseconds^2
    :return: None
    """
    bg_magnitude = bg_surface_brightness - 2.5 * np.log10(psf_area)
    return bg_magnitude


def get_wcc_snr_and_simulation(
    mag,
    texp,
    source_type="pickles",
    spt="",
    teff=None,
    source_file=None,
    wave_column=0,
    flux_column=1,
    wave_unit="AA",
    flux_unit="FLAM",
    source_bandpass="johnson_r",
    source_mag_type="Vega",
    sensor_and_filter="zwo:r",
    bg_surface_brightness=22.5,
    bg_bandpass="johnson_r",
    read_noise=None,
    jitter_sigma=10,
    r_aper_mas=70,
):
    """
    Get the SNR for a given set of parameters.

    INPUT:
        source_type: Type of the source (e.g. 'pickles', 'blackbody', 'file')
        spt: spectral type, only used for source_type == 'pickles'
        teff: Effective temperature in K, only used if source_type=='blackbody'
        source_file: Wavelength/flux spectrum file, only used if source_type=='file'
        wave_column: Wavelength column in source_file, only used if source_type=='file'
        flux_column: Flux column in source_file, only used if source_type=='file'
        wave_unit: Wavelength unit in source_file, only used if source_type=='file'
        flux_unit: Flux unit in source_file, only used if source_type=='file'

    EXAMPLE:
        get_wcc_snr(25.4,60)
    """
    if source_type == "pickles":
        scene = get_scene(
            spt,
            mag=mag,
            host=None,
            background="zodi",
            bandpass=source_bandpass,
            background_prop={"bandpass": bg_bandpass, "mag": bg_surface_brightness},
        )
    elif source_type == "blackbody":
        scene = get_scene(
            "blackbody",
            mag=mag,
            teff=teff,
            host=None,
            background="zodi",
            bandpass=source_bandpass,
            background_prop={"bandpass": bg_bandpass, "mag": bg_surface_brightness},
        )
    elif source_type == "file":
        scene = get_scene(
            "file",
            mag=mag,
            source_file=source_file,
            wave_column=wave_column,
            flux_column=flux_column,
            wave_unit=wave_unit,
            flux_unit=flux_unit,
            host=None,
            background="zodi",
            bandpass=source_bandpass,
            background_prop={"bandpass": bg_bandpass, "mag": bg_surface_brightness},
        )
    else:
        raise ValueError("Unknown source type")
    simu = Simulation.from_sensor_and_scene(sensor_and_filter, scene)
    simu.update(r_aper_mas=r_aper_mas)
    if read_noise is not None:
        simu.update(read_noise=read_noise)
    simu.update(jitter_sigma=jitter_sigma)

    snr = simu.get_snr(texp)["snr"]
    return snr, simu


def get_blackbody_flux(
    w, teff, mag, unit="FLAM", filter="johnson_v", plot=False, ax=None
):
    """
    Get a blackbody spectrum normalized to a given magnitude in the V band.

    INPUT:
        w in A
        teff in K
        mag in V band magnitude

    OUTPUT:
        f in erg/s/cm^2/Å if unit=='FLAM', else W/m^2/μm

    EXAMPLE:
        # Sun
        w = np.linspace(3000,10000,10000)
        Teff = 5777
        teff in K
        mag in V band magnitude

    OUTPUT:
        f in erg/s/cm^2/Å if unit=='FLAM', else W/m^2/μm

    EXAMPLE:
        # Sun
        w = np.linspace(3000,10000,10000)
        Teff = 5777
        mag = -26.8
        f = get_blackbody_spectrum(w,Teff, mag,filter='johnson_v',plot=True)
    """
    sp = SourceSpectrum(BlackBodyNorm1D, temperature=teff)
    from .io import resolve_bandpass
    bp = resolve_bandpass(filter)
    vega = SourceSpectrum.from_vega()  # For unit conversion
    sp_norm = sp.normalize(mag * units.VEGAMAG, bp, vegaspec=vega)

    if unit == "FLAM":
        unit = "erg/s/cm2/A"
        f = synphot.units.convert_flux(w, sp_norm(w), "FLAM")
    elif unit == "W/m2/micron":
        unit = "W/m2/micron"
        f = synphot.units.convert_flux(w, sp_norm(w), "FLAM").value * 10
    else:
        print("Unknown unit")
        return None
    if plot:
        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        ax.plot(w, f, label=unit)
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.set_ylabel(f"Flux [{unit}]")
        ax.legend()
    return f


def calc_moon_scatter_countrate_per_pixel(
    mag,
    P_out,
    P_in=1,
    pixel_size_micron=3.76,
    filter="johnson_v",
    obsbandpass="sony:g",
    scale_factor=1.3,
    plot=True,
    verbose=True,
):
    """
    Calculate the scatter countrate in a pixel

    INPUT:
        mag - magnitude of moon
        P_out - scattered power in W/m2
        P_in - incoming power in W/m2, default is 1 W/m2
        pixel_size_micron - size of the pixel in microns
        wstart - start wavelength in micron
        wend - end wavelength in micron

    OUTPUT:
        number of photons_per_pixel_per_s

    EXAMPLE:
        calc_moon_scatter_countrate_per_pixel(-12.8,1.8e-11*1e6)
    """

    def integrate_flux(ww, ff):
        return trapezoid(ff, ww)

    pixel_size_m = pixel_size_micron * 1e-6

    w = np.linspace(1000, 20000, 100000) * u.AA
    f_moon = get_blackbody_flux(
        w, 5777 * u.K, mag, unit="W/m2/micron", filter=filter, plot=False
    )

    # hack way to get the bandpass:
    scene = get_scene(
        name="G5V",
        mag=25.4,
        host=None,
        background="zodi",
        bandpass="johnson_r",
        background_prop={"bandpass": "johnson_r", "mag": 22.5},
    )
    simu = Simulation.from_sensor_and_scene(obsbandpass, scene)

    # this includes all of the mirror losses, which I should not have, just QE and filter response
    _, throughput_bandpass = simu.sensor.bandpass._get_arrays(wavelengths=w)
    f_moon_band = (
        f_moon * throughput_bandpass * scale_factor
    )  # hack scale factor to account for using total bandpass for now

    P_moon_total = integrate_flux(w.value / 10000, f_moon)  # W/m2
    P_moon_in_band = integrate_flux(w.value / 10000, f_moon_band)  # W/m2

    scatter_frac = P_out / P_in
    P_moon_in_band_scatter = P_moon_in_band * scatter_frac  # W/m2
    P_moon_in_band_scatter_per_pix = P_moon_in_band_scatter * (
        pixel_size_m**2
    )  # W/pixel
    w_mean = simu.sensor.bandpass.avgwave().value / 10000  # micron
    energy_photon = const.h.value * const.c.value / (w_mean * 1e-6)  # J
    phot_per_s_moon_in_band_scatter_per_pix = (
        P_moon_in_band_scatter_per_pix / energy_photon
    )  # phot/s/pix

    if verbose:
        print("Scatter frac: ", scatter_frac)
        print("P_moon_in_band: ", P_moon_in_band, "W/m2")
        print("P_moon_in_band_scatter: ", P_moon_in_band_scatter, "W/m2")
        print(
            "P_moon_in_band_scatter_per_pix: ", P_moon_in_band_scatter_per_pix, "W/pix"
        )
        print(
            "phot_per_s_moon_in_band_scatter_per_pix: ",
            phot_per_s_moon_in_band_scatter_per_pix,
            "phots/s/pix",
        )

    if plot:
        fig, ax = plt.subplots(dpi=200)
        ax.plot(
            w / 10000,
            f_moon,
            label="Moon, total, Blackbody: $T_{eff}=5777$K"
            + ", $V$ mag={}, Int. flux={:0.5f}W/m2".format(mag, P_moon_total),
        )
        ax.plot(
            w / 10000,
            f_moon_band,
            label="Moon, in-band, Blackbody: $T_{eff}=5777$K"
            + ", $V$ mag={}, Int. flux={:0.5f}W/m2".format(mag, P_moon_in_band),
        )
        ax.axvline(
            w_mean,
            color="gray",
            linestyle="--",
            label="Bandpass mean wavelength: {:0.2f} micron".format(w_mean),
        )
        ax.set_xlabel("Wavelength (micron)")
        ax.set_ylabel("Flux (W/m2/micron)")
        ax.minorticks_on()
        ax.legend(loc="upper right")
        ax.set_title(
            "Band: {}\nScale Factor={}\nPhoton flux: {:0.5f}photons/s/pixel".format(
                obsbandpass, scale_factor, phot_per_s_moon_in_band_scatter_per_pix
            )
        )

    return phot_per_s_moon_in_band_scatter_per_pix


if __name__ == "__main__":
    print("Main")
