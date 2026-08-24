import os
from dataclasses import dataclass
from typing import Optional

import astropy.io.fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import tifffile
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import (
    HistEqStretch,
    LogStretch,
)
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import shift, zoom
from scipy.signal import fftconvolve

from . import airy
from .radial_data import radial_data
from .simulation import Simulation

# Bundled Zemax Huygens defocus PSF data (monochromatic, 500 nm, 4 um spacing)
_PSF_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "psfs")
DEFOCUS_1WAVE_PATH = os.path.join(
    _PSF_DATA_DIR, "CAD_1-waves-defocus_500nm_Huygens-PSF-Data_Linear.txt"
)
DEFOCUS_2WAVE_PATH = os.path.join(
    _PSF_DATA_DIR, "CAD_2-waves-defocus_500nm_Huygens-PSF-Data_Linear.txt"
)


@dataclass
class DetectorPSFContext:
    """Detector + optics parameters a PSF source needs to render onto the grid."""

    npix: int
    pixel_size_um: float
    plate_scale_mas: float
    wavelength_m: float
    diameter_m: float
    fnum: float
    jitter_sigma_mas: float = 0.0
    center: Optional[tuple] = None
    oversample: int = 11


def normalize_psf(psf):
    """Clip negatives and normalize a 2D PSF so it sums to 1."""
    psf = np.clip(np.asarray(psf, dtype=float), 0.0, None)
    total = psf.sum()
    if total <= 0:
        raise ValueError("PSF total is non-positive; cannot normalize.")
    return psf / total


def center_crop_or_pad(img, npix, fill=0.0):
    """Center-crop or zero-pad a 2D array to (npix, npix), preserving the center."""
    img = np.asarray(img, dtype=float)
    ny, nx = img.shape
    out = np.full((npix, npix), fill, dtype=float)
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    y0 = int(np.floor(cy - (npix - 1) / 2.0 + 0.5))
    x0 = int(np.floor(cx - (npix - 1) / 2.0 + 0.5))
    y1, x1 = y0 + npix, x0 + npix
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(ny, y1), min(nx, x1)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def recenter(psf, center):
    """Sub-pixel shift a grid-centered PSF so its center lands at (cx, cy)."""
    npix = psf.shape[0]
    grid_center = (npix - 1) / 2.0
    cx, cy = float(center[0]), float(center[1])
    return shift(
        psf,
        shift=(cy - grid_center, cx - grid_center),
        order=3,
        mode="constant",
        cval=0.0,
    )


class PSFSource:
    """Base class: render a normalized (sum=1) PSF onto a DetectorPSFContext."""

    def render(self, ctx):
        raise NotImplementedError("Subclasses must implement render(ctx).")

    def cache_key(self):
        """
        Hashable key identifying this PSF *source's* parameters (not the
        rendered output). The render also depends on the DetectorPSFContext
        (wavelength, optics, jitter, npix, oversample), so a render cache must
        combine this key with the context and/or be invalidated when the
        simulation state changes — see Simulation._image_render_bundle.
        """
        return (type(self).__name__,)


class AiryPSF(PSFSource):
    """Diffraction-limited Airy PSF rendered on the detector grid (default)."""

    def render(self, ctx):
        psf, _ = airy.render_detector_psf(
            wavelength=ctx.wavelength_m,
            fnum=ctx.fnum,
            D=ctx.diameter_m,
            pixel_size=ctx.pixel_size_um,
            jitter_sigma_mas=ctx.jitter_sigma_mas,
            n_pixels=ctx.npix,
            oversample=ctx.oversample,
        )
        if psf.shape != (ctx.npix, ctx.npix):  # render_detector_psf forces odd n_pixels
            psf = center_crop_or_pad(psf, ctx.npix)
        if ctx.center is not None:
            psf = recenter(psf, ctx.center)
        return normalize_psf(psf)


def load_huygens_psf(path, encoding="utf-16"):
    """Load a Zemax Huygens PSF text file into a 2D float array of intensities."""
    with open(path, encoding=encoding) as fh:
        rows = [
            ln
            for ln in fh.read().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    data = np.array([[float(x) for x in ln.split()] for ln in rows], dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError(f"Huygens PSF file did not parse to a 2D array: {path}")
    return data


class _ResampledPSF(PSFSource):
    """A PSF defined as a sampled image at a known source pixel scale (microns)."""

    def __init__(self, data, src_um_per_pix):
        self._data = np.asarray(data, dtype=float)
        if self._data.ndim != 2:
            raise ValueError(f"PSF data must be 2D, got shape {self._data.shape}")
        self.src_um_per_pix = float(src_um_per_pix)

    def render(self, ctx):
        zoom_factor = self.src_um_per_pix / ctx.pixel_size_um
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be positive (check pixel sizes).")
        zoomed = zoom(self._data, zoom_factor, order=1, mode="constant", cval=0.0)
        psf = normalize_psf(center_crop_or_pad(zoomed, ctx.npix))
        if ctx.jitter_sigma_mas and ctx.jitter_sigma_mas > 0:
            psf = normalize_psf(
                apply_jitter(psf, ctx.jitter_sigma_mas, ctx.plate_scale_mas)
            )
        if ctx.center is not None:
            psf = normalize_psf(recenter(psf, ctx.center))
        return psf

    def cache_key(self):
        return (type(self).__name__, self.src_um_per_pix, id(self._data))


class DefocusPSF(_ResampledPSF):
    """A defocused PSF loaded from a Zemax Huygens text file."""

    def __init__(self, path, src_um_per_pix=4.0, encoding="utf-16"):
        super().__init__(load_huygens_psf(path, encoding), src_um_per_pix)
        self.path = path

    def cache_key(self):
        return (type(self).__name__, self.src_um_per_pix, self.path)


class CustomPSF(_ResampledPSF):
    """A custom PSF from an ndarray or a Huygens-format text file (future hook)."""

    def __init__(self, source, src_um_per_pix, encoding="utf-16"):
        data = (
            source
            if isinstance(source, np.ndarray)
            else load_huygens_psf(source, encoding)
        )
        super().__init__(data, src_um_per_pix)


def saturation_mask_from_image_e(sensor, image_e):
    """Boolean mask of pixels at/over the ADC full scale or the full well.

    Parameters
    ----------
    sensor : Sensor
        Provides gain, adc_max, and (optionally) meta['well_depth'].
    image_e : ndarray
        Per-pixel charge in electrons (per frame for saturation tests).

    Returns
    -------
    ndarray of bool
        True where (image_e / gain) >= adc_max, OR image_e >= well_depth
        when a well_depth is configured.
    """
    gain = sensor.gain.to(u.electron / u.ct).value
    mask = (image_e / gain) >= sensor.adc_max.to(u.ct).value
    well_depth = sensor.meta.get("well_depth")
    if well_depth is not None:
        mask = mask | (image_e >= well_depth)
    return mask


@dataclass
class SimulatedImage:
    """Result of an ImageSimulator.simulate() call."""

    image_e: np.ndarray  # detector image in electrons (noisy unless add_noise=False)
    image_clean: np.ndarray  # noiseless electrons
    saturation_mask: np.ndarray  # bool: pixels at/over adc_max or full well
    gain: float  # electron / ct
    bias_level: float  # ct
    npix: int
    pixel_scale_mas: float
    psf: "PSFSource"

    def to_adu(self):
        """Electrons -> ADU via gain, plus the bias level."""
        return self.image_e / self.gain + self.bias_level

    def to_fitsimg(self):
        """Wrap the electron image in a FitsImg for photometry/plotting."""
        return FitsImg(data=self.image_e)

    def plot_image(self, backend="mpl", **kwargs):
        """Plot this image (single panel). backend='mpl' or 'bokeh'."""
        from . import plotting

        if backend == "mpl":
            return plotting.plot_image_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_image_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")

    def plot_image_row(self, backend="mpl", **kwargs):
        """Plot the 3-panel row (PSF+noise, PSF, saturation mask)."""
        from . import plotting

        if backend == "mpl":
            return plotting.plot_image_row_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_image_row_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")

    def plot_radial(self, backend="mpl", **kwargs):
        """Plot the azimuthally-averaged radial profile."""
        from . import plotting

        if backend == "mpl":
            return plotting.plot_radial_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_radial_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")

    def plot_encircled_energy(self, backend="mpl", **kwargs):
        """Plot the encircled-energy curve (marks the 90% EE radius by default;
        pass ee_target=None to disable or another fraction to override)."""
        from . import plotting

        if backend == "mpl":
            return plotting.plot_encircled_energy_mpl(self, **kwargs)
        if backend == "bokeh":
            return plotting.plot_encircled_energy_bokeh(self, **kwargs)
        raise ValueError("backend must be 'mpl' or 'bokeh'")


class ImageSimulator:
    """Render a point source onto a detector grid with noise, driven by an ETC Simulation."""

    def __init__(self, simulation, npix=300, oversample=11):
        """Build from a prebuilt Simulation; npix is the (square) detector grid size."""
        self.sim = simulation
        self.npix = int(npix)
        self.oversample = int(oversample)

    @classmethod
    def from_sensor_and_scene(cls, sensor, scene, npix=300, oversample=11):
        """Build an ImageSimulator from a sensor name (e.g. 'sony:r') and a Scene."""
        sim = Simulation.from_sensor_and_scene(sensor, scene)
        return cls(sim, npix=npix, oversample=oversample)

    @classmethod
    def from_sensorfilter(cls, sensorfilter, scene, npix=300, oversample=11):
        """Build an ImageSimulator from a sensorfilter label (e.g. 'zwo:r+1').

        Sets _default_psf on the underlying Simulation (accessible via
        self.sim._default_psf) based on the sensor's focus_level in sensor_info.
        Note: simulate() still requires an explicit psf= argument; _default_psf
        is used by Simulation.get_image_snr and get_image_exptime_for_snr.
        """
        sim = Simulation.from_sensorfilter(sensorfilter, scene)
        return cls(sim, npix=npix, oversample=oversample)

    def _context(self, jitter_sigma_mas=None, center=None):
        sim = self.sim
        plate_scale_mas = (
            sim.sensor.get_plate_scale(sim.telescope).to("arcsec/pix").value * 1000.0
        )
        if jitter_sigma_mas is None:
            jitter_sigma_mas = sim.telescope.jitter_sigma.to("mas").value
        return DetectorPSFContext(
            npix=self.npix,
            pixel_size_um=sim.sensor.pixel_size.value,
            plate_scale_mas=plate_scale_mas,
            wavelength_m=sim.sensor.wavelength.to("m").value,
            diameter_m=sim.telescope.diameter_primary.to("m").value,
            fnum=sim.telescope.f_num,
            jitter_sigma_mas=jitter_sigma_mas,
            center=center,
            oversample=self.oversample,
        )

    def simulate(
        self,
        time=None,
        psf=None,
        jitter_sigma_mas=None,
        center=None,
        add_noise=True,
        seed=None,
    ):
        """
        Simulate a detector image for the given exposure time and PSF.

        Parameters
        ----------
        time : float or Quantity, optional
            Exposure time (seconds if a bare float). Defaults to the
            Simulation's meta['time'].
        psf : PSFSource, optional
            PSF model to render. Defaults to AiryPSF() (diffraction limited).
        jitter_sigma_mas : float, optional
            Override the telescope jitter (mas). Defaults to the telescope value.
        center : tuple, optional
            Sub-pixel (cx, cy) center for the PSF. Defaults to the grid center.
        add_noise : bool, optional
            If True, apply Poisson shot + dark noise and Gaussian read noise.
            If False, return the noiseless electron image. Default True.
        seed : int, optional
            Seed for the random generator, for reproducible noise.

        Returns
        -------
        SimulatedImage
            Holds the electron image, the noiseless image, and a saturation
            mask (computed from the returned image_e against adc_max / well).
        """
        sim = self.sim
        if time is None:
            time = sim.meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second

        if psf is None:
            psf = AiryPSF()
        ctx = self._context(jitter_sigma_mas=jitter_sigma_mas, center=center)
        psf_norm = psf.render(ctx)  # sum = 1

        comps = sim._count_rate_components()
        source_e_total = comps["source_rate_total"] * time.to(u.s).value
        source_image = source_e_total * psf_norm
        background_per_pix = comps["background_rate_per_pix"] * time.to(u.s).value
        diffuse_per_pix = comps["diffuse_rate_per_pix"] * time.to(u.s).value
        # The rendered image includes diffuse/host flux; the saturation budget
        # (get_peak_pixel / is_saturated / _per_frame_clean_image_e) deliberately
        # does NOT (approved budget = source + background + dark, host excluded).
        # Do not "align" these — the difference is intentional.
        bkg_per_pix = background_per_pix + diffuse_per_pix
        # dark current per pixel (uniform)
        dark_per_pix = (sim.sensor.dark_current * time).to(u.electron / u.pix).value

        image_clean = source_image + bkg_per_pix + dark_per_pix

        if add_noise:
            rng = np.random.default_rng(seed)
            image_e = rng.poisson(np.clip(image_clean, 0.0, None)).astype(float)
            read_noise = sim.sensor.read_noise.to(u.electron / u.pix).value
            image_e = image_e + rng.normal(0.0, read_noise, size=image_e.shape)
        else:
            image_e = image_clean.copy()

        gain = sim.sensor.gain.to(u.electron / u.ct).value
        bias_level = sim.sensor.bias_level.to(u.ct).value

        saturation_mask = saturation_mask_from_image_e(sim.sensor, image_e)

        return SimulatedImage(
            image_e=image_e,
            image_clean=image_clean,
            saturation_mask=saturation_mask,
            gain=gain,
            bias_level=bias_level,
            npix=self.npix,
            pixel_scale_mas=ctx.plate_scale_mas,
            psf=psf,
        )


def howell_center(postage_stamp):
    """
    Howell centroiding, from Howell's Handbook of CCD astronomy

    INPUT:
     postage_stamp - A 2d numpy array to do the centroiding

    OUTPUT:
     x and y center of the numpy array

    NOTES:
    Many thanks to Thomas Beatty and the MINERVAphot.py pipeline for this method
    see here: https://github.com/TGBeatty/MINERVAphot/blob/master/MINERVAphot.py
    """
    xpixels = np.arange(postage_stamp.shape[1])
    ypixels = np.arange(postage_stamp.shape[0])
    I = np.sum(postage_stamp, axis=0)
    J = np.sum(postage_stamp, axis=1)

    Isub = I - np.sum(I) / I.size
    Isub[Isub < 0] = 0
    Jsub = J - np.sum(J) / J.size
    Jsub[Jsub < 0] = 0
    xc = np.sum(Isub * xpixels) / np.sum(Isub)
    yc = np.sum(Jsub * ypixels) / np.sum(Jsub)
    return xc, yc


def psf_center(image):
    """
    Measured center ``(xc, yc)`` of a PSF image for radial reducers.

    Uses Howell centroiding so radii are measured from the actual light
    distribution rather than a fixed grid index. This matters because
    ``render_detector_psf`` forces an odd grid (peak on the exact center pixel)
    and ``center_crop_or_pad`` then brings an even target grid back down, landing
    the centroid on an integer pixel — a full pixel away from ``n//2`` and half a
    pixel from the geometric center ``(n-1)/2``. It also tracks the centroid of
    asymmetric defocused (DefocusPSF) bands, which are not guaranteed symmetric
    about the grid center.

    Falls back to the geometric center ``((nx-1)/2, (ny-1)/2)`` for an empty or
    degenerate image where the centroid is undefined (zero/negative total flux or
    a non-finite centroid, e.g. a flat field).
    """
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    geom = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    if not np.isfinite(image).all() or image.sum() <= 0:
        return geom
    with np.errstate(invalid="ignore", divide="ignore"):
        xc, yc = howell_center(image)  # may divide 0/0 on a flat field -> nan
    if not (np.isfinite(xc) and np.isfinite(yc)):
        return geom
    return xc, yc


def apply_jitter(data, jitter_mas, pixel_scale):
    """
    Apply jitter to the input data.

    INPUT:
        data: 2d data
        jitter_mas: jitter in mas
        pixel_scale: mas/pixel
    """
    total_flux = data.sum()
    sigma_pix = jitter_mas / pixel_scale
    ker = airy.gaussian_kernel_2d(sigma_pix)
    data_blur = fftconvolve(data, ker, mode="same")
    s = data_blur.sum()
    if s > 0:
        data_blur /= s
    return data_blur * total_flux


# make a function that converts a tiff file to a series of fits files
def tiff_to_fits(tiff_file, output_dir):
    """
    Convert a multi-frame TIFF file to a series of FITS files.
    """
    # Read the TIFF file
    image_data = tifffile.imread(tiff_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the image data is 3D (multiple frames)
    if image_data.ndim == 3:
        for i in range(image_data.shape[0]):
            fits_filename = os.path.join(output_dir, f"frame_{i:03d}.fits")
            fits.writeto(fits_filename, image_data[i], overwrite=True)
            print(f"Saved {fits_filename}")
    else:
        fits_filename = os.path.join(output_dir, "image.fits")
        fits.writeto(fits_filename, image_data, overwrite=True)
        print(f"Saved {fits_filename}")


# make a function that reads in fits files from a directory and averages them to create a master flat
def create_master_flat_from_fits(directory, output_filename):
    fits_files = [f for f in os.listdir(directory) if f.endswith(".fits")]
    if not fits_files:
        raise ValueError("No FITS files found in the specified directory.")

    flat_frames = []
    for f in fits_files:
        with fits.open(os.path.join(directory, f)) as hdul:
            flat_frames.append(hdul[0].data.astype(float))

    master_flat = np.nanmedian(flat_frames, axis=0)
    master_flat = master_flat / np.nanmedian(master_flat)

    # Save the master flat
    hdu = fits.PrimaryHDU(data=master_flat.astype(np.float32))
    hdu.writeto(output_filename, overwrite=True)
    print(f"Master flat saved to {output_filename}")


def apply_nl_scaling(df_nl, data, how="makenonlinear", scale=1):
    """
    Apply non-linear scaling to the input data using the provided scaling factors.

    Parameters:
        df_nl (pd.DataFrame): DataFrame containing 'mean_value' and 'scaling_factor' columns.
        data (np.ndarray): Input data array to be scaled.
        how (str): 'makenonlinear' to apply non-linearity, 'correctnonlinear' to reverse it.

    RETURNS:
        data_scaled (np.ndarray): Scaled data array.

    EXAMPLE:

    NOTES:
        - only applies scaling to positive values; zero or negative values are unchanged.
        - Should only be applied after bias subtraction.
    """
    f_nl = scipy.interpolate.interp1d(
        df_nl["mean_value"].values,
        df_nl["scaling_factor"].values * scale,
        kind="linear",
        fill_value="extrapolate",
    )
    data_flat = data.flatten()
    m = data_flat > 0
    data_flat_scaled = data_flat.copy()
    if how == "makenonlinear":
        data_flat_scaled[m] = data_flat[m] / f_nl(data_flat[m])
    elif how == "correctnonlinear":
        data_flat_scaled[m] = data_flat[m] * f_nl(data_flat[m])
    # data_flat_scaled = data_flat.copy()
    # if how == 'makenonlinear':
    #    data_flat_scaled = data_flat / f_nl(np.abs(data_flat))
    # elif how == 'correctnonlinear':
    #    data_flat_scaled = data_flat * f_nl(np.abs(data_flat))
    data_scaled = data_flat_scaled.reshape(data.shape)
    return data_scaled


class FitsImgList(object):
    def __init__(self, data_list, center_list=None, **kwargs):
        """
        Initialize the FitsImgList with a list of data arrays and optional centers.
        """
        self.imglist = []
        for i, data in enumerate(data_list):
            center = center_list[i] if center_list is not None else None
            self.imglist.append(FitsImg(data=data, center=center, **kwargs))

    def aperture_photometry(self, **kwargs):
        """
        Perform aperture photometry on all images in the list.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the `aperture_photometry` method of `FitsImg`.

        Returns
        -------
        results_list : list of dict
            List of dictionaries containing photometry results for each image.
        """
        results_list = []
        for img in self.imglist:
            result = img.aperture_photometry(**kwargs)
            results_list.append(result)
        self.df_phot = pd.DataFrame(results_list)
        self.df_phot["flux_norm"] = self.df_phot["net_flux"] / np.abs(
            np.median(self.df_phot["net_flux"])
        )
        self.df_phot["flux_err_norm"] = self.df_phot["flux_err"] / np.abs(
            np.median(self.df_phot["net_flux"])
        )
        return self.df_phot

    def plot_photometry(self, axes=None):
        """
        Plot the photometry results stored in self.df_phot.
        """
        if not hasattr(self, "df_phot"):
            raise ValueError(
                "No photometry data found. Please run aperture_photometry() first."
            )

        if axes is None:
            fig, axes = plt.subplots(dpi=200, nrows=3, sharex=True)
        ax, bx, cx = axes
        label = r"$\sigma$={:0.0f}ppm, MedErr={:0.0f}ppm".format(
            1e6 * np.std(self.df_phot.flux_norm),
            1e6 * np.median(self.df_phot.flux_err_norm),
        )
        ax.errorbar(
            np.arange(len(self.df_phot)),
            self.df_phot.flux_norm,
            yerr=self.df_phot.flux_err_norm,
            marker="o",
            lw=0,
            mew=0.5,
            capsize=4,
            elinewidth=0.5,
            label=label,
        )
        bx.plot(
            np.arange(len(self.df_phot)),
            self.df_phot["xcen"],
            marker="o",
            lw=0.5,
            mew=0.5,
        )
        cx.plot(
            np.arange(len(self.df_phot)),
            self.df_phot["ycen"],
            marker="o",
            lw=0.5,
            mew=0.5,
        )
        for xx in [ax, bx, cx]:
            xx.grid(lw=0.3, alpha=0.3)
            xx.minorticks_on()
        ax.set_ylabel("Normalized flux", fontsize=15)
        bx.set_ylabel("x-centroid", fontsize=15)
        cx.set_ylabel("y-centroid", fontsize=15)

        ax.legend(fontsize=8, loc="upper right")
        cx.set_xlabel("Exposure number", fontsize=15)


class FitsImg(object):
    def __init__(
        self,
        filename=None,
        data=None,
        header=None,
        center=None,
        dark_current_rate=0.0,
        exp_time=1.0,
        read_noise_rms=0.0,
        imgnumber=0,
    ):
        """
        Initialize the FitsImg with data and optional center.
        """
        if filename != None:
            self.filename = filename
            self.hdulist = astropy.io.fits.open(self.filename)
            self.header = self.hdulist[imgnumber].header
            data = self.hdulist[imgnumber].data
            self.data = data.astype(float)
        else:
            self.filename = ""
            self.hdulist = None
            self.header = header
            self.data = data

        self.center = center
        self.dark_current_rate = dark_current_rate
        self.exp_time = exp_time
        self.read_noise_rms = read_noise_rms

    def crop(self, x, y, w, h):
        """
        Crop to a box centered at (x,y), of size w x h
        """
        x, y = int(x), int(y)
        self.data = self.data[
            int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)
        ]

    def cropcenter(self, w, h, points=False):
        """
        Returns an image array around the center of an image array.
        """
        shape = self.data.shape
        x, y = (int(shape[0] / 2), int(shape[1] / 2))
        self.crop(x, y, w, h)

    def cropcentroid(self, w, h):
        """
        Crop to a box centered at the centroid, of size w x h
        """
        x, y = self.get_centroid()
        self.crop(x, y, w, h)

    def get_centroid(self, plot_cross=False, ax=None, plot_lines=False):
        """
        Find centroid using Howell centroiding.

        See phothelp for the method
        """
        self.xcenter, self.ycenter = howell_center(self.data)
        if plot_cross:
            self.plot(ax=ax)
            self.ax.scatter(self.xcenter, self.ycenter, marker="+", s=50, color="green")
        if plot_lines:
            self.plot(ax=ax)
            self.ax.hlines(
                int(self.ycenter), 0, self.data.shape[1], color="#1f77b4", lw=1
            )
            self.ax.vlines(
                int(self.xcenter), 0, self.data.shape[0], color="#1f77b4", lw=1
            )
        return self.xcenter, self.ycenter

    def get_centroid_line_cut(self, line="X", plot=False, ax=None):
        """
        Horizontal (``line='X'``) or vertical (``line='Y'``) cut through the
        centroid.

        INPUT:
            line - 'X' for the row through the centroid, 'Y' for the column
            plot - draw the cut on ``ax`` (or a new axis)

        OUTPUT:
            cut  - 1D array along the requested direction
        """
        x, y = self.get_centroid()
        x, y = int(x), int(y)
        if line == "Y":
            cut = self.data[:, x]
        elif line == "X":
            cut = self.data[y, :]
        else:
            raise ValueError("line must be 'X' (a row) or 'Y' (a column)")

        self.cut_x = np.arange(len(cut))
        self.cut_y = cut

        if plot:
            if ax is None:
                self.fig, self.ax = plt.subplots()
            else:
                self.ax = ax
            self.ax.plot(self.cut_x, self.cut_y, label=f"{line} cut")
            self.ax.minorticks_on()
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Counts")
        return cut

    def plot(
        self,
        stretch="hist",
        cmap="gray",
        origin="lower",
        ax=None,
        colorbar=False,
        title="",
        vmin=None,
        vmax=None,
        dpi=200,
    ):
        if ax == None:
            self.fig, self.ax = plt.subplots(dpi=dpi)
        else:
            self.ax = ax
        if stretch == "hist":
            print("hist stretch")
            norm = ImageNormalize(stretch=HistEqStretch(self.data))
            self.im = self.ax.imshow(
                self.data, cmap=cmap, origin=origin, norm=norm, vmin=vmin, vmax=vmax
            )
        elif stretch == "log":
            print("log stretch")
            norm = ImageNormalize(self.data, stretch=LogStretch())
            self.im = self.ax.imshow(
                self.data, cmap=cmap, origin=origin, norm=norm, vmin=vmin, vmax=vmax
            )
        else:
            print("linear stretch")
            self.im = self.ax.imshow(
                self.data, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax
            )
        self.ax.set_xlim(0, self.data.shape[1])  # cols
        self.ax.set_ylim(0, self.data.shape[0])  # rows
        self.ax.set_title(title, y=1.02)
        self.ax.set_xlabel("X pixels")
        self.ax.set_ylabel("Y pixels")
        if colorbar:
            self.fig.colorbar(self.im)

    def get_radial_profile(
        self,
        rmax=None,
        plot=False,
        z=2.0,
        return_hwzm=False,
        ax=None,
        xcen=None,
        ycen=None,
        annulus_width=1,
        subtract_min=False,
    ):
        """
        Plot radial profile
        """
        if not hasattr(self, "radial"):
            print("Calculating radial data")
            self.radial = radial_data(
                self.data, rmax=rmax, x=xcen, y=ycen, annulus_width=annulus_width
            )
        else:
            print("Warning: using stored radial data")
        if subtract_min:
            print(
                "Subtracting min value from azimuthal average", np.min(self.radial.mean)
            )
            self.radial.mean -= np.min(self.radial.mean)
        self.radial_hwhm = calc_hwhm(self.radial.r, self.radial.mean)
        self.radial_hwzm = calc_hwzm(self.radial.r, self.radial.mean, z=z)
        z = float(z)

        print("HWZM", self.radial_hwzm)

        if plot:
            if ax == None:
                self.fig, self.ax = plt.subplots()
            else:
                self.ax = ax
            self.ax.plot(self.radial.r, self.radial.mean)
            ymin, ymax = self.ax.get_ylim()
            self.ax.vlines(
                self.radial_hwhm,
                ymin,
                ymax,
                label="HWHM={}".format(self.radial_hwhm),
                color="orange",
                linestyle="--",
                lw=1,
            )
            self.ax.vlines(
                self.radial_hwzm,
                ymin,
                ymax,
                label="HWZM(z={})={}".format(z, self.radial_hwzm),
                color="red",
                linestyle="--",
                lw=1,
            )
            self.ax.legend(loc="upper right", fontsize=14)
            self.ax.grid(lw=0.5, alpha=0.3)

        if return_hwzm:
            return self.radial.r, self.radial.mean, self.radial_hwzm
        else:
            return self.radial.r, self.radial.mean

    def aperture_photometry(
        self,
        r_ap=3.0,
        r_in=6.0,
        r_out=8.0,
        gain=1.0,
        center=None,
        bkg_sigma_clip=3.0,
        bkg_maxiters=5,
        plot=True,
        ax=None,
        verbose=True,
        vmin=None,
        vmax=None,
        cmap="viridis",
        origin="lower",
        stretch="hist",
        colorbar=True,
    ):
        """
        Perform circular aperture photometry on the current `self.data` image using photutils.

        Parameters
        ----------
        r_ap : float
            Aperture radius in pixels.
        r_in, r_out : float
            Inner and outer radii for background annulus in pixels. If None or invalid,
            the image median is used as a background estimate.
        gain : float
            e/ADU (set to 1.0 if `self.data` is already in electrons).
        center : None or (y,x)
            If provided, use this center; otherwise the method will try `self.center`,
            then the peak pixel, then image center.
        bkg_sigma_clip : float
            Sigma for sigma-clipped background estimation in the annulus.
        bkg_maxiters : int
            Max iterations for sigma clipping.

        Returns
        -------
        dict
            Dictionary containing ap_sum, bkg_mean_per_pix, bkg_std_per_pix, bkg_sum,
            net_flux, flux_err, snr, ap_area, ann_area, ann_pixels_used, center
        """
        img = np.asarray(self.data)

        if center is None:
            cx, cy = howell_center(img)
            if verbose:
                print(f"Using centroid at (x={cx:.2f}, y={cy:.2f}) for photometry.")

        # determine center
        # if center is None:
        #    if hasattr(self, 'center') and self.center is not None:
        #        try:
        #            cy, cx = self.center if len(self.center) == 2 else (float(self.center), float(self.center))
        #        except Exception:
        #            cy = (img.shape[0] - 1) / 2.0
        #            cx = (img.shape[1] - 1) / 2.0
        #    else:
        #        try:
        #            cy, cx = np.unravel_index(np.argmax(img), img.shape)
        #            cy, cx = float(cy), float(cx)
        #        except Exception:
        #            cy = (img.shape[0] - 1) / 2.0
        #            cx = (img.shape[1] - 1) / 2.0
        # else:
        #    cy, cx = float(center[0]), float(center[1]) if len(center) == 2 else (float(center), float(center))

        pos = [(cx, cy)]  # photutils uses (x, y)

        aper = CircularAperture(pos, r=r_ap)

        use_ann = (r_in is not None) and (r_out is not None) and (r_out > r_in)
        if use_ann:
            ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
            phot_table = aperture_photometry(img, [aper, ann])
        else:
            phot_table = aperture_photometry(img, [aper])

        # aperture sums
        ap_sum = float(phot_table["aperture_sum_0"][0])
        ap_area = float(aper.area)
        bkg_mean = 0.0
        bkg_median = 0.0
        bkg_std = 0.0
        ann_area = 0.0
        ann_pixels_used = 0

        if use_ann:
            try:
                ann_sum = float(phot_table["aperture_sum_1"][0])
            except Exception:
                ann_sum = 0.0

            mask = ann.to_mask(method="exact")[0]
            annulus_data = mask.multiply(img)
            ann_pixels = annulus_data[mask.data > 0]
            ann_pixels_used = int(ann_pixels.size)

            if ann_pixels_used > 0:
                bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(
                    ann_pixels, sigma=bkg_sigma_clip, maxiters=bkg_maxiters
                )
                ann_area = float(ann.area)
            else:
                # fallback to image statistics
                bkg_mean = float(np.median(img))
                bkg_std = float(np.std(img))
                ann_area = 0.0
        else:
            # fallback: use image median as background estimate
            bkg_mean = float(np.median(img))
            bkg_std = float(np.std(img))

        bkg_sum = bkg_mean * ap_area
        net_flux = ap_sum - bkg_sum

        # helper to extract value from astropy Quantity or bare number
        def _val(x):
            try:
                return float(x.value)
            except Exception:
                return float(x)

        dark_rate = _val(getattr(self, "dark_current_rate"))
        exp_time_val = _val(getattr(self, "exp_time"))
        read_noise = _val(getattr(self, "read_noise_rms"))

        dark_per_pix = dark_rate * exp_time_val

        # noise model (electrons)
        shot_var = net_flux
        dark_var = ap_area * dark_per_pix
        read_var = ap_area * (read_noise**2)
        bkg_var = ap_area * (bkg_std**2)

        total_var = shot_var + dark_var + read_var + bkg_var
        flux_err = np.sqrt(total_var) / float(gain)
        snr = net_flux / flux_err if flux_err > 0 else np.nan

        if plot:
            if ax is None:
                fig, ax = plt.subplots(dpi=100)
            # --- 4. Plot the Ideal PSF --
            if stretch == "hist":
                norm = ImageNormalize(stretch=HistEqStretch(self.data))
                self.im = ax.imshow(
                    self.data, cmap=cmap, origin=origin, norm=norm, vmin=vmin, vmax=vmax
                )
            elif stretch == "log":
                print("log stretch")
                norm = ImageNormalize(self.data, stretch=LogStretch())
                self.im = ax.imshow(
                    self.data, cmap=cmap, origin=origin, norm=norm, vmin=vmin, vmax=vmax
                )
            else:
                print("linear stretch")
                self.im = ax.imshow(
                    self.data, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax
                )

            ax.imshow(
                self.data, origin="lower", interpolation="nearest", cmap="viridis"
            )
            # ax.set_title(f"PSF with Detector Noise (Flux={self.total_flux} e-)")
            ax.set_xlabel("Pixel")
            ax.set_ylabel("Pixel")
            if colorbar:
                ax.figure.colorbar(ax.images[0], ax=ax, label="Signal (electrons)")
            ax.grid(lw=0)
            # Draw aperture and annulus outlines
            aper_patch = aper.plot(ax=ax, color="red", lw=1.6, alpha=0.9)[0]
            ann_patch = ann.plot(ax=ax, color="white", lw=1.2, alpha=0.9)[0]
            # Mark the center
            ax.plot(cx, cy, marker="+", color="yellow", markersize=10, mew=1.5)
            # Annotation: show aperture radii in pixels
            ax.text(
                0.02,
                0.98,
                f"r_ap={r_ap:.1f}px, r_in={r_in:.1f}px, r_out={r_out:.1f}px",
                transform=ax.transAxes,
                color="white",
                fontsize=9,
                va="top",
            )
            ax.set_title("Centroid: (x={:.2f}, y={:.2f})".format(cx, cy), y=1.02)

        return {
            "xcen": cx,
            "ycen": cy,
            "ap_sum": ap_sum,
            "bkg_mean_per_pix": float(bkg_mean),
            "bkg_median_per_pix": float(bkg_median),
            "bkg_std_per_pix": float(bkg_std),
            "bkg_sum": float(bkg_sum),
            "net_flux": float(net_flux),
            "flux_err": float(flux_err),
            "snr": float(snr),
            "ap_area": ap_area,
            "ann_area": ann_area,
            "ann_pixels_used": int(ann_pixels_used),
        }


def solve_time_for_snr(snr, A, B, C):
    """
    Solve SNR = A*t / sqrt(B*t + C) for the positive root t.

    A, B, C may be scalars or broadcastable arrays (A = signal rate, B = variance
    rate, C = constant read-noise variance). Returns t in the same shape (a float
    if all inputs are scalar). Entries with A <= 0 return +inf.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    s2 = float(snr) ** 2
    disc = s2 * s2 * B**2 + 4.0 * A**2 * s2 * C
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (s2 * B + np.sqrt(disc)) / (2.0 * A**2)
    t = np.where(A > 0, t, np.inf)
    return t.item() if t.ndim == 0 else t


def _radial_cumulative(psf_norm, plate_scale_mas):
    """
    Radius-sorted cumulative geometry of a normalized PSF.

    Returns (r_mas, enclosed_fraction, n_pix) as ascending-radius arrays:
    enclosed_fraction is the cumulative PSF sum (psf sums to 1) and n_pix is the
    number of pixels enclosed (1..N).
    """
    psf_norm = np.asarray(psf_norm, dtype=float)
    if psf_norm.ndim != 2 or psf_norm.shape[0] != psf_norm.shape[1]:
        raise ValueError(
            f"psf_norm must be a square 2D array, got shape {psf_norm.shape}"
        )
    npix = psf_norm.shape[0]
    xc, yc = psf_center(psf_norm)  # shared centroid convention (see psf_center)
    yy, xx = np.mgrid[0:npix, 0:npix]
    r_pix = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2).ravel()
    order = np.argsort(r_pix, kind="stable")
    r_sorted = r_pix[order]
    enclosed = np.cumsum(psf_norm.ravel()[order])
    n_pix = np.arange(1, r_sorted.size + 1)
    return r_sorted * plate_scale_mas, enclosed, n_pix


def aperture_snr_radial(
    psf_norm, plate_scale_mas, source_e_total, diffuse_per_pix, dark_per_pix, read_noise
):
    """
    SNR as a function of circular-aperture radius for a rendered PSF.

    The PSF (sum=1) sets how much source light falls inside each radius; the
    per-pixel diffuse (sky+host), dark, and read-noise terms set the background
    noise that grows with the number of aperture pixels.

    Parameters
    ----------
    psf_norm : ndarray
        Normalized (sum=1) PSF on the detector grid.
    plate_scale_mas : float
        Detector plate scale, mas/pixel (to report radii in mas).
    source_e_total : float
        Total source electrons (all of the PSF, before aperture clipping).
    diffuse_per_pix, dark_per_pix : float
        Per-pixel sky+host and dark-current electrons.
    read_noise : float
        Read noise (electrons rms per pixel).

    Returns
    -------
    dict of ndarrays, sorted by ascending radius:
        'r_mas', 'enclosed_fraction', 'n_pix', 'signal_e', 'noise_e', 'snr'.
    """
    r_mas, enclosed, n_pix = _radial_cumulative(psf_norm, plate_scale_mas)

    signal = source_e_total * enclosed
    per_pix_var = diffuse_per_pix + dark_per_pix + read_noise**2
    noise = np.sqrt(signal + per_pix_var * n_pix)
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)

    return {
        "r_mas": r_mas,
        "enclosed_fraction": enclosed,
        "n_pix": n_pix,
        "signal_e": signal,
        "noise_e": noise,
        "snr": snr,
    }


def select_aperture(profile, r_aper_mas=None, ee_frac=None, optimize=False):
    """
    Index into an `aperture_snr_radial` profile for the chosen aperture mode.

    Precedence: optimize (max SNR) > explicit r_aper_mas > ee_frac.
    Raises ValueError if no mode is given.
    """
    if optimize:
        return int(np.argmax(profile["snr"]))
    if r_aper_mas is not None:
        r_mas = profile["r_mas"]
        idx = int(np.searchsorted(r_mas, r_aper_mas, side="right") - 1)
        return int(np.clip(idx, 0, r_mas.size - 1))
    if ee_frac is not None:
        enc = profile["enclosed_fraction"]
        idx = int(np.searchsorted(enc, ee_frac))
        return int(np.clip(idx, 0, enc.size - 1))
    raise ValueError("select_aperture: specify optimize, r_aper_mas, or ee_frac.")


def aperture_time_for_snr(
    psf_norm,
    plate_scale_mas,
    source_rate_total,
    diffuse_rate_per_pix,
    dark_rate_per_pix,
    read_noise,
    n_reads=1,
    snr=None,
    r_aper_mas=None,
    ee_frac=None,
    optimize=False,
):
    """
    Exposure time (s) to reach `snr` for a rendered PSF, per aperture mode.

    Rates are per second (the time dependence is solved for analytically). The
    read-noise variance is incurred n_reads times. Aperture precedence matches
    select_aperture: optimize (fastest radius) > r_aper_mas > ee_frac.

    Returns {'time_s', 'snr', 'r_aper_mas', 'enclosed_fraction', 'n_pix'}.
    """
    if snr is None:
        raise ValueError("snr is required")
    r_mas, enclosed, n_pix = _radial_cumulative(psf_norm, plate_scale_mas)

    A = source_rate_total * enclosed
    B = A + (diffuse_rate_per_pix + dark_rate_per_pix) * n_pix
    C = n_reads * read_noise**2 * n_pix
    t = solve_time_for_snr(snr, A, B, C)  # array over radii

    if optimize:
        idx = int(np.argmin(t))  # radius reaching snr fastest
    elif r_aper_mas is not None:
        idx = int(
            np.clip(
                np.searchsorted(r_mas, r_aper_mas, side="right") - 1, 0, r_mas.size - 1
            )
        )
    elif ee_frac is not None:
        idx = int(np.clip(np.searchsorted(enclosed, ee_frac), 0, enclosed.size - 1))
    else:
        raise ValueError("specify optimize, r_aper_mas, or ee_frac.")

    return {
        "time_s": float(t[idx]),
        "snr": float(snr),
        "r_aper_mas": float(r_mas[idx]),
        "enclosed_fraction": float(enclosed[idx]),
        "n_pix": int(n_pix[idx]),
    }


def calc_hwzm(x, y, z=20):
    """
    Calculates the HWHM at the Z-th maximum for a given dataset, by finding the roots of splines.

    INPUTS:
        x - x input array
        y - y input array

    OUTPUT:
        HWZM The Half Width at Z-th Max of the data

    EXAMPLE:
    """
    spline = UnivariateSpline(x, y - np.max(y) / z, s=0)
    roots = spline.roots()
    return roots


def calc_hwhm(x, y):
    """
    Calculates the HWHM for a given dataset, by finding the roots of splines.

    INPUTS:
        x - x input array
        y - y input array

    OUTPUT:
        HWZM The Half Width at Z-th Max of the data

    EXAMPLE:
    """
    roots = calc_hwzm(x, y, z=2)
    return roots
