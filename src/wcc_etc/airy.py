import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from astropy.io import fits
from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import j1

# render_detector_psf rounds an even n_pixels up by one (to center the PSF peak).
# This is routine and happens on most grids, so we note it quietly at most once
# per session instead of warning on every call (which floods notebooks).
_ODD_NPIX_NOTIFIED = False


def get_airy_psf(D, rr, wavelength, normalize=True):
    """
    Unobscured circular-aperture Airy PSF on grid of radial coords rr [rad].

    INPUT:
        D - diameter in m
        rr - rad?
        wavelength - in m

    OUTPUT:

    EXAMPLE:
        psf2d = wcc_etc.airy.airy_psf(D, rr, k)
        psf2d /= psf2d.sum()
    """
    # Ensure all arguments are unitless floats
    D = float(D.value) if hasattr(D, "unit") else float(D)
    wavelength = (
        float(wavelength.value) if hasattr(wavelength, "unit") else float(wavelength)
    )
    rr = np.array(rr)
    k = 2.0 * np.pi / wavelength
    x = (k * D * rr) / 2.0
    psf = np.ones_like(x)
    nz = x != 0
    psf[nz] = (2.0 * j1(x[nz]) / x[nz]) ** 2
    if normalize:
        psf = psf / psf.sum()
    return psf


def get_ee_value_at_radius(r_mas, ee, r_value):
    """
    Interpolate EE curve to get EE value at a given radius
    """
    return scipy.interpolate.interp1d(r_mas, ee)([r_value])[0]


def gaussian_kernel_2d(sigma_pix, size=None):
    """
    Return normalized 2D Gaussian kernel with given sigma in pixels.

    INPUT:
        sigma_pix -
        size -

    OUTPUT:
        kernel -

    EXAMPLE:
        sigma_pix = sig / px_scale_mean
        ker = gaussian_kernel_2d(sigma_pix)
        psf_blur = fftconvolve(psf2d, ker, mode='same')
    """
    if sigma_pix <= 0:
        return np.array([[1.0]])
    if size is None:
        size = int(np.ceil(8.0 * sigma_pix))
        size = max(size, 3)
        if size % 2 == 0:
            size += 1
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    ker = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
    ker /= ker.sum()
    return ker


def psf_to_encircled_energy(psf, pixel_scale_x_mas, pixel_scale_y_mas):
    """
    Convert a 2D PSF to encircled energy vs radius using annular sums.
    Allows for possibly non-square images by using separate pixel scales in x and y.

    INPUT:
        psf - input PSF
        pixel_scale_x_mas
        pixel_scale_y_mas

    OUTPUT:
        r_mas - radii in mas
        ee - encircled energy

    EXAMPLE:

    """
    ny, nx = psf.shape
    # Measure radii from the PSF's true centroid, not the n//2 index. On even
    # grids the rendered+cropped PSF lands on an integer pixel a full pixel from
    # n//2, biasing the curve of growth outward. psf_center shares one convention
    # with psfsim._radial_cumulative so the two reducers can't drift apart.
    from .psfsim import psf_center  # lazy import: psfsim imports airy at module load

    cx, cy = psf_center(psf)
    y, x = np.indices(psf.shape)
    # Anisotropic pixel scale handled here:
    r_mas = np.sqrt(
        ((x - cx) * pixel_scale_x_mas) ** 2 + ((y - cy) * pixel_scale_y_mas) ** 2
    )
    # Bin edges at 1 mas resolution based on min pixel scale to get smooth curves
    dr = min(pixel_scale_x_mas, pixel_scale_y_mas)
    r_max = r_mas.max()
    edges = np.arange(0, r_max + dr, dr)

    # --- KEY CHANGES START HERE ---

    # 1. Sum of the pixel values (power) in each annular bin
    hist_power, _ = np.histogram(r_mas.ravel(), bins=edges, weights=psf.ravel())

    # 2. Count of the number of pixels in each annular bin
    hist_counts, _ = np.histogram(r_mas.ravel(), bins=edges)

    # 3. Calculate the 1D average PSF by dividing the sum by the count
    # We use np.divide to safely handle bins with zero pixels
    psf1d = np.divide(
        hist_power,
        hist_counts,
        out=np.zeros_like(hist_power, dtype=float),
        where=hist_counts != 0,
    )
    psf1d = psf1d / psf1d.max()

    # --- KEY CHANGES END HERE ---

    # hist_power, edges = np.histogram(r_mas.ravel(), bins=edges, weights=psf.ravel())
    ee = np.cumsum(hist_power)
    if ee[-1] > 0:
        ee /= ee[-1]
    r_centers = 0.5 * (edges[1:] + edges[:-1])
    return r_centers, psf1d, ee


def load_custom_psf(custom_psf_path, custom_psf_hdu=0):
    """
    Load a custom PSF from FITS or image. Returns (psf2d, pixel_scale_x_mas, pixel_scale_y_mas).
    Pixel scale is not in file; user must set custom_psf_extent_mas externally. We compute mas/pixel from the
    provided angular half-extent and the image size.
    """
    path = custom_psf_path
    if path.lower().endswith((".fits", ".fit", ".fts")):
        if fits is None:
            raise RuntimeError("astropy is required to read FITS files.")
        data = fits.getdata(path, ext=custom_psf_hdu).astype(float)
    else:
        if Image is None:
            raise RuntimeError("Pillow is required to read image files (PNG/JPG/BMP).")
        img = Image.open(path).convert("F")  # 32-bit float
        data = np.array(img, dtype=float)

    # Normalize to peak=1.0
    peak = np.max(data)
    if peak > 0:
        data = data / peak

    ny, nx = data.shape
    # Compute mas/pixel from user-provided half-extent
    px_scale_x = (2.0 * custom_psf_extent_mas) / nx
    px_scale_y = (2.0 * custom_psf_extent_mas) / ny
    return data, px_scale_x, px_scale_y


def render_detector_psf(
    wavelength,
    fnum,
    D,
    pixel_size,
    jitter_sigma_mas=0,
    n_pixels=21,
    oversample=11,
    verbose=False,
):
    """
    Render the (optionally jittered) Airy PSF onto the detector pixel grid.

    The PSF is computed on a grid oversampled by ``oversample`` per detector
    pixel, optionally convolved with a Gaussian jitter kernel, then binned down
    to detector pixels. The grid uses an odd number of detector pixels so the
    PSF peak sits on the central pixel (worst case), making the brightest-pixel
    fraction conservative for a saturation check. The result is normalized so
    the rendered window sums to 1.

    Defocus extension: a defocus kernel can be convolved alongside the jitter
    kernel at the oversampled stage without changing this function's interface.

    INPUT:
        wavelength       - wavelength in m
        fnum             - f-number (focal ratio)
        D                - primary diameter in m
        pixel_size       - detector pixel size in microns
        jitter_sigma_mas - jitter sigma in mas (0 = none)
        n_pixels         - detector pixels per side (forced odd)
        oversample       - Sub-pixel sampling factor per detector pixel. Any value works (centering is by symmetry); odd values place a sample exactly on the PSF peak.
        verbose          - print diagnostics

    OUTPUT:
        psf_detector     - (n_pixels, n_pixels) array summing to 1; its max
                           is the brightest-pixel energy fraction
        pscale_mas       - detector plate scale in mas/pixel
    """
    # Strip units to plain floats
    wavelength = (
        float(wavelength.value) if hasattr(wavelength, "unit") else float(wavelength)
    )
    fnum = float(fnum.value) if hasattr(fnum, "unit") else float(fnum)
    D = float(D.value) if hasattr(D, "unit") else float(D)
    pixel_size = (
        float(pixel_size.value) if hasattr(pixel_size, "unit") else float(pixel_size)
    )
    jitter_sigma_mas = (
        float(jitter_sigma_mas.value)
        if hasattr(jitter_sigma_mas, "unit")
        else float(jitter_sigma_mas)
    )
    n_pixels = int(n_pixels)
    oversample = int(oversample)

    # Force odd pixel count so a pixel is centered on the PSF peak.
    if n_pixels % 2 == 0:
        n_pixels += 1
        global _ODD_NPIX_NOTIFIED
        if not _ODD_NPIX_NOTIFIED:
            _ODD_NPIX_NOTIFIED = True
            print(
                "[wcc_etc] note: even n_pixels is rounded up by 1 to center the "
                "PSF peak (shown once).",
                file=sys.stderr,
            )

    # Detector and oversampled plate scales (mas/pix)
    pscale_mas = calc_plate_scale_from_flength(fnum * D, pixel_size) * 1000.0
    fine_pscale_mas = pscale_mas / oversample

    fine_n = (
        n_pixels * oversample
    )  # oversampled grid; centering is by symmetry (peak in the central block)
    half = (fine_n - 1) / 2.0
    coord_mas = (np.arange(fine_n) - half) * fine_pscale_mas

    arcsec_per_radian = 206265.0
    mas_per_radian = arcsec_per_radian * 1000.0
    coord_rad = coord_mas / mas_per_radian
    xx, yy = np.meshgrid(coord_rad, coord_rad, indexing="xy")
    rr = np.sqrt(xx**2 + yy**2)

    # Raw Airy intensity (do not normalize on the truncated grid)
    psf_fine = get_airy_psf(D, rr, wavelength, normalize=False)

    # Jitter broadening at the oversampled scale
    if jitter_sigma_mas != 0:
        sigma_pix_fine = jitter_sigma_mas / fine_pscale_mas
        ker = gaussian_kernel_2d(sigma_pix_fine)
        psf_fine = fftconvolve(psf_fine, ker, mode="same")

    # Bin oversample x oversample blocks into detector pixels
    psf_detector = psf_fine.reshape(n_pixels, oversample, n_pixels, oversample).sum(
        axis=(1, 3)
    )

    # Normalize the rendered window to 1 (conservative: window truncates wings)
    total = psf_detector.sum()
    if total > 0:
        psf_detector = psf_detector / total

    if verbose:
        print(
            f"render_detector_psf: pscale={pscale_mas:.3f} mas/pix, "
            f"n_pixels={n_pixels}, oversample={oversample}, "
            f"peak_fraction={psf_detector.max():.4f}"
        )

    return psf_detector, pscale_mas


def calc_plate_scale_from_flength(focal_length, pix_size):
    """
    Calculate the plate scale from the focal length

    INPUT:
        focal length in m
        pixel size in microns

    OUTPUT:
        plate scale in arcsec/pix

    NOTES:
        focal_length has to be in m
        pix_size is in microns
    """
    plate_scale_arcsec_pix = 206265.0 / (focal_length * 1000.0) * pix_size / 1000.0
    return plate_scale_arcsec_pix


def get_airy_and_ee_curve_pixel_grid(
    wavelength,
    r_aper_mas,
    grid_size=100,
    verbose=False,
    jitter_sigma_mas=0,
    plot=False,
    ax1=None,
    ax2=None,
    pixel_size=3.74,
    fnum=15,
    D=3,
):
    """
    INPUT:
        wavelength - wavelength in m
        D - diameter in m
        grid_size -
        extent_mas -
    EXAMPLE:
        r_mas, ee_base = get_ee_curve(wavelength=0.6e-6)
        r_mas, psf1d, ee = wcc_etc.airy.get_airy_and_ee_curve(wavelength=wavelength*1e-6,plot=True,jitter_sigma_mas=JITTER_MAS,ax=ax,r_aper_mas=70)
    """
    # Strip units from all arguments
    wavelength = (
        float(wavelength.value) if hasattr(wavelength, "unit") else float(wavelength)
    )
    r_aper_mas = (
        float(r_aper_mas.value) if hasattr(r_aper_mas, "unit") else float(r_aper_mas)
    )
    grid_size = int(grid_size)
    # extent_mas = float(extent_mas.value) if hasattr(extent_mas, 'unit') else float(extent_mas)
    jitter_sigma_mas = (
        float(jitter_sigma_mas.value)
        if hasattr(jitter_sigma_mas, "unit")
        else float(jitter_sigma_mas)
    )
    pixel_size = (
        float(pixel_size.value) if hasattr(pixel_size, "unit") else float(pixel_size)
    )
    fnum = float(fnum.value) if hasattr(fnum, "unit") else float(fnum)
    D = float(D.value) if hasattr(D, "unit") else float(D)

    pscale = calc_plate_scale_from_flength(fnum * D, pixel_size) * 1000  # mas/pix
    extent_mas = pscale * grid_size / 2

    arcsec_per_radian = 206265.0
    mas_per_radian = arcsec_per_radian * 1000.0
    extent_rad = extent_mas / mas_per_radian

    x = np.linspace(-extent_rad, extent_rad, grid_size)
    y = np.linspace(-extent_rad, extent_rad, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    rr = np.sqrt(xx**2 + yy**2)
    psf2d = get_airy_psf(D, rr, wavelength, normalize=True)
    pixel_scale_mas = (2.0 * extent_mas) / grid_size
    px_scale_x_mas = pixel_scale_mas
    px_scale_y_mas = pixel_scale_mas
    if verbose:
        source_desc = f"Airy PSF (D={D:.2f} m, λ={wavelength * 1e6} μm), grid={grid_size}², px={pixel_scale_mas:.3f} mas"
        print("Source:", source_desc)
        print(
            f"Pixel scales: {px_scale_x_mas:.3f} mas/px (x), {px_scale_y_mas:.3f} mas/px (y)"
        )

    # Baseline, no jitter
    r_mas, psf1d, ee = psf_to_encircled_energy(psf2d, px_scale_x_mas, px_scale_y_mas)

    if jitter_sigma_mas != 0:
        if verbose:
            print("Broadening with {}mas".format(jitter_sigma_mas))

        # Assuming symmetric
        px_scale_mean = np.sqrt(px_scale_x_mas * px_scale_y_mas)
        sigma_pix = jitter_sigma_mas / px_scale_mean
        ker = gaussian_kernel_2d(sigma_pix)
        psf_blur = fftconvolve(psf2d, ker, mode="same")
        s = psf_blur.sum()
        if s > 0:
            psf_blur /= s
        r_mas, psf1d, ee = psf_to_encircled_energy(
            psf_blur, px_scale_x_mas, px_scale_y_mas
        )

    ee_aper = get_ee_value_at_radius(r_mas, ee, r_aper_mas)
    if plot:
        # EE plot
        if ax1 is None:
            fig, ax1 = plt.subplots()
        ax1.plot(r_mas, ee)
        ax1.set_xlabel("Radius [mas]", fontsize=16)
        ax1.set_ylabel("Encircled Energy", fontsize=16)
        # ax1.axhline(0.9,color='crimson',ls='--')
        ax1.grid(lw=0.3, alpha=0.3)
        ax1.set_title(
            "EE as a function of radius.\nWavelength={:0.3f}micron, Jitter={:0.1f}mas".format(
                wavelength * 1e6, jitter_sigma_mas
            ),
            fontsize=14,
        )
        ax1.axvline(
            r_aper_mas,
            color="k",
            ls="--",
            label="EE={:0.3f} at r={:0.1f}mas, r={:0.1f}pixels".format(
                ee_aper, r_aper_mas, r_aper_mas / (1000 * pscale)
            ),
        )
        ax1.axhline(ee_aper, color="k", ls="--")
        ax1.legend()
        ax1.set_xlim(0, 300)
        bx = ax1.twiny()
        bx.set_xticks(ax1.get_xticks() / (1000 * pscale))
        bx.set_xlabel("Pixels", fontsize=16)

        # EE plot
        if ax2 is None:
            fig, ax2 = plt.subplots()
        ax2.plot(r_mas, psf1d)
        ax2.set_xlabel("Radius [mas]", fontsize=16)
        ax2.set_ylabel("Normalized Flux", fontsize=16)
        ax2.set_title(
            "Airy disk as a function of radius.\nWavelength={:0.3f}micron, Jitter={:0.1f}mas".format(
                wavelength * 1e6, jitter_sigma_mas
            ),
            fontsize=14,
        )
        # ax2.axhline(0.9,color='crimson',ls='--')
        ax2.grid(lw=0.3, alpha=0.3)
        ax2.axvline(
            r_aper_mas,
            color="k",
            ls="--",
            label="EE={:0.3f} at r={:0.1f}mas, r={:0.1f}pixels".format(
                ee_aper, r_aper_mas, r_aper_mas / (1000 * pscale)
            ),
        )
        ax2.legend()
        ax2.set_xlim(0, 300)
        bx = ax2.twiny()
        bx.set_xticks(ax2.get_xticks() / (1000 * pscale))
        bx.set_xlabel("Pixels", fontsize=16)

    if jitter_sigma_mas == 0:
        psf_blur = psf2d

    return r_mas, psf1d, ee, ee_aper, psf_blur


def get_airy_and_ee_curve(
    wavelength,
    r_aper_mas,
    grid_size=1024,
    extent_mas=500,
    verbose=False,
    jitter_sigma_mas=0,
    plot=False,
    ax1=None,
    ax2=None,
    pixel_size=3.74,
    fnum=15,
    D=3,
):
    """
    Calculate airy curve

    INPUT:
        wavelength - wavelength in m
        grid_size - number of pixels in the grid (square)
        extent_mas - size of the grid in milliarcseconds
        D - diameter in m

    EXAMPLE:
        r_mas, ee_base = get_ee_curve(wavelength=0.6e-6)
        r_mas, psf1d, ee = wcc_etc.airy.get_airy_and_ee_curve(wavelength=wavelength*1e-6,plot=True,jitter_sigma_mas=JITTER_MAS,ax=ax,r_aper_mas=70)
    """
    # Strip units from all arguments
    wavelength = (
        float(wavelength.value) if hasattr(wavelength, "unit") else float(wavelength)
    )
    r_aper_mas = (
        float(r_aper_mas.value) if hasattr(r_aper_mas, "unit") else float(r_aper_mas)
    )
    grid_size = int(grid_size)
    extent_mas = (
        float(extent_mas.value) if hasattr(extent_mas, "unit") else float(extent_mas)
    )
    jitter_sigma_mas = (
        float(jitter_sigma_mas.value)
        if hasattr(jitter_sigma_mas, "unit")
        else float(jitter_sigma_mas)
    )
    pixel_size = (
        float(pixel_size.value) if hasattr(pixel_size, "unit") else float(pixel_size)
    )
    fnum = float(fnum.value) if hasattr(fnum, "unit") else float(fnum)
    D = float(D.value) if hasattr(D, "unit") else float(D)

    arcsec_per_radian = 206265.0
    mas_per_radian = arcsec_per_radian * 1000.0
    extent_rad = extent_mas / mas_per_radian

    x = np.linspace(-extent_rad, extent_rad, grid_size)
    y = np.linspace(-extent_rad, extent_rad, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    rr = np.sqrt(xx**2 + yy**2)
    psf2d = get_airy_psf(D, rr, wavelength, normalize=True)
    pixel_scale_mas = (2.0 * extent_mas) / grid_size
    px_scale_x_mas = pixel_scale_mas
    px_scale_y_mas = pixel_scale_mas
    if verbose:
        source_desc = f"Airy PSF (D={D:.2f} m, λ={wavelength * 1e6} μm), grid={grid_size}², px={pixel_scale_mas:.3f} mas"
        print("Source:", source_desc)
        print(
            f"Pixel scales: {px_scale_x_mas:.3f} mas/px (x), {px_scale_y_mas:.3f} mas/px (y)"
        )

    # Baseline, no jitter
    r_mas, psf1d, ee = psf_to_encircled_energy(psf2d, px_scale_x_mas, px_scale_y_mas)

    if jitter_sigma_mas != 0:
        if verbose:
            print("Broadening with {}mas".format(jitter_sigma_mas))

        # Assuming symmetric
        px_scale_mean = np.sqrt(px_scale_x_mas * px_scale_y_mas)
        sigma_pix = jitter_sigma_mas / px_scale_mean
        ker = gaussian_kernel_2d(sigma_pix)
        psf_blur = fftconvolve(psf2d, ker, mode="same")
        s = psf_blur.sum()
        if s > 0:
            psf_blur /= s
        r_mas, psf1d, ee = psf_to_encircled_energy(
            psf_blur, px_scale_x_mas, px_scale_y_mas
        )

    ee_aper = get_ee_value_at_radius(r_mas, ee, r_aper_mas)
    if plot:
        pscale = calc_plate_scale_from_flength(fnum * D, pixel_size)  # arcsec/pix
        # EE plot
        if ax1 is None:
            fig, ax1 = plt.subplots()
        ax1.plot(r_mas, ee)
        ax1.set_xlabel("Radius [mas]", fontsize=16)
        ax1.set_ylabel("Encircled Energy", fontsize=16)
        # ax1.axhline(0.9,color='crimson',ls='--')
        ax1.grid(lw=0.3, alpha=0.3)
        ax1.set_title(
            "EE as a function of radius.\nWavelength={:0.3f}micron, Jitter={:0.1f}mas".format(
                wavelength * 1e6, jitter_sigma_mas
            ),
            fontsize=14,
        )
        ax1.axvline(
            r_aper_mas,
            color="k",
            ls="--",
            label="EE={:0.3f} at r={:0.1f}mas, r={:0.1f}pixels".format(
                ee_aper, r_aper_mas, r_aper_mas / (1000 * pscale)
            ),
        )
        ax1.axhline(ee_aper, color="k", ls="--")
        ax1.legend()
        ax1.set_xlim(0, 300)
        bx = ax1.twiny()
        bx.set_xticks(ax1.get_xticks() / (1000 * pscale))
        bx.set_xlabel("Pixels", fontsize=16)

        # EE plot
        if ax2 is None:
            fig, ax2 = plt.subplots()
        ax2.plot(r_mas, psf1d)
        ax2.set_xlabel("Radius [mas]", fontsize=16)
        ax2.set_ylabel("Normalized Flux", fontsize=16)
        ax2.set_title(
            "Airy disk as a function of radius.\nWavelength={:0.3f}micron, Jitter={:0.1f}mas".format(
                wavelength * 1e6, jitter_sigma_mas
            ),
            fontsize=14,
        )
        # ax2.axhline(0.9,color='crimson',ls='--')
        ax2.grid(lw=0.3, alpha=0.3)
        ax2.axvline(
            r_aper_mas,
            color="k",
            ls="--",
            label="EE={:0.3f} at r={:0.1f}mas, r={:0.1f}pixels".format(
                ee_aper, r_aper_mas, r_aper_mas / (1000 * pscale)
            ),
        )
        ax2.legend()
        ax2.set_xlim(0, 300)
        bx = ax2.twiny()
        bx.set_xticks(ax2.get_xticks() / (1000 * pscale))
        bx.set_xlabel("Pixels", fontsize=16)

    return r_mas, psf1d, ee, ee_aper


# Multi-jitter list
def jittered_ee_list(psf2d, jitter_sigmas_mas, px_scale_x_mas, px_scale_y_mas):
    """
    Loop through different PSFs
    """
    results = {}
    # Use geometric mean pixel scale as a representative scale for isotropic jitter kernel
    px_scale_mean = np.sqrt(px_scale_x_mas * px_scale_y_mas)
    for sig in jitter_sigmas_mas:
        sigma_pix = sig / px_scale_mean
        ker = gaussian_kernel_2d(sigma_pix)
        psf_blur = fftconvolve(psf2d, ker, mode="same")
        s = psf_blur.sum()
        if s > 0:
            psf_blur /= s
        r, psf1d, ee = psf_to_encircled_energy(psf_blur, px_scale_x_mas, px_scale_y_mas)
        results[sig] = (r, ee)
    return results
