"""Real galaxies as hosts: NGC images and spectra moved to any redshift.

Ported from slicersim (``scene/sources/galaxy.py``, M. Rigault). The data come
from the ``lazuli_astrosim_data`` package: for each NGC galaxy an r-band
thumbnail (``PIXSCALE``, ``REDSHIFT``, ``XCENTER``/``YCENTER`` in the header) and
a spectral template (``WAVELENGTH`` in Angstrom, ``FLUX`` in flam).

- :func:`render_galaxy` projects a thumbnail onto the detector grid at a
  redshift, keeping the galaxy's physical size (angular-diameter-distance
  ratio), with an arcsec offset and a rotation. It is the ``"ngc"`` spatial
  model of :class:`wcc_etc.scene.SceneElement` (``profile="ngc0628"``).
- :func:`load_galaxy_spectrum` is the galaxy's own template, used when a scene
  element's spectrum is an NGC name.
"""

import dataclasses
from functools import lru_cache

import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits
from synphot import Empirical1D, SourceSpectrum

__all__ = [
    "COSMOLOGY",
    "GalaxyImage",
    "galaxy_filepath",
    "load_galaxy_spectrum",
    "load_galaxy_image",
    "galaxy_pixel_scale",
    "project_image",
    "render_galaxy",
]

#: Cosmology for distance moduli, kpc-to-arcsec and angular-size scaling.
COSMOLOGY = Planck18


def galaxy_filepath(name, which="spectrum"):
    """Path to the ``"spectrum"`` or ``"image"`` file of an NGC galaxy.

    ``name`` is case insensitive and accepts ``"ngc0628"``, ``"NGC 0628"``,
    ``"ngc_0628"``. Raises FileNotFoundError for a galaxy not in the collection.
    """
    from lazuli_astrosim_data.io import get_galaxy_filepath

    path = get_galaxy_filepath(name.replace(" ", ""), which=which)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not in the galaxy collection")
    return path


@lru_cache(maxsize=None)
def load_galaxy_spectrum(name):
    """The galaxy's spectral template as a synphot SourceSpectrum (flam).

    Memoized per (lower-cased) name; the same object is returned every time.
    """
    data = fits.getdata(galaxy_filepath(name.lower(), which="spectrum"))
    wave = np.asarray(data["WAVELENGTH"], dtype=float)
    flux = np.asarray(data["FLUX"], dtype=float)
    order = np.argsort(wave)
    return SourceSpectrum(
        Empirical1D,
        points=wave[order],
        lookup_table=flux[order],
        meta={"name": name.lower()},
    )


@dataclasses.dataclass(frozen=True)
class GalaxyImage:
    """A flux-normalised galaxy thumbnail with its sampling and distance.

    ``image`` sums to 1; ``pixel_scale`` is arcsec/pixel; ``redshift`` is the
    galaxy's own; ``center`` is the (x, y) 0-based pixel of the nucleus.
    """

    image: np.ndarray
    pixel_scale: float
    redshift: float
    center: tuple


@lru_cache(maxsize=None)
def load_galaxy_image(name):
    """Load an NGC thumbnail as a :class:`GalaxyImage` (memoized per name).

    NaNs become 0 and negative sky residuals are clipped to 0 before the
    normalisation.
    """
    data, header = fits.getdata(
        galaxy_filepath(name.lower(), which="image"), header=True
    )
    image = np.clip(np.nan_to_num(np.asarray(data, dtype=float)), 0.0, None)
    image = image / image.sum()
    image.setflags(write=False)
    return GalaxyImage(
        image=image,
        pixel_scale=float(header["PIXSCALE"]),
        redshift=float(header["REDSHIFT"]),
        center=(float(header["XCENTER"]) - 1.0, float(header["YCENTER"]) - 1.0),
    )


def galaxy_pixel_scale(
    image_pixel_scale, galaxy_redshift, redshift, cosmology=COSMOLOGY
):
    """Thumbnail pixel scale (arcsec) once the galaxy is moved to ``redshift``.

    The physical size is kept, so the angular size scales with the ratio of
    angular diameter distances. ``redshift`` must exceed ``galaxy_redshift``.
    """
    if redshift <= galaxy_redshift:
        raise ValueError(
            f"redshift must be larger than the galaxy's own ({galaxy_redshift})"
        )
    ratio = cosmology.angular_diameter_distance(galaxy_redshift) / (
        cosmology.angular_diameter_distance(redshift)
    )
    return float(image_pixel_scale * ratio.value)


def project_image(
    image,
    center,
    scale,
    shape,
    out_center,
    position=(0.0, 0.0),
    rotation=0.0,
    subsampling=8,
):
    """Project ``image`` onto an ``(ny, nx)`` grid, conserving flux.

    Each image pixel is split into k x k sub-pixels (k such that a sub-pixel is
    at most 1/``subsampling`` of an output pixel) and each sub-pixel's flux is
    added to the output pixel containing its centre. Light that lands off the
    grid is lost, not renormalised.

    Parameters
    ----------
    image : ndarray
        Source image, ``(ny_in, nx_in)``.
    center : (float, float)
        (x, y) image pixel placed at ``out_center + position``.
    scale : float
        Size of one image pixel in output pixels.
    shape : (int, int)
        Output ``(ny, nx)``.
    out_center : (float, float)
        (x, y) output pixel the galaxy centre is measured from.
    position : (float, float), optional
        (dx, dy) offset of the galaxy centre from ``out_center``, in output
        pixels.
    rotation : float, optional
        Rotation of the galaxy in degrees, counterclockwise.
    subsampling : int, optional
        Minimum sub-pixels per output pixel. Default 8.
    """
    ny, nx = shape
    cx, cy = out_center
    theta = np.deg2rad(rotation)
    cos, sin = np.cos(theta), np.sin(theta)

    # 1. crop the image to what can fall onto the grid: a disk around the grid
    #    centre, whatever the rotation
    dx, dy = (
        -np.asarray(position, float) / scale
    )  # grid centre from galaxy centre [image px]
    x0, y0 = center[0] + cos * dx + sin * dy, center[1] - sin * dx + cos * dy
    radius = np.hypot(nx, ny) / 2 / scale + 1
    xmin, xmax = max(int(x0 - radius), 0), min(int(x0 + radius) + 2, image.shape[1])
    ymin, ymax = max(int(y0 - radius), 0), min(int(y0 + radius) + 2, image.shape[0])
    if xmin >= xmax or ymin >= ymax:
        return np.zeros(shape)
    sub = image[ymin:ymax, xmin:xmax]
    v, u = np.nonzero(sub)  # only the galaxy pixels (0 elsewhere)
    flux = sub[v, u]

    # 2. sub-pixels: offsets from each pixel centre [image pixels]
    k = max(1, int(np.ceil(subsampling * scale)))
    offset = (np.arange(k) + 0.5) / k - 0.5
    du, dv = np.meshgrid(offset, offset)
    u = (u + xmin - center[0])[
        :, None
    ] + du.ravel()  # (npix, k*k), from the galaxy centre
    v = (v + ymin - center[1])[:, None] + dv.ravel()

    # 3. to output pixels: scale, rotate and shift
    x = cx + position[0] + scale * (cos * u - sin * v)
    y = cy + position[1] + scale * (sin * u + cos * v)
    ix, iy = np.floor(x + 0.5).astype(int), np.floor(y + 0.5).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    weights = np.broadcast_to(flux[:, None] / k**2, ix.shape)

    return np.bincount(
        (iy * nx + ix)[inside], weights=weights[inside], minlength=ny * nx
    ).reshape(shape)


def render_galaxy(
    name,
    redshift,
    plate_scale_arcsec,
    npix,
    dx=0.0,
    dy=0.0,
    pa=0.0,
    center=None,
    cosmology=COSMOLOGY,
):
    """An NGC galaxy at ``redshift`` on an ``(npix, npix)`` detector grid.

    Returns the fraction of the galaxy's total flux in each pixel (light off
    the grid is lost). ``center`` is the (cx, cy) pixel the ``dx``/``dy``
    offset (arcsec) is measured from; default is ``psfsim.grid_center``, the
    convention every PSF render shares. ``pa`` rotates the galaxy in degrees
    counterclockwise.
    """
    gal = load_galaxy_image(name)
    if center is None:
        center = ((npix - 1) // 2, (npix - 1) // 2)
    scale = galaxy_pixel_scale(gal.pixel_scale, gal.redshift, redshift, cosmology)
    return project_image(
        gal.image,
        gal.center,
        scale / plate_scale_arcsec,
        (npix, npix),
        center,
        position=(dx / plate_scale_arcsec, dy / plate_scale_arcsec),
        rotation=pa,
    )
