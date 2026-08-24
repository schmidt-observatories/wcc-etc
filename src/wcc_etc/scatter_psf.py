"""
scatter_psf -- combine a FRED stray-light map with a core (Airy) PSF.

What it does
------------
A FRED ``.fgd`` grid gives the scattered irradiance (W/mm^2) across the WCC
focal plane for a 1 W on-axis source.  ``make_total_psf`` resamples that map
onto a detector pixel grid (flux-conserving), builds the diffraction core on
the same grid, and adds them with

    total = (1 - desired_power) * core + desired_power * scatter / P_full

where ``P_full`` is the integral over the **whole** FRED map.  So
``desired_power`` is the *instrument-wide* scattered fraction -- a property of the
optics, which is what FRED measured -- and **not** a per-stamp quantity.

That matters: because the halo's surface brightness is then pinned to FRED's, the
Airy-peak-to-scatter-peak contrast and the radius where the two cross are
**independent of the stamp size**.  Normalizing against the power inside the
window instead would make both drift with the extent, which is why there is no
option to do so.  A finite stamp simply holds ``scatter_enclosed_frac`` of the
halo and therefore carries ``desired_power * scatter_enclosed_frac`` of the flux;
both numbers are reported, and ``renormalize=True`` (default) then divides the
array through so it sums to exactly 1.

Two conventions worth knowing
-----------------------------
*Plate scale.*  mm -> mas is derived from the telescope config
(f = f_num * D = 45.975 m, 4.4864 arcsec/mm), using the same 206265 arcsec/rad
constant as ``wcc_etc.airy`` so the scatter and the core share one plate scale
exactly.  ``wcc_etc.scatter_help.FredResult`` hardcodes 4.584 arcsec/mm
(f = 45.0 m); the two differ by 2.1%.

*Grid.*  FRED treats ``A_AXIS_MIN/MAX`` as the outer **edges** of ``A_AXIS_DIM``
cells, so ``dx = (max - min) / DIM`` and the samples sit at cell centers.  Using
that (see :func:`fgd_grid`) reproduces the header's ``integrated power`` to all
15 printed digits.  ``FredResult`` uses ``linspace(min, max, DIM)`` -- a node
convention -- which puts its ``current_power`` 0.2% high.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage
from astropy.io import fits
from scipy.special import j1

from wcc_etc import airy, io, psfsim
from wcc_etc.radial_data import radial_data
from wcc_etc.scatter_help import read_fgd
from wcc_etc.telescope import Telescope

__all__ = [
    "SENSORS",
    "DEFAULT_SCATTER_FGD",
    "TotalPSF",
    "make_total_psf",
    "resample_scatter",
    "enclosed_power",
    "get_sensor_geometry",
    "render_airy_core",
    "airy_irradiance",
    "plot_scatter_verify",
    "write_psf_report",
    "report_diagnostics",
    "crossover_radius",
    "use_plot_style",
    "format_call",
    "fgd_grid",
    "fgd_integrated_power",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

DEFAULT_SCATTER_FGD = os.path.join(
    io.PACKAGE_PATH,
    "psfs",
    "scatter",
    "26-0212_tele image plane, scatter rays, full area, center_450nm_scatter=all.fgd",
)

#: Physical chip geometry.  ``pixel_size`` is pulled from the package config.
SENSORS = {
    "imx455": {
        "config": "zwo.toml",
        "width_mm": 36.0,
        "height_mm": 24.0,
        "label": "IMX455",
        "color": "#d1495b",
    },
    "hwk4123": {
        "config": "qcmos.toml",
        "width_mm": 18.8416,
        "height_mm": 10.5984,
        "label": "HWK4123",
        "color": "#30638e",
    },
}

SENSOR_ALIASES = {
    "sony": "imx455",
    "imx": "imx455",
    "zwo": "imx455",
    "imx455": "imx455",
    "hwk": "hwk4123",
    "qcmos": "hwk4123",
    "hwk4123": "hwk4123",
}

#: Only used to turn a pixel radius into an angle for ``airy.get_airy_psf``.
#: Matches the constant inside ``wcc_etc.airy`` (206265.0, not 206264.806) so the
#: two never drift.
ARCSEC_PER_RAD = 206265.0

DEFAULT_NPIX = 2001  # "default" extent
CHUNK_ROWS = 512  # output rows processed at a time (memory cap)
KEEP_COMPONENTS_MAX_MB = 200  # above this, drop .core/.scatter to save memory
AIRY_CACHE = {}  # rendering the core is the slow part; memoize it


def get_sensor_geometry(sensor):
    """
    Resolve a sensor spec to a geometry dict.

    Parameters
    ----------
    sensor : str or dict
        ``'imx455'`` / ``'hwk4123'`` (aliases: sony, imx, zwo, hwk, qcmos), or a
        dict with at least ``pixel_size`` (micron) and optionally
        ``width_mm`` / ``height_mm``.

    Returns
    -------
    dict
        ``pixel_size``, ``width_mm``, ``height_mm``, ``label``, ``color``.
    """
    if isinstance(sensor, dict):
        geom = {
            "label": "custom",
            "color": "#00798c",
            "width_mm": None,
            "height_mm": None,
        }
        geom.update(sensor)
        if "pixel_size" not in geom:
            raise ValueError("custom sensor dict needs a 'pixel_size' (micron)")
        return geom

    key = SENSOR_ALIASES.get(str(sensor).lower())
    if key is None:
        raise ValueError(
            f"unknown sensor {sensor!r}; use one of {sorted(set(SENSOR_ALIASES))}"
        )
    geom = dict(SENSORS[key])
    geom["pixel_size"] = io.read_config(geom["config"])["sensor"]["pixel_size"]
    geom["name"] = key
    return geom


def plate_scale_mas_per_pixel(pixel_size_um, D, fnum):
    """Detector plate scale [mas/pix].  Thin wrapper on the package's own formula."""
    return airy.calc_plate_scale_from_flength(fnum * D, pixel_size_um) * 1000.0


def arcsec_per_mm(D, fnum):
    """Focal-plane scale [arcsec/mm] -- the same formula at a 1 mm pixel."""
    return airy.calc_plate_scale_from_flength(fnum * D, 1000.0)


def fgd_grid(header, data):
    """
    Cell-center coordinates (mm) for a ``.fgd`` grid, FRED's own convention.

    ``A_AXIS_MIN``/``A_AXIS_MAX`` bound the *outer edges* of ``A_AXIS_DIM``
    cells, so ``dx = (max - min) / DIM`` and sample i sits at
    ``min + (i + 0.5) * dx``.  This reproduces the header's ``integrated
    power`` exactly; ``linspace(min, max, DIM)`` does not.
    """
    ny, nx = data.shape
    x0, x1 = float(header["A_AXIS_MIN"]), float(header["A_AXIS_MAX"])
    y0, y1 = float(header["B_AXIS_MIN"]), float(header["B_AXIS_MAX"])
    x = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    y = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    return x, y


def fgd_integrated_power(filename, max_lines=200):
    """Pull FRED's own ``integrated power`` [W] out of the header, or ``None``."""
    with open(filename) as fh:
        for i, line in enumerate(fh):
            if "BeginData" in line or i > max_lines:
                break
            m = re.search(r"integrated power:\s*([0-9eE+.\-]+)", line)
            if m:
                return float(m.group(1))
    return None


# --------------------------------------------------------------------------- #
# Flux-conserving resampling                                                   #
# --------------------------------------------------------------------------- #


def cell_edges(centers):
    """Edges of uniformly spaced cell centers -> (array of len+1, spacing)."""
    centers = np.asarray(centers, dtype=float)
    d = float(np.mean(np.diff(centers)))
    return np.concatenate([centers - d / 2.0, [centers[-1] + d / 2.0]]), d


def cumulative_power(data, dx, dy):
    """
    2-D cumulative integral of an irradiance map, evaluated on the cell edges.

    ``C[j, i]`` is the total power in ``[xe[0], xe[i]] x [ye[0], ye[j]]``.
    Because ``data`` is piecewise-constant over its cells, ``C`` is *exactly*
    piecewise-bilinear, so bilinear interpolation of ``C`` is exact and the
    4-corner difference gives the exact power in any rectangle.
    """
    c = np.cumsum(np.cumsum(np.asarray(data, dtype=float), axis=0), axis=1)
    out = np.zeros((c.shape[0] + 1, c.shape[1] + 1), dtype=float)
    out[1:, 1:] = c * dx * dy
    return out


def interp_weights(out_coords, e0, step, n_cells):
    """Index/weight pairs for linear interpolation on a uniform edge grid."""
    t = np.clip((np.asarray(out_coords, dtype=float) - e0) / step, 0.0, n_cells)
    i0 = np.clip(np.floor(t).astype(int), 0, n_cells - 1)
    return i0, (t - i0)


def enclosed_power(cum, xe_src, ye_src, x0, x1, y0, y1):
    """Exact power inside the rectangle ``[x0, x1] x [y0, y1]`` (scalar)."""
    dx = xe_src[1] - xe_src[0]
    dy = ye_src[1] - ye_src[0]
    nx, ny = len(xe_src) - 1, len(ye_src) - 1
    ix, wx = interp_weights([x0, x1], xe_src[0], dx, nx)
    iy, wy = interp_weights([y0, y1], ye_src[0], dy, ny)
    corners = np.empty((2, 2))
    for a in range(2):  # a indexes y, b indexes x
        for b in range(2):
            lo = cum[iy[a], ix[b]] * (1 - wx[b]) + cum[iy[a], ix[b] + 1] * wx[b]
            hi = cum[iy[a] + 1, ix[b]] * (1 - wx[b]) + cum[iy[a] + 1, ix[b] + 1] * wx[b]
            corners[a, b] = lo * (1 - wy[a]) + hi * wy[a]
    return corners[1, 1] - corners[1, 0] - corners[0, 1] + corners[0, 0]


def rebin_exact(
    cum, xe_src, ye_src, xe_out, ye_out, dtype=np.float32, chunk=CHUNK_ROWS
):
    """
    Exact area-overlap rebin: power per output pixel, conserving flux exactly.

    Correct for both up- and downsampling.  At large upsample factors the result
    is blocky (each source cell becomes a block of identical output pixels) --
    the honest answer for piecewise-constant input.  Use ``resample_smooth``
    when you want a smooth halo.
    """
    dx, dy = xe_src[1] - xe_src[0], ye_src[1] - ye_src[0]
    nx_src, ny_src = len(xe_src) - 1, len(ye_src) - 1

    ix, wx = interp_weights(xe_out, xe_src[0], dx, nx_src)
    # cumulative in y, differenced in x -> still piecewise-linear in y
    dxcum = np.diff(cum[:, ix] * (1 - wx) + cum[:, ix + 1] * wx, axis=1)

    iy, wy = interp_weights(ye_out, ye_src[0], dy, ny_src)
    ny_out = len(ye_out) - 1
    out = np.empty((ny_out, dxcum.shape[1]), dtype=dtype)
    for r0 in range(0, ny_out, chunk):
        r1 = min(r0 + chunk, ny_out)
        s = slice(r0, r1 + 1)
        rows = dxcum[iy[s]] * (1 - wy[s])[:, None] + dxcum[iy[s] + 1] * wy[s][:, None]
        out[r0:r1] = np.diff(rows, axis=0).astype(dtype)
    return out


def resample_smooth(
    data, x_src, y_src, x_out, y_out, order=3, dtype=np.float32, chunk=CHUNK_ROWS
):
    """
    Smooth (spline) interpolation of the irradiance onto the output pixel
    centers, converted to power per output pixel.  The total is *not* conserved
    here -- ``resample_scatter`` rescales the result to the exact integral.
    """
    data = np.asarray(data, dtype=float)
    dx = float(np.mean(np.diff(x_src)))
    dy = float(np.mean(np.diff(y_src)))
    coeffs = (
        scipy.ndimage.spline_filter(data, order=order, mode="constant")
        if order > 1
        else data
    )

    col = (np.asarray(x_out, dtype=float) - x_src[0]) / dx
    px_area = abs(float(np.mean(np.diff(x_out))) * float(np.mean(np.diff(y_out))))

    out = np.empty((len(y_out), len(x_out)), dtype=dtype)
    for r0 in range(0, len(y_out), chunk):
        r1 = min(r0 + chunk, len(y_out))
        row = (np.asarray(y_out[r0:r1], dtype=float) - y_src[0]) / dy
        rr, cc = np.meshgrid(row, col, indexing="ij")
        vals = scipy.ndimage.map_coordinates(
            coeffs,
            np.stack([rr.ravel(), cc.ravel()]),
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(rr.shape)
        np.clip(vals, 0.0, None, out=vals)  # splines can undershoot
        out[r0:r1] = (vals * px_area).astype(dtype)
    return out


def resample_scatter(
    data, x_src, y_src, x_out, y_out, smooth=True, interp_order=3, dtype=np.float32
):
    """
    Resample an irradiance map to power-per-pixel on a new grid, conserving flux.

    Returns
    -------
    stamp : ndarray
        Power (data units x area) in each output pixel.
    p_exact : float
        Exact power inside the output footprint, from the area-overlap integral.
    p_full : float
        Exact power in the entire source map.
    """
    xe_src, dx = cell_edges(x_src)
    ye_src, dy = cell_edges(y_src)
    xe_out, _ = cell_edges(x_out)
    ye_out, _ = cell_edges(y_out)

    cum = cumulative_power(data, dx, dy)
    p_full = float(cum[-1, -1])
    p_exact = float(
        enclosed_power(
            cum, xe_src, ye_src, xe_out[0], xe_out[-1], ye_out[0], ye_out[-1]
        )
    )

    if smooth:
        stamp = resample_smooth(
            data, x_src, y_src, x_out, y_out, order=interp_order, dtype=dtype
        )
        total = float(stamp.sum(dtype=np.float64))
        if total > 0:
            stamp *= dtype(p_exact / total)  # exact total, smooth shape
    else:
        stamp = rebin_exact(cum, xe_src, ye_src, xe_out, ye_out, dtype=dtype)
    return stamp, p_exact, p_full


# --------------------------------------------------------------------------- #
# Grid / extent bookkeeping                                                    #
# --------------------------------------------------------------------------- #


def dp_pct(desired_power, enclosed):
    """Format the flux fraction a stamp actually carries, for log lines."""
    return f"{100 * desired_power * enclosed:.4g}%"


def make_odd(n):
    """Round to the nearest odd integer >= 1 (so a pixel sits on the center)."""
    n = int(round(n))
    return max(n + 1 if n % 2 == 0 else n, 1)


def resolve_extent(extent, extent_mm, geom, pixel_size_mm):
    """Return ``(nx, ny)``, both odd."""
    if extent_mm is not None:
        hx, hy = (extent_mm, extent_mm) if np.isscalar(extent_mm) else extent_mm
        return make_odd(2 * hx / pixel_size_mm), make_odd(2 * hy / pixel_size_mm)

    if isinstance(extent, (tuple, list, np.ndarray)):
        return make_odd(extent[0]), make_odd(extent[1])
    if isinstance(extent, (int, np.integer)) and not isinstance(extent, bool):
        return make_odd(extent), make_odd(extent)

    key = str(extent).lower()
    if key == "default":
        return DEFAULT_NPIX, DEFAULT_NPIX
    if key in ("chip", "full", "diagonal"):
        if geom.get("width_mm") is None:
            raise ValueError(
                f"extent={extent!r} needs width_mm/height_mm on the sensor"
            )
        if key == "diagonal":
            n = make_odd(np.hypot(geom["width_mm"], geom["height_mm"]) / pixel_size_mm)
            return n, n
        nx = make_odd(geom["width_mm"] / pixel_size_mm)
        ny = make_odd(geom["height_mm"] / pixel_size_mm)
        # "full": minimal stamp that never clips for a star anywhere on the chip
        return (
            (nx, ny) if key == "chip" else (make_odd(2 * nx - 1), make_odd(2 * ny - 1))
        )
    raise ValueError(f"unrecognised extent {extent!r}")


def airy_irradiance(r_mm, D, fnum, wavelength, power=1.0, envelope=False):
    """
    Analytic unobscured-Airy irradiance [W/mm^2] at focal-plane radius ``r_mm``.

    Independent of the PSF pipeline -- this is the closed form, used to verify it.
    Peak is ``P * A / (lambda**2 * f**2)`` with ``A = pi D^2 / 4`` and ``f = fnum*D``;
    for Lazuli at 450 nm with 0.993 W in the core that is 1.712e4 W/mm^2.

    ``envelope=True`` returns the large-x mean ``4 / (pi x^3)`` instead of the
    oscillating ``(2 J1(x) / x)^2`` -- i.e. what an azimuthal average measures.
    """
    f_m = fnum * D
    i0 = power * (np.pi * D**2 / 4.0) / (wavelength**2 * f_m**2) / 1e6
    x = np.pi * D * (np.abs(np.asarray(r_mm, dtype=float)) * 1e-3) / (wavelength * f_m)
    out = np.full(x.shape, i0, dtype=float)
    nz = x > 0
    if envelope:
        # 4/(pi x^3) is the large-x mean of (2 J1/x)^2; it diverges as x -> 0, so
        # clip it at the true peak rather than letting it run above the Airy.
        out[nz] = np.minimum(i0 * 4.0 / (np.pi * x[nz] ** 3), i0)
    else:
        out[nz] = i0 * (2.0 * j1(x[nz]) / x[nz]) ** 2
    return out


def render_inner_core(
    wavelength, fnum, D, pixel_size_um, jitter_sigma_mas, npix, oversample
):
    """Memoized ``airy.render_detector_psf`` -- the slow step when called in a loop."""
    key = (wavelength, fnum, D, pixel_size_um, jitter_sigma_mas, npix, oversample)
    if key not in AIRY_CACHE:
        AIRY_CACHE[key] = airy.render_detector_psf(
            wavelength=wavelength,
            fnum=fnum,
            D=D,
            pixel_size=pixel_size_um,
            jitter_sigma_mas=jitter_sigma_mas,
            n_pixels=npix,
            oversample=oversample,
        )
    return AIRY_CACHE[key]


def render_airy_core(
    wavelength,
    fnum,
    D,
    pixel_size_um,
    shape,
    jitter_sigma_mas=0.0,
    inner_npix=201,
    oversample=11,
    radial_oversample=32,
    dtype=np.float32,
    chunk=CHUNK_ROWS,
):
    """
    Unobscured Airy PSF on a detector grid of ``shape``, at arbitrary size.

    Why not just call ``airy.render_detector_psf`` on the whole grid?  Its cost
    is ``(npix * oversample)**2``, so a 8193-pixel core at oversample 11 would
    need a 6.5 Tpix fine grid.  Truncating the core instead leaves a hard square
    edge where the Airy wing is still an order of magnitude above the scatter
    halo -- a visible step, and a square halo around every simulated star.

    So: the inner ``inner_npix`` block comes from ``render_detector_psf``
    (exact 2-D pixel integration, so the peak pixel and the first rings are
    right), and everything outside it from a pixel-width radial average of the
    Airy intensity, mapped by pixel-center radius.  That is O(N) rather than
    O(N * oversample^2), and averaging over a pixel gives the smooth ~r^-3
    envelope instead of aliasing the lambda/D fringes.  The two are matched in
    an annulus at the seam, and the whole array is normalized to sum 1.

    ``jitter_sigma_mas`` is applied exactly inside the inner block only; at
    typical jitter (~10 mas, well under a pixel) its effect on the wings is
    negligible.
    """
    ny, nx = shape
    pscale_mas = plate_scale_mas_per_pixel(pixel_size_um, D, fnum)
    mas_per_rad = ARCSEC_PER_RAD * 1000.0

    # ---- radial, pixel-averaged Airy over the whole grid ------------------ #
    r_max_px = float(np.hypot(nx, ny)) / 2.0 + 2.0
    ros = int(radial_oversample)
    rp = np.arange(0.0, r_max_px * ros + ros) / ros  # radius in pixels
    inten = airy.get_airy_psf(
        D, rp * pscale_mas / mas_per_rad, wavelength, normalize=False
    )
    kernel = np.ones(ros) / ros  # one-pixel box
    inten = np.convolve(inten, kernel, mode="same")

    core = np.empty((ny, nx), dtype=dtype)
    xpx = np.arange(nx) - (nx - 1) / 2.0
    for r0 in range(0, ny, chunk):
        r1 = min(r0 + chunk, ny)
        ypx = np.arange(r0, r1) - (ny - 1) / 2.0
        rr = np.hypot(xpx[None, :], ypx[:, None])
        core[r0:r1] = np.interp(rr, rp, inten).astype(dtype)

    # ---- exact inner block, matched to the radial scale at the seam ------- #
    n_in = make_odd(min(inner_npix, nx, ny))
    inner, pscale_check = render_inner_core(
        wavelength, fnum, D, pixel_size_um, jitter_sigma_mas, n_in, oversample
    )
    assert np.isclose(pscale_check, pscale_mas, rtol=1e-9)

    half = (n_in - 1) // 2
    gy, gx = np.mgrid[-half : half + 1, -half : half + 1]
    rr_in = np.hypot(gx, gy)
    annulus = (rr_in >= half - 3) & (rr_in <= half - 1)  # just inside the seam
    scale = float(np.interp(rr_in[annulus], rp, inten).mean() / inner[annulus].mean())
    oy, ox = (ny - 1) // 2 - half, (nx - 1) // 2 - half
    core[oy : oy + n_in, ox : ox + n_in] = (inner * scale).astype(dtype)

    total = float(core.sum(dtype=np.float64))
    core /= dtype(total)
    return core, pscale_mas


# --------------------------------------------------------------------------- #
# Result container                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class TotalPSF:
    """Combined core + scatter PSF on a detector pixel grid."""

    data: np.ndarray
    pixel_size_um: float
    pixel_scale_mas: float
    mm_per_pixel: float
    desired_power: float
    scatter_enclosed_frac: float
    core_enclosed_frac: float
    scatter_in_array: float = 1.0
    renorm_scale: float = 1.0
    core: np.ndarray = field(repr=False, default=None)
    scatter: np.ndarray = field(repr=False, default=None)
    meta: dict = field(repr=False, default_factory=dict)
    #: The call that produced this PSF, verbatim -- what the PDF report prints.
    settings: dict = field(repr=False, default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    @property
    def extent_mm(self):
        """``(xmin, xmax, ymin, ymax)`` in mm, for ``imshow(extent=...)``."""
        ny, nx = self.data.shape
        return (
            -nx / 2 * self.mm_per_pixel,
            nx / 2 * self.mm_per_pixel,
            -ny / 2 * self.mm_per_pixel,
            ny / 2 * self.mm_per_pixel,
        )

    @property
    def irradiance_scale(self):
        """
        Multiply ``.data`` (fraction of total flux per pixel) by this to get
        **irradiance in W/mm^2 per watt of source flux**: one over the pixel area.
        """
        return 1.0 / self.mm_per_pixel**2

    def halo_profile(self):
        """``(radius_mm, irradiance)`` of the scatter alone along the +x half-row.

        Irradiance is W/mm^2 per watt of source flux.  Only the positive half:
        ``abs(x_mm)`` is V-shaped, and interpolating on it mixes the two sides.
        Returns ``(None, None)`` when the components were not retained.
        """
        if self.scatter is None:
            return None, None
        xm = self.x_mm
        keep = xm >= 0
        row = np.asarray(self.scatter[self.shape[0] // 2], dtype=float)[keep]
        return xm[keep], row * self.desired_power * self.irradiance_scale

    @property
    def x_mm(self):
        nx = self.data.shape[1]
        return (np.arange(nx) - (nx - 1) / 2.0) * self.mm_per_pixel

    def header(self):
        """FITS header describing the sampling and the provenance."""
        ny, nx = self.data.shape
        h = fits.Header()
        h["CRPIX1"] = ((nx + 1) / 2.0, "PSF center (1-indexed)")
        h["CRPIX2"] = ((ny + 1) / 2.0, "PSF center (1-indexed)")
        h["CRVAL1"] = (0.0, "mas")
        h["CRVAL2"] = (0.0, "mas")
        h["CDELT1"] = (self.pixel_scale_mas, "mas / pixel")
        h["CDELT2"] = (self.pixel_scale_mas, "mas / pixel")
        h["CUNIT1"] = ("mas", "")
        h["CUNIT2"] = ("mas", "")
        h["BUNIT"] = ("fraction", "fraction of source energy per pixel")
        h["PIXSZUM"] = (self.pixel_size_um, "detector pixel size [micron]")
        h["PSCLMAS"] = (self.pixel_scale_mas, "plate scale [mas/pix]")
        h["MMPERPIX"] = (self.mm_per_pixel, "mm per pixel at the focal plane")
        for key, card in self.meta.items():
            h[key] = card
        h["DPOWER"] = (self.desired_power, "instrument-wide scatter fraction")
        h["FENCLOSE"] = (self.scatter_enclosed_frac, "frac of FRED halo inside stamp")
        h["FSCATTER"] = (self.scatter_in_array, "scatter fraction inside this stamp")
        h["CENCLOSE"] = (self.core_enclosed_frac, "core fraction inside stamp")
        h["PSFSUM"] = (float(self.data.sum(dtype=np.float64)), "sum of the array")
        h.add_history("built by scatter_psf.make_total_psf")
        h.add_history("total = (1-DPOWER)*core + DPOWER*scatter/P_ref, see SCATREF")
        return h

    def save(self, path, overwrite=False, compress=None, quantize_level=0.0):
        """
        Write the PSF to ``path`` as float32.

        ``compress=None`` writes a plain ``PrimaryHDU`` (HDU 0, memory-mappable).
        ``compress='GZIP_2'`` writes a tile-compressed ``CompImageHDU`` -- the
        data then lives in **HDU 1**, and astropy reads it back transparently.

        Only ``quantize_level=0.0`` is lossless for floating-point data.  Measured
        on this PSF: GZIP_2 1.28x, GZIP_1 1.11x, file-level gzip 1.12x.  The
        quantized modes reach ~2.8x but at ~8e-3 relative error per pixel --
        bigger than the scatter fraction itself, so they are refused below unless
        you ask for them explicitly.
        """
        hdr = self.header()
        if compress is None:
            fits.PrimaryHDU(data=self.data, header=hdr).writeto(
                path, overwrite=overwrite
            )
        else:
            if compress.upper().startswith("RICE") and quantize_level == 0.0:
                raise ValueError(
                    "RICE_1 with quantize_level=0 does not round-trip float data "
                    "(measured max relative error 1.0). Use GZIP_2 for lossless."
                )
            if quantize_level != 0.0:
                print(
                    f"  WARNING: quantize_level={quantize_level} is LOSSY "
                    f"(~8e-3 relative error at 16)."
                )
            hdr["FZALGOR"] = (compress, "tile compression used")
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.CompImageHDU(
                        data=self.data,
                        header=hdr,
                        compression_type=compress,
                        quantize_level=quantize_level,
                    ),
                ]
            ).writeto(path, overwrite=overwrite)
        mb = os.path.getsize(path) / 1e6
        raw = self.data.nbytes / 1e6
        note = f"  ({mb:,.1f} MB on disk, {raw:,.1f} MB in memory, {raw / mb:.2f}x)"
        print(f"  wrote {path}" + (note if compress else f"  ({mb:,.1f} MB)"))
        return path

    def plot(self, **kwargs):
        """Two-panel figure: log-stretched image + log-y central cut."""
        return plot_total_psf(self, **kwargs)


# --------------------------------------------------------------------------- #
# The public entry point                                                       #
# --------------------------------------------------------------------------- #


def format_call(call_locals, name="make_total_psf", width=58):
    """
    Render a call as reproducible source, from ``locals()`` taken at entry.

    The output is valid Python you can paste back into a session. Bulk arrays are
    elided rather than dumped -- ``scatter_data`` becomes a note, ``core_psf`` its
    shape -- and long strings are split with implicit concatenation so nothing
    runs off the edge of the report page.
    """
    import inspect

    def render(key, value):
        text = repr(value)
        if not isinstance(value, str) or len(text) <= width:
            return [f"    {key}={text},"]
        chunks = [value[k : k + width] for k in range(0, len(value), width)]
        out = [f"    {key}=("]
        out += [f'        "{c}"' for c in chunks]
        out.append("    ),")
        return out

    params = inspect.signature(globals()[name]).parameters
    lines = [f"{name}("]
    for key in params:
        if key not in call_locals:
            continue
        value = call_locals[key]
        if key == "scatter_data":
            if value is None:
                continue
            value = "<pre-parsed; source named under 'scatter file' above>"
        elif key == "core_psf" and value is not None:
            value = f"<array {np.asarray(value).shape}>"
        elif key == "dtype":
            lines.append(f"    {key}=np.{getattr(value, '__name__', value)},")
            continue
        elif value is params[key].default and key != "scatter_file":
            continue
        lines += render(key, value)
    lines.append(")")
    return "\n".join(lines)


def use_plot_style():
    """
    Apply the ``gks`` matplotlib style if it is installed, else the package's own.

    ``scatter_psf`` is library code: ``gks`` lives in a personal matplotlib
    stylelib, so it is absent in CI and for collaborators, where a bare
    ``plt.style.use("gks")`` raises OSError. Falling back to ``WCC_STYLE`` keeps
    the figures looking right everywhere.
    """
    import matplotlib.pyplot as plt

    from wcc_etc.plotting import WCC_STYLE

    try:
        plt.style.use("gks")
    except OSError:
        plt.rcParams.update(WCC_STYLE)


def crossover_radius(
    r_mm,
    halo_irradiance,
    D,
    fnum,
    wavelength,
    power=1.0,
    envelope=False,
    rmax=None,
    n=200001,
):
    """
    Largest radius [mm] at which the Airy still exceeds the scatter halo.

    ``envelope=False`` uses the fringed Airy -- the last radius at which a ring
    *maximum* pokes above the halo, which is what a peak-to-peak requirement
    means.  ``envelope=True`` uses the azimuthally-averaged envelope, which
    crosses earlier.  Returns NaN if they never cross inside ``rmax``.
    """
    r_mm = np.asarray(r_mm, dtype=float)
    halo_irradiance = np.asarray(halo_irradiance, dtype=float)
    rmax = float(r_mm.max()) if rmax is None else float(rmax)
    r = np.linspace(max(float(r_mm.min()), 1e-4), rmax, int(n))
    halo = np.interp(r, r_mm, halo_irradiance)

    # If the Airy envelope is still above the halo at the edge of the window, the
    # crossover lies outside it. Without this the last sign change sits at the
    # boundary and reads as a real crossover -- a stamp-size artefact.
    if airy_irradiance(rmax, D, fnum, wavelength, power=power, envelope=True) > (
        np.interp(rmax, r_mm, halo_irradiance)
    ):
        return float("nan")

    diff = (
        airy_irradiance(r, D, fnum, wavelength, power=power, envelope=envelope) - halo
    )
    i = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    return float(r[i[-1]]) if len(i) else float("nan")


def report_diagnostics(res):
    """
    Derived numbers describing a :class:`TotalPSF`, for a report or a log line.

    Peak irradiances (W/mm^2 per watt of source), the Airy-to-halo contrast, and
    both crossover radii.  Halo-dependent entries are ``None`` when the PSF was
    built without ``keep_components``.
    """
    d = float(res.meta["TELDIAM"][0])
    fnum = float(res.meta["FNUM"][0])
    lam = float(res.meta["WAVELEN"][0]) * 1e-9
    dp = res.desired_power

    out = {
        "shape": res.shape,
        "pixel_size_um": res.pixel_size_um,
        "pixel_scale_mas": res.pixel_scale_mas,
        "mm_per_pixel": res.mm_per_pixel,
        "half_width_mm": res.shape[1] / 2 * res.mm_per_pixel,
        "sum": float(res.data.sum(dtype=np.float64)),
        "peak_fraction": float(res.data.max()),
        "peak_irradiance": float(res.data.max()) * res.irradiance_scale,
        "airy_peak_analytic": airy_irradiance(0.0, d, fnum, lam, power=1.0 - dp),
        "scatter_enclosed_frac": res.scatter_enclosed_frac,
        "core_enclosed_frac": res.core_enclosed_frac,
        "scatter_in_array": res.scatter_in_array,
        "halo_peak": None,
        "contrast": None,
        "crossover_fringed_mm": None,
        "crossover_envelope_mm": None,
    }

    r_mm, halo = res.halo_profile()
    if halo is not None:
        out["halo_peak"] = float(halo.max())
        out["contrast"] = out["airy_peak_analytic"] / out["halo_peak"]
        for key, env in (
            ("crossover_fringed_mm", False),
            ("crossover_envelope_mm", True),
        ):
            out[key] = crossover_radius(
                r_mm, halo, d, fnum, lam, power=1.0 - dp, envelope=env
            )
    return out


def write_psf_report(res, path, fits_path=None, dpi=140):
    """
    Write a PDF describing this PSF: what went into it, what came out, how it looks.

    Page 1 is the three-panel diagnostic figure.  Page 2 is an irradiance cut of
    the core and halo against the closed-form Airy, so the contrast and the
    crossover are visible rather than merely tabulated.  Page 3 lists the input
    ingredients, the settings the PSF was built with, and the derived numbers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    use_plot_style()
    diag = report_diagnostics(res)
    d = float(res.meta["TELDIAM"][0])
    fnum = float(res.meta["FNUM"][0])
    lam = float(res.meta["WAVELEN"][0]) * 1e-9

    with PdfPages(path) as pdf:
        # -- page 1: the diagnostic figure -------------------------------- #
        fig, _ = plot_total_psf(res)
        fig.suptitle("Airy + scattered-light PSF", fontsize=13, y=1.0)
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # -- page 2: irradiance cut vs the closed-form Airy ---------------- #
        r_mm, halo = res.halo_profile()
        if halo is not None:
            fig, ax = plt.subplots(figsize=(9.5, 6.5))
            rr = np.linspace(res.mm_per_pixel / 2, float(r_mm.max()), 40001)
            core_power = 1.0 - res.desired_power
            ax.plot(
                rr,
                airy_irradiance(rr, d, fnum, lam, power=core_power),
                color="#3b76af",
                lw=0.4,
                label="Airy, analytic, continuous (rings resolved)",
            )
            ax.plot(
                rr,
                airy_irradiance(rr, d, fnum, lam, power=core_power, envelope=True),
                color="#003d5b",
                lw=1.4,
                ls=":",
                label=r"Airy ring-averaged envelope $4/(\pi x^3)$",
            )
            ax.plot(
                r_mm,
                halo,
                color="#66a182",
                lw=2.0,
                label=f"scatter halo ({100 * res.desired_power:.3g}% of flux)",
            )
            if res.core is not None:
                core_row = np.asarray(res.core[res.shape[0] // 2], dtype=float)
                keep = res.x_mm >= 0
                ax.plot(
                    res.x_mm[keep],
                    core_row[keep]
                    * (1 - res.desired_power)
                    * res.renorm_scale
                    * res.irradiance_scale,
                    color="0.25",
                    lw=1.2,
                    ls="--",
                    label="core AS SAMPLED on detector pixels (this PSF)",
                )
            for key, style in (
                ("crossover_fringed_mm", "-"),
                ("crossover_envelope_mm", ":"),
            ):
                if diag[key] and np.isfinite(diag[key]):
                    ax.axvline(
                        diag[key],
                        color="#d1495b",
                        lw=1.2,
                        ls=style,
                        label=f"{key.replace('_mm', '').replace('_', ' ')}"
                        f" = {diag[key]:.2f} mm",
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(res.mm_per_pixel / 2, float(r_mm.max()))
            ax.set_ylim(float(halo.min()) / 30, diag["airy_peak_analytic"] * 5)
            ax.set_xlabel("radius [mm]")
            ax.set_ylabel("irradiance [W mm$^{-2}$ per W of source]")
            ax.set_title(
                f"Core vs halo   contrast = {diag['contrast']:.3e}", fontsize=12
            )
            fig.text(
                0.01,
                0.005,
                "Envelope is the analytic mean level the Airy rings oscillate "
                "about -- a reference curve, not a resampled one. The dashed "
                "core is what actually lands on the pixels: 2-D pixel-integrated "
                "inside inner_npix, pixel-width radial average outside. At "
                f"{res.pixel_scale_mas:.1f} mas/pix the ring spacing is "
                f"{1e3 * lam * fnum / res.mm_per_pixel:.1f} pixels, so the rings "
                "are only marginally sampled and partly survive in the array.",
                fontsize=6.5,
                color="0.35",
                wrap=True,
            )
            ax.legend(fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

        # -- page 3: ingredients, settings, diagnostics -------------------- #
        pdf.savefig(_report_text_page(res, diag, fits_path), dpi=dpi)
        plt.close("all")

    print(f"  wrote {path}")
    return path


def _report_text_page(res, diag, fits_path):
    """The tabular page of write_psf_report. Kept separate so it stays readable."""
    import matplotlib.pyplot as plt

    def fmt(value):
        if value is None:
            return "n/a (built without keep_components)"
        if isinstance(value, float):
            if value and (abs(value) < 1e-3 or abs(value) >= 1e5):
                return f"{value:.6e}"
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    st = res.settings
    ny, nx = res.shape
    aspmm = arcsec_per_mm(res.meta["TELDIAM"][0], res.meta["FNUM"][0])
    sections = [
        (
            "INPUT INGREDIENTS",
            [
                (
                    "scatter file",
                    st.get("scatter_file", res.meta.get("SCATFILE", ("?",))[0])
                    + (
                        "   [passed in pre-parsed]"
                        if st.get("scatter_preparsed")
                        else ""
                    ),
                ),
                ("scatter dir", st.get("scatter_dir") or "n/a"),
                ("FRED run", st.get("fred_run") or "n/a"),
                (
                    "rays traced",
                    f"{int(st['rays_traced']):,}" if st.get("rays_traced") else "n/a",
                ),
                (
                    "FRED integrated power",
                    f"{res.meta['PFULL'][0]:.9e} W per W in "
                    f"({100 * res.meta['PFULL'][0]:.4f}%)",
                ),
                ("core", st.get("core_kind", "Airy (analytic, rendered)")),
                (
                    "telescope",
                    f"D = {res.meta['TELDIAM'][0]} m, f/{res.meta['FNUM'][0]}, "
                    f"f = {res.meta['TELDIAM'][0] * res.meta['FNUM'][0]:.3f} m",
                ),
                ("detector", f"{res.meta['SENSOR'][0]}, {res.pixel_size_um} um pixels"),
            ],
        ),
        (
            "SETTINGS",
            [
                (
                    "desired_power",
                    f"{res.desired_power:g}  "
                    f"({100 * res.desired_power:g}% of total flux, "
                    f"instrument-wide)",
                ),
                (
                    "extent",
                    f"{st.get('extent', '?')}  ->  {nx} x {ny} px  = "
                    f"{nx * res.mm_per_pixel:.2f} x {ny * res.mm_per_pixel:.2f} mm",
                ),
                ("wavelength", f"{res.meta['WAVELEN'][0]:g} nm"),
                ("jitter_sigma_mas", f"{res.meta['JITTER'][0]:g}"),
                (
                    "inner_npix / oversample",
                    f"{st.get('inner_npix', '?')} / {st.get('oversample', '?')}",
                ),
                (
                    "smooth / interp_order",
                    f"{res.meta['SMOOTH'][0]} / {st.get('interp_order', '?')}",
                ),
                ("renormalize", str(res.meta["RENORM"][0])),
                ("dtype", str(res.data.dtype)),
            ],
        ),
        (
            "SAMPLING",
            [
                (
                    "plate scale",
                    f"{res.pixel_scale_mas:.4f} mas/pix  ({aspmm:.4f} arcsec/mm)",
                ),
                ("mm per pixel", fmt(res.mm_per_pixel)),
                (
                    "half-width",
                    f"{diag['half_width_mm']:.3f} mm  "
                    f"({diag['half_width_mm'] * aspmm:.1f} arcsec)",
                ),
                ("in memory", f"{res.data.nbytes / 1e6:,.1f} MB"),
            ],
        ),
        (
            "DIAGNOSTICS",
            [
                ("array sum", fmt(diag["sum"])),
                (
                    "peak pixel",
                    f"{diag['peak_fraction']:.6f} of total flux  "
                    f"= {diag['peak_irradiance']:.4e} W/mm^2",
                ),
                ("Airy peak (analytic)", f"{diag['airy_peak_analytic']:.4e} W/mm^2"),
                (
                    "halo peak",
                    fmt(diag["halo_peak"]) + (" W/mm^2" if diag["halo_peak"] else ""),
                ),
                ("contrast (Airy/halo)", fmt(diag["contrast"])),
                (
                    "crossover, fringed",
                    fmt(diag["crossover_fringed_mm"])
                    + (" mm" if diag["crossover_fringed_mm"] else ""),
                ),
                (
                    "crossover, envelope",
                    fmt(diag["crossover_envelope_mm"])
                    + (" mm" if diag["crossover_envelope_mm"] else ""),
                ),
                (
                    "halo enclosed by stamp",
                    f"{res.scatter_enclosed_frac:.6f}  "
                    f"({100 * res.scatter_enclosed_frac:.2f}% of the map)",
                ),
                ("core enclosed by stamp", fmt(res.core_enclosed_frac)),
                (
                    "scatter carried by array",
                    f"{res.scatter_in_array:.6e}  "
                    f"({100 * res.scatter_in_array:.4f}% of flux)",
                ),
            ],
        ),
    ]
    if fits_path:
        size = os.path.getsize(fits_path) / 1e6 if os.path.exists(fits_path) else None
        sections.append(
            (
                "OUTPUT",
                [
                    ("FITS", os.path.basename(str(fits_path))),
                    ("on disk", f"{size:,.1f} MB" if size else "not written"),
                ],
            )
        )

    lines = []
    for title, rows in sections:
        lines.append(title)
        lines.append("-" * 78)
        for key, value in rows:
            lines.append(f"  {key:<26s} {value}")
        lines.append("")

    if st.get("call"):
        lines += ["REPRODUCE", "-" * 78]
        lines += ["  " + ln for ln in st["call"].splitlines()]
        lines.append("")

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(
        0.06, 0.965, "Airy + scattered-light PSF - build report", fontsize=14, va="top"
    )
    fig.text(
        0.06,
        0.935,
        "wcc_etc.scatter_psf.make_total_psf",
        fontsize=9,
        color="0.4",
        va="top",
    )
    fig.text(
        0.06,
        0.905,
        "\n".join(lines),
        fontsize=7.4,
        va="top",
        family="monospace",
        linespacing=1.45,
    )
    return fig


def make_total_psf(
    scatter_file=DEFAULT_SCATTER_FGD,
    desired_power=5.542e-3,
    sensor="hwk4123",
    extent="default",
    extent_mm=None,
    core_psf=None,
    wavelength=450e-9,
    jitter_sigma_mas=0.0,
    inner_npix=201,
    oversample=11,
    D=None,
    fnum=None,
    telescope_config="lazuli.toml",
    smooth=True,
    interp_order=3,
    renormalize=True,
    max_size_mb=1000.0,
    keep_components=None,
    dtype=np.float32,
    output=None,
    overwrite=False,
    compress=None,
    report=None,
    plot=False,
    scatter_data=None,
    verbose=True,
):
    """
    Build a combined (core + scattered-light) PSF on a detector pixel grid.

    Parameters
    ----------
    scatter_file : str
        FRED ``.fgd`` irradiance map.  Ignored if ``scatter_data`` is given.
    desired_power : float
        Fraction of the *total* source energy carried by scattered light.
        ``5.542e-3`` reproduces the 26-0212 FRED run as-is.
    sensor : str or dict
        ``'imx455'`` / ``'hwk4123'`` (or a dict with ``pixel_size`` in micron).
        Sets the output pixel scale.
    extent : {'default', 'chip', 'diagonal', 'full'}, int, or (nx, ny)
        Output size in pixels (forced odd).  ``'default'`` = 2001 x 2001;
        ``'chip'`` = the chip itself; ``'diagonal'`` = a square whose *side* is
        the chip diagonal; ``'full'`` = ``(2Nx-1, 2Ny-1)``, the minimal stamp
        that never clips for a star anywhere on the chip.  Note ``'diagonal'``
        does **not** meet the non-clipping condition -- its half-width is only
        half the chip diagonal, less than the chip's own width.
    extent_mm : float or (hx, hy), optional
        Half-width(s) in mm; overrides ``extent``.
    core_psf : ndarray, optional
        Use this instead of an Airy core.  Must already be on the output pixel
        scale; it is normalized to sum 1 and embedded at the center.
    wavelength, jitter_sigma_mas, inner_npix, oversample
        Airy core controls, see :func:`render_airy_core`.  The core always fills
        the whole output grid (no square truncation edge); ``inner_npix`` only
        sets how big the exactly-pixel-integrated central block is, at a cost
        quadratic in ``inner_npix * oversample``.  ``jitter_sigma_mas`` defaults
        to 0 so the saved base PSF is purely optical; apply jitter downstream.
    D, fnum : float, optional
        Telescope diameter [m] and f-number.  Default: read from
        ``telescope_config`` (Lazuli: 3.065 m, f/15).
    smooth : bool
        ``True`` (default) interpolates the irradiance smoothly and rescales to
        the exact enclosed integral.  ``False`` uses pure area-overlap rebinning
        (exact but blocky at ~100x upsample).
    renormalize : bool
        Divide the output so it sums to exactly 1.  Enclosed fractions are
        reported either way.
    max_size_mb : float
        Refuse to allocate an output larger than this.
    keep_components : bool, optional
        Retain ``.core`` / ``.scatter`` for plotting.  Default: automatic --
        dropped above ``KEEP_COMPONENTS_MAX_MB`` to save memory.
    output : str, optional
        Path to write a float32 FITS file.
    compress : str, optional
        ``'GZIP_2'`` for a lossless tile-compressed file (data moves to HDU 1).
    report : str, optional
        Path to write a PDF build report -- the diagnostic figure, an irradiance
        cut against the closed-form Airy, and a page listing the input
        ingredients, the settings used and the derived numbers.  Implies
        ``keep_components`` unless you set it explicitly.
    plot : bool
        Draw the two-panel diagnostic figure.
    scatter_data : tuple, optional
        ``(header, data, x_src, y_src)`` from a previous ``read_fgd``, to skip
        re-parsing the 24 MB ASCII file.

    Returns
    -------
    TotalPSF
    """
    call_kwargs = format_call(locals())

    def say(*a):
        if verbose:
            print(*a)

    # ---- geometry & plate scale ------------------------------------------- #
    geom = get_sensor_geometry(sensor)
    pixel_size_um = float(geom["pixel_size"])
    mm_per_pixel = pixel_size_um / 1000.0

    tel = Telescope.from_config(io.read_config(telescope_config)["telescope"])
    D = float(tel.diameter_primary.value) if D is None else float(D)
    fnum = float(tel.f_num) if fnum is None else float(fnum)
    flength_m = fnum * D
    aspmm = arcsec_per_mm(D, fnum)
    pixel_scale_mas = plate_scale_mas_per_pixel(pixel_size_um, D, fnum)

    say(f"Sensor        : {geom.get('label', 'custom')}  pixel = {pixel_size_um:g} um")
    say(f"Telescope     : D = {D:g} m, f/{fnum:g} -> f = {flength_m:.3f} m")
    say(f"Plate scale   : {aspmm:.4f} arcsec/mm = {pixel_scale_mas:.3f} mas/pix")
    say(
        "                (scatter_help.FredResult hardcodes 4.584 arcsec/mm, "
        "f = 45.0 m)"
    )

    # ---- output grid ------------------------------------------------------ #
    nx, ny = resolve_extent(extent, extent_mm, geom, mm_per_pixel)
    n_mb = nx * ny * np.dtype(dtype).itemsize / 1e6
    say(
        f"Output grid   : {nx} x {ny} pix = "
        f"{nx * mm_per_pixel:.2f} x {ny * mm_per_pixel:.2f} mm "
        f"-> {n_mb:,.1f} MB ({np.dtype(dtype).name})"
    )
    if n_mb > max_size_mb:
        raise ValueError(
            f"output would be {n_mb:,.0f} MB > max_size_mb={max_size_mb:,.0f}. "
            "Shrink `extent`/`extent_mm`, or raise max_size_mb deliberately."
        )
    if keep_components is None:
        # the report's cut and contrast need .core / .scatter
        keep_components = bool(report) or n_mb <= KEEP_COMPONENTS_MAX_MB
        if not keep_components:
            say(
                f"                >{KEEP_COMPONENTS_MAX_MB} MB: dropping "
                ".core/.scatter (pass keep_components=True to override)"
            )

    x_out = (np.arange(nx) - (nx - 1) / 2.0) * mm_per_pixel
    y_out = (np.arange(ny) - (ny - 1) / 2.0) * mm_per_pixel

    # ---- scatter ---------------------------------------------------------- #
    if scatter_data is None:
        say(f"Reading       : {os.path.basename(scatter_file)}")
        hdr, sdata = read_fgd(scatter_file)
        x_src, y_src = fgd_grid(hdr, sdata)
    else:
        hdr, sdata, x_src, y_src = scatter_data

    scatter, p_exact, p_full = resample_scatter(
        sdata,
        x_src,
        y_src,
        x_out,
        y_out,
        smooth=smooth,
        interp_order=interp_order,
        dtype=dtype,
    )
    f_encl = p_exact / p_full
    # Always normalize against the WHOLE map: that pins the halo to FRED's
    # surface brightness, which is what makes the contrast and the crossover
    # radius independent of the stamp size.
    scatter /= dtype(p_full)  # now sums to f_encl
    scatter_sum = f_encl
    say(
        f"Scatter       : P_full = {p_full:.6e} W over "
        f"{x_src.min():g}..{x_src.max():g} x {y_src.min():g}..{y_src.max():g} mm"
    )
    say(
        f"                stamp holds {100 * f_encl:.3f}% of the FRED halo, so it "
        f"carries"
    )
    say(
        f"                {dp_pct(desired_power, f_encl)} of the source flux "
        f"(FRED surface brightness kept)"
    )

    # ---- core ------------------------------------------------------------- #
    if core_psf is not None:
        core_small = psfsim.normalize_psf(core_psf)
        say(f"Core          : user-supplied, {core_small.shape}")
        # square grids could use psfsim.center_crop_or_pad; this also handles
        # rectangular ones without allocating a max(nx, ny)**2 float64 scratch.
        core = np.zeros((ny, nx), dtype=dtype)
        sy, sx = core_small.shape
        oy, ox = (ny - sy) // 2, (nx - sx) // 2
        core[oy : oy + sy, ox : ox + sx] = core_small.astype(dtype)
        del core_small
    else:
        say(
            f"Core          : Airy, lam = {wavelength * 1e9:g} nm, jitter = "
            f"{jitter_sigma_mas:g} mas, radial + {inner_npix} pix exact core "
            f"@ x{oversample}"
        )
        core, pscale_check = render_airy_core(
            wavelength,
            fnum,
            D,
            pixel_size_um,
            (ny, nx),
            jitter_sigma_mas=jitter_sigma_mas,
            inner_npix=inner_npix,
            oversample=oversample,
            dtype=dtype,
        )
        assert np.isclose(pscale_check, pixel_scale_mas, rtol=1e-9)

    c_encl = float(core.sum(dtype=np.float64))
    if c_encl < 0.999:
        say(f"                core clipped by the stamp: {c_encl:.6f} enclosed")

    # ---- combine (in place, to keep peak memory near 2x the output) -------- #
    dp = float(desired_power)
    core_keep = core.copy() if keep_components else None
    scatter_keep = scatter.copy() if keep_components else None

    total = core  # reuse the buffer
    total *= dtype(1.0 - dp)
    scatter *= dtype(dp)
    total += scatter
    del scatter
    if not keep_components:
        core = None

    raw_sum = float(total.sum(dtype=np.float64))
    say(
        f"Combined      : sum = {raw_sum:.8f}   "
        f"[(1-dp)*{c_encl:.6f} + dp*{scatter_sum:.6f}]"
    )
    if renormalize:
        total /= dtype(raw_sum)
        say(
            f"                renormalized -> sum = "
            f"{float(total.sum(dtype=np.float64)):.8f}"
        )

    meta = {
        "SCATFILE": (os.path.basename(scatter_file)[:47], "source FRED map"),
        "SENSOR": (str(geom.get("label", "custom")), "detector"),
        "WAVELEN": (float(wavelength) * 1e9, "core wavelength [nm]"),
        "JITTER": (float(jitter_sigma_mas), "jitter sigma [mas]"),
        "TELDIAM": (D, "primary diameter [m]"),
        "FNUM": (fnum, "f-number"),
        "SMOOTH": (bool(smooth), "smooth interp (else exact area rebin)"),
        "RENORM": (bool(renormalize), "output renormalized to sum 1"),
        "PFULL": (p_full, "integral of the full FRED map [W]"),
    }

    res = TotalPSF(
        data=total,
        pixel_size_um=pixel_size_um,
        pixel_scale_mas=pixel_scale_mas,
        mm_per_pixel=mm_per_pixel,
        desired_power=dp,
        scatter_enclosed_frac=f_encl,
        core_enclosed_frac=c_encl,
        scatter_in_array=dp * scatter_sum / raw_sum
        if renormalize
        else dp * scatter_sum,
        core=core_keep,
        scatter=scatter_keep,
        meta=meta,
        renorm_scale=(1.0 / raw_sum) if renormalize else 1.0,
    )

    res.settings = {
        # Prefer the FRED run recorded in the .fgd header over the file name:
        # it survives the map being passed in pre-parsed via scatter_data.
        "scatter_file": os.path.basename(scatter_file),
        "scatter_dir": os.path.dirname(scatter_file),
        "scatter_preparsed": scatter_data is not None,
        "call": call_kwargs,
        "fred_run": (
            f"{str(hdr.get('FRED_FILENAME', '?')).strip(chr(34))}   "
            f"{str(hdr.get('DATETIME', '')).strip(chr(34))}"
            if "FRED_FILENAME" in hdr
            else None
        ),
        "rays_traced": hdr.get("NUMBER_OF_RAYS_USED"),
        "core_kind": "user-supplied array"
        if core_psf is not None
        else "Airy (analytic, radial + exact inner block)",
        "extent": extent if extent_mm is None else f"extent_mm={extent_mm}",
        "inner_npix": inner_npix,
        "oversample": oversample,
        "interp_order": interp_order,
        "smooth": bool(smooth),
        "renormalize": bool(renormalize),
        "max_size_mb": max_size_mb,
        "compress": compress,
    }

    if output is not None:
        res.save(output, overwrite=overwrite, compress=compress)
    if report is not None:
        write_psf_report(res, report, fits_path=output)
    if plot:
        res.plot()
    return res


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #


def plot_scatter_verify(
    scatter_data,
    desired_power=0.007,
    D=3.065,
    fnum=15.0,
    wavelength=450e-9,
    half_mm=6.0,
    n=400001,
    n_plot=6001,
    psf=None,
    path=None,
    ax=None,
):
    """
    Reproduce Scott's "Comparison of PSFs" check, in irradiance.

    Everything here is computed **independently of the PSF pipeline**: the core
    from the closed-form Airy (:func:`airy_irradiance`), the halo straight from
    the ``.fgd`` cut, each scaled to its share of a 1 W source.  Optionally
    overplots a :class:`TotalPSF` so the pipeline can be checked against both.

    Returns a dict of the measured quantities: peak irradiances, their contrast,
    and the two crossover radii (the fringed one, where Airy ring maxima stop
    exceeding the halo, and the envelope one, which is what an azimuthal average
    sees).
    """
    import matplotlib.pyplot as plt

    use_plot_style()
    hdr, sdata, x_src, y_src = scatter_data
    dp = float(desired_power)

    dx = float(np.mean(np.diff(x_src)))
    dy = float(np.mean(np.diff(y_src)))
    p_full = float((np.asarray(sdata, dtype=float) * dx * dy).sum())
    k = dp / p_full  # rescale FRED to carry dp of 1 W
    cut = np.asarray(sdata[sdata.shape[0] // 2], dtype=float) * k

    # measure on a fine grid, draw on a coarse one -- at 0.00675 mm fringe
    # spacing a 12 mm span holds ~1800 rings, which render as a solid block.
    r = np.linspace(-half_mm, half_mm, int(n))
    core = airy_irradiance(r, D, fnum, wavelength, power=1.0 - dp)
    env = airy_irradiance(r, D, fnum, wavelength, power=1.0 - dp, envelope=True)
    halo = np.interp(np.abs(r), x_src[x_src >= 0], cut[x_src >= 0])

    rp = np.linspace(-half_mm, half_mm, int(n_plot))
    core_p = airy_irradiance(rp, D, fnum, wavelength, power=1.0 - dp)
    env_p = airy_irradiance(rp, D, fnum, wavelength, power=1.0 - dp, envelope=True)
    halo_p = np.interp(np.abs(rp), x_src[x_src >= 0], cut[x_src >= 0])

    def cross(a, b):
        d = a - b
        i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        return float(abs(r[i[-1]])) if len(i) else float("nan")

    out = {
        "airy_peak": float(core.max()),
        "scatter_peak": float(halo.max()),
        "contrast": float(core.max() / halo.max()),
        "crossover_fringed_mm": cross(core, halo),
        "crossover_envelope_mm": cross(env, halo),
        "p_full": p_full,
        "rescale": k,
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7.5))
    else:
        fig = ax.figure

    ax.plot(
        rp,
        core_p,
        color="#3b76af",
        lw=0.5,
        label=f"Airy PSF (analytic), norm={100 * (1 - dp):.1f}%",
    )
    ax.plot(rp, env_p, color="#003d5b", lw=1.5, ls=":", label="Airy envelope")
    ax.plot(
        rp,
        halo_p,
        color="#66a182",
        lw=2.2,
        label=f"Scatter PSF from FRED, norm={100 * dp:.2f}%",
    )
    if psf is not None:
        rr, prof, _ = radial_multiscale(
            psf.data, psf.mm_per_pixel, max(1, int(np.ceil(max(psf.shape) / 1100)))
        )
        pipe = prof * psf.irradiance_scale
        m = rr <= half_mm
        ax.plot(
            rr[m],
            pipe[m],
            color="0.15",
            lw=1.5,
            ls="--",
            label="this pipeline (azimuthal mean)",
        )
        ax.plot(-rr[m], pipe[m], color="0.15", lw=1.5, ls="--")
        # peak PIXEL, not the first annulus mean (which annulus-averaging dilutes)
        out["pipeline_peak"] = float(psf.data.max()) * psf.irradiance_scale
        out["pipeline_peak_ratio"] = out["pipeline_peak"] / out["airy_peak"]
        # agreement against analytic envelope + halo over 0.05..half_mm
        ref = airy_irradiance(
            rr[m], D, fnum, wavelength, power=1.0 - dp, envelope=True
        ) + np.interp(rr[m], x_src[x_src >= 0], cut[x_src >= 0])
        w = rr[m] > 0.05
        out["pipeline_vs_analytic_med"] = float(np.median(pipe[m][w] / ref[w]))
        out["pipeline_vs_analytic_max"] = float(
            np.max(np.abs(pipe[m][w] / ref[w] - 1.0))
        )

    # contrast arrow between the two peaks
    ax.annotate(
        "",
        xy=(0.35, out["airy_peak"]),
        xytext=(0.35, out["scatter_peak"]),
        arrowprops=dict(arrowstyle="<->", color="#d1495b", lw=2),
    )
    ax.text(
        0.55,
        np.sqrt(out["airy_peak"] * out["scatter_peak"]),
        f"{out['contrast']:.2e}",
        color="#d1495b",
        fontsize=17,
        va="center",
    )
    for xc, style, lbl in (
        (
            out["crossover_fringed_mm"],
            "-",
            f"fringe crossover {out['crossover_fringed_mm']:.2f} mm",
        ),
        (
            out["crossover_envelope_mm"],
            ":",
            f"envelope crossover {out['crossover_envelope_mm']:.2f} mm",
        ),
    ):
        ax.axvline(xc, color="#d1495b", lw=1.2, ls=style)
        ax.axvline(-xc, color="#d1495b", lw=1.2, ls=style, label=lbl)

    ax.set_yscale("log")
    ax.set_xlim(-half_mm, half_mm)
    ax.set_ylim(1e-11, 1e6)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_xlabel("Distance [mm]")
    ax.set_ylabel("Irradiance [W mm$^{-2}$ per W of source]")
    ax.set_title("Comparison of PSFs — Airy core vs FRED scatter halo", fontsize=13)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

    bx = ax.twiny()
    aspmm = arcsec_per_mm(D, fnum)
    bx.set_xlim(-half_mm * aspmm, half_mm * aspmm)
    bx.set_xlabel("Distance [arcsec]")
    bx.grid(False)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
        print(f"  wrote {path}")
    return fig, ax, out


def block_reduce(a, factor, how="mean"):
    """
    Decimate for display / radial profiling.  Nothing in wcc_etc does this, and it
    is unavoidable here: ``plotting.plot_image_mpl`` would try to draw 244 Mpix and
    ``radial_data`` loops once per annulus, which is hopeless on a full-size stamp.
    ``how='max'`` keeps peaks visible under a log stretch; ``'mean'`` preserves the
    surface brightness, which is what a radial profile wants.
    """
    if factor <= 1:
        return a
    ny = (a.shape[0] // factor) * factor
    nx = (a.shape[1] // factor) * factor
    blocks = a[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    return blocks.max(axis=(1, 3)) if how == "max" else blocks.mean(axis=(1, 3))


def radial_multiscale(arr, mm_per_pixel, fac, inner_npix=601):
    """
    Radial profile that keeps the core at full resolution.

    ``radial_data`` on a block-mean-decimated array dilutes the peak by ``fac**2``
    and smears the first arcsecond, which makes both the profile and the curve of
    growth wrong where they matter most.  So: profile the full-resolution central
    crop out to ``inner_npix//2``, the decimated array beyond that, and splice.

    Returns ``(r_mm, mean_per_full_res_pixel, power_per_annulus)``; the power is
    already in units of "fraction of total flux", so ``cumsum`` is the EE curve.
    """
    arr = np.asarray(arr, dtype=float)
    ny, nx = arr.shape
    r_all, m_all, p_all = [], [], []

    r_start = -1.0
    n_in = min(int(inner_npix), ny, nx)
    if fac > 1 and n_in >= 9:
        h = n_in // 2
        cy, cx = ny // 2, nx // 2
        rd = radial_data(arr[cy - h : cy + h + 1, cx - h : cx + h + 1])
        r = np.asarray(rd.r, dtype=float)
        k = r <= h
        r_all.append(r[k] * mm_per_pixel)
        m_all.append(np.asarray(rd.mean, dtype=float)[k])
        p_all.append(
            np.asarray(rd.mean, dtype=float)[k] * np.asarray(rd.numel, dtype=float)[k]
        )
        r_start = float(h)

    rd = radial_data(block_reduce(arr, fac, how="mean"))
    r = np.asarray(rd.r, dtype=float) * fac  # back to full-res pixels
    k = r > r_start
    r_all.append(r[k] * mm_per_pixel)
    m_all.append(np.asarray(rd.mean, dtype=float)[k])
    p_all.append(
        np.asarray(rd.mean, dtype=float)[k]
        * np.asarray(rd.numel, dtype=float)[k]
        * fac**2
    )

    return (np.concatenate(r_all), np.concatenate(m_all), np.concatenate(p_all))


def plot_total_psf(res, max_display_side=1100, dyn_range=1e11, figsize=(11, 9.5)):
    """
    Three-panel diagnostic, drawn with the package's own helpers --
    ``plotting.plot_image_mpl`` for the image, ``radial_data.radial_data`` for the
    azimuthal averages, ``airy.psf_to_encircled_energy`` for the curve of growth.

    * log-stretched image with the IMX455 / HWK4123 footprints overlaid;
    * radial profiles of core, scatter and total, so the scatter contribution and
      the radius where it takes over are both readable;
    * encircled energy, marking the fraction of flux the scatter carries.
    """
    import matplotlib.pyplot as plt

    use_plot_style()
    ny, nx = res.data.shape
    fac = max(1, int(np.ceil(max(ny, nx) / max_display_side)))
    mm_disp = res.mm_per_pixel * fac
    aspmm = arcsec_per_mm(res.meta["TELDIAM"][0], res.meta["FNUM"][0])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax_img = fig.add_subplot(gs[0, :])
    ax_rad = fig.add_subplot(gs[1, 0])
    ax_ee = fig.add_subplot(gs[1, 1])

    # ---- image ------------------------------------------------------------ #
    # NOT plotting.plot_image_mpl here: its stretch='log' goes through
    # ImageNormalize(data, LogStretch()), which is built for detector images of
    # modest dynamic range and crushes an 11-decade PSF to a flat field. LogNorm
    # is the right tool. The other two panels do use the package reducers.
    from matplotlib.colors import LogNorm

    disp = block_reduce(res.data, fac, how="max") * res.irradiance_scale
    vmax = float(disp.max())
    vmin = max(
        float(disp[disp > 0].min()) if np.any(disp > 0) else vmax / dyn_range,
        vmax / dyn_range,
    )
    ny_d, nx_d = disp.shape
    ext = [
        -nx_d / 2 * mm_disp,
        nx_d / 2 * mm_disp,
        -ny_d / 2 * mm_disp,
        ny_d / 2 * mm_disp,
    ]
    im = ax_img.imshow(
        np.clip(disp, vmin, None),
        origin="lower",
        extent=ext,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax_img, pad=0.02, fraction=0.046)
    cb.ax.set_ylabel("irradiance [W mm$^{-2}$ per W in]", fontsize=10)
    ax_img.set_anchor("C")  # keep a square stamp centred in its slot
    ax_img.set_xlabel("X [mm]")
    ax_img.set_ylabel("Y [mm]")
    ax_img.grid(False)
    for geom in SENSORS.values():
        w, h = geom["width_mm"], geom["height_mm"]
        ax_img.add_patch(
            plt.Rectangle(
                (-w / 2, -h / 2),
                w,
                h,
                edgecolor=geom["color"],
                facecolor="none",
                lw=1.6,
                ls="--",
                label=geom["label"],
            )
        )
    ax_img.legend(fontsize=9, loc="upper right")
    ax_img.set_title(
        f"{res.meta.get('SENSOR', ('?',))[0]}   {nx} x {ny} pix @ "
        f"{res.pixel_size_um:g} um  ({res.pixel_scale_mas:.2f} mas/pix, "
        f"{nx * res.mm_per_pixel:.1f} x {ny * res.mm_per_pixel:.1f} mm)\n"
        f"scatter = {100 * res.desired_power:.3g}% of total flux instrument-wide; "
        f"this stamp holds {100 * res.scatter_enclosed_frac:.1f}% of the halo "
        f"= {100 * res.scatter_in_array:.4g}% of the flux"
        + (f"   [display block-max x{fac}]" if fac > 1 else ""),
        fontsize=11,
    )

    # ---- radial profiles --------------------------------------------------- #
    dp = res.desired_power
    w_scale = res.renorm_scale
    series = [("total", res.data, "0.25", 1.0, "--")]
    if res.core is not None:
        series.append(("core (Airy)", res.core, "#00798c", (1 - dp) * w_scale, "-"))
    if res.scatter is not None:
        series.append(("scatter", res.scatter, "#d1495b", dp * w_scale, "-"))

    ee_curve = None
    irr = res.irradiance_scale  # fraction/pixel -> W/mm^2 per W of source
    for label, arr, colour, weight, ls in series:
        r_mm, prof, power = radial_multiscale(arr, res.mm_per_pixel, fac)
        ax_rad.plot(r_mm, prof * weight * irr, color=colour, ls=ls, lw=1.6, label=label)
        if label == "total":
            ee_curve = (r_mm, np.cumsum(power))
    ax_rad.set_xscale("log")
    ax_rad.set_yscale("log")
    top = float(res.data.max()) * irr
    ax_rad.set_ylim(top / dyn_range, top * 3)
    ax_rad.set_xlabel("radius [mm]")
    ax_rad.set_ylabel("irradiance [W mm$^{-2}$ per W of source]")
    ax_rad.set_title("Radial profile", fontsize=11)
    ax_rad.legend(fontsize=9)
    bx = ax_rad.twiny()
    bx.set_xscale("log")
    bx.set_xlim(*[v * aspmm for v in ax_rad.get_xlim()])
    bx.set_xlabel("radius [arcsec]")
    bx.grid(False)

    # ---- encircled energy --------------------------------------------------- #
    # From the same multiscale annulus powers, so the inner half of the curve is
    # not diluted by the display decimation. (airy.psf_to_encircled_energy is the
    # package equivalent, but it needs np.indices on the full grid -- 3.9 GB here.)
    r_ee, ee = ee_curve
    ax_ee.plot(r_ee, ee, color="#00798c", lw=1.8, label="total")
    ax_ee.axhline(
        1 - res.scatter_in_array,
        color="0.4",
        ls="--",
        lw=1,
        label=f"core share = {1 - res.scatter_in_array:.4f}",
    )
    if res.scatter is not None:
        _, _, p_sc = radial_multiscale(res.scatter, res.mm_per_pixel, fac)
        ax_ee.plot(
            r_ee,
            np.cumsum(p_sc) * dp * w_scale,
            color="#d1495b",
            lw=1.6,
            label=f"scatter only ({res.scatter_in_array:.2e})",
        )
    ax_ee.set_xscale("log")
    ax_ee.set_yscale("log")
    ax_ee.set_ylim(1e-4, 1.5)
    ax_ee.set_xlabel("radius [mm]")
    ax_ee.set_ylabel("enclosed fraction of total flux")
    ax_ee.set_title("Curve of growth", fontsize=11)
    ax_ee.legend(fontsize=8, loc="lower right")
    cx = ax_ee.twiny()
    cx.set_xscale("log")
    cx.set_xlim(*[v * aspmm for v in ax_ee.get_xlim()])
    cx.set_xlabel("radius [arcsec]")
    cx.grid(False)

    fig.tight_layout()
    return fig, (ax_img, ax_rad, ax_ee)
