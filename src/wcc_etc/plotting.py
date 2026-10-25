# src/wcc_etc/plotting.py
"""Convenience plotting helpers for SimulatedImage (matplotlib + bokeh).

Every function accepts either a SimulatedImage (positional `source`) or explicit
raw arrays as keywords, so the Flask web portal can pass arrays directly.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.colors import ListedColormap, LogNorm

from .airy import psf_to_encircled_energy
from .psfsim import calc_hwhm
from .radial_data import radial_data

__all__ = [
    "WCC_STYLE",
    "set_wcc_style",
    "apply_ax_settings",
    "plot_image_mpl",
    "plot_image_row_mpl",
    "plot_radial_mpl",
    "plot_encircled_energy_mpl",
    "plot_bandpass_mpl",
    "plot_lightcurve_mpl",
    "plot_image_bokeh",
    "plot_image_row_bokeh",
    "plot_radial_bokeh",
    "plot_encircled_energy_bokeh",
    "plot_lightcurve_bokeh",
    "plot_snr",
    "plot_snr_vs_mag",
    "plot_sweep",
    "plot_peak_pixel",
    "plot_exptime_for_snr",
    "plot_snr_vs_aperture",
    "plot_psf",
    "plot_host",
]

# Shared look-and-feel for WCC matplotlib plots: STIX math/text font, higher dpi,
# inward minor+major ticks, and faint gridlines. Grid is enabled globally (1D line
# plots pick it up automatically); the 2D image plotters explicitly disable it.
WCC_STYLE = {
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.formatter.useoffset": False,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.3,
}


def set_wcc_style():
    """Apply the WCC plotting style to matplotlib's global rcParams.

    Call once (e.g. at the top of a notebook) so every subsequent matplotlib
    figure shares the same look. Mutates global state by design."""
    plt.rcParams.update(WCC_STYLE)


def apply_ax_settings(ax, ticksize=None):
    """Apply the standard per-axis tweaks: minor ticks, tick padding, faint grid.

    Unlike set_wcc_style this touches one axis rather than global rcParams, so it
    is safe to call on a figure that also carries someone else's style."""
    ax.minorticks_on()
    ax.tick_params(pad=3, labelsize=12 if ticksize is None else ticksize)
    ax.grid(lw=0.3, alpha=0.3)
    return ax


def _resolve_inputs(
    source=None,
    *,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
):
    """Resolve (image_e, image_clean, saturation_mask, pixel_scale_mas) from a
    SimulatedImage or explicit arrays."""
    if source is not None and hasattr(source, "image_e"):
        return (
            source.image_e,
            source.image_clean,
            source.saturation_mask,
            source.pixel_scale_mas,
        )
    return image_e, image_clean, saturation_mask, pixel_scale_mas


def _make_norm(data, stretch, vmin=None, vmax=None, dyn_range=1e4):
    """Normalization for the given stretch ('log'|'hist'|'linear').

    'log' is a matplotlib LogNorm from the data minimum to its maximum, with the
    floor no lower than `vmax / dyn_range` so a bare PSF (zeros far out) keeps a
    readable range and the colorbar ticks read as decades. Pixels at or below
    the floor (read-noise negatives) are clipped to it, not masked. `vmin` and
    `vmax` override the data-derived range. An image with no positive pixel
    cannot be log-scaled and gets None (linear)."""
    data = np.asarray(data, dtype=float)
    if stretch == "hist":
        return ImageNormalize(stretch=HistEqStretch(data))
    if stretch == "log":
        if vmax is None:
            vmax = float(np.nanmax(data))
        if vmax <= 0:
            return None
        if vmin is None:
            positive = data[data > 0]
            vmin = float(positive.min()) if positive.size else vmax / dyn_range
            vmin = max(vmin, vmax / dyn_range)
        return LogNorm(vmin=vmin, vmax=vmax, clip=True)
    return None  # linear: no normalization object


ANGULAR_UNITS = {"mas": 1.0, "arcsec": 1e-3}


def _image_extent(ny, nx, pixel_scale_mas, units):
    """imshow extent centered on the grid in mas or arcsec, or None for pixels."""
    if units in ANGULAR_UNITS and pixel_scale_mas:
        scale = pixel_scale_mas * ANGULAR_UNITS[units]
        hx = nx / 2.0 * scale
        hy = ny / 2.0 * scale
        return [-hx, hx, -hy, hy]
    return None


def axis_unit(units, extent):
    """The unit printed on the image axes: the angular unit, or 'pix'."""
    return units if extent else "pix"


def _saturation_overlay(saturation_mask):
    """A masked array of ones, masked everywhere the pixel is NOT saturated, so an
    imshow of it colors only the saturated pixels."""
    mask = np.asarray(saturation_mask, dtype=bool)
    return np.ma.masked_where(~mask, np.ones(mask.shape, dtype=float))


def plot_image_mpl(
    source=None,
    *,
    noise=True,
    show_saturation=False,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    stretch="log",
    cmap="viridis",
    vmin=None,
    vmax=None,
    colorbar=True,
    title="",
    units="pix",
    sat_color="red",
    sat_alpha=0.6,
    ax=None,
):
    """Plot a single simulated detector image with equal x/y scale.

    noise=True shows the noisy image_e; noise=False shows the noiseless
    image_clean. show_saturation overlays the saturation mask. units='mas' or
    'arcsec' labels the axes in angle using pixel_scale_mas; 'pix' in pixels.
    Returns (fig, ax)."""
    ie, ic, sat, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    data = ie if noise else ic
    ny, nx = data.shape
    extent = _image_extent(ny, nx, ps, units)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    norm = _make_norm(data, stretch, vmin=vmin, vmax=vmax)
    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        extent=extent,
        aspect="equal",
    )

    if show_saturation and sat is not None:
        ax.imshow(
            _saturation_overlay(sat),
            origin="lower",
            cmap=ListedColormap([sat_color]),
            alpha=sat_alpha,
            extent=extent,
            aspect="equal",
        )

    ax.set_xlabel(f"X [{axis_unit(units, extent)}]")
    ax.set_ylabel(f"Y [{axis_unit(units, extent)}]")
    ax.set_title(title)
    ax.grid(False)  # no gridlines over a 2D image
    if colorbar:
        fig.colorbar(im, ax=ax)
    return fig, ax


def plot_image_row_mpl(
    source=None,
    *,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    stretch="log",
    cmap="viridis",
    units="pix",
    sat_cmap="gray",
    figsize=(15, 5),
    axes=None,
):
    """Three panels: PSF+noise, PSF (no noise), and the saturation mask.

    The two image panels share a common color scale (computed from the noisy
    image). Returns (fig, axes) where axes has length 3."""
    ie, ic, sat, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    ny, nx = ie.shape
    extent = _image_extent(ny, nx, ps, units)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].figure

    norm = _make_norm(ie, stretch)
    vmin = float(np.min(ie)) if norm is None else None
    vmax = float(np.max(ie)) if norm is None else None

    titles = ["PSF + Noise", "PSF (No Noise)", "Saturation Mask"]
    for ax, data, title in zip(axes[:2], [ie, ic], titles[:2]):
        ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel(f"X [{axis_unit(units, extent)}]")
        ax.set_ylabel(f"Y [{axis_unit(units, extent)}]")

    axes[2].imshow(
        np.asarray(sat, dtype=float),
        origin="lower",
        cmap=sat_cmap,
        extent=extent,
        aspect="equal",
    )
    axes[2].set_title(titles[2])
    axes[2].set_xlabel(f"X [{axis_unit(units, extent)}]")
    axes[2].set_ylabel(f"Y [{axis_unit(units, extent)}]")
    for ax in axes:
        ax.grid(False)  # no gridlines over the 2D image panels
    return fig, axes


def plot_radial_mpl(
    source=None,
    *,
    noise=False,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    units="mas",
    annulus_width=1,
    show_hwhm=True,
    title="",
    label=None,
    ax=None,
    **kwargs,
):
    """Azimuthally-averaged radial profile of the simulated image.

    Uses image_clean by default (noise=True uses image_e). units='mas' scales the
    radius by pixel_scale_mas. `label` and any extra keyword arguments are passed
    to ax.plot, so several profiles can be overlaid on one axis with a legend.
    Returns (fig, ax, (radius, profile))."""
    ie, ic, _, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    data = ie if noise else ic
    rd = radial_data(np.asarray(data), annulus_width=annulus_width)
    r = np.asarray(rd.r, dtype=float)
    prof = np.asarray(rd.mean, dtype=float)
    if units == "mas" and ps:
        r = r * ps

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(r, prof, label=label, **kwargs)
    if show_hwhm:
        hwhm = calc_hwhm(r, prof)
        if len(hwhm):
            ax.axvline(
                hwhm[0],
                color="orange",
                linestyle="--",
                lw=1,
                label="HWHM={:.2f}".format(hwhm[0]),
            )
            ax.legend(loc="upper right")
    ax.set_xlabel("Radius [mas]" if (units == "mas" and ps) else "Radius [pix]")
    ax.set_ylabel("Azimuthally-Averaged Signal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    return fig, ax, (r, prof)


def plot_encircled_energy_mpl(
    source=None,
    *,
    noise=False,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    units="mas",
    ee_target=0.9,
    title="",
    label=None,
    ax=None,
    **kwargs,
):
    """Encircled-energy curve (normalized to 1) of the simulated image.

    Uses image_clean by default. units='mas' uses the mas radius from
    psf_to_encircled_energy; units='pix' divides by pixel_scale_mas. ee_target
    defaults to 0.9 (marks the 90% encircled-energy radius); pass ee_target=None
    to disable the marker or another fraction to override. `label` and any extra
    keyword arguments are passed to ax.plot, so several curves can be overlaid on
    one axis with a legend. Returns (fig, ax, (radius, ee))."""
    ie, ic, _, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    data = ie if noise else ic
    scale = ps if ps else 1.0
    r_mas, _psf1d, ee = psf_to_encircled_energy(np.asarray(data), scale, scale)
    r = r_mas if (units == "mas" and ps) else r_mas / scale
    if ee[-1] > 0:
        ee = ee / ee[-1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(r, ee, label=label, **kwargs)
    if ee_target is not None:
        idx = int(np.searchsorted(ee, ee_target))
        if 0 < idx < len(r):
            ax.axvline(
                r[idx],
                color="red",
                linestyle="--",
                lw=1,
                label="EE={:.2f} @ {:.1f}".format(ee_target, r[idx]),
            )
            ax.axhline(ee_target, color="gray", linestyle=":", lw=1)
            ax.legend(loc="lower right")
    ax.set_xlabel("Radius [mas]" if (units == "mas" and ps) else "Radius [pix]")
    ax.set_ylabel("Encircled Energy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    return fig, ax, (r, ee)


def resolve_bandpass_curve(bandpass, wave=None):
    """Return (wavelength_angstrom, throughput) for a bandpass-like object.

    Accepts a synphot SpectralElement, or anything carrying one on a `.bandpass`
    attribute (e.g. a Sensor). `wave` overrides the sampling grid and may be a
    plain array in Angstrom or an astropy Quantity."""
    bp = getattr(bandpass, "bandpass", bandpass)
    if wave is None:
        wave = bp.waveset
    if not hasattr(wave, "to"):
        wave = np.asarray(wave, dtype=float) * u.AA
    return wave.to(u.AA).value, np.asarray(bp(wave).value, dtype=float)


def plot_bandpass_mpl(
    bandpass,
    *,
    wave=None,
    title="",
    label=None,
    ax=None,
    **kwargs,
):
    """Plot a filter throughput curve.

    `bandpass` is a synphot SpectralElement or an object carrying one (a Sensor).
    `wave` overrides the sampling grid; `label` and extra keyword arguments go to
    ax.plot so several filters can be overlaid. Returns
    (fig, ax, (wavelength_angstrom, throughput))."""
    w, t = resolve_bandpass_curve(bandpass, wave=wave)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(w, t, label=label, **kwargs)
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Throughput")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    return fig, ax, (w, t)


def _finish_bokeh(obj, return_):
    """Return a bokeh object as the figure ('obj'), a standalone HTML string
    ('html'), or an (script, div) components tuple ('components')."""
    if return_ == "obj":
        return obj
    if return_ == "html":
        from bokeh.embed import file_html
        from bokeh.resources import CDN

        return file_html(obj, CDN)
    if return_ == "components":
        from bokeh.embed import components

        return components(obj)
    raise ValueError("return_ must be 'obj', 'html', or 'components'")


def plot_image_bokeh(
    source=None,
    *,
    noise=True,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    units="pix",
    palette="Viridis256",
    title="",
    width=400,
    height=400,
    return_="obj",
):
    """Bokeh single-image plot with equal x/y scale (match_aspect=True).

    ``return_`` selects the output form (see _finish_bokeh)."""
    from bokeh.plotting import figure

    ie, ic, _sat, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    data = np.asarray(ie if noise else ic, dtype=float)
    ny, nx = data.shape
    if units == "mas" and ps:
        x0, y0, dw, dh = -nx / 2.0 * ps, -ny / 2.0 * ps, nx * ps, ny * ps
        axis_label = "mas"
    else:
        x0, y0, dw, dh = 0, 0, nx, ny
        axis_label = "pix"

    p = figure(
        width=width,
        height=height,
        match_aspect=True,
        title=title,
        x_axis_label="X [{}]".format(axis_label),
        y_axis_label="Y [{}]".format(axis_label),
    )
    p.image(image=[data], x=x0, y=y0, dw=dw, dh=dh, palette=palette)
    return _finish_bokeh(p, return_)


def plot_image_row_bokeh(
    source=None,
    *,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    units="pix",
    palette="Viridis256",
    sat_palette="Greys256",
    width=300,
    height=300,
    return_="obj",
):
    """Bokeh 3-panel row: PSF+noise, PSF (no noise), saturation mask.

    ``return_`` selects the output form (see _finish_bokeh)."""
    from bokeh.layouts import row
    from bokeh.plotting import figure

    ie, ic, sat, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    ie = np.asarray(ie, dtype=float)
    ic = np.asarray(ic, dtype=float)
    sat = np.asarray(sat, dtype=float)
    ny, nx = ie.shape
    if units == "mas" and ps:
        x0, y0, dw, dh, lbl = -nx / 2.0 * ps, -ny / 2.0 * ps, nx * ps, ny * ps, "mas"
    else:
        x0, y0, dw, dh, lbl = 0, 0, nx, ny, "pix"

    panels = []
    for data, title, pal in [
        (ie, "PSF + Noise", palette),
        (ic, "PSF (No Noise)", palette),
        (sat, "Saturation Mask", sat_palette),
    ]:
        p = figure(
            width=width,
            height=height,
            match_aspect=True,
            title=title,
            x_axis_label="X [{}]".format(lbl),
            y_axis_label="Y [{}]".format(lbl),
        )
        p.image(image=[data], x=x0, y=y0, dw=dw, dh=dh, palette=pal)
        panels.append(p)
    return _finish_bokeh(row(*panels), return_)


def plot_radial_bokeh(
    source=None,
    *,
    noise=False,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    units="mas",
    annulus_width=1,
    title="",
    width=500,
    height=350,
    return_="obj",
):
    """Bokeh radial profile. ``return_`` selects the output form (see _finish_bokeh)."""
    from bokeh.plotting import figure

    ie, ic, _sat, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    data = ie if noise else ic
    rd = radial_data(np.asarray(data), annulus_width=annulus_width)
    r = np.asarray(rd.r, dtype=float)
    prof = np.asarray(rd.mean, dtype=float)
    use_mas = units == "mas" and ps
    if use_mas:
        r = r * ps
    p = figure(
        width=width,
        height=height,
        title=title,
        x_axis_label="Radius [mas]" if use_mas else "Radius [pix]",
        y_axis_label="Azimuthally-Averaged Signal",
    )
    p.line(r, prof, line_width=2)
    return _finish_bokeh(p, return_)


def plot_encircled_energy_bokeh(
    source=None,
    *,
    noise=False,
    image_e=None,
    image_clean=None,
    saturation_mask=None,
    pixel_scale_mas=None,
    units="mas",
    ee_target=0.9,
    title="",
    width=500,
    height=350,
    return_="obj",
):
    """Bokeh encircled-energy curve. ee_target defaults to 0.9 (marks the 90%
    encircled-energy radius); pass ee_target=None to disable. ``return_`` selects the
    output form."""
    from bokeh.plotting import figure

    ie, ic, _sat, ps = _resolve_inputs(
        source,
        image_e=image_e,
        image_clean=image_clean,
        saturation_mask=saturation_mask,
        pixel_scale_mas=pixel_scale_mas,
    )
    data = ie if noise else ic
    scale = ps if ps else 1.0
    r_mas, _psf1d, ee = psf_to_encircled_energy(np.asarray(data), scale, scale)
    use_mas = units == "mas" and ps
    r = r_mas if use_mas else r_mas / scale
    if ee[-1] > 0:
        ee = ee / ee[-1]
    p = figure(
        width=width,
        height=height,
        title=title,
        y_range=(0, 1.02),
        x_axis_label="Radius [mas]" if use_mas else "Radius [pix]",
        y_axis_label="Encircled Energy",
    )
    p.line(r, ee, line_width=2)
    if ee_target is not None:
        from bokeh.models import Span

        idx = int(np.searchsorted(ee, ee_target))
        if 0 < idx < len(r):
            p.add_layout(
                Span(
                    location=r[idx],
                    dimension="height",
                    line_color="red",
                    line_dash="dashed",
                )
            )
            p.add_layout(
                Span(
                    location=ee_target,
                    dimension="width",
                    line_color="gray",
                    line_dash="dotted",
                )
            )
    return _finish_bokeh(p, return_)


def _resolve_lightcurve(
    source=None, *, time=None, flux=None, flux_clean=None, flux_err=None
):
    """Resolve (time, flux, flux_clean, flux_err) from a LightCurve or arrays."""
    if source is not None and hasattr(source, "flux_clean"):
        return source.time, source.flux, source.flux_clean, source.flux_err
    return time, flux, flux_clean, flux_err


def plot_lightcurve_mpl(
    source=None,
    *,
    show_noise=True,
    show_model=True,
    time=None,
    flux=None,
    flux_clean=None,
    flux_err=None,
    ax=None,
    **kw,
):
    """Matplotlib light-curve plot: clean model line, noisy points with error
    bars, or both. Accepts a LightCurve (positional) or raw arrays."""
    time, flux, flux_clean, flux_err = _resolve_lightcurve(
        source, time=time, flux=flux, flux_clean=flux_clean, flux_err=flux_err
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=kw.pop("figsize", (7, 4)))
    else:
        fig = ax.figure
    if show_noise and flux is not None:
        ax.errorbar(
            time,
            flux,
            yerr=flux_err,
            fmt="o",
            ms=3,
            color="0.35",
            ecolor="0.7",
            elinewidth=0.8,
            capsize=0,
            zorder=1,
            label="Simulated",
        )
    if show_model and flux_clean is not None:
        ax.plot(time, flux_clean, "-", color="C3", lw=1.8, zorder=2, label="Model")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Flux")
    ax.set_title("Transit Light Curve")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", frameon=False)
    return fig, ax


def plot_lightcurve_bokeh(
    source=None,
    *,
    show_noise=True,
    show_model=True,
    time=None,
    flux=None,
    flux_clean=None,
    flux_err=None,
    width=600,
    height=350,
    return_="obj",
):
    """Bokeh light-curve plot. ``return_`` selects the output form (see
    _finish_bokeh). Accepts a LightCurve (positional) or raw arrays."""
    from bokeh.plotting import figure

    time, flux, flux_clean, flux_err = _resolve_lightcurve(
        source, time=time, flux=flux, flux_clean=flux_clean, flux_err=flux_err
    )
    p = figure(
        width=width,
        height=height,
        title="Transit Light Curve",
        x_axis_label="Time",
        y_axis_label="Relative Flux",
    )
    if show_noise and flux is not None:
        p.scatter(
            time, flux, size=4, color="#595959", alpha=0.8, legend_label="Simulated"
        )
        if flux_err is not None:
            lower = np.asarray(flux) - flux_err
            upper = np.asarray(flux) + flux_err
            p.segment(time, lower, time, upper, color="#b3b3b3", line_width=0.8)
    if show_model and flux_clean is not None:
        p.line(time, flux_clean, color="crimson", line_width=2, legend_label="Model")
    if p.legend:
        p.legend.location = "bottom_right"
    return _finish_bokeh(p, return_)


# --------------------------------------------------------------------------- #
#  Simulation-level curves (SNR, saturation, exposure time) and bare PSFs      #
# --------------------------------------------------------------------------- #


def seconds(time):
    """Exposure time(s) as a float array in seconds; a Quantity is converted."""
    return np.atleast_1d(u.Quantity(time, u.s).value).astype(float)


def resolve_ax(ax=None, **subplots_kw):
    """Return (fig, ax): a new figure when `ax` is None, else `ax` and its figure."""
    if ax is None:
        return plt.subplots(**subplots_kw)
    return ax.figure, ax


def legend_if_labelled(ax):
    """Draw a legend when something on the axis carries a label, one entry per
    distinct label (overlaid calls redraw the same ceiling line)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys())


def plot_snr(sim, time, *, psf=None, n_reads=None, warn=False, ax=None, **kw):
    """SNR vs exposure time for a Simulation.

    `time` is an array of exposure times (seconds, or a Quantity). `psf`,
    `n_reads` and `warn` go to `get_snr`; `kw` goes to ax.plot, so several
    simulations overlay on one axis with `label=`. Returns (fig, ax)."""
    time = seconds(time)
    snr = sim.get_snr(time=time, psf=psf, n_reads=n_reads, warn=warn)["snr"]
    fig, ax = resolve_ax(ax)
    ax.plot(time, snr, **kw)
    ax.set_xlabel("Exposure time [s]")
    ax.set_ylabel("SNR")
    legend_if_labelled(ax)
    return fig, ax


def plot_snr_vs_mag(
    sim, mags, *, time=None, psf=None, n_reads=None, warn=False, ax=None, **kw
):
    """SNR vs source magnitude at a fixed exposure time.

    Uses `get_image_snr(mags=...)`, which rescales the source off one cached
    render instead of rebuilding the scene per point. `psf`, `n_reads` and
    `warn` go to `get_image_snr`; `kw` goes to ax.plot. Returns (fig, ax)."""
    mags = np.asarray(mags, dtype=float)
    snr = sim.get_image_snr(time=time, mags=mags, psf=psf, n_reads=n_reads, warn=warn)[
        "snr"
    ]
    fig, ax = resolve_ax(ax)
    ax.plot(mags, snr, **kw)
    ax.set_xlabel("Source magnitude")
    ax.set_ylabel("SNR")
    legend_if_labelled(ax)
    return fig, ax


def plot_sweep(
    sim, key, values, *, time=None, psf=None, n_reads=None, warn=False, ax=None, **kw
):
    """SNR vs any `update()` key (e.g. 'host__mag', 'host__r_eff', 'n_reads').

    Each value is set with `sim.update(key=value)` and the SNR evaluated at
    `time`; the original value is restored afterwards, even if a point raises.
    `psf`, `n_reads` and `warn` go to `get_snr`; `kw` goes to ax.plot.
    Returns (fig, ax)."""
    original = sim.get_parameter(key)[0]
    snr = []
    try:
        for value in values:
            sim.update(**{key: value})
            snr.append(
                sim.get_snr(time=time, psf=psf, n_reads=n_reads, warn=warn)["snr"]
            )
    finally:
        sim.update(**{key: original})
    fig, ax = resolve_ax(ax)
    ax.plot(values, snr, **kw)
    ax.set_xlabel(key)
    ax.set_ylabel("SNR")
    legend_if_labelled(ax)
    return fig, ax


def plot_peak_pixel(
    sim,
    time,
    *,
    units="adu",
    psf=None,
    n_reads=None,
    show_ceiling=True,
    ax=None,
    **kw,
):
    """Peak pixel vs exposure time on log axes, with the saturation ceiling.

    The dashed line is `sensor.saturation_e` (the smaller of full well and ADC
    clip) in the chosen units; a dotted vertical line marks the earliest time
    at which `is_saturated` is True. `time` is in seconds or a Quantity; `psf`
    and `n_reads` go to `get_peak_pixel`. Returns (fig, ax)."""
    time = seconds(time)
    peak = sim.get_peak_pixel(time, units=units, n_reads=n_reads, psf=psf).value
    fig, ax = resolve_ax(ax)
    (line,) = ax.loglog(time, peak, **kw)
    if show_ceiling:
        sensor = sim.sensor
        if units.lower() == "adu":
            ceiling = (sensor.saturation_e / sensor.gain).to(u.ct) + sensor.bias_level
            unit_label = "ADU"
        else:
            ceiling, unit_label = sensor.saturation_e, "e-"
        ceiling = float(getattr(ceiling, "value", ceiling))
        ax.axhline(
            ceiling,
            color="0.3",
            ls="--",
            lw=1,
            label=f"saturation ({sensor.saturation_limit}) = {ceiling:.0f} {unit_label}",
        )
        sat = np.asarray(sim.is_saturated(time, n_reads=n_reads, psf=psf), dtype=bool)
        if sat.any():
            t_sat = time[sat].min()
            ax.axvline(
                t_sat,
                color=line.get_color(),
                ls=":",
                lw=1,
                label=f"saturates at {t_sat:.2g} s",
            )
    ax.set_xlabel("Exposure time [s]")
    ax.set_ylabel("Peak pixel [ADU]" if units.lower() == "adu" else "Peak pixel [e-]")
    legend_if_labelled(ax)
    return fig, ax


def plot_exptime_for_snr(
    sim, snr, *, psf=None, n_reads=None, warn=False, ax=None, **kw
):
    """Exposure time needed to reach each target SNR (PSF-aware inverse).

    `psf`, `n_reads` and `warn` go to `get_image_exptime_for_snr`; `kw` goes to
    ax.plot. Returns (fig, ax)."""
    snr = np.asarray(snr, dtype=float)
    times = [
        sim.get_image_exptime_for_snr(s, psf=psf, n_reads=n_reads, warn=warn)["time_s"]
        for s in snr
    ]
    fig, ax = resolve_ax(ax)
    ax.plot(snr, times, **kw)
    ax.set_xlabel("Target SNR")
    ax.set_ylabel("Exposure time [s]")
    legend_if_labelled(ax)
    return fig, ax


def plot_snr_vs_aperture(
    sim,
    radii,
    *,
    time=None,
    psf=None,
    n_reads=None,
    warn=False,
    show_markers=True,
    ax=None,
    **kw,
):
    """SNR vs fixed aperture radius, with the default and optimized apertures.

    A square marks the aperture `get_image_snr` picks by default and a circle
    the `optimize=True` radius. `psf`, `n_reads` and `warn` go to
    `get_image_snr`; `kw` goes to ax.plot. Returns (fig, ax)."""
    radii = np.asarray(radii, dtype=float)
    common = dict(time=time, psf=psf, n_reads=n_reads, warn=warn)
    snr = [sim.get_image_snr(r_aper_mas=r, **common)["snr"] for r in radii]
    fig, ax = resolve_ax(ax)
    (line,) = ax.plot(radii, snr, **kw)
    if show_markers:
        default = sim.get_image_snr(**common)
        best = sim.get_image_snr(optimize=True, **common)
        ax.plot(
            default["r_aper_mas"], default["snr"], "s", color=line.get_color(), ms=7
        )
        ax.plot(best["r_aper_mas"], best["snr"], "o", color=line.get_color(), ms=9)
    ax.set_xlabel("Aperture radius [mas]")
    ax.set_ylabel("SNR")
    legend_if_labelled(ax)
    return fig, ax


def plot_psf(
    imsim, psf=None, *, jitter_sigma_mas=None, npix=None, units="mas", ax=None, **kw
):
    """The bare normalized PSF on the detector grid (log stretch by default).

    Wraps `ImageSimulator.render_psf` and `plot_image_mpl`; `kw` reaches the
    latter (`title=`, `cmap=`, `colorbar=`, `stretch=` ...). Returns (fig, ax)."""
    img = imsim.render_psf(psf, jitter_sigma_mas=jitter_sigma_mas, npix=npix)
    kw.setdefault("stretch", "log")
    return plot_image_mpl(
        image_clean=img,
        noise=False,
        pixel_scale_mas=imsim.plate_scale_mas,
        units=units,
        ax=ax,
        **kw,
    )


def plot_host(
    sim,
    time,
    *,
    psf=None,
    seed=0,
    npix=128,
    oversample=11,
    units="mas",
    axes=None,
    **kw,
):
    """A transient in its host: the simulated frame next to the host alone.

    Left: one noisy ``ImageSimulator.simulate`` realization of the whole scene
    for ``time`` seconds (``seed`` fixes the noise). Right: the PSF-convolved,
    noiseless host image for the same time (``Simulation.extended_rate_image``),
    i.e. what the galaxy contributes under the transient. Both panels share
    ``units`` and ``kw`` (``cmap=``, ``stretch=`` ...). Returns (fig, axes).
    """
    from .psfsim import ImageSimulator

    imsim = ImageSimulator(sim, npix=npix, oversample=oversample)
    frame = imsim.simulate(time, psf=psf, seed=seed)
    bundle = sim._image_render_bundle(
        psf if psf is not None else sim.default_psf, None, npix, oversample
    )
    t_sec = float(np.squeeze(seconds(time)))
    host = bundle["extended_rate_image"] * t_sec

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    else:
        fig = axes[0].figure
    plot_image_mpl(
        frame,
        noise=True,
        units=units,
        ax=axes[0],
        **({"title": f"{t_sec:g} s frame"} | kw),
    )
    plot_image_mpl(
        image_clean=host,
        noise=False,
        pixel_scale_mas=imsim.plate_scale_mas,
        units=units,
        ax=axes[1],
        **({"title": "host alone, no noise"} | kw),
    )
    return fig, axes
