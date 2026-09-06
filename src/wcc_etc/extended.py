"""Extended (spatially resolved) scene elements: Sersic-profile hosts.

Angular parameterization only — ``r_eff``, ``dx``, ``dy`` in arcsec, ``pa`` in
degrees CCW from +x — the convention shared with wcc-sim's SersicComponent,
Pandeia and the HST ETC. Distance / redshift are deliberately not modelled.
"""

import numpy as np
from astropy.modeling.models import Sersic2D
from scipy.special import gamma, gammaincinv

__all__ = ["SERSIC_DEFAULTS", "sersic_total_over_amplitude", "render_sersic"]

SERSIC_DEFAULTS = {"n": 1.0, "ellip": 0.0, "pa": 0.0, "dx": 0.0, "dy": 0.0}


def sersic_total_over_amplitude(n, r_eff_pix, ellip=0.0):
    """F_total / I_e for a Sersic2D profile with ``r_eff`` in pixels (analytic).

    Sersic2D's ``amplitude`` is the surface brightness at r_eff per pixel area;
    integrating over the plane gives 2 pi n r_eff^2 (1-ellip) e^bn bn^(-2n) Gamma(2n).
    """
    bn = float(gammaincinv(2.0 * n, 0.5))
    return float(
        2.0
        * np.pi
        * n
        * r_eff_pix**2
        * (1.0 - ellip)
        * np.exp(bn)
        * bn ** (-2.0 * n)
        * gamma(2.0 * n)
    )


def render_sersic(
    profile, plate_scale_arcsec, npix, oversample=11, center=None, total=True
):
    """Sersic profile on an (npix, npix) detector grid, attached to ``center``.

    ``profile`` holds ``r_eff`` (arcsec, required) and any of SERSIC_DEFAULTS.
    ``center`` is the source position (cx, cy) the ``dx``/``dy`` offset is
    measured from; default is the grid centre, as in ImageSimulator.
    Returns the per-pixel profile: unit total flux (analytic, so light off the
    grid is lost, not renormalized) when ``total`` is True, else I/I_e.
    """
    p = SERSIC_DEFAULTS | profile
    if center is None:
        center = ((npix - 1) / 2.0, (npix - 1) / 2.0)
    r_eff_pix = p["r_eff"] / plate_scale_arcsec
    amp = (
        1.0 / sersic_total_over_amplitude(p["n"], r_eff_pix, p["ellip"])
        if total
        else 1.0
    )
    model = Sersic2D(
        amplitude=amp,
        r_eff=r_eff_pix,
        n=p["n"],
        x_0=center[0] + p["dx"] / plate_scale_arcsec,
        y_0=center[1] + p["dy"] / plate_scale_arcsec,
        ellip=p["ellip"],
        theta=np.radians(p["pa"]),
    )
    xx = np.arange(npix, dtype=float)
    img = model(xx[None, :], xx[:, None])  # pixel centres, native grid

    # A Sersic cusp changes across a pixel; the rest of the profile does not.
    # Re-evaluate a core box on the oversampled sub-grid and pixel-average it.
    # ponytail: box half-width capped at 64 px (wcc-sim's choice); beyond that
    # the profile is smooth at pixel scale. Raise the cap if r_eff < 3 px ever
    # matters at large npix.
    half = int(np.clip(np.ceil(2.0 * r_eff_pix), 8, 64))
    x0, y0 = int(round(model.x_0.value)), int(round(model.y_0.value))
    xlo, xhi = max(x0 - half, 0), min(x0 + half + 1, npix)
    ylo, yhi = max(y0 - half, 0), min(y0 + half + 1, npix)
    if xlo < xhi and ylo < yhi:
        off = (np.arange(oversample) + 0.5) / oversample - 0.5
        fx = (np.arange(xlo, xhi, dtype=float)[:, None] + off).ravel()
        fy = (np.arange(ylo, yhi, dtype=float)[:, None] + off).ravel()
        fine = model(fx[None, :], fy[:, None])
        img[ylo:yhi, xlo:xhi] = fine.reshape(
            yhi - ylo, oversample, xhi - xlo, oversample
        ).mean(axis=(1, 3))
    return img
