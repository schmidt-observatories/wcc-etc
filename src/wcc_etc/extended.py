"""Extended (spatially resolved) scene elements: Sersic-profile hosts.

Angular parameterization only — ``r_eff``, ``dx``, ``dy`` in arcsec, ``pa`` in
degrees CCW from +x — the convention shared with wcc-sim's SersicComponent,
Pandeia and the HST ETC. Distance / redshift are deliberately not modelled.
"""

import numpy as np
from astropy.modeling.models import Sersic2D
from scipy.integrate import dblquad
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
    measured from; default is the integer grid centre ``(npix-1)//2``
    (``psfsim.grid_center``), the convention every PSF render shares.
    Returns the per-pixel profile: unit total flux (analytic, so light off the
    grid is lost, not renormalized) when ``total`` is True, else I/I_e.

    Pixel integration: the 5x5 pixels around the cusp are integrated exactly
    (adaptive quadrature split at the cusp), a core box around them is
    pixel-averaged on an ``oversample`` x ``oversample`` sub-grid, and the
    rest is sampled at pixel centres. Validated against analytic enclosed-flux
    bounds for n in [0.5, 8] and r_eff down to 0.18 pixel: the on-grid total
    is within ~1e-4 of the analytic value at the default ``oversample``, and
    never exceeds 1 (see tests/imaging/test_extended.py).
    """
    p = SERSIC_DEFAULTS | profile
    if center is None:
        center = ((npix - 1) // 2, (npix - 1) // 2)
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

    # The cusp itself (I(0) = I_e e^bn, ~2000 I_e at n=4) falls off within a
    # fraction of a sub-pixel for compact hosts, so no uniform grid converges.
    # Integrate the 5x5 pixels around it exactly, splitting each pixel at the
    # cusp so the quadrature only ever sees it at a corner. Uniform sampling of
    # the remaining pixels is converged to <1e-4 of the total at oversample=11.
    cx, cy = model.x_0.value, model.y_0.value
    args = tuple(float(v) for v in model.parameters)

    def sersic(y, x):
        return Sersic2D.evaluate(x, y, *args)

    def pixel_integral(ix, iy):
        xs = sorted(
            {ix - 0.5, ix + 0.5} | ({cx} if ix - 0.5 < cx < ix + 0.5 else set())
        )
        ys = sorted(
            {iy - 0.5, iy + 0.5} | ({cy} if iy - 0.5 < cy < iy + 0.5 else set())
        )
        return sum(
            dblquad(sersic, xa, xb, ya, yb, epsabs=0.0, epsrel=1e-6)[0]
            for xa, xb in zip(xs, xs[1:])
            for ya, yb in zip(ys, ys[1:])
        )

    for iy in range(max(y0 - 2, 0), min(y0 + 3, npix)):
        for ix in range(max(x0 - 2, 0), min(x0 + 3, npix)):
            img[iy, ix] = pixel_integral(ix, iy)
    return img
