# Changelog

## Unreleased

- Packaging: `MANIFEST.in` is an explicit allowlist and `package-data` names only the two
  bundled defocus PSF FITS, so a build no longer depends on what is in the developer's
  working tree (188 MB from a dev tree before). The unused `data/flats/` (36 MB) and the
  Pickles `dat_uvi/` atlas are removed; `hsiao07.dat` is now the phase-0 slice (2 columns,
  130 kB instead of 8 MB) and loads with `get_scene("hsiao07", ...)` instead of raising.
  Wheel and sdist are 8 MB; CI fails a build over 20 MB or carrying local products (#62, #101).
- Sersic hosts: the 5x5 pixels around the cusp are now integrated exactly (adaptive
  quadrature split at the cusp) instead of sampled on the `oversample` sub-grid, which
  overshot the analytic total by up to 3x (n=4) and 4 orders of magnitude (n=8) for
  hosts with r_eff below a pixel. On-grid totals are now within ~1e-4 of analytic for
  n in [0.5, 8] and r_eff >= 0.18 pixel, and never exceed 1 (#84).
- One centre convention for PSFs and hosts: a render with `center=None` puts the source
  on the integer pixel `(npix-1)//2` (`psfsim.grid_center`), on odd and even grids;
  `recenter` shifts from the PSF's actual centre, so `center=(cx, cy)` is honoured to
  sub-pixel precision for Airy and custom PSFs alike; and `extended_rate_image` places
  the host by convolution with the source PSF, so a zero-offset host sits on the source
  (it was half a pixel off on even grids, including the default 128-pixel SNR grid) (#84).

- Sersic-profile hosts (structured background):
  `host_prop={"profile": "sersic", "r_eff": 1.0, "n": 1, "ellip": 0.3, "pa": 45, "dx": 0.8, "dy": 0}`
  with angular parameters (arcsec / degrees). `mag` is the total host magnitude, or
  μ_e (mag/arcsec² at r_eff) with `surface_brightness=True`. The profile is rendered and
  PSF-convolved once and shared by `ImageSimulator.simulate`, `get_snr`,
  `get_image_exptime_for_snr`, `get_peak_pixel` and `is_saturated`. New module
  `wcc_etc.extended`; the deprecated analytic Airy path warns that it ignores profiles.

## v0.1.0
Initial public release.

- Scene abstraction managing source, host, and background elements
- Selectable magnitude system (Vega by default, AB optional)
- Separate sensor and telescope objects
- 2-D image SNR by default, plus exposure-time inversion
- PSF simulator (Airy, defocus, custom) and PSF-aware SNR with aperture optimization
- Saturation flagging
- Parametric source spectra (blackbody, flat, power law, emission)
- Background spectrum support and filter selection
- Convenience plotting (matplotlib and bokeh)
