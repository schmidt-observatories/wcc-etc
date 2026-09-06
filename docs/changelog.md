# Changelog

## Unreleased

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
