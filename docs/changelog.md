# Changelog

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
