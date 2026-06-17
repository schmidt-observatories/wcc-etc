Changelog
=========

This page summarizes notable changes. The authoritative history is the git log
and the design specs under ``docs/superpowers/``.

v0.5.x
------

- **2-D image SNR is now the default.** :meth:`~wcc_etc.Simulation.get_snr`
  delegates to :meth:`~wcc_etc.Simulation.get_image_snr` and returns a
  **dict** (index ``["snr"]``). The analytic Airy form remains as
  ``get_snr_airy`` (deprecated) for cross-checks, paired with
  :meth:`~wcc_etc.Simulation.get_exptime_for_snr`.
- **Convenience plotting** (:mod:`wcc_etc.plotting`): matplotlib and bokeh
  variants of image, image-row, radial-profile, and encircled-energy plots,
  plus a shared house style (``set_wcc_style`` / ``WCC_STYLE``) and dispatch
  methods on :class:`~wcc_etc.SimulatedImage`.
- **Parametric source spectra** in :func:`~wcc_etc.get_scene`: ``blackbody``,
  ``flat`` (:math:`F_\nu`/:math:`F_\lambda`), ``powerlaw``, and ``emission``;
  shape parameters are rebuildable via ``update``.
- **PSF simulator and PSF-aware SNR**: :class:`~wcc_etc.AiryPSF`,
  :class:`~wcc_etc.DefocusPSF`, :class:`~wcc_etc.CustomPSF`,
  :class:`~wcc_etc.ImageSimulator`, and
  :meth:`~wcc_etc.Simulation.get_image_snr` with aperture optimization.
- **Saturation flagging**: :meth:`~wcc_etc.Simulation.get_peak_pixel` and
  :meth:`~wcc_etc.Simulation.is_saturated`, plus sensor ``bit_depth`` /
  ``bias_level`` / ``adc_max`` properties.

v0.4.0
------

- Refactored into a :class:`~wcc_etc.Scene` abstraction managing source, host,
  and background elements.

v0.3.0
------

- Substantial refactoring into separate ``Sensor`` and ``Telescope`` objects.

v0.2.0
------

- Added background spectrum support.
- Improved documentation.

v0.1.0
------

- Initial release: basic ETC functionality, Airy-disk / PSF support, and SNR
  calculation.
