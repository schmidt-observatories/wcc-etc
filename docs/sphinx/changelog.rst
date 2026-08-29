Changelog
=========

This page summarizes notable changes. The authoritative history is the git log
and the design specs under ``docs/superpowers/``.

v0.1.0
------

Initial public release. Highlights:

- **Scene abstraction**: :class:`~wcc_etc.Scene` and
  :class:`~wcc_etc.scene.SceneElement` manage source, host, and background
  elements, built through :func:`~wcc_etc.get_scene`.
- **Selectable magnitude system**: ``magsys="vegamag"`` (default) or
  ``magsys="abmag"`` (case-insensitive) on
  :class:`~wcc_etc.scene.SceneElement` and :func:`~wcc_etc.get_scene`. Note
  that magnitudes without an explicit ``magsys`` — including the built-in
  ``zodi`` background (``mag=22.5``) — are Vega magnitudes. The AB offset is
  negligible in Johnson *V* (~0.002 mag) but grows toward the red (~0.26 mag
  in *R*, ~1.9 mag in *K*).
- **Separate** :class:`~wcc_etc.Sensor` **and** :class:`~wcc_etc.Telescope`
  **objects**, with sensor ``bit_depth`` / ``bias_level`` / ``adc_max``
  properties.
- **2-D image SNR by default**: :meth:`~wcc_etc.Simulation.get_snr` delegates
  to :meth:`~wcc_etc.Simulation.get_image_snr` and returns a **dict** (index
  ``["snr"]``). The analytic Airy form remains as ``get_snr_airy``
  (deprecated) for cross-checks.
- **Exposure-time inversion**:
  :meth:`~wcc_etc.Simulation.get_exptime_for_snr`.
- **PSF simulator and PSF-aware SNR**: :class:`~wcc_etc.AiryPSF`,
  :class:`~wcc_etc.DefocusPSF`, :class:`~wcc_etc.CustomPSF`,
  :class:`~wcc_etc.ImageSimulator`, and aperture optimization in
  :meth:`~wcc_etc.Simulation.get_image_snr`.
- **Saturation flagging**: :meth:`~wcc_etc.Simulation.get_peak_pixel` and
  :meth:`~wcc_etc.Simulation.is_saturated`.
- **Parametric source spectra** in :func:`~wcc_etc.get_scene`: ``blackbody``,
  ``flat`` (:math:`F_\nu`/:math:`F_\lambda`), ``powerlaw``, and ``emission``;
  shape parameters are rebuildable via ``update``.
- **Background spectrum support** and a selection of filters.
- **Convenience plotting** (:mod:`wcc_etc.plotting`): matplotlib and bokeh
  variants of image, image-row, radial-profile, and encircled-energy plots,
  plus a shared house style (``set_wcc_style`` / ``WCC_STYLE``) and dispatch
  methods on :class:`~wcc_etc.SimulatedImage`.
