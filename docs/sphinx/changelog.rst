Changelog
=========

This page summarizes notable changes. The authoritative history is the git log
and the design specs under ``docs/superpowers/``.

Unreleased
----------

- **Source-weighted PSF wavelength** (issue #65, option **A**). The PSF is now
  rendered at the photon-weighted effective wavelength of the bandpass times
  the source SED — :attr:`~wcc_etc.Simulation.effective_wavelength` — instead
  of the filter pivot wavelength, which is the same for every source. Through
  the WCC broad band this moves the render wavelength from 582.1 nm to 540.7 nm
  (O5V) or 716.5 nm (M5V) and corrects the modelled central-pixel fraction by
  −9 % to +36 %; the pivot-only model predicted premature saturation for red
  sources. New :mod:`wcc_etc.spectral` module with
  :func:`~wcc_etc.effective_wavelength` and
  :func:`~wcc_etc.photon_weighted_subbands`.
- **Polychromatic PSFs**: :class:`~wcc_etc.PolychromaticPSF` coadds a base PSF
  rendered at ``n_sub`` equal-photon-weight sub-bands;
  :meth:`~wcc_etc.Simulation.polychromatic_psf` builds one for a simulation's
  band and source.
- **Defocus products carry their optics.** The bundled Zemax Huygens defocus
  PSFs now ship as FITS with ``PIXSCALE`` / ``WAVELEN`` / ``FNUM`` /
  ``DEFOCUSW`` headers (regenerate with
  :func:`wcc_etc.psfsim.huygens_txt_to_fits`), and
  ``DEFOCUS_1WAVE_PATH`` / ``DEFOCUS_2WAVE_PATH`` point at them; the raw
  ``.txt`` exports remain as ``DEFOCUS_*_TXT_PATH``.
  :class:`~wcc_etc.DefocusPSF` and :class:`~wcc_etc.CustomPSF` gained a
  ``wavelength_scaling`` argument (``"despace"`` — the default — ``"waves"``,
  or ``"none"``) and now **raise** instead of silently ignoring
  ``ctx.wavelength_m`` and ``ctx.fnum`` when they carry no reference metadata.
  See :ref:`resampled-psf-scaling`.

  *Migration*: a :class:`~wcc_etc.CustomPSF` built from a bare array needs
  either ``ref_wavelength_m`` and ``ref_fnum``, or
  ``wavelength_scaling="none"``.

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
