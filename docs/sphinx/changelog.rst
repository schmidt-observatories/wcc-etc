Changelog
=========

This page summarizes notable changes. The authoritative history is the git log
and the design specs under ``docs/superpowers/``.

Unreleased
----------

Plotting and PSF introspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New public API, each replacing something the docs or a notebook previously had
to hand-roll:

- :attr:`~wcc_etc.Simulation.default_psf` — the PSF used when ``psf=`` is
  omitted, replacing reads of the private ``_default_psf``. Falls back to a
  diffraction-limited :class:`~wcc_etc.AiryPSF`. Every internal call site,
  including :meth:`~wcc_etc.Simulation.polychromatic_psf`, now goes through it.
- :meth:`~wcc_etc.ImageSimulator.render_psf` — the bare normalized PSF on the
  detector grid, replacing manual construction of the private render context.
  It builds that context the same way ``simulate`` does, so it honours the
  source-weighted effective wavelength from issue #65.
- :attr:`~wcc_etc.ImageSimulator.plate_scale_mas` and
  :attr:`~wcc_etc.ImageSimulator.default_psf`.
- ``plot_bandpass_mpl`` and :meth:`~wcc_etc.Sensor.show` — filter throughput
  curves.
- ``label=`` and ``**kwargs`` passthrough on ``plot_radial_mpl``,
  ``plot_encircled_energy_mpl`` and ``plot_bandpass_mpl``, so overlaid curves
  can carry a legend without reaching into ``ax.lines``.
- ``wave=`` and ``flux_unit=`` on
  :meth:`~wcc_etc.scene.SceneElement.get_spectrum` and
  :meth:`~wcc_etc.scene.SceneElement.show`, so a source can be plotted over a
  chosen grid in F\ :sub:`lambda` rather than only PHOTLAM.

Behaviour change
~~~~~~~~~~~~~~~~

- :meth:`~wcc_etc.ImageSimulator.simulate` now defaults to ``default_psf``
  instead of always :class:`~wcc_etc.AiryPSF`. For a simulator built by
  ``from_sensorfilter`` with a defocused label, ``simulate()`` without ``psf=``
  now renders that filter's defocus PSF, matching what ``get_image_snr`` has
  always done. Every other construction path is unaffected. Pass ``psf=``
  explicitly to override.

- **Host and diffuse elements are classified by** ``surface_brightness``
  **, not by element name** (issue #63). A host given as an ordinary
  integrated magnitude used to have its *total* count rate added to
  **every** detector pixel, overstating its shot-noise contribution by
  ``n_pix / enclosed_fraction`` — 64.6x for the representative mag 20 source /
  mag 16 host scene (38.6 M e- of host charge instead of 597 k e-). Such an
  element is now treated as unresolved and co-located with the source and is
  rendered through the same PSF; only ``surface_brightness=True`` elements are
  per-pixel. See :ref:`host-spatial-treatment`.

  This also unifies the charge budget: :meth:`~wcc_etc.Simulation.get_peak_pixel`,
  :meth:`~wcc_etc.Simulation.is_saturated`, the
  :meth:`~wcc_etc.Simulation.get_image_snr` saturation check and
  ``ImageSimulator.simulate`` previously disagreed about whether host charge
  counted, so identical scenes gave different saturation answers depending on
  which API you called. They now all see the same electrons, and
  ``get_peak_pixel`` includes an unresolved host (it no longer drops it).

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
