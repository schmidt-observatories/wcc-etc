Changelog
=========

This page summarizes notable changes. The authoritative history is the git log
and the design specs under ``docs/superpowers/``.

Unreleased
----------

Fixes
~~~~~

- The qCMOS (HWK4123) now carries its two gain modes. ``qcmos.toml`` has a
  ``[sensor.gain_modes.<mode>]`` table per mode and a ``gain_mode`` default of
  ``"high"`` (32x: 7.42 ADU/e, 0.25 e- read noise); ``"low"`` (1x: 0.242 ADU/e,
  2.33 e- read noise) is selected with ``gain_mode="low"`` on
  :meth:`~wcc_etc.Simulation.from_sensorfilter`,
  :meth:`~wcc_etc.Simulation.from_sensor_and_scene` or
  :meth:`~wcc_etc.Sensor.from_name`. The 12-bit ADC is real: in high gain it
  clips at 552 e-, far below the 7500 e- full well (was 7000), so faint-source
  "saturation" there is physical; low gain is well-limited. Gain is stored in
  e-/ADU with its provenance stated; the old ``gain = 0.112`` was a single
  unlabelled value (#94, #61).
- Defocused filters (``zwo:r+1``, ``zwo:bb2``, ...) no longer use the 70 mas
  Airy-core aperture by default, which held only 2 to 8 percent of the flux and
  under-reported SNR by 3 to 7x. With no aperture argument the aperture now
  follows the PSF: 70 mas for the Airy PSF, the SNR-optimized radius otherwise.
  ``Simulation(r_aper_mas=...)`` still pins a fixed radius. The default
  ``r_aper_mas`` is ``None``; ``sim.meta`` only carries it when set (#95).
- ``get_scene("emission", mag=..., lines=...)`` now raises instead of silently
  renormalizing the absolute line fluxes to the magnitude (a 400x error for a
  1e-15 erg/s/cm^2 line). ``get_scene_element("emission", ...)`` defaults to
  ``mag=None``, and ``surface_brightness=True`` with ``mag=None`` scales the
  per-arcsec^2 spectrum by the pixel area instead of crashing (#96).
- The x2 read-noise margin on the Sony/ZWO path is now the config key
  ``read_noise_margin`` (2.0 in ``zwo.toml`` and ``qcmos.toml``, whose
  ``read_noise`` is back to the measured 0.28 e-) instead of a hidden factor
  in ``Sensor.from_config``. Results are unchanged. ``get_snr`` reports the
  effective per-pixel read noise as ``read_noise_e`` (#97).

H-alpha filters
~~~~~~~~~~~~~~~

- Three H-alpha bands on the Sony (IMX455) sensors now have EOL telescope + WCC
  throughput curves and are usable with
  :meth:`~wcc_etc.Simulation.from_sensorfilter`: ``zwo:halpha2`` (2 nm,
  position 10), ``zwo:halpha6`` (6 nm, position 9, formerly the N-II
  placeholder) and ``zwo:halpha20`` (20 nm, position 21, formerly the O-III
  placeholder). The ``nii`` / ``oiii`` keys are removed; ``hbeta`` / ``heii``
  remain unimplemented placeholders.

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
