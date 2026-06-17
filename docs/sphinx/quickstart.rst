Quick start
===========

This page walks through the core workflow in a few minutes. Every snippet
below uses only the public top-level API exported from ``wcc_etc``.

The mental model
----------------

A calculation is built from three pieces, combined into a
:class:`~wcc_etc.Simulation`:

.. code-block:: text

   Scene  (what is on the sky)        ─┐
   Sensor (the detector + filter)      ├──►  Simulation  ──►  SNR / exptime / images
   Telescope (aperture, jitter)       ─┘

- A :class:`~wcc_etc.Scene` holds a **source**, an optional **host**, and an
  optional **background**, each an astronomical spectrum with a magnitude.
- A :class:`~wcc_etc.Sensor` couples a detector configuration (read noise, dark
  current, gain, full well, pixel size, bit depth) with a **filter** bandpass.
- A :class:`~wcc_etc.Simulation` ties them to a telescope and exposes the
  photometry: SNR, exposure time, saturation, simulated images.

1. Build a scene
----------------

.. code-block:: python

   from wcc_etc import get_scene

   scene = get_scene(
       "G5V",                 # a Pickles stellar template
       mag=15,                # source magnitude
       background="zodi",     # zodiacal-light sky background
       bandpass="johnson_r",
   )

See :doc:`user_guide/scenes` for stellar/galaxy templates, parametric spectra
(blackbody, flat, power law, emission lines), host galaxies, and backgrounds.

2. Create a simulation for a sensor
-----------------------------------

Sensors are addressed as ``"kind:band"`` strings:

.. code-block:: python

   from wcc_etc import Simulation

   sim = Simulation.from_sensor_and_scene("sony:r", scene)

Available sensors include ``sony:r`` / ``sony:bb`` (Sony IMX455, 16-bit) and
``qcmos:r`` (Hamamatsu qCMOS, 12-bit). See :doc:`configuration` for the full
list and how the TOML configs are structured.

3. Compute signal-to-noise
--------------------------

``get_snr`` returns a **dictionary**. The signal-to-noise itself is under the
``"snr"`` key:

.. code-block:: python

   result = sim.get_snr(time=60)        # 60-second exposure
   result["snr"]                        # signal-to-noise ratio
   result["signal_e"]                   # source electrons in the aperture
   result["noise_e"]                    # total noise (electrons)
   result["r_aper_mas"]                 # aperture radius used (mas)
   result["enclosed_fraction"]          # PSF fraction inside the aperture

``time`` can be an array to sweep exposure time in one call:

.. code-block:: python

   import numpy as np
   sweep = sim.get_snr(time=np.array([10, 30, 60, 120]))
   sweep["snr"]                         # ndarray of SNR values

.. note::

   ``get_snr`` now delegates to the PSF-aware, image-based
   :meth:`~wcc_etc.Simulation.get_image_snr`. The old analytic Airy-disk
   formula is still available as ``get_snr_airy`` (deprecated) for cross-checks.
   See :doc:`user_guide/snr`.

4. Invert: exposure time for a target SNR
-----------------------------------------

.. code-block:: python

   t = sim.get_image_exptime_for_snr(snr=100)   # seconds to reach SNR = 100

5. Change parameters and recompute
----------------------------------

Update scene or instrument parameters in place with double-underscore keys,
then recompute. The render cache is invalidated automatically:

.. code-block:: python

   sim.update(source__mag=22)            # fainter source
   sim.get_snr(60)["snr"]                # lower than before

6. Check saturation
-------------------

.. code-block:: python

   sim.is_saturated(60)                  # True / False for a 60 s frame
   sim.get_peak_pixel(60, units="adu")   # brightest pixel value

7. Simulate a detector image and plot it
----------------------------------------

.. code-block:: python

   import wcc_etc
   from wcc_etc import ImageSimulator, AiryPSF

   wcc_etc.set_wcc_style()               # shared house plotting style

   imsim = ImageSimulator.from_sensor_and_scene("sony:r", scene, npix=300)
   img = imsim.simulate(time=30, psf=AiryPSF(), add_noise=True, seed=0)

   img.image_e                           # noisy electron image (ndarray)
   img.saturation_mask                   # boolean saturation mask
   img.to_adu()                          # image in ADU
   img.to_fitsimg()                      # FitsImg for photometry

   img.plot_image_row()                  # PSF+noise / PSF / saturation panels
   img.plot_radial(units="mas")          # azimuthally-averaged radial profile
   img.plot_encircled_energy(units="mas", ee_target=0.8)

Where to go next
----------------

- :doc:`user_guide/index` — the conceptual reference for each subsystem.
- :doc:`tutorials` — the six runnable notebooks.
- :doc:`api/index` — the full auto-generated API reference.
