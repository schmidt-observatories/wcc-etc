PSFs and image simulation
=========================

``wcc-etc`` can render a realistic detector image of the scene, including the
point-spread function, telescope jitter, sky background, Poisson and read
noise, and per-pixel saturation.

PSF models
----------

All PSF models subclass :class:`~wcc_etc.psfsim.PSFSource` and implement
``render(ctx)`` and ``cache_key()``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Class
     - Description
   * - :class:`~wcc_etc.AiryPSF`
     - Diffraction-limited Airy disk computed for the sensor's wavelength,
       plate scale, and aperture. The default when no PSF is given.
   * - :class:`~wcc_etc.DefocusPSF`
     - A resampled Zemax Huygens PSF loaded from disk. Two defocus files ship
       with the package (see below).
   * - :class:`~wcc_etc.CustomPSF`
     - An arbitrary PSF from a NumPy array or a file, with a given source pixel
       scale (``src_um_per_pix``).

Bundled defocus PSFs
~~~~~~~~~~~~~~~~~~~~~

Two Zemax Huygens PSFs (500 nm, 4 µm source pixels) ship with the package and
are exposed as path constants:

.. code-block:: python

   from wcc_etc import DefocusPSF, DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH

   psf1 = DefocusPSF(DEFOCUS_1WAVE_PATH)   # 1-wave defocus
   psf2 = DefocusPSF(DEFOCUS_2WAVE_PATH)   # 2-wave defocus

Custom PSFs
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from wcc_etc import CustomPSF

   array = np.load("my_psf.npy")
   psf = CustomPSF(array, src_um_per_pix=4.0)

Automatic PSF selection by focus level
--------------------------------------

When you build a simulation with
:meth:`~wcc_etc.Simulation.from_sensorfilter`, an appropriate default PSF
(in-focus vs. defocused) is selected for that sensor:filter and used whenever
you omit ``psf=`` in the SNR / image methods.

Simulating an image
-------------------

Use :class:`~wcc_etc.ImageSimulator`:

.. code-block:: python

   from wcc_etc import ImageSimulator, AiryPSF

   imsim = ImageSimulator.from_sensor_and_scene("sony:r", scene, npix=300)
   img = imsim.simulate(
       time=30,
       psf=AiryPSF(),
       jitter_sigma_mas=None,    # None → use the telescope's jitter
       add_noise=True,
       seed=0,                   # reproducible noise
   )

``simulate`` returns a :class:`~wcc_etc.SimulatedImage` with:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Attribute / method
     - Description
   * - ``image_e``
     - Noisy image in electrons.
   * - ``image_clean``
     - Noiseless image (mean electrons).
   * - ``saturation_mask``
     - Boolean mask of saturated pixels.
   * - ``to_adu()``
     - Image converted to ADU using the sensor gain / bit depth.
   * - ``to_fitsimg()``
     - A :class:`~wcc_etc.psfsim.FitsImg` for aperture photometry and plotting.
   * - ``plot_image()`` … ``plot_encircled_energy()``
     - Convenience plotting dispatchers (see :doc:`plotting`).

Grid size and oversampling
--------------------------

``npix`` sets the rendered grid and must be large enough to contain the PSF;
``oversample`` controls sub-pixel rendering before binning to detector pixels.
The default ``npix`` used inside the SNR path (128) contains the bundled
defocus PSFs for the current sensors, but very small pixels or stronger defocus
may need a larger value.

Telescope jitter
----------------

Pointing jitter is applied as a Gaussian blur with standard deviation
``jitter_sigma_mas``. Passing ``None`` uses the telescope's configured
``jitter_sigma`` (10 mas for Lazuli by default); pass a value to override it
for a single call.
