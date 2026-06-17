Signal-to-noise and exposure time
=================================

``wcc-etc`` offers two SNR paths that agree to ~1% for an in-focus point
source, plus their exposure-time inverses.

Image-based SNR (the default)
-----------------------------

:meth:`~wcc_etc.Simulation.get_snr` delegates to
:meth:`~wcc_etc.Simulation.get_image_snr`. It renders the PSF on the detector
grid, performs circular-aperture photometry, and returns a **dictionary**:

.. code-block:: python

   r = sim.get_snr(time=60)
   r["snr"]                 # signal-to-noise ratio
   r["signal_e"]            # source electrons inside the aperture
   r["noise_e"]             # total noise (electrons)
   r["enclosed_fraction"]   # PSF fraction inside the aperture
   r["r_aper_mas"]          # aperture radius (mas)
   r["n_pix"]               # number of pixels in the aperture
   r["n_saturated"]         # pixels at/above full well in the per-frame image
   r["saturated"]           # bool

.. important::

   The return value is a **dict** — always index ``["snr"]``. (Earlier versions
   returned a Quantity with ``.value``; that is gone.)

Choosing the aperture
~~~~~~~~~~~~~~~~~~~~~~~

Aperture precedence is ``optimize`` > ``r_aper_mas`` > ``ee_frac``; if none is
given, the Simulation's stored ``r_aper_mas`` is used.

.. code-block:: python

   sim.get_snr(60, r_aper_mas=300)          # fixed aperture radius (mas)
   sim.get_snr(60, ee_frac=0.8)             # aperture enclosing 80% of the PSF
   sim.get_snr(60, optimize=True)           # radius that maximizes SNR

Sweeping time or magnitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``time`` may be a scalar or an array. ``get_image_snr`` additionally accepts
``mags`` to sweep the *source* brightness off a single cached render (host,
background, dark, and read noise held fixed). ``time`` and ``mags`` may not
both be arrays.

.. code-block:: python

   import numpy as np
   sim.get_snr(time=np.array([10, 30, 60, 120]))["snr"]      # array of SNR
   sim.get_image_snr(time=60, mags=np.arange(16, 24))["snr"] # SNR vs. magnitude

Specifying the PSF and jitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass any :class:`~wcc_etc.psfsim.PSFSource`; override jitter per call:

.. code-block:: python

   from wcc_etc import DefocusPSF, DEFOCUS_2WAVE_PATH
   sim.get_image_snr(time=60, psf=DefocusPSF(DEFOCUS_2WAVE_PATH))
   sim.get_image_snr(time=60, jitter_sigma_mas=15)

See :doc:`psf` for the PSF models and the ``npix`` / ``oversample`` grid
parameters.

Exposure time for a target SNR
------------------------------

:meth:`~wcc_etc.Simulation.get_image_exptime_for_snr` inverts the image-based
SNR; :meth:`~wcc_etc.Simulation.get_exptime_for_snr` is the analytic Airy
inverse that pairs with ``get_snr_airy``.

.. code-block:: python

   sim.get_image_exptime_for_snr(snr=100)        # seconds (PSF-aware)
   sim.get_exptime_for_snr(snr=100)              # seconds (analytic Airy)

Multiple reads (``n_reads``)
----------------------------

Detectors that co-add or sample multiple reads per exposure reduce the
effective read noise. All SNR and exposure-time methods accept ``n_reads``;
the default comes from the sensor/meta:

.. code-block:: python

   sim.get_snr(60, n_reads=4)
   sim.get_image_exptime_for_snr(snr=100, n_reads=4)

The analytic Airy SNR
---------------------

:meth:`~wcc_etc.Simulation.get_snr_airy` is the original closed-form
Airy-disk approximation. It is retained for cross-checking the image-based
path and **emits a** :class:`DeprecationWarning`. To use it quietly:

.. code-block:: python

   import warnings
   with warnings.catch_warnings():
       warnings.simplefilter("ignore", DeprecationWarning)
       snr_airy = sim.get_snr_airy(60)

For an in-focus :class:`~wcc_etc.AiryPSF` with the default aperture, the
image-based and analytic results agree to about 1%.
