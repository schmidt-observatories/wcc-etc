Saturation
==========

``wcc-etc`` tracks detector saturation at the brightest-pixel level so you can
tell whether an exposure clips before you commit to it.

The peak pixel
--------------

:meth:`~wcc_etc.Simulation.get_peak_pixel` returns the value of the brightest
pixel for a given exposure time, summing source, background, dark current, and
bias:

.. code-block:: python

   sim.get_peak_pixel(60, units="e-")    # electrons
   sim.get_peak_pixel(60, units="adu")   # ADU (uses gain + bias)

``time`` may be a scalar or an array.

The saturation flag
--------------------

:meth:`~wcc_etc.Simulation.is_saturated` compares the peak pixel in electrons to
the sensor's per-frame saturation ceiling, :attr:`~wcc_etc.Sensor.saturation_e`:

.. code-block:: python

   sim.is_saturated(60)                   # True / False
   sim.is_saturated(np.array([10, 60]))   # array of bools

``saturation_e`` is the smaller of the full well (``well_depth``) and the charge
at which the ADC clips, ``(adc_max - bias_level) * gain`` with
``adc_max = 2**bit_depth - 1``. Bias consumes ADC headroom; it does not raise
the well. :attr:`~wcc_etc.Sensor.saturation_limit` tells you which one binds
(``"well"`` or ``"adc"``). The same ceiling is used by the image mask and the
SNR paths below, so every entry point gives the same answer on the same frame.

For example, the Sony IMX455 (``sony:*``) is 16-bit (``adc_max = 65535``) and
the full well (~16.3 ke-) binds. The qCMOS (``qcmos:*``) is 12-bit
(``adc_max = 4095``) and which limit binds depends on the gain mode: in the
default high-gain (32x) mode the ADC clips at 4095 / 7.42 = 552 e-, well below
the 7500 e- full well; in low-gain (1x) mode the ADC reaches 16,900 e- and the
full well clips first. Pass ``gain_mode="low"`` to
:meth:`~wcc_etc.Simulation.from_sensorfilter` for bright targets.

Saturation in the image and SNR paths
-------------------------------------

A simulated image carries a per-pixel mask:

.. code-block:: python

   img = imsim.simulate(time=60, add_noise=True)
   img.saturation_mask.any()              # did anything saturate?

The mask is evaluated on the returned ``image_e``. With ``add_noise=True`` that
is a noisy realization, which can flag pixels the clean prediction does not;
pass ``add_noise=False`` for the same clean frame that
:meth:`~wcc_etc.Simulation.is_saturated` and the SNR methods test.

The image-based SNR methods also report saturation of the rendered *per-frame*
image via the ``n_saturated`` (count) and ``saturated`` (bool) keys in their
result dict, and emit a warning when a pixel clips (unless ``warn=False``):

.. code-block:: python

   r = sim.get_snr(60)
   r["saturated"], r["n_saturated"]

.. tip::

   When using multiple reads, saturation is evaluated per frame — a long total
   exposure split into several reads may avoid clipping that a single long
   frame would hit.
