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

:meth:`~wcc_etc.Simulation.is_saturated` compares the peak pixel in ADU to the
sensor's ADC maximum (``adc_max = 2**bit_depth - 1``):

.. code-block:: python

   sim.is_saturated(60)                   # True / False
   sim.is_saturated(np.array([10, 60]))   # array of bools

The relevant sensor properties are :attr:`~wcc_etc.Sensor.bit_depth`,
:attr:`~wcc_etc.Sensor.bias_level`, and :attr:`~wcc_etc.Sensor.adc_max`. For
example, the Sony IMX455 (``sony:*``) is 16-bit
(``adc_max = 65535``) and the qCMOS (``qcmos:*``) is 12-bit
(``adc_max = 4095``).

Saturation in the image and SNR paths
-------------------------------------

A simulated image carries a per-pixel mask:

.. code-block:: python

   img = imsim.simulate(time=60, add_noise=True)
   img.saturation_mask.any()              # did anything saturate?

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
