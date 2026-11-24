Slitless grating
================

``wcc_etc.grating`` models the low-resolution slitless grating studied as a
replacement for one 2.0 mm WCC filter (WCC grating trade memo, 2026-11-16): a
16-level fused-silica staircase 70 mm in front of the sensor, 4.414 lines/mm,
R of about 20. The grating has two halves, each behind an edge coating so that
first and second orders do not overlap:

========  ============  ==========  =========  ====================
Half      Passband      Blaze       Bands      Dispersion direction
========  ============  ==========  =========  ====================
``blue``  350-700 nm    430 nm      u, g, r    pixel rows
``red``   600-1100 nm   775 nm      i, z       pixel columns
========  ============  ==========  =========  ====================

A :class:`~wcc_etc.GratingSimulation` covers one half on one sensor, the Sony
IMX455 (``"zwo"``) or the HWK4123 (``"qcmos"``):

.. code-block:: python

   import wcc_etc

   scene = wcc_etc.get_scene("G2V", mag=15, bandpass="johnson_v")
   grating = wcc_etc.GratingSimulation.from_sensor_and_scene("zwo", scene, half="blue")

   grating.get_band_snr(60, n_reads=1)        # {'u': ..., 'g': ..., 'r': ...}
   grating.get_band_exptime_for_snr(10)       # seconds per band
   grating.is_saturated(60, n_reads=1)

   image = grating.simulate(60, seed=1)       # a SimulatedImage
   image.plot_image()

Throughput
----------

The sensor curve is ``"zwo:nofilter"`` or ``"qcmos:nofilter"``: the end-of-life
telescope (bare-aluminium M1, three protected-silver mirrors, contamination, 5%
reserve) times the WCC contamination and 5% reserve times the detector QE, from
the Lazuli throughput budget v0.4. The grating multiplies it by the edge coating
(0.95 in band, 15 nm edges) and by the scalar diffraction efficiency of the
staircase in each order
(:func:`~wcc_etc.grating.staircase_efficiency`); the mean first-order efficiency
is 0.85 in the blue half and 0.89 in the red half.

Image
-----

Each star becomes a zeroth-order image plus a short spectrum: the first order
of wavelength :math:`\lambda` lands ``grating.offset_px(wavelength)`` pixels
from the zeroth order (29-58 px for 350-700 nm on the IMX455). The image holds
orders -1 to +2 by default (``orders=``). The PSF is rendered every 25 nm, with
the telescope jitter; pass ``psf=`` for anything other than an Airy pattern.
The sky is slitless too: every pixel collects zodiacal light from the whole
passband of the half.

Per-band SNR
------------

Each band is summed in a box of ``aperture_mas`` (70 mas) either side of the
trace by the length of the band along the first order. The signal is the
first-order light in the box; the noise adds the other orders, the sky, the
dark current and the read noise of the box pixels, once per read.

Limits
------

The static wavefront error, the FSM smear for bright hosts, the 3.6% change of
the dispersion across the tilted sensor, and extended sources are not modelled.
