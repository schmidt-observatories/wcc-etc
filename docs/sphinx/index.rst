wcc-etc: Exposure Time Calculator for the Lazuli WCC
====================================================

.. image:: _static/logo_schmidt_observatory_system.png
   :alt: Schmidt Observatory System
   :width: 320px

``wcc-etc`` is the exposure time calculator for the Wide-field Context Camera
(WCC) on *Lazuli*, the flagship instrument of the Schmidt Observatory System.
You describe what is on the sky, pick a sensor and filter, and it returns the
signal-to-noise ratio, the exposure time that reaches a target SNR, a simulated
detector image, a saturation flag, and a plot of any of these.

A hosted web version of the ETC is available at
`simulators.schmidtobservatorysystem.org/wcc <https://simulators.schmidtobservatorysystem.org/wcc/>`_.

.. note::

   New here? Read :doc:`installation`, then work through :doc:`quickstart`.
   The :doc:`tutorials` are Jupyter notebooks that cover each major feature end
   to end, with their figures and printed numbers saved in place.

What it models
--------------

- A scene: a stellar or galaxy template, or a parametric spectrum (blackbody,
  flat :math:`F_\nu` or :math:`F_\lambda`, power law, emission lines), with an
  optional host galaxy that may have a Sersic profile, and a sky background
  (zodiacal light, moon scatter).
- Signal-to-noise from circular-aperture photometry on a rendered image, with
  the aperture fixed, set by enclosed energy, or optimized for SNR. The older
  analytic Airy formula is kept for cross-checks.
- The exposure time that reaches a target SNR, including the effect of
  splitting an exposure into ``n_reads`` frames.
- PSFs and images: diffraction-limited Airy PSFs rendered at the
  source-weighted effective wavelength, polychromatic coadds, bundled Zemax
  defocus PSFs, custom PSF arrays, telescope jitter, Poisson and read noise,
  and per-pixel saturation masks.
- Plots: SNR against time, magnitude, aperture or any parameter; peak pixel
  against time with the saturation ceiling; images, radial profiles,
  encircled energy and filter curves, in matplotlib and bokeh, under one house
  style.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   configuration
   changelog

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
