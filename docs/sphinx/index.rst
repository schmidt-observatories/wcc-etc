wcc-etc: Exposure Time Calculator for the Lazuli WCC
====================================================

.. image:: _static/logo_schmidt_observatory_system.png
   :alt: Schmidt Observatory System
   :width: 320px

**wcc-etc** is the exposure time calculator (ETC) for the **Wide-field Context
Camera (WCC)** on *Lazuli*, the flagship instrument of the Schmidt Observatory
System. It models a sky *scene*, propagates it through a *telescope* and
*sensor*, and returns photometric signal-to-noise ratios, required exposure
times, simulated detector images, saturation flags, and ready-made diagnostic
plots.

A hosted web version of the ETC is available at
`simulators.schmidtobservatorysystem.org/wcc <https://simulators.schmidtobservatorysystem.org/wcc/>`_.

.. note::

   New here? Read :doc:`installation` and then work through
   :doc:`quickstart`. The :doc:`tutorials` are runnable Jupyter notebooks that
   cover every major feature end to end.

Highlights
----------

- **Scene model** — stellar/galaxy templates *or* parametric source spectra
  (blackbody, flat :math:`F_\nu`/:math:`F_\lambda`, power law, emission lines),
  optional host galaxy, and a sky background (zodiacal light, moon scatter).
- **Signal-to-noise** — analytic Airy approximation *and* a PSF-aware,
  image-based SNR with circular-aperture photometry and aperture optimization.
- **Exposure-time inversion** — solve for the exposure time that reaches a
  target SNR, with multi-read (``n_reads``) support.
- **PSF & image simulation** — diffraction-limited Airy PSFs, bundled Zemax
  defocus PSFs, custom PSFs, telescope jitter, Poisson + read noise, and
  per-pixel saturation masks.
- **Plotting** — matplotlib *and* bokeh helpers for images, radial profiles,
  and encircled-energy curves, plus a shared house style.

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
