Plotting
========

The :mod:`wcc_etc.plotting` module provides ready-made diagnostic plots in both
**matplotlib** and **bokeh**, plus a shared house style. Every plotting
function comes in a ``*_mpl`` and a ``*_bokeh`` variant.

The house style
---------------

Apply the shared theme once per session/notebook:

.. code-block:: python

   import wcc_etc
   wcc_etc.set_wcc_style()        # global matplotlib rcParams
   wcc_etc.WCC_STYLE              # the underlying rcParams dict

The style uses STIX math/text fonts, 150 dpi, inward ticks, and faint
gridlines on 1-D plots (image plots turn the grid off).

The four plots
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Function pair
     - What it shows
   * - ``plot_image_mpl`` / ``plot_image_bokeh``
     - A single image with equal x/y scale, a ``noise=`` toggle, and an
       optional ``show_saturation=`` mask overlay.
   * - ``plot_image_row_mpl`` / ``plot_image_row_bokeh``
     - A three-panel row: PSF+Noise / PSF (no noise) / saturation mask, on a
       shared color scale.
   * - ``plot_radial_mpl`` / ``plot_radial_bokeh``
     - The azimuthally-averaged radial profile with a half-width marker.
   * - ``plot_encircled_energy_mpl`` / ``plot_encircled_energy_bokeh``
     - The encircled-energy curve (normalized to 1) with an optional
       ``ee_target=`` marker.

Two ways to call them
---------------------

Each function accepts either a :class:`~wcc_etc.SimulatedImage` as the first
positional argument, **or** raw arrays via keywords
(``image_e=``, ``image_clean=``, ``saturation_mask=``, ``pixel_scale_mas=``):

.. code-block:: python

   from wcc_etc import plot_image_mpl, plot_radial_mpl

   # From a SimulatedImage
   plot_image_mpl(img, show_saturation=True)

   # From raw arrays
   plot_radial_mpl(image_e=arr, pixel_scale_mas=18.0, units="mas")

Convenience methods on ``SimulatedImage``
-----------------------------------------

The image object dispatches to the functions above, with a ``backend=`` switch:

.. code-block:: python

   img.plot_image(backend="mpl", show_saturation=True)
   img.plot_image_row()                       # the 3-panel row
   img.plot_radial(units="mas")
   img.plot_encircled_energy(units="mas", ee_target=0.8)

Return values and embedding
---------------------------

- The matplotlib variants return ``(fig, ax)`` (plus ``(x, y)`` data for the
  1-D plots), so you can pass an existing ``ax=`` and compose figures.
- The bokeh variants take ``return_="obj" | "html" | "components"``: use
  ``"obj"`` for ``bokeh.io.show()`` in a notebook, or ``"components"`` to get
  ``(script, div)`` for embedding in the Flask web portal.

.. code-block:: python

   # matplotlib: overplot two profiles on one axis
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots()
   plot_radial_mpl(img_a, ax=ax)
   plot_radial_mpl(img_b, ax=ax)

   # bokeh: components for a web template
   script, div = img.plot_image(backend="bokeh", return_="components")
