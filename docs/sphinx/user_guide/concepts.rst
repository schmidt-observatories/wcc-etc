Concepts and architecture
=========================

This page describes the objects that make up a calculation and how they relate.

The three building blocks
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Object
     - Role
   * - :class:`~wcc_etc.Scene`
     - What is on the sky: a **source**, an optional **host** galaxy, and an
       optional sky **background**. Each is a
       :class:`~wcc_etc.scene.SceneElement` wrapping a ``synphot`` spectrum and
       a magnitude.
   * - :class:`~wcc_etc.Sensor`
     - The detector + filter: read noise, dark current, gain, full well, pixel
       size, bit depth, bias level, and the filter **bandpass**.
   * - :class:`~wcc_etc.telescope.Telescope`
     - The optics: primary diameter, focal ratio, throughput surface, and
       pointing **jitter**.

A :class:`~wcc_etc.Simulation` owns one of each and turns them into photometry.

Constructing a simulation
--------------------------

There are several convenience constructors, from lowest to highest level:

.. code-block:: python

   from wcc_etc import Simulation

   # From a sensor spec ("kind:band") and a Scene:
   sim = Simulation.from_sensor_and_scene("sony:r", scene)

   # Auto-select the PSF focus level appropriate to a given sensor:filter:
   sim = Simulation.from_sensorfilter("sony:r", scene)

   # From an explicit Sensor object, scene optional:
   sim = Simulation.from_sensor(sensor, scene)

   # From a configuration dict / file:
   sim = Simulation.from_config(config)

``from_sensorfilter`` additionally records a *default PSF* for that sensor and
filter (in-focus vs. defocused), which is then used by the SNR and image
methods when you do not pass an explicit ``psf=``.

Inspecting and updating parameters
----------------------------------

Every sub-object is a "meta holder": its tunable parameters are exposed through
``mutable_parameters`` and can be changed with :meth:`~wcc_etc.Simulation.update`
using ``element__key`` double-underscore syntax. The element prefix selects
which sub-object the key belongs to:

.. code-block:: python

   sim.mutable_parameters             # everything you can change
   sim.update(source__mag=22)         # scene source magnitude
   sim.update(telescope__jitter_sigma=15)
   sim.update(time=120)               # stored default exposure time

   sim.get_parameter("source__mag")   # read a single parameter
   sim.reset()                        # restore the original configuration

You can also target a single sub-object explicitly with
:meth:`~wcc_etc.Simulation.update_scene`,
:meth:`~wcc_etc.Simulation.update_sensor`, and
:meth:`~wcc_etc.Simulation.update_telescope`.

.. important::

   Updating a parameter invalidates the internal PSF / count-rate **render
   cache**, so the next SNR or image call reflects the change. This fixed an
   earlier bug where a stale render survived a jitter update.

Count rates and the electron budget
------------------------------------

Beneath the SNR helpers, :meth:`~wcc_etc.Simulation.get_countrates` returns the
per-second electron rates for the source, host, background, and dark current.
These feed the signal-and-variance accounting in
:meth:`~wcc_etc.Simulation.get_signal_and_variance` and, ultimately, the SNR.
You rarely need them directly, but they are handy for debugging a surprising
SNR:

.. code-block:: python

   sim.get_countrates(units="e-/s")      # dict of electron rates
   sim.get_signal_and_variance(time=60)  # signal and per-source variance terms

Accessor properties
--------------------

A :class:`~wcc_etc.Simulation` exposes its parts as read-only properties:
``sim.scene``, ``sim.sensor``, ``sim.telescope``, ``sim.meta``, and
``sim.psf_profile``. Likewise a :class:`~wcc_etc.Scene` exposes ``scene.source``,
``scene.host``, ``scene.background``, and ``scene.element_names``.
