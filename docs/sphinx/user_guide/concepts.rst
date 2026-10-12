Concepts and architecture
=========================

A calculation is built from three objects and a :class:`~wcc_etc.Simulation`
that owns one of each.

The three building blocks
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Object
     - Role
   * - :class:`~wcc_etc.Scene`
     - What is on the sky: a source, an optional host galaxy, and an optional
       sky background. Each is a :class:`~wcc_etc.scene.SceneElement` that
       wraps a ``synphot`` spectrum and a magnitude.
   * - :class:`~wcc_etc.Sensor`
     - The detector and its filter: read noise, dark current, gain, full well,
       pixel size, bit depth, bias level, and the filter bandpass.
   * - :class:`~wcc_etc.telescope.Telescope`
     - The optics: primary diameter, focal ratio, throughput surface, and
       pointing jitter.

The :class:`~wcc_etc.Simulation` turns the three into photometry: count
rates, SNR, exposure time, the peak pixel and simulated images.

Constructing a simulation
--------------------------

There are several constructors, from lowest to highest level:

.. code-block:: python

   from wcc_etc import Simulation

   # From a sensor spec ("kind:band") and a Scene:
   sim = Simulation.from_sensor_and_scene("sony:r", scene)

   # Auto-select the PSF focus level appropriate to a given sensor:filter.
   # Only canonical labels are accepted here ("zwo:r", not the "sony:r" nickname):
   sim = Simulation.from_sensorfilter("zwo:r", scene)

   # From an explicit Sensor object, scene optional:
   sim = Simulation.from_sensor(sensor, scene)

   # From a configuration dict / file:
   sim = Simulation.from_config(config)

``from_sensorfilter`` also records a default PSF for that sensor and filter,
in focus or defocused, which the SNR and image methods use whenever you do not
pass ``psf=`` yourself.

Inspecting and updating parameters
----------------------------------

Every sub-object exposes its tunable parameters through
``mutable_parameters``. Change them with :meth:`~wcc_etc.Simulation.update`
using ``element__key`` double-underscore syntax, where the prefix names the
sub-object the key belongs to:

.. code-block:: python

   sim.mutable_parameters             # everything you can change
   sim.update(source__mag=22)         # scene source magnitude
   sim.update(telescope__jitter_sigma=15)
   sim.update(time=120)               # stored default exposure time

   sim.get_parameter("source__mag")   # read a single parameter
   sim.reset()                        # restore the original configuration

You can also target one sub-object with
:meth:`~wcc_etc.Simulation.update_scene`,
:meth:`~wcc_etc.Simulation.update_sensor` and
:meth:`~wcc_etc.Simulation.update_telescope`. A key that matches no
parameter, or more than one, raises ``KeyError`` instead of being ignored.

.. important::

   Updating a parameter clears the internal PSF and count-rate render cache,
   so the next SNR or image call reflects the change.

To see how the SNR responds to a parameter, sweep it with
:meth:`~wcc_etc.Simulation.plot_sweep`, which updates the key for each value
and restores the original afterwards. See :doc:`plotting`.

Count rates and the electron budget
------------------------------------

Beneath the SNR helpers, :meth:`~wcc_etc.Simulation.get_countrates` returns
the per-second electron rates for the source, host, background and dark
current. These feed :meth:`~wcc_etc.Simulation.get_signal_and_variance` and
from there the SNR. You rarely need them directly, but they are the first
place to look when an SNR surprises you:

.. code-block:: python

   sim.get_countrates(units="e-/s")      # dict of electron rates
   sim.get_signal_and_variance(time=60)  # signal and per-source variance terms

Accessor properties
--------------------

A :class:`~wcc_etc.Simulation` exposes its parts as read-only properties:
``sim.scene``, ``sim.sensor``, ``sim.telescope``, ``sim.meta`` and
``sim.psf_profile``. A :class:`~wcc_etc.Scene` exposes ``scene.source``,
``scene.host``, ``scene.background`` and ``scene.element_names``.
