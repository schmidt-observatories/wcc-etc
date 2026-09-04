API reference
=============

This reference is generated automatically from the docstrings in the source.
It is organized by module. The most commonly used objects are re-exported at
the top level of the ``wcc_etc`` package, so e.g. ``wcc_etc.Simulation`` and
``wcc_etc.simulation.Simulation`` are the same class.

Top-level public API
--------------------

.. currentmodule:: wcc_etc

.. autosummary::
   :nosignatures:

   get_scene
   get_scene_from_file
   get_scene_element
   Scene
   Simulation
   Sensor
   ImageSimulator
   SimulatedImage
   AiryPSF
   PolychromaticPSF
   DefocusPSF
   CustomPSF
   effective_wavelength
   photon_weighted_subbands
   get_moon_magnitude
   read_config
   get_sensor_config
   set_wcc_style

Modules
-------

.. toctree::
   :maxdepth: 2

   simulation
   scene
   sensor
   telescope
   psfsim
   spectral
   plotting
   astro
   io
