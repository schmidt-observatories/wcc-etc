Configuration and sensors
=========================

Detectors, the telescope, and astrophysical defaults are described by TOML
configuration files bundled with the package under
``src/wcc_etc/data/config/``. Notebooks and scripts can also override any of
these values on the fly with :meth:`~wcc_etc.Simulation.update`.

Sensor strings
--------------

Sensors are addressed as ``"kind:band"``. The available combinations:

.. list-table::
   :header-rows: 1
   :widths: 18 26 56

   * - String
     - Detector
     - Notes
   * - ``sony:r`` / ``sony:bb``
     - Sony IMX455 (ZWO ASI6200MM)
     - 16-bit, 3.76 µm pixels. ``adc_max = 65535``.
   * - ``qcmos:r``
     - Hamamatsu qCMOS
     - 12-bit, 4.6 µm pixels. ``adc_max = 4095``.

(The ``zwo`` and ``hwk`` configs back the Sony and qCMOS sensors respectively.)

.. note::

   The narrowband bands ``sony:halpha`` / ``sony:nii`` / ``sony:oiii`` /
   ``sony:heii`` exist as placeholders but have no throughput file yet, so they
   are not usable until those CSVs are added.

The configuration files
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - File
     - Contents
   * - ``zwo.toml``
     - Sony IMX455 detector: pixel size, gain setting, sensor area, bias level,
       bit depth, and paths to the dark-current, gain, read-noise, and
       well-depth CSV calibration curves.
   * - ``qcmos.toml``
     - Hamamatsu qCMOS detector (12-bit) — same structure as ``zwo.toml``.
   * - ``lazuli.toml``
     - Telescope: primary diameter (3.065 m), focal ratio (f/15), pointing
       jitter (10 mas), and the default zodiacal background magnitude.
   * - ``astro.toml``
     - Astrophysical defaults, e.g. the background surface brightness.

Example — the Sony detector config
-----------------------------------

.. code-block:: toml

   [sensor]
   pixel_size   = 3.76     # micron / pix
   gain_setting = 100      # 0.1 dB
   sensor_temp  = 0.0      # C
   sensor_area  = 864      # mm^2  (36 mm x 24 mm)
   bias_level   = 0        # ADU, additive offset
   bit_depth    = 16       # ADC bit depth; adc_max = 2^bit_depth - 1
   path_dark_current = "sensors/.../Dark_Current_vs_Sensor_Temperature.csv"
   path_gain_curve   = "sensors/.../Gain_vs_Gain_Setting.csv"
   path_read_noise   = "sensors/.../Read_Noise_vs_Gain_Setting.csv"
   path_well_depth   = "sensors/.../Well_Depth_vs_Gain_Setting.csv"

Example — the telescope config
-------------------------------

.. code-block:: toml

   [telescope]
   diameter_primary = 3.065   # m
   f_num            = 15.0    # unitless
   jitter_sigma     = 10      # mas

   [zodi]
   zodi_mag_r = 22.5

Overriding configuration at runtime
-----------------------------------

You do not need to edit the TOML files to explore parameter changes. Use
``update`` with ``element__key`` syntax:

.. code-block:: python

   sim.update(telescope__jitter_sigma=15)     # mas
   sim.update(sensor__gain_setting=200)
   sim.update(source__mag=22)

Reading configs programmatically
--------------------------------

Lower-level helpers are exported for working with the configs directly:

.. code-block:: python

   from wcc_etc import read_config, get_sensor_config

   cfg = get_sensor_config("sony", "r")    # the resolved sensor config dict
   tel = read_config("lazuli", source="config")
