Tutorials
=========

These tutorials are runnable Jupyter notebooks from the ``notebooks/``
directory of the repository, rendered here with their saved outputs. Each one
is self-contained — download it and run it interactively to experiment.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it covers
   * - :doc:`notebooks/01_getting_started`
     - The scene → simulation → SNR workflow, updating parameters, and plotting.
   * - :doc:`notebooks/02_saturation_flag`
     - Peak-pixel and saturation flagging (``get_peak_pixel``, ``is_saturated``).
   * - :doc:`notebooks/03_from_sensorfilter`
     - ``from_sensorfilter`` — automatic PSF selection by sensor:filter focus level.
   * - :doc:`notebooks/04_psf_and_image_snr`
     - The PSF simulator, ``ImageSimulator``, and PSF-aware SNR with aperture
       optimization.
   * - :doc:`notebooks/05_n_reads_exptime`
     - ``n_reads`` and the exposure-time-for-SNR inverses; per-frame saturation.
   * - :doc:`notebooks/06_source_spectra`
     - Parametric source spectra (blackbody / flat / power law / emission) and
       rebuilding them via ``update``.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/01_getting_started
   notebooks/02_saturation_flag
   notebooks/03_from_sensorfilter
   notebooks/04_psf_and_image_snr
   notebooks/05_n_reads_exptime
   notebooks/06_source_spectra
