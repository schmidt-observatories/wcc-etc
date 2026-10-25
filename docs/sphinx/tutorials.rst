Tutorials
=========

The tutorials are Jupyter notebooks from the ``notebooks/`` directory of the
repository, rendered here with their outputs. Each one stands on its own, so
download it and change the numbers to see what happens.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it covers
   * - :doc:`notebooks/01_getting_started`
     - Building a scene and a simulation, reading the SNR, updating parameters,
       and the plot methods.
   * - :doc:`notebooks/02_saturation_flag`
     - The peak pixel and the saturation flag (``get_peak_pixel``,
       ``is_saturated``).
   * - :doc:`notebooks/03_from_sensorfilter`
     - ``from_sensorfilter``: the PSF that goes with each sensor and filter,
       in focus or defocused.
   * - :doc:`notebooks/04_psf_and_image_snr`
     - The PSF simulator, ``ImageSimulator``, and PSF-aware SNR with aperture
       optimization.
   * - :doc:`notebooks/05_n_reads_exptime`
     - ``n_reads``, the exposure time for a target SNR, and per-frame
       saturation.
   * - :doc:`notebooks/06_source_spectra`
     - Parametric source spectra (blackbody, flat, power law, emission lines)
       and rebuilding them with ``update``.
   * - :doc:`notebooks/07_sersic_host`
     - A transient on a galaxy: Sersic hosts (``profile="sersic"``), total
       magnitude against :math:`\mu_e` normalization, and SNR at the nucleus
       against SNR in the outskirts.
   * - :doc:`notebooks/08_galaxy_host`
     - A supernova in a real galaxy: NGC hosts moved to the SN redshift
       (``host="ngc0628"``, ``redshift``, ``magabs``), the host under the
       transient, and SNR against offset.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/01_getting_started
   notebooks/02_saturation_flag
   notebooks/03_from_sensorfilter
   notebooks/04_psf_and_image_snr
   notebooks/05_n_reads_exptime
   notebooks/06_source_spectra
   notebooks/07_sersic_host
   notebooks/08_galaxy_host
