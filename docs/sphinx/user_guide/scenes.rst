Scenes, sources, and backgrounds
================================

A :class:`~wcc_etc.Scene` describes everything on the sky. It is built from up
to three :class:`~wcc_etc.scene.SceneElement` objects:

- **source** — the object you are observing (required),
- **host** — an optional underlying/host spectrum,
- **background** — an optional sky background.

The fastest way to build one is :func:`~wcc_etc.get_scene`.

Building a scene
----------------

.. code-block:: python

   from wcc_etc import get_scene

   scene = get_scene(
       "G5V",                   # source spectrum name
       mag=15,                  # source magnitude
       bandpass="johnson_r",    # band the magnitude is defined in
       host=None,               # optional host spectrum name
       background="zodi",       # sky background
       background_prop={"bandpass": "johnson_r", "mag": 22.5},
   )

``host_prop`` and ``background_prop`` are dictionaries of keyword arguments
forwarded to the host and background elements, mirroring the source keywords.

Magnitude systems
-----------------

Magnitudes are interpreted in the **Vega** system by default
(``magsys="vegamag"``). Pass ``magsys="abmag"`` (case-insensitive) to use AB
magnitudes instead. The setting applies per element, so source, ``host_prop``,
and ``background_prop`` can each carry their own ``magsys``:

.. code-block:: python

   scene = get_scene(
       "G5V", mag=15, magsys="abmag", bandpass="johnson_r",
       background="zodi",
       background_prop={"bandpass": "johnson_r", "mag": 22.5, "magsys": "abmag"},
   )

The AB\ :math:`-`\ Vega offset is negligible in Johnson *V* (~0.002 mag) but
grows toward the red (~0.26 mag in *R*, ~1.9 mag in *K*), so the chosen system
matters most for red bandpasses.

Stellar and galaxy templates
----------------------------

Any name recognized by the bundled spectral libraries works as a ``source`` or
``host`` — for example Pickles stellar types (``"G5V"``, ``"K3IV"``,
``"M0V"`` …) and the Brown galaxy templates. These are normalized to the
requested ``mag`` in the requested ``bandpass``.

Parametric source spectra
--------------------------

Instead of a template name, ``get_scene`` accepts reserved **parametric**
names that synthesize a ``synphot`` spectrum from keyword arguments:

.. list-table::
   :header-rows: 1
   :widths: 16 26 58

   * - ``name``
     - Key arguments
     - Meaning
   * - ``"blackbody"``
     - ``teff=<K>``
     - Planck blackbody at the given effective temperature.
   * - ``"flat"``
     - ``flat_unit="fnu"`` | ``"flam"``
     - Flat in :math:`F_\nu` (AB-flat, default) or flat in :math:`F_\lambda`.
   * - ``"powerlaw"``
     - ``alpha=<exp>``, ``lambda_ref=5500``
     - Power law :math:`F_\lambda \propto \lambda^{\alpha}` about a reference
       wavelength.
   * - ``"emission"``
     - ``lines=[{"wave","flux","fwhm"}, ...]``, ``mag=None``
     - Sum of Gaussian emission lines; ``flux`` is the absolute integrated line
       flux (erg s\ :sup:`-1` cm\ :sup:`-2`), ``fwhm`` defaults to 2 Å.

Examples:

.. code-block:: python

   # A 3000 K blackbody source at mag 18
   scene = get_scene("blackbody", mag=18, teff=3000, bandpass="johnson_r")

   # An AB-flat source
   scene = get_scene("flat", mag=20, flat_unit="fnu")

   # A blue power law, F_lambda ~ lambda^(-1)
   scene = get_scene("powerlaw", mag=19, alpha=-1.0, lambda_ref=5500)

   # An emission-line source (absolute flux; mag may be None)
   scene = get_scene(
       "emission",
       mag=None,
       lines=[{"wave": 6563, "flux": 1e-15, "fwhm": 3.0}],
   )

.. note::

   For ``"emission"`` you can pass ``mag=None`` to use the absolute integrated
   line fluxes directly (no magnitude normalization).

.. warning::

   The narrowband filters ``sony:halpha`` / ``sony:nii`` / ``sony:oiii`` /
   ``sony:heii`` currently have placeholder throughput files, so emission-line
   sources cannot yet be observed through them until those CSVs are added.

Rebuilding a parametric spectrum with ``update``
-------------------------------------------------

Shape parameters of a parametric source are mutable; changing one rebuilds the
underlying spectrum and changes the SNR:

.. code-block:: python

   sim.update(source__teff=3000)        # rebuilds the blackbody, SNR changes

The spectrum **type** is locked once chosen — you can change ``teff`` of a
blackbody, but you cannot swap a blackbody for a power law via ``update``
(rebuild the scene instead).

Backgrounds
-----------

The default ``background="zodi"`` adds zodiacal light. Backgrounds are
surface-brightness elements; their magnitude is interpreted per unit area. The
moon-scatter contribution is modelled separately (see
:func:`~wcc_etc.get_moon_magnitude` and the moon-scatter count-rate helper).

Working with scene elements directly
-------------------------------------

For finer control, build elements yourself and assemble a :class:`~wcc_etc.Scene`:

.. code-block:: python

   from wcc_etc import get_scene_element, Scene

   source = get_scene_element("G5V", mag=15, bandpass="johnson_r")
   bg = get_scene_element("zodi", mag=22.5, bandpass="johnson_r")
   scene = Scene(source=source, background=bg)

   scene.get_mag()          # magnitudes of the elements
   scene.get_spectrum()     # synphot spectra
   scene.has_source()       # True
