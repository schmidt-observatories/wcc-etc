Scenes, sources, and backgrounds
================================

A :class:`~wcc_etc.Scene` describes everything on the sky. It is built from
exactly three :class:`~wcc_etc.scene.SceneElement` slots — there is no way to
add a fourth:

- **source** — what you are measuring (required),
- **host** — any *other* object whose light lands in the same aperture,
- **background** — the diffuse sky.

The fastest way to build one is :func:`~wcc_etc.get_scene`.

.. _which-slot:

What goes in which slot
-----------------------

**The slots are roles, not object types.** The same star is a ``source`` when
you are measuring it and a ``host`` when it is sitting next to something else
you are measuring. Ask two questions:

**1. Am I measuring it?** Only ``source`` contributes *signal*. ``host`` and
``background`` contribute photons — and therefore shot noise and saturation
charge — but never signal. There is one ``source`` per scene.

**2. Is it resolved?** This is set by ``surface_brightness`` and decides how
the light is spread over pixels. It matters far more than which slot the
element sits in; see :ref:`host-spatial-treatment`.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * -
     - I am measuring it
     - It is contaminating my aperture
   * - **Unresolved / compact**
     - ``source``
     - ``host`` (default, ``surface_brightness=False``) — rides the source PSF
   * - **Resolved / extended**
     - not expressible (the source is always a point)
     - ``host`` with ``profile="sersic"`` — a PSF-convolved galaxy profile
       (see :ref:`sersic-host`); or ``surface_brightness=True`` alone —
       uniform per pixel, mag/arcsec²

Worked examples
~~~~~~~~~~~~~~~

.. code-block:: python

   # A star you are measuring. Nothing else in the aperture.
   get_scene("G5V", mag=25.4, bandpass="johnson_r", background="zodi")

   # A neighbouring star contaminating that aperture: same kind of object,
   # different role. Unresolved, so it follows the same PSF as the target.
   get_scene("G5V", mag=25.4, bandpass="johnson_r",
             host="G5V", host_prop={"mag": 22, "bandpass": "johnson_r"},
             background="zodi")

   # A resolved background galaxy behind the target: extended, so give it a
   # surface brightness in mag/arcsec² instead of an integrated magnitude.
   get_scene("G5V", mag=25.4, bandpass="johnson_r",
             host={"name": "ngc_0628", "mag": 22.5,
                   "surface_brightness": True},
             background="zodi")

   # A transient in a galaxy: the host has a shape. Total host mag 16, n=4
   # bulge, half-light radius 0.5", source 0.3" from the nucleus.
   get_scene("G5V", mag=21, bandpass="johnson_r",
             host="G5V",
             host_prop={"mag": 16, "bandpass": "johnson_r", "profile": "sersic",
                        "r_eff": 0.5, "n": 4, "dx": 0.3},
             background="zodi")

Answers to the usual questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Is a star a source or a host?*
   Whichever role it is playing. Measuring it → ``source``. Contaminating
   someone else's aperture → ``host``.

*Is a background galaxy a host?*
   Yes. Then decide whether it is resolved: a compact/unresolved one keeps the
   ``surface_brightness=False`` default; an extended one gets a
   ``profile="sersic"`` (or, for a galaxy much larger than the field,
   ``surface_brightness=True`` alone). Declaring an extended galaxy as a bare
   integrated magnitude concentrates all of its light into the PSF core and
   overstates the peak pixel.

*Can I have two contaminants?*
   Not currently — there is one ``host`` slot. Combine them into a single
   equivalent magnitude, or model the dominant one.

*Why is it called "host"?*
   Inherited terminology. This ETC grew out of supernova cosmology, where the
   ``source`` is a supernova and the ``host`` is its host galaxy (the bundled
   ``hsiao07`` SN Ia template and the Brown galaxy atlas are from that
   lineage). Read it as **"the other object in the aperture"**, not
   specifically as a host galaxy.

.. note::

   ``host`` has no special-cased physics. Since issue #63 the simulation
   classifies every element by its own ``surface_brightness`` flag, never by
   its slot name; the name only selects which defaults you get
   (``background`` defaults to the zodiacal sky, ``surface_brightness=True``).

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

.. _host-spatial-treatment:

How an element is distributed on the detector
---------------------------------------------

Every element is classified by its own ``surface_brightness`` flag — never by
its name — and that flag decides how its light is spread over pixels:

``surface_brightness=False`` (the default for source and host)
    The magnitude is an **integrated** magnitude, so the count rate is a
    *total*. The element is treated as **unresolved and co-located with the
    source**: it is rendered through the same PSF, contributing
    ``total x enclosed_fraction`` inside an aperture. A host declared this way
    adds shot noise and peak-pixel charge, but never counts as signal.

``surface_brightness=True`` (the default for ``zodi``/``background``)
    The magnitude is per square arcsecond, so the count rate is **per pixel**
    and is applied uniformly across the detector — the right treatment for the
    sky and for a host much larger than the field of view.

``profile="sersic"`` (any ``surface_brightness``)
    The element has a **spatial profile**: it is rendered as a Sersic galaxy,
    convolved with the PSF, and added pixel by pixel. See :ref:`sersic-host`.

.. code-block:: python

   # unresolved companion/host: total rate, follows the source PSF
   get_scene("G5V", mag=20, host={"name": "G5V", "mag": 16})

   # resolved, extended host: mag per arcsec^2, uniform per pixel
   get_scene("G5V", mag=20,
             host={"name": "G5V", "mag": 22, "surface_brightness": True})

   # resolved host with a shape: total mag 16 spread over a Sersic profile
   get_scene("G5V", mag=20,
             host={"name": "G5V", "mag": 16, "profile": "sersic", "r_eff": 0.5})

.. _sersic-host:

Hosts with a spatial profile: Sersic galaxies
---------------------------------------------

At the WCC plate scale (16.9 mas/pix) a 1″ galaxy spans ~60 pixels, so how
much host light falls under the aperture depends on the galaxy's shape and on
where the source sits in it — a TDE on the cusp of a bulge sees far more host
shot noise than a supernova an arcsecond out on a disk. ``profile="sersic"``
renders the host as an elliptical Sersic profile,

.. math::

   I(r) = I_e \exp\left\{-b_n\left[(r/r_e)^{1/n} - 1\right]\right\},

convolves it with the rendered PSF, and shares that one image with
:meth:`~wcc_etc.psfsim.ImageSimulator.simulate`,
:meth:`~wcc_etc.Simulation.get_snr`,
:meth:`~wcc_etc.Simulation.get_image_exptime_for_snr`,
:meth:`~wcc_etc.Simulation.get_peak_pixel` and
:meth:`~wcc_etc.Simulation.is_saturated`.

.. list-table::
   :header-rows: 1
   :widths: 14 10 12 64

   * - key
     - unit
     - default
     - meaning
   * - ``profile``
     - str
     - —
     - ``"sersic"`` (the only profile so far)
   * - ``r_eff``
     - arcsec
     - required
     - half-light radius along the major axis
   * - ``n``
     - –
     - 1
     - Sersic index: 1 is an exponential disk, 4 a de Vaucouleurs bulge
   * - ``ellip``
     - –
     - 0
     - ellipticity :math:`1 - b/a`
   * - ``pa``
     - deg
     - 0
     - major-axis position angle, counter-clockwise from +x
   * - ``dx``, ``dy``
     - arcsec
     - 0
     - offset of the host centre from the source (+x right, +y up)

The magnitude keeps its existing meaning: with the default
``surface_brightness=False``, ``mag`` is the **total** integrated host
magnitude; with ``surface_brightness=True`` it is :math:`\mu_e`, the surface
brightness at ``r_eff`` in mag/arcsec². Parameters are angular only — convert
physical sizes and absolute magnitudes yourself. Typical r-band values:
:math:`n` = 1–4, :math:`r_e` ≈ 0.3–1″ for hosts at :math:`z \approx 0.5`–1
and 1–10″ for nearby galaxies, :math:`\mu_e` ≈ 20–24 mag/arcsec².

.. code-block:: python

   from wcc_etc import get_scene, Simulation

   scene = get_scene(
       "G5V", mag=21, bandpass="johnson_r",
       host="G5V",
       host_prop={"mag": 16, "bandpass": "johnson_r",
                  "profile": "sersic", "r_eff": 0.5, "n": 4,
                  "ellip": 0.3, "pa": 45, "dx": 0.3},
       background="zodi",
   )
   sim = Simulation.from_sensorfilter("zwo:r", scene)
   sim.get_snr(60)["snr"]                 # host shot noise included

   sim.update(host__dx=0.0)               # move the source onto the nucleus
   sim.update(host__r_eff=1.0, host__n=1) # profile parameters are mutable

The profile parameters are mutable like any other element parameter, so
``update`` sweeps over ``host__mag``, ``host__r_eff`` or ``host__dx`` work as
expected. The deprecated analytic path (``get_snr_airy``) cannot place a
profile and warns that it ignores it. See the
:doc:`Sersic host tutorial </notebooks/07_sersic_host>` for images and SNR
sweeps.

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
