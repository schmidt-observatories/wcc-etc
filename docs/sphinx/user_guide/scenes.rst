Scenes, sources, and backgrounds
================================

A :class:`~wcc_etc.Scene` describes everything on the sky. It has exactly
three :class:`~wcc_etc.scene.SceneElement` slots, and there is no way to add
a fourth:

- ``source``: what you are measuring (required),
- ``host``: any other object whose light lands in the same aperture,
- ``background``: the diffuse sky.

The fastest way to build one is :func:`~wcc_etc.get_scene`.

.. _which-slot:

What goes in which slot
-----------------------

The slots are roles, not object types. The same star is a ``source`` when you
are measuring it and a ``host`` when it sits next to something else you are
measuring. Ask two questions.

Am I measuring it? Only ``source`` contributes signal. ``host`` and
``background`` contribute photons, and therefore shot noise and saturation
charge, but never signal. There is one ``source`` per scene.

Is it resolved? This is set by ``surface_brightness`` and decides how the
light is spread over pixels. It matters far more than which slot the element
sits in; see :ref:`host-spatial-treatment`.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * -
     - I am measuring it
     - It is contaminating my aperture
   * - Unresolved or compact
     - ``source``
     - ``host`` (default, ``surface_brightness=False``), which rides the
       source PSF
   * - Resolved or extended
     - not expressible (the source is always a point)
     - ``host`` with ``profile="sersic"``, a PSF-convolved galaxy profile (see
       :ref:`sersic-host`), or ``host="ngc0628"``, a real galaxy image moved
       to ``redshift`` (see :ref:`galaxy-host`); or ``surface_brightness=True``
       alone, uniform per pixel in mag/arcsec²

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
   Whichever role it is playing. If you are measuring it, it is the
   ``source``. If it is contaminating someone else's aperture, it is the
   ``host``.

*Is a background galaxy a host?*
   Yes. Then decide whether it is resolved. A compact or unresolved one keeps
   the ``surface_brightness=False`` default. An extended one gets
   ``profile="sersic"``, or ``surface_brightness=True`` alone if the galaxy
   is much larger than the field. Declaring an extended galaxy as a bare
   integrated magnitude concentrates all of its light into the PSF core and
   overstates the peak pixel.

*Can I have two contaminants?*
   Not at present, since there is one ``host`` slot. Combine them into a
   single equivalent magnitude, or model the dominant one.

*Why is it called "host"?*
   The ETC grew out of supernova cosmology, where the ``source`` is a
   supernova and the ``host`` is its host galaxy. The bundled ``hsiao07`` SN
   Ia template and the Brown galaxy atlas come from that lineage. Read it as
   "the other object in the aperture" rather than as a host galaxy in
   particular.

.. note::

   ``host`` has no physics of its own. The simulation classifies every
   element by its ``surface_brightness`` flag, never by its slot name. The
   name only selects the defaults you get: ``background`` defaults to the
   zodiacal sky with ``surface_brightness=True``.

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

Every element is classified by its own ``surface_brightness`` flag, and that
flag decides how its light is spread over pixels:

``surface_brightness=False`` (the default for source and host)
    The magnitude is an integrated magnitude, so the count rate is a total.
    The element is treated as unresolved and co-located with the source: it
    is rendered through the same PSF and contributes
    ``total x enclosed_fraction`` inside an aperture. A host declared this
    way adds shot noise and peak-pixel charge, and never counts as signal.

``surface_brightness=True`` (the default for ``zodi``/``background``)
    The magnitude is per square arcsecond, so the count rate is per pixel and
    is applied uniformly across the detector. This is the right treatment for
    the sky and for a host much larger than the field of view.

``profile="sersic"`` (any ``surface_brightness``)
    The element has a spatial profile. It is rendered as a Sersic galaxy,
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

At the WCC plate scale (16.9 mas/pix) a 1″ galaxy spans about 60 pixels, so
how much host light falls under the aperture depends on the galaxy's shape
and on where the source sits in it. A TDE on the cusp of a bulge sees far
more host shot noise than a supernova an arcsecond out on a disk.
``profile="sersic"`` renders the host as an elliptical Sersic profile,

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
     - none
     - ``"sersic"``, or an NGC name such as ``"ngc0628"`` (see :ref:`galaxy-host`)
   * - ``r_eff``
     - arcsec
     - required unless ``r_kpc``
     - half-light radius along the major axis
   * - ``r_kpc``
     - kpc
     - none
     - physical half-light radius, converted to ``r_eff`` at the element's
       ``redshift``; give one of ``r_eff`` or ``r_kpc``, not both
   * - ``n``
     - none
     - 1
     - Sersic index: 1 is an exponential disk, 4 a de Vaucouleurs bulge
   * - ``ellip``
     - none
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

The magnitude keeps its usual meaning. With the default
``surface_brightness=False``, ``mag`` is the total integrated host magnitude.
With ``surface_brightness=True`` it is :math:`\mu_e`, the surface brightness
at ``r_eff`` in mag/arcsec². Parameters are angular only, so convert physical
sizes and absolute magnitudes yourself. Typical r-band values are
:math:`n` = 1 to 4, :math:`r_e` of 0.3 to 1″ for hosts at
:math:`z \approx 0.5` to 1 and 1 to 10″ for nearby galaxies, and
:math:`\mu_e` of 20 to 24 mag/arcsec².

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
``update`` sweeps over ``host__mag``, ``host__r_eff`` or ``host__dx`` work.
Moving a mag-21 transient out from the nucleus of a mag-16 bulge recovers most
of the SNR within an arcsecond:

.. plot::
   :context: reset

   import matplotlib.pyplot as plt
   import numpy as np
   import wcc_etc
   from wcc_etc import get_scene, Simulation

   wcc_etc.set_wcc_style()

   scene = get_scene(
       "G5V", mag=21, bandpass="johnson_r",
       host="G5V",
       host_prop={"mag": 16, "bandpass": "johnson_r",
                  "profile": "sersic", "r_eff": 0.5, "n": 4, "dx": 0.0},
       background="zodi",
   )
   sim = Simulation.from_sensorfilter("zwo:r", scene)

   fig, ax = plt.subplots()
   sim.plot_sweep("host__dx", [0, 0.25, 0.5, 1.0, 1.5, 2.0], time=60, ax=ax)
   ax.set_xlabel("Source offset from the nucleus [arcsec]")

The deprecated analytic path (``get_snr_airy``) cannot place a profile and
warns that it ignores it. The :doc:`Sersic host tutorial
</notebooks/07_sersic_host>` has images and more sweeps.

.. _galaxy-host:

Real galaxies as hosts
----------------------

``host="ngc0628"`` (any NGC galaxy in ``lazuli-astrosim-data``) uses the
galaxy's own r-band image as the spatial profile and its own spectrum as the
template. The image keeps its physical size, so the host needs a ``redshift``
larger than the galaxy's own; ``get_scene(..., redshift=z)`` gives the source
redshift to the host unless ``host_prop`` sets one. The default normalization
is an absolute magnitude, ``magabs=-21`` in ``sdss_r`` AB, dimmed by the
distance modulus and K-corrected through the template; pass ``mag`` instead
for an observed magnitude. ``dx``, ``dy`` (arcsec) and ``pa`` (degrees)
place and rotate the galaxy. ``redshift`` and ``magabs`` work on any element,
so the transient can be given the same way:

.. code-block:: python

   scene = get_scene(
       "hsiao07", mag=None, magabs=-19.3, redshift=0.9,
       bandpass="sdss_r", magsys="abmag",
       host="ngc0628", host_prop={"dx": 0.3},
   )
   scene.host.show_image(0.0169)   # the host alone, arcsec/pixel
   sim = Simulation.from_sensorfilter("zwo:r", scene)
   sim.plot_host(300)              # a 300 s frame next to the host

See the :doc:`galaxy host tutorial </notebooks/08_galaxy_host>` and
:mod:`wcc_etc.galaxy`.

Magnitude systems
-----------------

Magnitudes are interpreted in the Vega system by default
(``magsys="vegamag"``). Pass ``magsys="abmag"`` (case-insensitive) to use AB
magnitudes. The setting applies per element, so source, ``host_prop`` and
``background_prop`` can each carry their own ``magsys``:

.. code-block:: python

   scene = get_scene(
       "G5V", mag=15, magsys="abmag", bandpass="johnson_r",
       background="zodi",
       background_prop={"bandpass": "johnson_r", "mag": 22.5, "magsys": "abmag"},
   )

The AB\ :math:`-`\ Vega offset is negligible in Johnson *V* (about 0.002 mag)
and grows toward the red (about 0.26 mag in *R* and 1.9 mag in *K*), so the
choice matters most for red bandpasses.

Stellar and galaxy templates
----------------------------

Any name recognized by the bundled spectral libraries works as a ``source``
or ``host``: Pickles stellar types (``"G5V"``, ``"K3IV"``, ``"M0V"`` and so
on) and the Brown galaxy templates. They are normalized to the requested
``mag`` in the requested ``bandpass``. Every element draws itself with
:meth:`~wcc_etc.scene.SceneElement.show`, which is the quickest check that
you picked the template you meant:

.. plot::
   :context: close-figs

   fig, ax = plt.subplots()
   for spectral_type in ("A0V", "G5V", "M5V"):
       source = get_scene(spectral_type, mag=15, bandpass="johnson_r").source
       source.show(
           ax=ax, wave=np.arange(3500, 9500, 5.0), flux_unit="flam",
           label=spectral_type,
       )
   ax.legend()

Three stars of the same r-band magnitude put very different amounts of light
at the blue and red ends of the WCC band, which is why the PSF is rendered at
a source-weighted wavelength (see :ref:`source-weighted-psf`).

Parametric source spectra
--------------------------

Instead of a template name, ``get_scene`` accepts reserved parametric names
that synthesize a ``synphot`` spectrum from keyword arguments:

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
   line fluxes directly, with no magnitude normalization.

.. note::

   Three H-alpha bands are available on the Sony sensors: ``zwo:halpha2``,
   ``zwo:halpha6`` and ``zwo:halpha20`` (2, 6 and 20 nm wide). The ``sony:``
   nickname works with ``from_sensor_and_scene``; ``from_sensorfilter`` takes
   only the canonical ``zwo:`` form. The ``zwo:hbeta`` / ``zwo:heii`` bands
   are placeholders with no throughput file, so emission-line sources cannot
   be observed through them until those CSVs are added.

Rebuilding a parametric spectrum with ``update``
-------------------------------------------------

Shape parameters of a parametric source are mutable. Changing one rebuilds
the underlying spectrum and changes the SNR:

.. code-block:: python

   sim.update(source__teff=3000)        # rebuilds the blackbody, SNR changes

The spectrum type is locked once chosen. You can change the ``teff`` of a
blackbody, but swapping a blackbody for a power law means rebuilding the
scene.

Backgrounds
-----------

The default ``background="zodi"`` adds zodiacal light. Backgrounds are
surface-brightness elements, so their magnitude is per unit area. Moon
scatter is modelled separately (see :func:`~wcc_etc.get_moon_magnitude` and
the moon-scatter count-rate helper).

Working with scene elements directly
-------------------------------------

For finer control, build the elements yourself and assemble a
:class:`~wcc_etc.Scene`:

.. code-block:: python

   from wcc_etc import get_scene_element, Scene

   source = get_scene_element("G5V", mag=15, bandpass="johnson_r")
   bg = get_scene_element("zodi", mag=22.5, bandpass="johnson_r")
   scene = Scene(source=source, background=bg)

   scene.get_mag()          # magnitudes of the elements
   scene.get_spectrum()     # synphot spectra
   scene.has_source()       # True
