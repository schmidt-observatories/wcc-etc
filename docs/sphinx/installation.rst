Installation
============

Requirements
------------

- **Python 3.11 or newer.**
- The scientific stack pulled in automatically as dependencies: ``numpy``,
  ``astropy``, ``synphot``, ``matplotlib``, ``pandas``, ``photutils``,
  ``bokeh``, ``flask``, ``toml``, and ``tifffile``.

Install from source (recommended)
---------------------------------

The package is developed on GitHub. Clone it and install in *editable* mode so
local changes to ``src/`` are picked up immediately:

.. code-block:: bash

   git clone git@github.com:schmidt-observatories/wcc-etc.git
   cd wcc-etc
   pip install -e .

If you only need the pinned extra dependencies that are not declared in
``pyproject.toml`` you can additionally run:

.. code-block:: bash

   pip install -r requirements.txt

Using a conda environment
-------------------------

The synphot / astropy / photutils stack is happiest in a dedicated
environment:

.. code-block:: bash

   conda create -n wcc-etc python=3.11
   conda activate wcc-etc
   pip install -e .

Verify the installation
-----------------------

.. code-block:: python

   import wcc_etc
   print(wcc_etc.__version__)

   from wcc_etc import get_scene, Simulation
   scene = get_scene("K3IV", mag=20)
   sim = Simulation.from_sensor_and_scene("sony:bb", scene)
   print(sim.get_snr(10)["snr"])

If that prints a version string and a signal-to-noise value, you are ready to
go. Continue with :doc:`quickstart`.

Running the test suite
----------------------

The package ships with a ``pytest`` suite. From the repository root:

.. code-block:: bash

   pip install -e .      # editable install so tests see local src/ changes
   pytest

Building the documentation locally
-----------------------------------

The documentation you are reading is built with `Sphinx
<https://www.sphinx-doc.org>`_ and the *Read the Docs* theme. The tutorial
pages are rendered from the Jupyter notebooks via `nbsphinx
<https://nbsphinx.readthedocs.io>`_, which requires `pandoc
<https://pandoc.org>`_ to be installed.

.. code-block:: bash

   # from the repository root
   pip install -e .                              # so autodoc can import the package
   pip install -r docs/sphinx/requirements.txt   # Sphinx + theme + nbsphinx
   # plus a pandoc binary, e.g.  conda install pandoc   or   brew install pandoc

   cd docs/sphinx
   make html

Open ``docs/sphinx/_build/html/index.html`` in a browser. To force a clean
rebuild, run ``make clean html``.

.. note::

   A legacy `MkDocs <https://www.mkdocs.org>`_ site also lives under ``docs/``
   (``mkdocs.yml`` + ``docs/*.md``). The Sphinx site documented here is the
   richer, autodoc-driven build and lives entirely under ``docs/sphinx/``; the
   two do not interfere.
