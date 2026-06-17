# Configuration file for the Sphinx documentation builder.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Make the package importable for autodoc without requiring an install.
DOCS_DIR = Path(__file__).resolve().parent          # docs/sphinx
REPO_ROOT = DOCS_DIR.parent.parent                  # wcc-etc/
sys.path.insert(0, str(REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "wcc-etc"
author = "Gudmundur Stefansson"
copyright = "2026, Schmidt Sciences / Schmidt Observatory System"

# Pull the version straight from the installed/importable package.
try:
    import wcc_etc

    release = wcc_etc.__version__
except Exception:  # pragma: no cover - docs should still build
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # pull docstrings from the source
    "sphinx.ext.autosummary",    # generate per-object summary tables/stubs
    "sphinx.ext.napoleon",       # parse NumPy/Google style docstrings
    "sphinx.ext.viewcode",       # add "[source]" links to highlighted source
    "sphinx.ext.intersphinx",    # cross-link to numpy/astropy/etc docs
    "sphinx.ext.mathjax",        # render LaTeX math
    "nbsphinx",                  # render the tutorial Jupyter notebooks
]

# Optional: enable Markdown authoring if myst-parser is installed.
try:
    import myst_parser  # noqa: F401

    extensions.append("myst_parser")
    source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
except Exception:  # pragma: no cover
    source_suffix = {".rst": "restructuredtext"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ---------------------------------------------------
# The autosummary tables in api/index.rst are summary-only (:nosignatures:);
# the full docs come from the per-module automodule pages. Generating stub
# pages here would document each object twice (duplicate-object warnings), so
# keep stub generation off.
autosummary_generate = False
autoclass_content = "class"          # merge __init__ docstring into the class
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
# Heavy / optional third-party deps that need not be importable just to build
# the docs locally. (On Read the Docs the package + deps are installed.)
autodoc_mock_imports = []

# -- napoleon ----------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_use_param = True
# Render docstring "Attributes" sections as :ivar: fields inside the class
# docstring rather than separate attribute directives. Without this, a class
# whose Attributes section names the same identifiers as its @property methods
# produces "duplicate object description" warnings.
napoleon_use_ivar = True

# -- nbsphinx ----------------------------------------------------------------
# Use the outputs already saved in the notebooks; do not re-execute at build
# time (keeps RTD/CI fast and free of the full scientific stack at run time).
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_prolog = """
.. note::

   This page is generated from the Jupyter notebook
   ``notebooks/{{ env.doc2path(env.docname, base=None)|basename }}`` in the
   repository. You can download it and run it interactively.
"""

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "style_external_links": True,
}
html_static_path = ["_static"]
html_logo = "_static/logo_schmidt_observatory_system.png"
html_title = f"wcc-etc {release}"
html_css_files = ["custom.css"]


# -- Copy the tutorial notebooks into the source tree at build time ----------
# Single source of truth stays in <repo>/notebooks/; nbsphinx requires the
# notebooks to live under the Sphinx source directory, so we copy them in.
def _sync_notebooks(app):
    src = REPO_ROOT / "notebooks"
    dst = DOCS_DIR / "notebooks"
    dst.mkdir(exist_ok=True)
    if not src.is_dir():
        return
    for nb in sorted(src.glob("*.ipynb")):
        shutil.copy2(nb, dst / nb.name)


def setup(app):
    app.connect("builder-inited", lambda app: _sync_notebooks(app))
