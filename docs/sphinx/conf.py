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

# Pull the version from package *metadata* rather than importing wcc_etc here:
# importing it would pull in lazuli_transit, which is mocked for the docs build
# (see autodoc_mock_imports below). Fall back to pyproject.toml if the package
# is not installed in the docs environment.
def _detect_release():
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("wcc_etc")
    except Exception:  # pragma: no cover
        pass
    try:
        import tomllib

        with open(REPO_ROOT / "pyproject.toml", "rb") as _fh:
            return tomllib.load(_fh)["project"]["version"]
    except Exception:  # pragma: no cover - docs should still build
        return "0.0.0"


release = _detect_release()
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
# the docs. lazuli_transit
# (https://github.com/schmidt-observatories/lazuli-transit) is a separate,
# currently *private* package that wcc_etc imports at top level; mocking it lets
# autodoc import wcc_etc without cloning that private repo in CI. Once
# lazuli-transit is public it can be installed normally and dropped from here.
autodoc_mock_imports = ["lazuli_transit"]

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
