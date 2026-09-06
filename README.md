![Logo](docs/logo_schmidt_observatory_system.png)

[![Tests](https://github.com/schmidt-observatories/wcc-etc/actions/workflows/tests.yml/badge.svg)](https://github.com/schmidt-observatories/wcc-etc/actions/workflows/tests.yml)

# wcc-etc
Exposure Time Calculator (ETC) for the Widefield Context Camera on Lazuli.

A web-based ETC is available here: https://simulators.schmidtobservatorysystem.org/wcc/

# Installation

## **1. Clone the Repository**
```sh
git clone git@github.com:schmidt-observatories/wcc-etc.git
cd wcc-etc
```

## **2. Install the Package**

Core install:
```sh
pip install -e .
```

This builds `lazuli_transit` from `packages/lazuli-transit/` as well — it is
vendored in this repo rather than published to PyPI.

With optional extras:
```sh
pip install -e ".[dev]"    # development tools (pytest, ruff, mypy, pre-commit)
pip install -e ".[docs]"   # documentation build (Sphinx, nbsphinx, etc.)
pip install -e ".[lightcurve]"   # transit light-curve modelling (jaxoplanet)
pip install -e ".[exoarchive]"   # NASA Exoplanet Archive queries
```

# Quick Start

```python
from wcc_etc import get_scene, Simulation

# Create a simple scene (stellar source only)
scene = get_scene("K3IV", mag=20, host=None, background="zodi")

# Create a simulation for a sensor (kind:band format) and the scene
sim = Simulation.from_sensor_and_scene("sony:bb", scene)

# Compute SNR for a single exposure time (seconds). get_snr uses the 2D image
# simulation and returns a dict; ["snr"] is the signal-to-noise ratio.
snr = sim.get_snr(10)["snr"]

# Update scene or instrument parameters (example: change source magnitude)
sim.update(source__mag=22)

# Recompute SNR after the change
snr_new = sim.get_snr(10)["snr"]

# A transient on a galaxy: give the host a Sersic profile (arcsec / degrees).
# mag is the total host magnitude; dx/dy offset the source from the nucleus.
scene = get_scene(
    "G5V", mag=21,
    host="G5V",
    host_prop={"mag": 16, "profile": "sersic", "r_eff": 0.5, "n": 4, "dx": 0.3},
)
```

# Tutorial
See the `notebooks/` directory for example tutorials:

| Notebook | What it covers |
|---|---|
| `01_getting_started.ipynb` | Getting started: the scene → simulation → SNR workflow, updating parameters, plotting. |
| `02_saturation_flag.ipynb` | Peak-pixel / saturation flagging (`get_peak_pixel`, `is_saturated`). |
| `03_from_sensorfilter.ipynb` | `from_sensorfilter` — auto PSF selection by sensor:filter focus level (in-focus vs defocused). |
| `04_psf_and_image_snr.ipynb` | PSF simulator (`AiryPSF`/`DefocusPSF`), `ImageSimulator`, and PSF-aware SNR (`get_image_snr`) with aperture optimization. |
| `05_n_reads_exptime.ipynb` | `n_reads` and the exposure-time-for-SNR inverses (`get_exptime_for_snr`, `get_image_exptime_for_snr`); per-frame saturation. |
| `06_source_spectra.ipynb` | Parametric source spectra in `get_scene` (blackbody/flat/powerlaw/emission) and rebuilding via `update`. |
| `07_sersic_host.ipynb` | A transient on a galaxy: Sersic-profile hosts (`profile="sersic"`), total-mag vs μ_e normalization, nuclear vs offset SNR and saturation. |


# Notes
Configuration files (.toml) are in the src/wcc_etc/data/config/ directory. Notebooks show that it is easy to change default configuration parameters on the fly.

## Documentation

The full documentation site (installation, a user guide, rendered tutorial
notebooks, and an auto-generated API reference) is built with **Sphinx** and
the *Read the Docs* theme. It lives under `docs/sphinx/`:

```bash
pip install -e ".[docs]"   # Sphinx + theme + nbsphinx (+ the package itself)
brew install pandoc        # or: conda install pandoc / apt-get install pandoc
cd docs/sphinx
make html                  # output in _build/html/index.html
```

A `.readthedocs.yaml` is included so the site builds automatically once the
repository is connected to Read the Docs.
