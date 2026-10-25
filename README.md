![Logo](docs/logo_schmidt_observatory_system.png)

[![Tests](https://github.com/schmidt-observatories/wcc-etc/actions/workflows/tests.yml/badge.svg)](https://github.com/schmidt-observatories/wcc-etc/actions/workflows/tests.yml)

# wcc-etc
Exposure Time Calculator (ETC) for the Widefield Context Camera on Lazuli.

A web-based ETC is available here: https://simulators.schmidtobservatorysystem.org/wcc/

# Installation

## **1. Clone the Repository**
```sh
git clone https://github.com/schmidt-observatories/wcc-etc.git
cd wcc-etc
```

## **2. Install the Package**

Core install:
```sh
pip install -e .
```

The transit models come from [`lazuli-exo`](https://github.com/schmidt-observatories/lazuli-exo)
(`lazuli_exo.transit`), pulled in as a dependency.

With optional extras:
```sh
pip install -e ".[dev]"    # development tools (pytest, ruff, mypy, pre-commit)
pip install -e ".[docs]"   # documentation build (Sphinx, nbsphinx, etc.)
pip install -e ".[lightcurve]"   # transit light-curve modelling (jaxoplanet)
pip install -e ".[exoarchive]"   # NASA Exoplanet Archive queries
```

# Quick Start

```python
import numpy as np
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

# Plot SNR vs exposure time, or the peak pixel against the saturation ceiling
sim.plot_snr(np.linspace(1, 300, 100))
sim.plot_peak_pixel(np.logspace(-1, 3, 200))

# A transient on a galaxy: give the host a Sersic profile (arcsec / degrees).
# mag is the total host magnitude; dx/dy offset the source from the nucleus.
scene = get_scene(
    "G5V", mag=21,
    host="G5V",
    host_prop={"mag": 16, "profile": "sersic", "r_eff": 0.5, "n": 4, "dx": 0.3},
)
```

## A supernova in its host galaxy

A host can also be a real galaxy: `host="ngc0628"` takes the r-band image and
the spectrum of NGC 0628 (from the `lazuli_astrosim_data` package, 76 NGC
galaxies). One `redshift` moves source and host together: the spectra are
stretched by 1+z, the brightness follows a rest-frame absolute magnitude
(`magabs`, default -21 in sdss r for a galaxy) through the distance modulus,
and the galaxy keeps its physical size, so it shrinks on the sky with distance.

```python
from wcc_etc import Simulation, get_scene

# A SN Ia (M = -19.3) at z = 0.9 in NGC 0628, 0.4" off the nucleus
scene = get_scene(
    "hsiao07", mag=None, magabs=-19.3, redshift=0.9, bandpass="sdss_r", magsys="abmag",
    host="ngc0628", host_prop={"dx": 0.4, "pa": 30},
)
sim = Simulation.from_sensorfilter("zwo:r", scene)
sim.get_image_snr(300)["snr"]      # the host adds flux and noise under the SN
sim.plot_host(300)                 # the simulated frame next to the host alone
scene.host.show_image(0.0169)      # the galaxy image on the WCC plate scale
```

A Sersic host can be sized in kpc the same way (`"r_kpc": 5` with a `redshift`).

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
| `08_galaxy_host.ipynb` | A supernova in a real galaxy: NGC hosts moved to the SN redshift (`host="ngc0628"`, `redshift`, `magabs`), the host under the transient, SNR against offset. |


# Known limitations (0.1.0)

- The zodiacal background is a fixed V = 22.5 mag/arcsec² (Vega) with no dependence on ecliptic latitude, solar elongation or season. The L3 requirement value is 22.1 (#98).
- The noise model has photon, dark and read terms only: no systematic floor (flat field, intrapixel × jitter, gain drift), no sky-estimation variance, no cosmic rays, no stray-light background (#100).
- The pupil is unobscured (correct for the WCC) but there is no wavefront-error / Strehl term, so the in-focus Airy PSF is diffraction-perfect and peak-pixel fractions (hence saturation limits) are optimistic (#99).
- Not yet validated against as-measured detector, throughput, PSF or on-sky data (#104).

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

## Releasing

Releases are tag-driven (`.github/workflows/publish.yml`); nothing is uploaded by hand.

1. Bump `version` in `pyproject.toml` and move the `Unreleased` entries in `docs/changelog.md` under the new version.
2. Tag and push. A tag containing `rc` (for example `v0.1.0rc1`) publishes to TestPyPI; a plain `vX.Y.Z` tag publishes to PyPI.
3. CI checks that the tag matches the version, that no dependency is a direct URL (PyPI rejects those), builds and audits the artifacts, and installs the wheel in a fresh venv before publishing.

Trusted publishing is used, so there are no API tokens: the repository, workflow file and GitHub environment (`pypi`, `testpypi`) are registered once on each index.
