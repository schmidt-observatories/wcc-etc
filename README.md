![Logo](docs/logo_schmidt_observatory_system.png)

[![Tests](https://github.com/schmidt-observatories/wcc-etc/actions/workflows/tests.yml/badge.svg)](https://github.com/schmidt-observatories/wcc-etc/actions/workflows/tests.yml)

# wcc-etc
Exposure Time Calculator (ETC) for the Widefield Context Camera on Lazuli.

A web-based ETC is available here: https://lazulisimulators-0b82f82d960f.herokuapp.com/wcc/

# Installation

## **1. Clone the Repository**
```sh
git clone git@github.com:schmidt-observatories/wcc-etc.git
cd wcc-etc
```

## **2. Install the Package**
Install the package itself:
```sh
pip install -e .
```

If you need to install additional dependencies, can run
```sh
pip install -r requirements.txt
```

# Quick Start

```python
from wcc_etc import get_scene, Simulation

# Create a simple scene (stellar source only)
scene = get_scene("K3IV", mag=20, host=None, background="zodi")

# Create a simulation for a sensor (kind:band format) and the scene
sim = Simulation.from_sensor_and_scene("sony:bb", scene)

# Compute SNR for a single exposure time (seconds)
snr = sim.get_snr(10)

# Update scene or instrument parameters (example: change source magnitude)
sim.update(source__mag=22)

# Recompute SNR after the change
snr_new = sim.get_snr(10)
```

# Tutorial
See notebooks/ directory for example tutorials.


# Notes
Configuration files (.toml) are in the sr/wcc_etc/data/config/ directory. Notebooks show that it is easy to change default configuration parameters on the fly.

## Viewing Documentation Locally

To view the documentation locally, make sure you have MkDocs installed:

```bash
pip install mkdocs
```

Then, from the project root directory, run:

```bash
mkdocs serve
```

This will start a local web server. Open your browser and go to:

    http://127.0.0.1:8000

to view the documentation.

Notes
- The documentation site uses the markdown files in the `docs/` folder.