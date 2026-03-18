
![Logo](logo_schmidt_observatory_system.png)

# WCC-ETC Documentation

Welcome to the documentation for the WCC-ETC package!

Web-based version is available here: <a href='https://lazulisimulators-0b82f82d960f.herokuapp.com/'>https://lazulisimulators-0b82f82d960f.herokuapp.com/</a>


## Installation

```bash
pip install wcc-etc
```

## Usage

```python
from wcc_etc import get_scene, Simulation

# Create a scene and a simulation for a sensor
scene = get_scene('K3IV', mag=20)
sim = Simulation.from_sensor_and_scene('sony:bb', scene)

# Compute SNR for a 60s exposure
snr = sim.get_snr(60)
```

## Features
- SNR calculations
- Choose different filters
- Fast exposure time calculator
- Background and source spectrum support

