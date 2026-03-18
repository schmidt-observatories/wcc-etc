# API Reference

## Public API

This package exposes a small programmatic API for building scenes and running simple simulations.

Key entry points:

- `get_scene(name, mag, host=None, background='zodi')` — build a `Scene` object.
- `Simulation.from_sensor_and_scene(sensor, scene)` — convenience constructor for `Simulation` using a sensor spec and a `Scene`.
- `Simulation.get_snr(time)` — compute SNR for a given exposure time.

For more detailed usage please consult `README.md` and the `examples/01_example.py` script in the repository.
