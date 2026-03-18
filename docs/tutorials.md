
# Tutorials

## Example: Calculate SNR for a given exposure time and magnitude

This example demonstrates how to use the package to calculate the SNR for a source.

```python
from wcc_etc import get_scene, Simulation
import numpy as np

# Build a scene and simulation
scene = get_scene('K3IV', mag=26)
sim = Simulation.from_sensor_and_scene('sony:bb', scene)

# Compute SNR for a 60s exposure
snr = sim.get_snr(60)
print(f"SNR={snr:.2f}")

```

Additional examples are shown in `notebooks/` in `examples/`

