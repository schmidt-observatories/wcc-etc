"""Minimal example script for wcc_etc
"""
from wcc_etc import get_scene, Simulation
import matplotlib.pyplot as plt
import numpy as np

# Create scene and simulation
scene = get_scene('K3IV', mag=20, host=None, background='zodi')
sim = Simulation.from_sensor_and_scene('sony:bb', scene)

# Compute SNR for a range of exposure times.
# get_snr now uses the 2D image simulation and returns a dict; ["snr"] is the
# (scalar or array) signal-to-noise ratio.
texp = np.logspace(0, 3, 30)  # 1s to 1000s
snr = sim.get_snr(texp)["snr"]

# Plot
plt.loglog(texp, snr)
plt.xlabel('Exposure time [s]')
plt.ylabel('SNR')
plt.grid(True)
plt.show()
