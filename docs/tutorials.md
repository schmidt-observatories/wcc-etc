
# Tutorials

## Example: Calculate SNR for a given exposure time and magnitude

This example demonstrates how to use the WCC ETC to calculate the SNR for a source, using a config file and included data files.

```python
from importlib import reload
import wcc_etc
reload(wcc_etc)

# Initialize ETC with config file
WCC = wcc_etc.WCCETC("../config/config_wcc_sdssg_from_final.toml")
WCC.setup(verbose=True, plot=True)

import os
TEXP = 60
MAG = 26 # AB mag
BKG_MAG = 180 # vmag / arcsec2 # make background irrelevant
PLOT = True

# Set source and background using package data
WCC.set_source(os.path.join(WCC.SOURCE_DIR, WCC.get_default_source_file()), plot=PLOT)
WCC.set_background(background_file=WCC.get_default_background_file(), support_data_path=WCC.PATH_SUPPORT_DATA_DIR, plot=PLOT)

print('########################')
snr = WCC.make_observation_and_calc_SNR(flux=MAG, r_aper_mas=70, texp=TEXP,
										jitter_sigma_mas=0, bg_flux=BKG_MAG,
										plot=PLOT, verbose=True)
print(f'SNR={{snr:0.2f}}: in {{TEXP:0.3f}}s')
```

## Advanced Usage
- Custom PSF
- Jitter modeling
- Batch calculations

---
Add more step-by-step guides as needed.
