# Tutorials

## Quickstart

```python
from wcc_etc import WCCETC
wcc = WCCETC('config.toml')
wcc.setup()
wcc.set_source('source.fits')
wcc.set_background('background.fits')
wcc.make_observation(flux=26, r_aper_mas=70)
snr = wcc.calc_SNR(int_time=60)
print(f'SNR: {snr}')
```

## Advanced Usage
- Custom PSF
- Jitter modeling
- Batch calculations

---
Add more step-by-step guides as needed.
