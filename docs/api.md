# API Reference

## wcc_etc.WCCETC

Main class for the exposure time calculator.

### Example
```python
from wcc_etc import WCCETC
wcc = WCCETC('config.toml')
wcc.setup()
```

### Methods
- `setup(verbose=True, plot=False)`: Initialize ETC
- `set_source(source_file, plot=True)`: Set source spectrum
- `set_background(background_file, ...)`: Set background
- `make_observation(...)`: Run ETC calculation
- `calc_SNR(int_time, ...)`: Calculate SNR

---
Add more details for each method as needed.
