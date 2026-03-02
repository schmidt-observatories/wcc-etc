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
# API Reference

## API Reference

The programmatic API documentation has been removed from the docs because the `api/` module was removed from this repository.

If you need API details, consult the package `README.md`, the `docs/usage.md`, or inspect the code in `src/wcc_etc/` directly.

Example quick import:

```python
from wcc_etc import WCCETC
# instantiate and use WCCETC as shown in the repository README and examples
```

If you want the automatic API extraction re-enabled later, I can re-add `mkdocstrings` and wire the `::: wcc_etc` directive back in.
