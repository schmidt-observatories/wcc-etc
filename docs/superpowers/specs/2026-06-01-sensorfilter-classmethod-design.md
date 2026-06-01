# Design: `Simulation.from_sensorfilter` — per-sensor PSF auto-selection

**Date:** 2026-06-01  
**Branch:** `sensorfilter-classmethod`

## Motivation

The WCC instrument has 23 physical sensors, each with a fixed filter and a fixed
focus level (in-focus, ±1 wave, or +2 wave defocus). The `sensor_info` dict in
`io.py` already encodes this per-sensor truth. Currently a user must manually
pass the right `DefocusPSF` to `get_image_snr`; there is no structured way to
say "simulate sensor 16 (r, +1 wave defocus)" without knowing the mapping by
heart.

This feature adds `Simulation.from_sensorfilter(sensorfilter, scene)` (and a
matching `ImageSimulator.from_sensorfilter`) that automatically selects the
correct throughput file and default PSF from the `sensorfilter` label stored in
`sensor_info`.

## Scope

- New classmethod `Simulation.from_sensorfilter`
- New classmethod `ImageSimulator.from_sensorfilter`
- `_default_psf` attribute on `Simulation` (and `ImageSimulator`), used as default in `get_image_snr` / `get_image_exptime_for_snr` when `psf=None`
- `SENSORS` dict cleanup: remove legacy `'r_defocus'`/`'bb_defocus'`; add `'r+1'`, `'r-1'`, `'bb2'`, `'hbeta'`
- `_SENSORFILTER_FOCUS` reverse-lookup built from `sensor_info` at import time in `io.py`

Out of scope: changes to `from_sensor_and_scene` (stays unchanged); changes to
the analytic SNR path (`get_snr`, `compute_psf_profile`).

## Data model changes (`io.py`)

### `SENSORS['zwo']` updates

Remove:
- `'r_defocus'` (replaced by `'r+1'` / `'r-1'` via `sensor_info`)
- `'bb_defocus'` (replaced by `'bb2'`)

Add:
- `'r+1'` → same throughput file as `'r'` (`wcc_imx_r_throughput.csv`)
- `'r-1'` → same throughput file as `'r'`
- `'bb2'` → same throughput file as `'bb'` (`wcc_imx_bb_throughput.csv`)
- `'hbeta'` → `None` (no throughput file yet; raises `NotImplementedError` if used)

### `_SENSORFILTER_FOCUS` reverse-lookup

Built once at module load from `sensor_info`:

```python
_SENSORFILTER_FOCUS: dict[str, str] = {}
for _entry in sensor_info.values():
    _sf = _entry["sensorfilter"]
    if _sf not in _SENSORFILTER_FOCUS:
        _SENSORFILTER_FOCUS[_sf] = _entry["focus_level"]
```

Where multiple sensors share a `sensorfilter` label (e.g. `'zwo:bb'` on
sensors 17 and 22), all have the same `focus_level`, so first-wins is correct.

### `focus_level` → PSF mapping

| `focus_level` | PSF |
|---|---|
| `'0wave'` | `AiryPSF()` |
| `'1wave'` | `DefocusPSF(DEFOCUS_1WAVE_PATH)` |
| `'2wave'` | `DefocusPSF(DEFOCUS_2WAVE_PATH)` |

A helper `_psf_from_focus_level(focus_level)` in `simulation.py` encodes this
mapping and raises `ValueError` for unknown values.

## `Simulation` changes (`simulation.py`)

### `__init__`

Add `self._default_psf = None` (set only by `from_sensorfilter`; `from_sensor_and_scene` leaves it `None`).

### `from_sensorfilter` classmethod

```python
@classmethod
def from_sensorfilter(cls, sensorfilter, scene):
    """
    Create a Simulation from a sensorfilter label, auto-selecting the PSF.

    Parameters
    ----------
    sensorfilter : str
        A label from sensor_info, e.g. 'zwo:r', 'zwo:r+1', 'qcmos:bb'.
    scene : Scene

    Returns
    -------
    Simulation
        With _default_psf set from the sensor's focus_level.
    """
```

Steps:
1. Validate `sensorfilter` is in `_SENSORFILTER_FOCUS`; raise `ValueError` with list of known labels if not.
2. Split `kind, band = sensorfilter.split(":", 1)`.
3. Call `get_sensor_config(kind, band)` and `cls.from_config(config)` (same path as `from_sensor_and_scene`).
4. Call `this.set_scene(scene)`.
5. Set `this._default_psf = _psf_from_focus_level(focus_level)`.
6. Return `this`.

### `get_image_snr` and `get_image_exptime_for_snr`

Replace `if psf is None: psf = AiryPSF()` with:

```python
if psf is None:
    psf = self._default_psf if self._default_psf is not None else AiryPSF()
```

Existing behavior for `from_sensor_and_scene` callers is preserved exactly.

## `ImageSimulator` changes (`psfsim.py`)

`ImageSimulator.from_sensorfilter` mirrors `Simulation.from_sensorfilter`:
1. Validate in `_SENSORFILTER_FOCUS`.
2. Build via `from_sensor_and_scene(sensorfilter, scene, npix=npix)` (reuse existing logic).
3. Set `self._default_psf` from `focus_level`.

`ImageSimulator.simulate` signature currently requires `psf` to be passed explicitly.
`_default_psf` is stored for discoverability / future use but `simulate` is not
changed in this spec (callers still pass `psf=` explicitly there, which is
appropriate since `simulate` is lower-level).

## Typical usage after this change

```python
# Sensor 15: zwo, r filter, in-focus — AiryPSF auto-selected
sim = wcc_etc.Simulation.from_sensorfilter('zwo:r', scene)
sim.get_image_snr(60)                     # uses AiryPSF automatically

# Sensor 16: zwo, r filter, +1 wave defocus — DefocusPSF auto-selected
sim = wcc_etc.Simulation.from_sensorfilter('zwo:r+1', scene)
sim.get_image_snr(60)                     # uses DefocusPSF(DEFOCUS_1WAVE_PATH)

# Sensor 12: zwo, BB filter, +2 wave defocus
sim = wcc_etc.Simulation.from_sensorfilter('zwo:bb2', scene)
sim.get_image_snr(60)                     # uses DefocusPSF(DEFOCUS_2WAVE_PATH)

# Override still works
sim.get_image_snr(60, psf=wcc_etc.AiryPSF())

# Old API completely unchanged
sim = wcc_etc.Simulation.from_sensor_and_scene('zwo:r', scene)
sim.get_image_snr(60)                     # AiryPSF (existing behavior)
```

## Testing

- `test_from_sensorfilter_psf_selection`: parametrized over `'zwo:r'` (0wave → Airy), `'zwo:r+1'` (1wave → Defocus1), `'zwo:bb2'` (2wave → Defocus2), `'qcmos:bb'` (0wave → Airy) — assert `sim._default_psf` is the right type.
- `test_from_sensorfilter_snr_uses_default_psf`: call `get_image_snr(60)` with no `psf=`; result should differ from `from_sensor_and_scene` result for defocused sensors (non-zero difference proves the PSF was applied).
- `test_from_sensorfilter_psf_overridable`: pass `psf=AiryPSF()` to `get_image_snr` on a defocused sensor; result matches in-focus Airy (override works).
- `test_from_sensorfilter_unknown_label`: assert `ValueError` is raised for an unknown string.
- `test_sensors_dict_no_legacy_defocus`: assert `'r_defocus'` and `'bb_defocus'` are not in `SENSORS['zwo']`.

## Branch and PR plan

Branch: `sensorfilter-classmethod` off `main`.  
Touches: `src/wcc_etc/io.py`, `src/wcc_etc/simulation.py`, `src/wcc_etc/psfsim.py`, `tests/`.
