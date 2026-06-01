# Sensorfilter Classmethod Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Simulation.from_sensorfilter(sensorfilter, scene)` and `ImageSimulator.from_sensorfilter(sensorfilter, scene)` that auto-select the correct throughput file and default PSF from the `focus_level` stored in `sensor_info`.

**Architecture:** `io.py` gains a `_SENSORFILTER_FOCUS` reverse-lookup built from `sensor_info` at import time. `Simulation` stores a `_default_psf` attribute (set only by `from_sensorfilter`) that `get_image_snr` and `get_image_exptime_for_snr` use when `psf=None`. `ImageSimulator.from_sensorfilter` mirrors the same pattern.

**Tech Stack:** Python, pytest, existing wcc_etc internals (`psfsim.AiryPSF`, `psfsim.DefocusPSF`, `psfsim.DEFOCUS_1WAVE_PATH`, `psfsim.DEFOCUS_2WAVE_PATH`).

**Branch:** `sensorfilter-classmethod` (already created).

**Run tests from:** `wcc-etc/` subdirectory (not the parent). The package is installed editable so test changes pick up immediately.

---

## File map

| File | Change |
|---|---|
| `src/wcc_etc/io.py` | Remove `r_defocus`/`bb_defocus` from SENSORS; add `r+1`, `r-1`, `bb2`, `hbeta`; add `_SENSORFILTER_FOCUS` |
| `src/wcc_etc/simulation.py` | `_default_psf = None` in `__init__`; `_psf_from_focus_level` helper; `from_sensorfilter` classmethod; update `get_image_snr` + `get_image_exptime_for_snr` |
| `src/wcc_etc/psfsim.py` | `ImageSimulator.from_sensorfilter` classmethod |
| `tests/test_sensorfilter.py` | New test file covering all of the above |

---

### Task 1: Clean up SENSORS and build `_SENSORFILTER_FOCUS` in `io.py`

**Files:**
- Modify: `src/wcc_etc/io.py`
- Test: `tests/test_sensorfilter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sensorfilter.py`:

```python
import pytest
from wcc_etc.io import SENSORS, _SENSORFILTER_FOCUS, sensor_info


def test_sensors_zwo_no_legacy_defocus():
    assert "r_defocus" not in SENSORS["zwo"]
    assert "bb_defocus" not in SENSORS["zwo"]


def test_sensors_zwo_has_new_bands():
    assert "r+1" in SENSORS["zwo"]
    assert "r-1" in SENSORS["zwo"]
    assert "bb2" in SENSORS["zwo"]
    assert "hbeta" in SENSORS["zwo"]


def test_sensors_zwo_r_variants_share_throughput():
    assert SENSORS["zwo"]["r+1"] == SENSORS["zwo"]["r"]
    assert SENSORS["zwo"]["r-1"] == SENSORS["zwo"]["r"]


def test_sensors_zwo_bb2_shares_throughput():
    assert SENSORS["zwo"]["bb2"] == SENSORS["zwo"]["bb"]


def test_sensors_zwo_hbeta_is_none():
    assert SENSORS["zwo"]["hbeta"] is None


def test_sensorfilter_focus_covers_all_sensor_info_labels():
    all_sf = {entry["sensorfilter"] for entry in sensor_info.values()}
    assert all_sf == set(_SENSORFILTER_FOCUS.keys())


def test_sensorfilter_focus_0wave_in_focus():
    assert _SENSORFILTER_FOCUS["zwo:r"] == "0wave"
    assert _SENSORFILTER_FOCUS["qcmos:bb"] == "0wave"


def test_sensorfilter_focus_1wave():
    assert _SENSORFILTER_FOCUS["zwo:r+1"] == "1wave"
    assert _SENSORFILTER_FOCUS["zwo:r-1"] == "1wave"


def test_sensorfilter_focus_2wave():
    assert _SENSORFILTER_FOCUS["zwo:bb2"] == "2wave"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sensorfilter.py -v
```

Expected: most tests FAIL (ImportError on `_SENSORFILTER_FOCUS`; assertions fail for missing/present keys).

- [ ] **Step 3: Update `SENSORS['zwo']` in `src/wcc_etc/io.py`**

Replace the `SENSORS` dict's `"zwo"` entry. The current entry is:
```python
"zwo": {"bb": "wcc_imx_bb_throughput.csv",
         "u":  "wcc_imx_u_throughput.csv",
         "g":  "wcc_imx_g_throughput.csv",
         "r":  "wcc_imx_r_throughput.csv",
         "i":  "wcc_imx_i_throughput.csv",
         "z":  "wcc_imx_z_throughput.csv",
         "r_defocus": None,
         "bb_defocus": None,
         "halpha": None,
         "nii": None,
         "oiii": None,
         "heii": None,
        },
```

Replace with:
```python
"zwo": {"bb":     "wcc_imx_bb_throughput.csv",
         "u":      "wcc_imx_u_throughput.csv",
         "g":      "wcc_imx_g_throughput.csv",
         "r":      "wcc_imx_r_throughput.csv",
         "i":      "wcc_imx_i_throughput.csv",
         "z":      "wcc_imx_z_throughput.csv",
         "r+1":    "wcc_imx_r_throughput.csv",
         "r-1":    "wcc_imx_r_throughput.csv",
         "bb2":    "wcc_imx_bb_throughput.csv",
         "halpha": None,
         "nii":    None,
         "oiii":   None,
         "heii":   None,
         "hbeta":  None,
        },
```

- [ ] **Step 4: Add `_SENSORFILTER_FOCUS` at the bottom of the `sensor_info` block in `src/wcc_etc/io.py`**

Immediately after the closing `}` of `sensor_info`, add:

```python
_SENSORFILTER_FOCUS: dict = {}
for _sf_entry in sensor_info.values():
    _sf_key = _sf_entry["sensorfilter"]
    if _sf_key not in _SENSORFILTER_FOCUS:
        _SENSORFILTER_FOCUS[_sf_key] = _sf_entry["focus_level"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_sensorfilter.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 6: Run the full suite to check no regressions**

```bash
pytest --tb=short -q
```

Expected: all previously passing tests still pass (124+).

- [ ] **Step 7: Commit**

```bash
git add src/wcc_etc/io.py tests/test_sensorfilter.py
git commit -m "Add _SENSORFILTER_FOCUS lookup and clean up SENSORS defocus entries"
```

---

### Task 2: `_default_psf` on `Simulation.__init__` and `_psf_from_focus_level` helper

**Files:**
- Modify: `src/wcc_etc/simulation.py`
- Test: `tests/test_sensorfilter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sensorfilter.py`:

```python
from wcc_etc.simulation import Simulation, _psf_from_focus_level
from wcc_etc.psfsim import AiryPSF, DefocusPSF


def test_psf_from_focus_level_0wave():
    psf = _psf_from_focus_level("0wave")
    assert isinstance(psf, AiryPSF)


def test_psf_from_focus_level_1wave():
    psf = _psf_from_focus_level("1wave")
    assert isinstance(psf, DefocusPSF)


def test_psf_from_focus_level_2wave():
    psf = _psf_from_focus_level("2wave")
    assert isinstance(psf, DefocusPSF)


def test_psf_from_focus_level_unknown():
    with pytest.raises(ValueError, match="Unknown focus_level"):
        _psf_from_focus_level("3wave")


def test_simulation_default_psf_is_none_by_default():
    sim = Simulation(telescope=None, sensor=None, scene=None)
    assert sim._default_psf is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sensorfilter.py::test_psf_from_focus_level_0wave \
       tests/test_sensorfilter.py::test_simulation_default_psf_is_none_by_default -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Add `_psf_from_focus_level` and `_default_psf = None` to `simulation.py`**

At the module level in `src/wcc_etc/simulation.py`, before the `Simulation` class definition, add:

```python
def _psf_from_focus_level(focus_level):
    from .psfsim import AiryPSF, DefocusPSF, DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH
    if focus_level == "0wave":
        return AiryPSF()
    if focus_level == "1wave":
        return DefocusPSF(DEFOCUS_1WAVE_PATH)
    if focus_level == "2wave":
        return DefocusPSF(DEFOCUS_2WAVE_PATH)
    raise ValueError(f"Unknown focus_level {focus_level!r}. Expected '0wave', '1wave', or '2wave'.")
```

In `Simulation.__init__`, add `self._default_psf = None` after the existing `self.set_scene(scene)` line:

```python
self._telescope = telescope
self._sensor = sensor
self.set_scene(scene)
self._default_psf = None          # set by from_sensorfilter only
```

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/test_sensorfilter.py -v -k "psf_from_focus or default_psf_is_none"
```

Expected: all 5 new tests PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_sensorfilter.py
git commit -m "Add _psf_from_focus_level helper and _default_psf=None on Simulation"
```

---

### Task 3: `Simulation.from_sensorfilter` classmethod

**Files:**
- Modify: `src/wcc_etc/simulation.py`
- Test: `tests/test_sensorfilter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sensorfilter.py`:

```python
import wcc_etc


def _make_scene():
    return wcc_etc.get_scene(name="G5V", mag=15, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})


def test_from_sensorfilter_0wave_uses_airy():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:r", scene)
    assert isinstance(sim._default_psf, AiryPSF)


def test_from_sensorfilter_1wave_uses_defocus():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:r+1", scene)
    assert isinstance(sim._default_psf, DefocusPSF)


def test_from_sensorfilter_2wave_uses_defocus():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:bb2", scene)
    assert isinstance(sim._default_psf, DefocusPSF)


def test_from_sensorfilter_qcmos():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("qcmos:bb", scene)
    assert isinstance(sim._default_psf, AiryPSF)


def test_from_sensorfilter_unknown_raises():
    scene = _make_scene()
    with pytest.raises(ValueError, match="Unknown sensorfilter"):
        Simulation.from_sensorfilter("zwo:nonexistent", scene)


def test_from_sensorfilter_builds_working_sim():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:r", scene)
    # basic sanity: can compute an analytic SNR
    snr = sim.get_snr(60)
    assert snr.value > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sensorfilter.py -v -k "from_sensorfilter"
```

Expected: FAIL with AttributeError (method does not exist).

- [ ] **Step 3: Add `from_sensorfilter` classmethod to `Simulation` in `simulation.py`**

Add after `from_sensor_and_scene`, before `set_scene`:

```python
@classmethod
def from_sensorfilter(cls, sensorfilter, scene):
    """
    Create a Simulation from a sensorfilter label, auto-selecting the PSF.

    The label must be a key in sensor_info (e.g. 'zwo:r', 'zwo:r+1',
    'qcmos:bb'). The PSF matching the sensor's focus_level is stored as
    _default_psf and used automatically by get_image_snr when psf=None.

    Parameters
    ----------
    sensorfilter : str
        Label from sensor_info, format 'kind:band'.
    scene : Scene

    Returns
    -------
    Simulation
    """
    from .io import _SENSORFILTER_FOCUS
    if sensorfilter not in _SENSORFILTER_FOCUS:
        known = sorted(_SENSORFILTER_FOCUS)
        raise ValueError(
            f"Unknown sensorfilter {sensorfilter!r}. "
            f"Known labels: {known}"
        )
    focus_level = _SENSORFILTER_FOCUS[sensorfilter]
    kind, band = sensorfilter.split(":", 1)
    config = get_sensor_config(kind, band)
    this = cls.from_config(config)
    this.set_scene(scene)
    this._default_psf = _psf_from_focus_level(focus_level)
    return this
```

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/test_sensorfilter.py -v -k "from_sensorfilter"
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_sensorfilter.py
git commit -m "Add Simulation.from_sensorfilter classmethod with auto PSF selection"
```

---

### Task 4: Wire `_default_psf` into `get_image_snr` and `get_image_exptime_for_snr`

**Files:**
- Modify: `src/wcc_etc/simulation.py`
- Test: `tests/test_sensorfilter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sensorfilter.py`:

```python
def test_get_image_snr_uses_default_psf_for_defocused_sensor():
    scene = _make_scene()
    sim_infocus = Simulation.from_sensorfilter("zwo:r", scene)
    sim_defocus = Simulation.from_sensorfilter("zwo:r+1", scene)
    snr_infocus = sim_infocus.get_image_snr(60)["snr"]
    snr_defocus = sim_defocus.get_image_snr(60)["snr"]
    # defocused SNR should differ from in-focus (PSF is wider → lower peak SNR)
    assert abs(snr_infocus - snr_defocus) > 0.01


def test_get_image_snr_psf_override_works():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:r+1", scene)
    sim_ref = Simulation.from_sensorfilter("zwo:r", scene)
    # override the defocused sim with AiryPSF — should match in-focus result
    snr_override = sim.get_image_snr(60, psf=AiryPSF())["snr"]
    snr_ref = sim_ref.get_image_snr(60)["snr"]
    assert abs(snr_override - snr_ref) < 0.01


def test_get_image_exptime_uses_default_psf_for_defocused_sensor():
    scene = _make_scene()
    sim_infocus = Simulation.from_sensorfilter("zwo:r", scene)
    sim_defocus = Simulation.from_sensorfilter("zwo:r+1", scene)
    t_infocus = sim_infocus.get_image_exptime_for_snr(10)["time_s"]
    t_defocus = sim_defocus.get_image_exptime_for_snr(10)["time_s"]
    assert abs(t_infocus - t_defocus) > 0.01


def test_from_sensor_and_scene_unaffected():
    # old API: _default_psf is None, get_image_snr falls back to AiryPSF
    scene = _make_scene()
    sim = Simulation.from_sensor_and_scene("sony:r", scene)
    assert sim._default_psf is None
    result = sim.get_image_snr(60)
    assert result["snr"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sensorfilter.py -v -k "get_image_snr or get_image_exptime or sensor_and_scene_unaffected"
```

Expected: `test_get_image_snr_uses_default_psf` and `test_get_image_exptime` FAIL because `get_image_snr` ignores `_default_psf` and always uses `AiryPSF`, so the two SNRs are identical.

- [ ] **Step 3: Update `get_image_snr` in `simulation.py`**

Find the line (around line 738):
```python
        if psf is None:
            psf = AiryPSF()
```

Replace with:
```python
        if psf is None:
            psf = self._default_psf if self._default_psf is not None else AiryPSF()
```

- [ ] **Step 4: Update `get_image_exptime_for_snr` in `simulation.py`**

Find the second occurrence (around line 797):
```python
        if psf is None:
            psf = AiryPSF()
```

Replace with:
```python
        if psf is None:
            psf = self._default_psf if self._default_psf is not None else AiryPSF()
```

- [ ] **Step 5: Run the new tests**

```bash
pytest tests/test_sensorfilter.py -v -k "get_image_snr or get_image_exptime or sensor_and_scene_unaffected"
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_sensorfilter.py
git commit -m "Wire _default_psf into get_image_snr and get_image_exptime_for_snr"
```

---

### Task 5: `ImageSimulator.from_sensorfilter` classmethod

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Test: `tests/test_sensorfilter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sensorfilter.py`:

```python
from wcc_etc.psfsim import ImageSimulator


def test_image_simulator_from_sensorfilter_0wave():
    scene = _make_scene()
    imsim = ImageSimulator.from_sensorfilter("zwo:r", scene)
    assert isinstance(imsim.sim._default_psf, AiryPSF)


def test_image_simulator_from_sensorfilter_1wave():
    scene = _make_scene()
    imsim = ImageSimulator.from_sensorfilter("zwo:r+1", scene)
    assert isinstance(imsim.sim._default_psf, DefocusPSF)


def test_image_simulator_from_sensorfilter_2wave():
    scene = _make_scene()
    imsim = ImageSimulator.from_sensorfilter("zwo:bb2", scene)
    assert isinstance(imsim.sim._default_psf, DefocusPSF)


def test_image_simulator_from_sensorfilter_unknown_raises():
    scene = _make_scene()
    with pytest.raises(ValueError, match="Unknown sensorfilter"):
        ImageSimulator.from_sensorfilter("zwo:nonexistent", scene)


def test_image_simulator_from_sensorfilter_has_npix():
    scene = _make_scene()
    imsim = ImageSimulator.from_sensorfilter("zwo:r", scene, npix=128)
    assert imsim.npix == 128
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sensorfilter.py -v -k "image_simulator_from_sensorfilter"
```

Expected: FAIL with AttributeError (method does not exist).

- [ ] **Step 3: Add `from_sensorfilter` to `ImageSimulator` in `psfsim.py`**

Add after the existing `from_sensor_and_scene` classmethod (around line 185):

```python
@classmethod
def from_sensorfilter(cls, sensorfilter, scene, npix=300, oversample=11):
    """Build an ImageSimulator from a sensorfilter label (e.g. 'zwo:r+1').

    The underlying Simulation's _default_psf is set from the sensor's
    focus_level in sensor_info.
    """
    sim = Simulation.from_sensorfilter(sensorfilter, scene)
    return cls(sim, npix=npix, oversample=oversample)
```

- [ ] **Step 4: Run the new tests**

```bash
pytest tests/test_sensorfilter.py -v -k "image_simulator_from_sensorfilter"
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_sensorfilter.py
git commit -m "Add ImageSimulator.from_sensorfilter mirroring Simulation.from_sensorfilter"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Remove `r_defocus`, `bb_defocus` from SENSORS | Task 1 |
| Add `r+1`, `r-1`, `bb2`, `hbeta` to SENSORS | Task 1 |
| `_SENSORFILTER_FOCUS` reverse-lookup from `sensor_info` | Task 1 |
| `_default_psf = None` on `Simulation.__init__` | Task 2 |
| `_psf_from_focus_level` helper | Task 2 |
| `Simulation.from_sensorfilter` classmethod | Task 3 |
| `get_image_snr` uses `_default_psf` when `psf=None` | Task 4 |
| `get_image_exptime_for_snr` uses `_default_psf` | Task 4 |
| Old `from_sensor_and_scene` API unaffected | Task 4 (tested) |
| `ImageSimulator.from_sensorfilter` | Task 5 |

All spec requirements covered. No placeholders. Method names consistent across all tasks.
