# Vega / AB Magnitude System Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the magnitude system selectable (`abmag` / `vegamag`) end-to-end in the modern `scene.py` path, with `vegamag` as the new default everywhere, fixing the two bugs that currently make Vega normalization raise.

**Architecture:** Two source-code touch points in `src/wcc_etc/scene.py` — a case-insensitive magsys string resolver (replacing the broken `getattr(u, magsys)`), and Vega-aware normalization (passing `vegaspec=` to synphot when the unit is `VEGAMAG`) backed by a lazily-cached Vega spectrum. One dead-code deletion in `src/wcc_etc/simulation.py`. The default magsys flips from AB to Vega; existing flux-asserting tests are pinned to `magsys="abmag"` to preserve their validated numbers.

**Tech Stack:** Python, synphot (`SourceSpectrum`, `SpectralElement`, `Observation`, `units as su`), astropy.units, pytest.

## Global Constraints

- Magsys strings accepted: `"abmag"` and `"vegamag"`, case-insensitive only. No loose aliases (`"vega"`, `"AB"`).
- Unit objects passed as `magsys` (anything with `.is_equivalent`) pass through unchanged.
- Default magsys is `vegamag` for source, host, AND the built-in zodi/background element.
- AB normalization behavior (the `u.ABmag` branch) must remain numerically identical to today.
- Scope is the modern `scene.py` / `simulation.py` path only. Do NOT modify `src/wcc_etc/wcc_etc.py` (legacy interface already supports both systems).
- synphot bundled bands only in tests (Johnson/Bessel/Cousins via `SpectralElement.from_filter`); 2MASS is not bundled — use `johnson_k` for a redder band.
- Known AB−Vega offsets (derived from synphot's Vega spectrum, for reference/tolerances): `johnson_v ≈ 0.0017`, `johnson_r ≈ 0.256`, `johnson_k ≈ 1.868`.

---

## File Structure

- `src/wcc_etc/scene.py` — add `_resolve_magsys()` and `_get_vega()` module-level helpers; change `_parse_mag_` to use the resolver; change `get_spectrum` to pass `vegaspec` for Vega; flip the `magsys` default in `SceneElement.__init__` and the `_parse_mag_` fallback.
- `src/wcc_etc/simulation.py` — delete the unused `_get_spectrum_observation` static method.
- `tests/test_scene.py` — add magsys unit tests; pin existing flux-asserting roundtrip tests to `magsys="abmag"`.
- Various `tests/test_*.py` — pin flux-asserting `get_scene` / `SceneElement` callsites to `magsys="abmag"` after the default flips.

---

### Task 1: Magsys resolver + Vega-aware normalization (capability, default unchanged)

Add the ability to normalize in Vega and to resolve `abmag`/`vegamag` strings. The default stays `"ABmag"` in this task, so the entire existing suite remains green; only new tests are added.

**Files:**
- Modify: `src/wcc_etc/scene.py` (imports already present: `u`, `su`, `SourceSpectrum`, `Observation`)
- Test: `tests/test_scene.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `wcc_etc.scene._resolve_magsys(magsys) -> astropy/synphot unit` — accepts `"abmag"`/`"vegamag"` (case-insensitive) or a unit object; raises `ValueError` on unknown strings.
  - `wcc_etc.scene._get_vega() -> SourceSpectrum` — lazily-cached Vega reference spectrum.
  - `SceneElement.get_spectrum(...)` now succeeds when the stored magnitude unit is `su.VEGAMAG`.

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_scene.py`:

```python
from synphot import SourceSpectrum


def _band_ab_vega_offset(band_name):
    """AB - Vega magnitude offset for a band, derived from synphot's Vega spectrum."""
    vega = SourceSpectrum.from_vega()
    band = SpectralElement.from_filter(band_name)
    obs = Observation(vega, band, force="extrap")
    return obs.effstim(u.ABmag).value - obs.effstim(su.VEGAMAG, vegaspec=vega).value


def _inband_flam(scene_element, band_name):
    band = SpectralElement.from_filter(band_name)
    return Observation(scene_element.get_spectrum(), band,
                       force="extrap").effstim(su.FLAM).value


def test_resolve_magsys_aliases_case_insensitive():
    from wcc_etc.scene import _resolve_magsys
    import pytest
    assert _resolve_magsys("abmag") == u.ABmag
    assert _resolve_magsys("ABMAG") == u.ABmag
    assert _resolve_magsys("AbMag") == u.ABmag
    assert _resolve_magsys("vegamag") == su.VEGAMAG
    assert _resolve_magsys("VEGAMAG") == su.VEGAMAG
    assert _resolve_magsys(u.ABmag) == u.ABmag          # unit object passthrough
    with pytest.raises(ValueError):
        _resolve_magsys("vega")                          # loose alias rejected
    with pytest.raises(ValueError):
        _resolve_magsys("nonsense")


def test_vegamag_normalization_runs_and_differs_from_ab():
    band_name = "johnson_r"
    se_ab = SceneElement.from_config({"spectrum": "flat", "mag": 15,
                                      "magsys": "abmag", "bandpass": band_name})
    se_vg = SceneElement.from_config({"spectrum": "flat", "mag": 15,
                                      "magsys": "vegamag", "bandpass": band_name})
    ratio = _inband_flam(se_vg, band_name) / _inband_flam(se_ab, band_name)
    expected = 10 ** (-0.4 * _band_ab_vega_offset(band_name))
    assert np.isclose(ratio, expected, rtol=1e-6)


def test_cross_system_offset_is_band_dependent():
    off_v = _band_ab_vega_offset("johnson_v")
    off_k = _band_ab_vega_offset("johnson_k")
    assert abs(off_v) < 0.05          # V offset is ~0
    assert off_k > 1.5                # K offset is ~1.9 mag, much larger
    for band_name in ("johnson_v", "johnson_k"):
        se_ab = SceneElement.from_config({"spectrum": "flat", "mag": 12,
                                          "magsys": "abmag", "bandpass": band_name})
        se_vg = SceneElement.from_config({"spectrum": "flat", "mag": 12,
                                          "magsys": "vegamag", "bandpass": band_name})
        ratio = _inband_flam(se_vg, band_name) / _inband_flam(se_ab, band_name)
        expected = 10 ** (-0.4 * _band_ab_vega_offset(band_name))
        assert np.isclose(ratio, expected, rtol=1e-6)


def test_vegamag_roundtrips():
    band_name = "johnson_r"
    se = SceneElement.from_config({"spectrum": "flat", "mag": 14.0,
                                   "magsys": "vegamag", "bandpass": band_name})
    vega = SourceSpectrum.from_vega()
    band = SpectralElement.from_filter(band_name)
    obs = Observation(se.get_spectrum(), band, force="extrap")
    assert abs(obs.effstim(su.VEGAMAG, vegaspec=vega).value - 14.0) < 0.01
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_scene.py -k "magsys or vegamag or cross_system" -v`
Expected: FAIL — `ImportError`/`AttributeError` for `_resolve_magsys`, and the vegamag tests raise inside `normalize` (synphot complains about missing `vegaspec`) or `AttributeError` from `getattr(u, "vegamag")`.

- [ ] **Step 3: Add the `_resolve_magsys` and `_get_vega` helpers**

In `src/wcc_etc/scene.py`, after the imports (around line 11, before the first class/function), add:

```python
# Magnitude-system resolution: 'abmag' lives in astropy.units, 'vegamag' in
# synphot.units. getattr(u, ...) cannot see VEGAMAG, so resolve explicitly.
_MAGSYS = {"abmag": u.ABmag, "vegamag": su.VEGAMAG}


def _resolve_magsys(magsys):
    """Resolve a magnitude-system spec to an astropy/synphot unit.

    Parameters
    ----------
    magsys : str or Unit
        Case-insensitive ``'abmag'`` or ``'vegamag'``, or a unit object
        (anything exposing ``is_equivalent``), which is returned unchanged.

    Returns
    -------
    Unit
        ``astropy.units.ABmag`` or ``synphot.units.VEGAMAG``.

    Raises
    ------
    ValueError
        If ``magsys`` is an unrecognized string.
    """
    if hasattr(magsys, "is_equivalent"):
        return magsys
    try:
        return _MAGSYS[str(magsys).lower()]
    except KeyError:
        raise ValueError(
            f"unknown magsys {magsys!r}; expected one of {sorted(_MAGSYS)} "
            "or an astropy/synphot magnitude unit"
        )


_VEGA = None


def _get_vega():
    """Lazily load and cache the Vega reference spectrum for VEGAMAG work."""
    global _VEGA
    if _VEGA is None:
        _VEGA = SourceSpectrum.from_vega()
    return _VEGA
```

- [ ] **Step 4: Use the resolver in `_parse_mag_`**

In `src/wcc_etc/scene.py`, replace the magsys resolution block in `_parse_mag_` (currently lines ~600-605):

```python
            # make sure mag has the correct units.
            magsys = self.meta.get("magsys", "ABmag")
            # make sure it is an astropy units.
            if not hasattr(magsys, "is_equivalent"):
                magsys = getattr(u, magsys)
            #         
            mag = mag * magsys
```

with:

```python
            # make sure mag has the correct units.
            magsys = _resolve_magsys(self.meta.get("magsys", "ABmag"))
            mag = mag * magsys
```

- [ ] **Step 5: Pass `vegaspec` in `get_spectrum`**

In `src/wcc_etc/scene.py`, replace the normalize call in `get_spectrum` (currently line ~506):

```python
            if apply_mag and mag is not None:
                spectrum = self.spectrum.normalize(mag, band=self.band)
```

with:

```python
            if apply_mag and mag is not None:
                if mag.unit == su.VEGAMAG:
                    spectrum = self.spectrum.normalize(mag, band=self.band,
                                                       vegaspec=_get_vega())
                else:
                    spectrum = self.spectrum.normalize(mag, band=self.band)
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `pytest tests/test_scene.py -k "magsys or vegamag or cross_system" -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Run the full existing suite to confirm no regressions**

Run: `pytest tests/test_scene.py -v`
Expected: PASS — all pre-existing tests still pass (default is still `"ABmag"`).

- [ ] **Step 8: Commit**

```bash
git add src/wcc_etc/scene.py tests/test_scene.py
git commit -m "feat: support vegamag normalization and abmag/vegamag magsys aliases"
```

---

### Task 2: Flip default magsys to vegamag + migrate flux-asserting tests

Change the default from AB to Vega everywhere, then pin existing flux-dependent tests to `magsys="abmag"` so their validated numeric assertions hold. Task ends with the full suite green.

**Files:**
- Modify: `src/wcc_etc/scene.py` (`SceneElement.__init__` default; `_parse_mag_` fallback; docstrings)
- Test: `tests/test_scene.py` (add default-is-vegamag test; pin existing roundtrip tests)
- Modify (pin to abmag, as needed for green suite): `tests/test_snr.py`, `tests/test_saturation_reads.py`, `tests/test_saturation_count.py`, `tests/test_exptime.py`, `tests/test_image_exptime.py`, `tests/test_simulation.py`, `tests/test_twod_countrate_migration.py`, `tests/test_psf_aware_saturation.py`, `tests/test_sensorfilter.py`, `tests/test_image_simulator.py`, `tests/test_count_rate_components.py`, `tests/test_analytic_deprecation.py`, `tests/test_source_physics.py`

**Interfaces:**
- Consumes: Task 1's `_resolve_magsys` / Vega-aware `get_spectrum`.
- Produces: `SceneElement(...)` with no `magsys` resolves to `su.VEGAMAG`; `get_scene(...)` source and background default to `su.VEGAMAG`.

- [ ] **Step 1: Write the failing default-is-vegamag test**

Add to `tests/test_scene.py`:

```python
def test_default_magsys_is_vegamag():
    import wcc_etc
    se = SceneElement.from_config({"spectrum": "flat", "mag": 15,
                                   "bandpass": "johnson_r"})
    assert se.mag.unit == su.VEGAMAG
    scene = wcc_etc.get_scene(name="G5V", mag=15, background="zodi")
    assert scene.source.mag.unit == su.VEGAMAG
    assert scene.background.mag.unit == su.VEGAMAG
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_scene.py::test_default_magsys_is_vegamag -v`
Expected: FAIL — units are `ABmag`, not `VEGAMAG`.

- [ ] **Step 3: Flip the default in `SceneElement.__init__`**

In `src/wcc_etc/scene.py`, change the `__init__` signature (line ~344):

```python
    def __init__(self, spectrum, mag, 
                 magsys="ABmag", bandpass="johnson_v", 
```

to:

```python
    def __init__(self, spectrum, mag, 
                 magsys="vegamag", bandpass="johnson_v", 
```

And update the `magsys` docstring (line ~356-357) from:

```python
        magsys : str, optional
            The magnitude system (e.g., 'ABmag'). Default is "ABmag".
```

to:

```python
        magsys : str, optional
            The magnitude system: 'vegamag' or 'abmag' (case-insensitive).
            Default is "vegamag".
```

- [ ] **Step 4: Flip the fallback default in `_parse_mag_`**

In `src/wcc_etc/scene.py`, change the line added in Task 1:

```python
            magsys = _resolve_magsys(self.meta.get("magsys", "ABmag"))
```

to:

```python
            magsys = _resolve_magsys(self.meta.get("magsys", "vegamag"))
```

- [ ] **Step 5: Run the default test to verify it passes**

Run: `pytest tests/test_scene.py::test_default_magsys_is_vegamag -v`
Expected: PASS.

- [ ] **Step 6: Pin existing magnitude-roundtrip tests in `tests/test_scene.py` to abmag**

These tests assert that the observed **AB** magnitude equals the input, so they must declare `magsys="abmag"` to stay system-consistent. Edit each config dict to add `"magsys": "abmag"`:

In `test_blackbody_source_roundtrips_magnitude`:
```python
    se = SceneElement.from_config({"spectrum": "blackbody", "teff": 5777,
                                   "mag": 15, "magsys": "abmag",
                                   "bandpass": "johnson_v"})
```

In `test_blackbody_shape_matches_get_blackbody_flux`:
```python
    se = SceneElement.from_config({"spectrum": "blackbody", "teff": 5777,
                                   "mag": 15, "magsys": "abmag",
                                   "bandpass": "johnson_v"})
```

In `test_flat_source_is_constant_fnu_and_roundtrips_mag`:
```python
    se = SceneElement.from_config({"spectrum": "flat", "mag": 18,
                                   "magsys": "abmag", "bandpass": "johnson_v"})
```

In `test_powerlaw_source_slope` and `test_name_and_config`, add `"magsys": "abmag"` to their `baseconfig`/config dicts the same way (these read flux/shape relative to an AB-normalized reference).

- [ ] **Step 7: Run the full suite to find every other flux-dependent failure**

Run: `pytest tests -q`
Expected: a set of FAILURES in the test files listed under **Files** above — each is a `get_scene(...)` / `SceneElement(...)` callsite whose numeric assertion (SNR, saturation count, exposure time, count rate) was validated against AB-normalized flux and now sees Vega-normalized flux. Note the failing test names.

- [ ] **Step 8: Pin each failing callsite to abmag**

For every failing test, preserve its original numeric assertion by declaring `magsys="abmag"` on the source and background. Apply these two mechanical transforms:

`get_scene(...)` source — add the `magsys` kwarg, and add `"magsys": "abmag"` to `background_prop`:
```python
# before
scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                          background_prop={"bandpass": "johnson_r", "mag": 22.5})
# after
scene = wcc_etc.get_scene(name="G5V", mag=mag, magsys="abmag", background="zodi",
                          background_prop={"bandpass": "johnson_r", "mag": 22.5,
                                           "magsys": "abmag"})
```

If a failing test calls `get_scene` with `background="zodi"` and no `background_prop`, add one: `background_prop={"magsys": "abmag"}`.

`SceneElement.from_config({...})` / host `host_prop={...}` — add `"magsys": "abmag"` to the dict:
```python
# before
host_prop={"mag": 16, "bandpass": "johnson_r"}
# after
host_prop={"mag": 16, "bandpass": "johnson_r", "magsys": "abmag"}
```

Do NOT change any asserted numeric expected values — pinning to abmag restores the original physics. Do NOT touch `tests/test_wcc_etc.py` (legacy path, unaffected).

- [ ] **Step 9: Re-run the full suite until green**

Run: `pytest tests -q`
Expected: PASS (all tests). If any flux-asserting test still fails, it has a source/host/background callsite not yet pinned — apply the Step 8 transform to it and re-run.

- [ ] **Step 10: Commit**

```bash
git add src/wcc_etc/scene.py tests/
git commit -m "feat!: default magsys to vegamag; pin existing flux tests to abmag"
```

---

### Task 3: Remove dead AB-hardcoded `_get_spectrum_observation`

Delete the unused static method that hardcoded `abmag * u.ABmag`, leaving the magsys-aware `scene.py` path as the single normalization route.

**Files:**
- Modify: `src/wcc_etc/simulation.py` (delete method at lines ~537-556)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (removal only; method has zero callers).

- [ ] **Step 1: Confirm there are no callers**

Run: `grep -rn "_get_spectrum_observation" src tests`
Expected: exactly one line — the `def _get_spectrum_observation` definition in `src/wcc_etc/simulation.py`. (If any other line appears, STOP — it is not dead and this task must be revised.)

- [ ] **Step 2: Delete the method**

In `src/wcc_etc/simulation.py`, remove the entire static method (including its `@staticmethod` decorator), currently:

```python
    @staticmethod
    def _get_spectrum_observation(spectrum, abmag, bandpass):
        """
        Normalize a spectrum and return an observation.

        Parameters
        ----------
        spectrum : SourceSpectrum
            The spectrum to normalize.
        abmag : float
            The AB magnitude to normalize to.
        bandpass : SpectralElement
            The bandpass filter.

        Returns
        -------
        Observation
        """
        spec_at_mag = spectrum.normalize(abmag * u.ABmag, bandpass, force='extrap')
        return Observation(spec_at_mag, bandpass, force='extrap')
```

- [ ] **Step 3: Confirm the module imports cleanly and the suite passes**

Run: `python -c "import wcc_etc.simulation" && pytest tests -q`
Expected: import succeeds; all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/wcc_etc/simulation.py
git commit -m "refactor: remove dead AB-hardcoded _get_spectrum_observation"
```

---

## Self-Review

**Spec coverage:**
- Magsys string resolution (`abmag`/`vegamag`, case-insensitive, unit passthrough, `ValueError`) → Task 1 Steps 3, 1 (`test_resolve_magsys_aliases_case_insensitive`).
- Vega-aware normalization + lazily-cached Vega → Task 1 Steps 3, 5.
- Default → vegamag everywhere (source, host, background) → Task 2 Steps 3, 4, 1 (`test_default_magsys_is_vegamag`).
- AB path unchanged → Task 1 Step 7 (existing suite green) + Task 2 abmag-pinned tests retain original numbers.
- Careful cross-system quantitative tests → Task 1 (`test_vegamag_normalization_runs_and_differs_from_ab`, `test_cross_system_offset_is_band_dependent` over V and K, `test_vegamag_roundtrips`).
- Delete dead `_get_spectrum_observation` → Task 3.
- Science ripple / existing-test migration acknowledged in spec → Task 2 Steps 6-9.
- Out of scope (legacy `wcc_etc.py`, loose aliases) → enforced by Global Constraints and the `with pytest.raises(ValueError): _resolve_magsys("vega")` assertion.

**Placeholder scan:** No TBD/TODO; every code step shows full content; the test-migration step uses an objective gate (green suite) with concrete transform patterns rather than vague "fix failures."

**Type consistency:** `_resolve_magsys`, `_get_vega`, `_MAGSYS`, `_VEGA`, `_band_ab_vega_offset`, `_inband_flam` names used consistently across tasks; `su.VEGAMAG` / `u.ABmag` unit comparisons consistent between `get_spectrum` and tests.
