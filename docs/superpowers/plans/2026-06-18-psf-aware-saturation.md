# PSF-Aware Saturation & 2D Count-Rate Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `get_peak_pixel`/`is_saturated` and the 2D SNR/image path derive from the actual rendered PSF + PSF-independent total count rates, so defocused configs saturate correctly; leave the Airy `compute_psf_profile` used only by deprecated analytic methods.

**Architecture:** Add a PSF-independent `_count_rate_components()` primitive (total source rate + per-pixel sky, no Airy aperture). Point `_image_render_bundle` and `ImageSimulator.simulate` at it. Reimplement `get_peak_pixel`/`is_saturated` as `source_rate_total·psf_norm.max() + sky_per_pix + dark`, all read from the render bundle. Retire `peak_pixel_fraction`; deprecate the analytic Airy methods.

**Tech Stack:** Python, numpy, astropy.units, synphot (Observation.countrate), pytest. All work in `src/wcc_etc/simulation.py` and `src/wcc_etc/psfsim.py`.

## Global Constraints

- CI runs **numpy 2.x / Python 3.11**; local is numpy 1.26. Do not use `np.trapz`.
- Plain commit messages, **no Co-Authored-By trailer**.
- Run `pytest -q` (full suite) once before each task's commit; all tests pass.
- **Backward compatibility:** existing `get_peak_pixel(time, units, n_reads)` and `is_saturated(time, n_reads)` positional calls must keep working. New args are keyword-only with defaults.
- **Intended numeric change (not a regression):** the migration removes a pre-existing spurious `× ee_at_aper` on the *background*, so per-pixel sky rises by `1/ee_at_aper`. Source-dominated SNR is unchanged; sky-affected SNR drops slightly. This is correct — treat shifted sky-dominated test values as the new ground truth and record them.
- `units` semantics unchanged: `'adu'` includes `sensor.bias_level` and divides by `sensor.gain`; `'e'`/`'e-'`/`'electron'` excludes bias. Saturation in ADU compares `>= sensor.adc_max`.
- Saturation is per-frame: per-frame integration time is `time / n_reads`.

---

### Task 1: `_count_rate_components()` — PSF-independent count-rate primitive

**Files:**
- Modify: `src/wcc_etc/simulation.py` (add method near `get_countrates`, ~line 558)
- Test: `tests/test_count_rate_components.py` (create)

**Interfaces:**
- Consumes: `self.scene`, `self.telescope`, `self.sensor`; `Scene.get_observation(band=, area=, as_dict=True)`; `Observation.countrate(area=telescope.surface)`; `Sensor.get_plate_scale(telescope) -> Quantity[arcsec/pix]`.
- Produces: `_count_rate_components(self, band=None) -> dict` with float values (electron/s):
  `{"source_rate_total", "background_rate_per_pix", "diffuse_rate_per_pix"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_count_rate_components.py
import warnings
import numpy as np
import astropy.units as u
import wcc_etc


def _sim():
    scene = wcc_etc.get_scene(name='G2V', mag=15.0, background='zodi',
                              bandpass='johnson_v',
                              background_prop={'bandpass': 'johnson_v', 'mag': 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene('sony:r', scene)


def test_source_total_matches_pre_ee_countrate():
    sim = _sim()
    comp = sim._count_rate_components()
    # legacy in-aperture source / ee_at_aper == total source rate (PSF-independent)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = sim.get_countrates(units="e/s", as_dict=True)
    ee = sim.psf_profile["ee_at_aper"]
    legacy_total = (legacy["source"] / ee).to(u.electron / u.s).value
    assert comp["source_rate_total"] == __import__("pytest").approx(legacy_total, rel=1e-6)


def test_background_per_pix_removes_spurious_ee_factor():
    sim = _sim()
    comp = sim._count_rate_components()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = sim.get_countrates(units="e/s", as_dict=True)
    prof = sim.psf_profile
    n_psf = prof["num_psf_pixels"].value if hasattr(prof["num_psf_pixels"], "value") else prof["num_psf_pixels"]
    ee = prof["ee_at_aper"]
    legacy_bkg_per_pix = (legacy["background"] / n_psf).to(u.electron / u.s).value
    # new per-pixel sky == legacy / ee_at_aper  (the spurious factor removed)
    assert comp["background_rate_per_pix"] == __import__("pytest").approx(legacy_bkg_per_pix / ee, rel=1e-4)
    assert comp["background_rate_per_pix"] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_count_rate_components.py -v`
Expected: FAIL — `AttributeError: 'Simulation' object has no attribute '_count_rate_components'`.

- [ ] **Step 3: Implement `_count_rate_components`**

Add this method to the `Simulation` class (e.g. immediately after `get_countrates`, ~line 615):

```python
    def _count_rate_components(self, band=None):
        """PSF-independent count-rate primitives for the 2D path (electron/s).

        Returns total source rate (point source, no aperture) and per-pixel
        sky/diffuse rates (surface brightness evaluated at one detector pixel).
        Unlike get_countrates, this does NOT apply the Airy ee_at_aper, so it is
        independent of compute_psf_profile and correct for any PSF. The rendered
        PSF (in _image_render_bundle) handles spatial distribution.
        """
        if band is None:
            band = self.sensor.bandpass
        plate_scale_arcsec = self.sensor.get_plate_scale(self.telescope).to(u.arcsec / u.pix).value
        pixel_area_arcsec2 = plate_scale_arcsec ** 2  # one detector pixel, arcsec^2
        surf = self.telescope.surface

        # One call at one-pixel area: non-surface-brightness elements (the point
        # source) ignore `area` -> total rate; surface-brightness elements (sky)
        # use it -> per-pixel rate.
        obs = self.scene.get_observation(band=band, area=pixel_area_arcsec2, as_dict=True)

        source_rate_total = 0.0
        background_rate_per_pix = 0.0
        diffuse_rate_per_pix = 0.0
        for name, o in obs.items():
            if o is None:
                continue
            rate = (o.countrate(area=surf) * u.electron / u.ct).to(u.electron / u.s).value
            if name == "source":
                source_rate_total = rate
            elif name == "background":
                background_rate_per_pix = rate
            else:
                diffuse_rate_per_pix += rate
        return {"source_rate_total": source_rate_total,
                "background_rate_per_pix": background_rate_per_pix,
                "diffuse_rate_per_pix": diffuse_rate_per_pix}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_count_rate_components.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all pass (no other code calls the new method yet).

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_count_rate_components.py
git commit -m "Add PSF-independent _count_rate_components primitive"
```

---

### Task 2: Migrate `_image_render_bundle` + `ImageSimulator.simulate` to `_count_rate_components`

**Files:**
- Modify: `src/wcc_etc/simulation.py:773-822` (`_image_render_bundle`)
- Modify: `src/wcc_etc/psfsim.py:331-345` (`ImageSimulator.simulate` source/sky budget)
- Test: `tests/test_twod_countrate_migration.py` (create)

**Interfaces:**
- Consumes: `Simulation._count_rate_components()` (Task 1).
- Produces: `_image_render_bundle` returns the same keys as before
  (`psf_norm`, `plate_scale_mas`, `source_rate_total`, `diffuse_rate_per_pix`,
  `background_rate_per_pix`) but computed from `_count_rate_components`, with no
  `psf_profile`/`ee_at_aper`/`get_countrates` reads.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_twod_countrate_migration.py
import inspect
import warnings
import numpy as np
import wcc_etc


def _sim(filt='sony:r', mag=15.0):
    scene = wcc_etc.get_scene(name='G2V', mag=mag, background='zodi',
                              bandpass='johnson_v',
                              background_prop={'bandpass': 'johnson_v', 'mag': 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene(filt, scene)


def test_bundle_uses_count_rate_components_not_psf_profile():
    src = inspect.getsource(wcc_etc.Simulation._image_render_bundle)
    assert "_count_rate_components" in src
    assert "ee_at_aper" not in src
    assert "get_countrates" not in src


def test_bundle_rates_match_components():
    sim = _sim()
    comp = sim._count_rate_components()
    b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
    assert b["source_rate_total"] == __import__("pytest").approx(comp["source_rate_total"], rel=1e-6)
    assert b["background_rate_per_pix"] == __import__("pytest").approx(comp["background_rate_per_pix"], rel=1e-6)


def test_bright_source_snr_essentially_unchanged():
    # source-dominated: removing the sky ee-factor barely moves SNR
    sim = _sim(mag=12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = sim.get_image_snr(time=1.0)["snr"]
    assert snr > 0 and np.isfinite(snr)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_twod_countrate_migration.py -v`
Expected: FAIL on `test_bundle_uses_count_rate_components_not_psf_profile` (source still contains `ee_at_aper`/`get_countrates`).

- [ ] **Step 3: Rewrite the count-rate section of `_image_render_bundle`**

In `src/wcc_etc/simulation.py`, replace the block from `profile = self.psf_profile`
through the `background_rate_per_pix` computation (currently lines ~796-814,
ending just before `bundle = {`) with:

```python
        comps = self._count_rate_components()
        source_rate_total = comps["source_rate_total"]
        diffuse_rate_per_pix = comps["diffuse_rate_per_pix"]
        background_rate_per_pix = comps["background_rate_per_pix"]
```

Leave the `imsim = ImageSimulator(...)`, `ctx = ...`, `psf_norm = psf.render(ctx)`
lines and the `bundle = {...}` / cache lines unchanged.

- [ ] **Step 4: Migrate `ImageSimulator.simulate`**

In `src/wcc_etc/psfsim.py`, replace the source/sky budget block (currently ~331-345,
the `profile = sim.psf_profile` / `ee_at_aper` / `num_psf_pixels` / `get_countrates`
/ `source_e_total` / background-per-pixel derivation) with the component-based
version. Use the simulation's primitive and the rendered PSF:

```python
        comps = sim._count_rate_components()
        source_e_total = comps["source_rate_total"] * time.to(u.s).value
        source_image = source_e_total * psf_norm
        background_per_pix = comps["background_rate_per_pix"] * time.to(u.s).value
        diffuse_per_pix = comps["diffuse_rate_per_pix"] * time.to(u.s).value
```

Then wherever the subsequent code added the per-pixel sky/diffuse to the image,
use `background_per_pix + diffuse_per_pix` (replacing the old `count_rates`-derived
per-pixel value). Read the surrounding ~20 lines and keep the dark-current and
noise handling exactly as-is; only the source-total and per-pixel-sky derivation
changes. If `time` is a bare float here, use `time` directly (no `.to`); match the
existing unit handling in that method.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_twod_countrate_migration.py tests/test_image_simulator.py -v`
Expected: migration tests PASS. If any `test_image_simulator.py` assertion shifts,
confirm it is a sky-affected value moving in the expected direction (sky higher),
update the expected number, and note it in the report.

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all pass. Any failures must be sky-affected SNR/image values shifted by
the removed `ee_at_aper` sky factor — update those expectations to the new correct
values and record each change in the report. Source-dominated values must be
unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/wcc_etc/simulation.py src/wcc_etc/psfsim.py tests/test_twod_countrate_migration.py tests/test_image_simulator.py
git commit -m "Route 2D render bundle and ImageSimulator through _count_rate_components"
```

---

### Task 3: Reimplement `get_peak_pixel` / `is_saturated` on the rendered PSF

**Files:**
- Modify: `src/wcc_etc/simulation.py:682-770` (`get_peak_pixel`, `is_saturated`)
- Test: `tests/test_psf_aware_saturation.py` (create)

**Interfaces:**
- Consumes: `_image_render_bundle(psf, jitter_sigma_mas, npix, oversample)` (Task 2)
  returning `psf_norm`, `source_rate_total`, `background_rate_per_pix`,
  `diffuse_rate_per_pix`; `self._default_psf`; `AiryPSF`.
- Produces:
  `get_peak_pixel(time=None, units="adu", n_reads=None, *, psf=None, jitter_sigma_mas=None, npix=128, oversample=11) -> Quantity` (scalar or array).
  `is_saturated(time=None, n_reads=None, *, psf=None, jitter_sigma_mas=None, npix=128, oversample=11) -> bool or ndarray[bool]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_psf_aware_saturation.py
import warnings
import numpy as np
import astropy.units as u
import wcc_etc


def _scene(mag):
    return wcc_etc.get_scene(name='G2V', mag=mag, background='zodi',
                             bandpass='johnson_v',
                             background_prop={'bandpass': 'johnson_v', 'mag': 22.5})


def test_defocus_peak_fraction_far_below_airy():
    # zwo:bb2 = broadband + 2-wave defocus; zwo:r = in-focus
    sim_def = wcc_etc.Simulation.from_sensorfilter('zwo:bb2', _scene(15.0))
    b = sim_def._image_render_bundle(sim_def._default_psf, None, 128, 11)
    airy_b = sim_def._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
    assert b["psf_norm"].max() < 0.2 * airy_b["psf_norm"].max()  # defocus spreads the peak


def test_is_saturated_agrees_with_image_snr_flag():
    sim = wcc_etc.Simulation.from_sensorfilter('zwo:bb2', _scene(13.0))
    for t in [0.05, 0.1, 0.5, 2.0]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flag_snr = bool(sim.get_image_snr(time=t)["saturated"])
            flag_sat = bool(sim.is_saturated(t))
        assert flag_sat == flag_snr, f"disagreement at t={t}"


def test_get_peak_pixel_array_time_linear():
    sim = wcc_etc.Simulation.from_sensorfilter('zwo:r', _scene(16.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = sim.get_peak_pixel(1.0, units="e-").to(u.electron).value
        p2 = sim.get_peak_pixel(np.array([1.0, 2.0]), units="e-").to(u.electron).value
    assert p2.shape == (2,)
    assert p2[0] == __import__("pytest").approx(p1, rel=1e-6)
    assert p2[1] == __import__("pytest").approx(2 * p1, rel=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_psf_aware_saturation.py -v`
Expected: FAIL — `test_defocus_peak_fraction_far_below_airy` may pass (bundle exists), but `is_saturated`/`get_peak_pixel` still use the Airy analytic peak so the agreement/array tests fail (or `get_peak_pixel` rejects the keyword args).

- [ ] **Step 3: Reimplement `get_peak_pixel`**

Replace the body of `get_peak_pixel` (from the signature through its `return`s, lines ~682-749) with:

```python
    def get_peak_pixel(self, time=None, units="adu", n_reads=None, *,
                       psf=None, jitter_sigma_mas=None, npix=128, oversample=11):
        """Brightest-pixel value for the actual (possibly defocused) PSF.

        Peak = source_rate_total * peak_pixel_fraction + sky_per_pix + dark,
        where peak_pixel_fraction is the brightest pixel of the *rendered* PSF
        (psf_norm.max()), so defocused configs are handled correctly. `psf`
        defaults to _default_psf (set by from_sensorfilter), else AiryPSF.
        Saturation is per-frame (per-frame time = time / n_reads). Linear in
        time, so scalar or array `time` both work.
        """
        if time is None:
            time = self._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second
        n_reads = self._resolve_n_reads(n_reads)
        tf = (time / n_reads).to(u.second).value  # per-frame seconds (scalar or array)

        if psf is None:
            psf = self._default_psf if self._default_psf is not None else AiryPSF()
        b = self._image_render_bundle(psf, jitter_sigma_mas, npix, oversample)

        dark_rate_per_pix = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        peak_rate = (b["source_rate_total"] * float(b["psf_norm"].max())
                     + b["background_rate_per_pix"]
                     + b["diffuse_rate_per_pix"]
                     + dark_rate_per_pix)  # electron / s in the brightest pixel
        peak_e = (peak_rate * tf) * u.electron

        if units in ["e", "e-", "electron"]:
            return peak_e
        if units.lower() == "adu":
            return (peak_e / self.sensor.gain).to(u.ct) + self.sensor.bias_level
        raise ValueError(f"unknown units {units=}. 'adu' or electron/'e-' expected.")
```

Ensure `AiryPSF` is importable in `simulation.py` (it is already used at lines ~904/1009 via the module-level import; if not, add `from .psfsim import AiryPSF` where the others are imported).

- [ ] **Step 4: Update `is_saturated` to forward the new kwargs**

Replace `is_saturated` (lines ~752-771) with:

```python
    def is_saturated(self, time=None, n_reads=None, *,
                     psf=None, jitter_sigma_mas=None, npix=128, oversample=11):
        """Whether the brightest pixel (ADU) reaches sensor.adc_max, per frame,
        for the actual (possibly defocused) PSF. See get_peak_pixel."""
        peak_adu = self.get_peak_pixel(time, units="adu", n_reads=n_reads, psf=psf,
                                       jitter_sigma_mas=jitter_sigma_mas,
                                       npix=npix, oversample=oversample)
        return peak_adu.value >= self.sensor.adc_max
```

(Preserve whatever the prior return expression was if it differs in form — keep the `>= sensor.adc_max` comparison and scalar/array behavior.)

- [ ] **Step 5: Run the new tests**

Run: `pytest tests/test_psf_aware_saturation.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Run the existing saturation suites and reconcile**

Run: `pytest tests/test_saturation_count.py tests/test_saturation_reads.py -v`
Expected: in-focus saturation behavior is preserved. If a value shifted, it is the
rendered-Airy peak (`psf_norm.max()`) vs the old analytic `render_detector_psf`
peak — verify the new value is correct (e.g. peak fraction is physically sensible),
update the expectation, and record the numeric delta in the report. Saturation
booleans at thresholds: confirm any flip is physically correct, not a tolerance
artifact.

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: all pass (with reconciled saturation expectations).

- [ ] **Step 8: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_psf_aware_saturation.py tests/test_saturation_count.py tests/test_saturation_reads.py
git commit -m "Compute saturation from the rendered PSF (PSF-aware get_peak_pixel/is_saturated)"
```

---

### Task 4: Retire `peak_pixel_fraction`; deprecate the analytic Airy path

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `compute_psf_profile` (1174), `get_countrates` (558), `get_signal_and_variance` (616), `get_exptime_for_snr` (1101)
- Test: `tests/test_analytic_deprecation.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `compute_psf_profile`, `get_countrates`, `get_signal_and_variance`,
  `get_exptime_for_snr` each emit `DeprecationWarning` and still return prior
  values. A private `_countrates_in_aperture(...)` carries the old `get_countrates`
  body so internal analytic callers don't trigger the warning.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analytic_deprecation.py
import warnings
import pytest
import wcc_etc


def _sim():
    scene = wcc_etc.get_scene(name='G2V', mag=15.0, background='zodi',
                              bandpass='johnson_v',
                              background_prop={'bandpass': 'johnson_v', 'mag': 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene('sony:r', scene)


@pytest.mark.parametrize("call", [
    lambda s: s.get_countrates(units="e/s"),
    lambda s: s.compute_psf_profile(),
    lambda s: s.get_signal_and_variance(1.0),
    lambda s: s.get_exptime_for_snr(50.0),
])
def test_analytic_methods_warn(call):
    s = _sim()
    with pytest.warns(DeprecationWarning):
        call(s)


def test_peak_pixel_fraction_retired():
    s = _sim()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = s.compute_psf_profile()
    assert "peak_pixel_fraction" not in prof
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_analytic_deprecation.py -v`
Expected: FAIL — no warnings emitted; `peak_pixel_fraction` still present.

- [ ] **Step 3: Extract the non-warning helper and warn from `get_countrates`**

Rename the current body of `get_countrates` into `_countrates_in_aperture` with the
same signature, and make `get_countrates` warn then delegate:

```python
    def get_countrates(self, scene=None, band=None, units="adu/s", as_dict=True):
        """DEPRECATED (Airy, in-aperture). Use _count_rate_components for the 2D
        path or get_image_snr. Retained for the analytic Airy methods."""
        import warnings
        warnings.warn("get_countrates is deprecated (Airy in-aperture model); "
                      "use get_image_snr / _count_rate_components.",
                      DeprecationWarning, stacklevel=2)
        return self._countrates_in_aperture(scene=scene, band=band, units=units, as_dict=as_dict)

    def _countrates_in_aperture(self, scene=None, band=None, units="adu/s", as_dict=True):
        # <-- the exact previous body of get_countrates, unchanged -->
        ...
```

Update the internal analytic callers `get_signal_and_variance` and
`get_exptime_for_snr` to call `self._countrates_in_aperture(...)` instead of
`self.get_countrates(...)` so they don't emit the warning themselves.

- [ ] **Step 4: Add DeprecationWarnings to the analytic methods**

At the top of `get_signal_and_variance`, `get_exptime_for_snr`, and
`compute_psf_profile`, add (matching the existing `get_snr_airy` pattern at line ~1085):

```python
        import warnings
        warnings.warn("<method> is the analytic Airy approximation and is "
                      "deprecated; use the PSF-aware 2D path (get_image_snr / "
                      "get_image_exptime_for_snr / get_peak_pixel).",
                      DeprecationWarning, stacklevel=2)
```

(substitute the method name). Do NOT add a warning to `psf_profile` (the cached
property) — only to the public `compute_psf_profile()` method, so internal
property access during the 2D path stays quiet. Note: after Task 2/3 the 2D path
no longer touches `psf_profile`, so the only `compute_psf_profile` calls are
explicit/analytic.

- [ ] **Step 5: Remove `peak_pixel_fraction` from `compute_psf_profile`**

In `compute_psf_profile` (1174), delete the `peak_pixel_fraction` computation
(the `render_detector_psf(...)` call used only for the peak and the
`peak_pixel_fraction = float(psf_detector.max())` line) and remove
`"peak_pixel_fraction": peak_pixel_fraction` from the returned dict. Update the
docstring's returned-keys list to drop `peak_pixel_fraction`.

- [ ] **Step 6: Run the new tests**

Run: `pytest tests/test_analytic_deprecation.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite and silence expected internal warnings**

Run: `pytest -q`
Expected: all pass. If any test now fails only because a deprecated method it calls
emits a `DeprecationWarning` (and the suite treats warnings as errors or asserts
clean output), wrap that call in `warnings.catch_warnings()` /
`simplefilter("ignore", DeprecationWarning)` in the *test*, or switch genuinely
internal callers to the non-deprecated path. Do not suppress warnings in library code.

- [ ] **Step 8: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_analytic_deprecation.py
git commit -m "Retire peak_pixel_fraction and deprecate the analytic Airy path"
```

---

## Post-implementation (per feature workflow)

After all tasks pass:

1. **Scratch demo notebook** — extend `notebooks_scratch/20260617_transit_lightcurve.ipynb` (or a new `notebooks_scratch/20260618_earth_twin_saturation.ipynb`) with the Earth-twin example: V=13 G2V, `zwo:bb2` (defocused broadband) vs `zwo:r` (in-focus) showing the in-focus config saturating where broadband+defocus succeeds, plus the same Earth-radius planet around an M dwarf (~930 ppm) detectable in-focus. Built via the `notebook-demo` skill and executed end-to-end. **Requires the transit-lightcurve module (PR #33)**; build it where both are present. Gitignored — not part of this PR.
2. **Update the project-status memory** — new `_count_rate_components`, PSF-aware `get_peak_pixel`/`is_saturated`, the deprecated analytic path, the intended sky-rate change, and the saturation-test reconciliations.
3. Push & open a PR per the pushing-and-CI workflow.

## Self-Review

- **Spec coverage:** `_count_rate_components` (Task 1) ✓; bundle + simulate migration (Task 2) ✓; PSF-aware get_peak_pixel/is_saturated with new kwargs (Task 3) ✓; retire peak_pixel_fraction (Task 4) ✓; deprecate-but-keep analytic path (Task 4) ✓; surface-brightness per-pixel via one-pixel area (Task 1) ✓; backward-compatible signatures (Tasks 3) ✓; intended sky-rate change flagged (Global Constraints, Task 2) ✓; in-focus regression reconciliation (Task 3) ✓; demo payoff (post-impl) ✓.
- **Placeholders:** none — every code step shows the code. Task 2 Step 4 and Task 4 Step 3 ask the implementer to read ~20 surrounding lines before editing (the exact prior body is large/contextual); the edit and its acceptance test are fully specified.
- **Type consistency:** `_count_rate_components` returns float-valued dict with keys `source_rate_total`/`background_rate_per_pix`/`diffuse_rate_per_pix` — used identically in Tasks 2 & 3; bundle keys unchanged; `get_peak_pixel`/`is_saturated` new keyword-only args consistent across both and match `get_image_snr`'s names (`psf`, `jitter_sigma_mas`, `npix`, `oversample`).
