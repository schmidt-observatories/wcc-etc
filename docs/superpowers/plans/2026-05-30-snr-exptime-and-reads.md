# SNR ↔ exposure-time inverse with coadded reads — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a time-for-target-SNR inverse on both the analytic and PSF-aware SNR paths, plus a coadded-`n_reads` parameter that scales read noise and per-frame saturation.

**Architecture:** Every case reduces to `SNR(t) = A·t/√(B·t+C)`, whose positive-root inverse is closed-form. A shared `solve_time_for_snr` helper serves both paths. `n_reads = N` coadds N frames over total time `t`: read-noise variance becomes `N·RN²` (signal/Poisson/dark unchanged), and saturation is evaluated per-frame at `t/N`. `n_reads` is a settable, updatable `Simulation` parameter (default 1; `N=1` reproduces current behavior exactly).

**Tech Stack:** Python, numpy, astropy.units, synphot; pytest. Work in `src/wcc_etc/simulation.py` and `src/wcc_etc/psfsim.py`. Run all commands from the repo root `wcc-etc/` (the inner git repo). Branch: `snr-exptime-nreads`.

**Spec:** `docs/superpowers/specs/2026-05-30-snr-exptime-and-reads-design.md`

---

## File Structure

- `src/wcc_etc/psfsim.py` — add `solve_time_for_snr` (pure quadratic solver, scalar/array), refactor a `_radial_cumulative` geometry helper out of `aperture_snr_radial`, add `aperture_time_for_snr` (per-radius inverse).
- `src/wcc_etc/simulation.py` — `n_reads` constructor param + `_mutable_parameters`; `_resolve_n_reads`/`_snr_coefficients` helpers; thread `n_reads` through `get_signal_and_variance`, `get_snr`, `get_peak_pixel`, `is_saturated`, `get_image_snr`; add `get_exptime_for_snr` and `get_image_exptime_for_snr`.
- `tests/test_exptime.py` — new: analytic round-trip, N=1 parity, monotonicity, √N scaling, update, edge.
- `tests/test_image_exptime.py` — new: PSF-aware round-trip, parity, analytic cross-check.
- `tests/test_saturation_reads.py` — new: per-frame saturation vs n_reads.
- Existing `tests/test_psfsource.py`, `tests/test_snr.py`, `tests/test_simulation.py` must stay green (regression).

---

## Task 1: `solve_time_for_snr` quadratic helper

**Files:**
- Modify: `src/wcc_etc/psfsim.py` (add module-level function near `aperture_snr_radial`, ~line 778)
- Test: `tests/test_psfsource.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_psfsource.py`:

```python
def test_solve_time_for_snr_scalar_roundtrips():
    from wcc_etc.psfsim import solve_time_for_snr
    A, B, C, snr = 10.0, 5.0, 100.0, 25.0
    t = solve_time_for_snr(snr, A, B, C)
    # forward SNR at t must recover the target
    assert abs(A * t / (B * t + C) ** 0.5 - snr) < 1e-9


def test_solve_time_for_snr_array_and_zero_signal():
    import numpy as np
    from wcc_etc.psfsim import solve_time_for_snr
    A = np.array([10.0, 0.0, 4.0])
    B = np.array([5.0, 5.0, 2.0])
    C = np.array([100.0, 100.0, 50.0])
    t = solve_time_for_snr(30.0, A, B, C)
    assert np.isinf(t[1])                       # zero source -> infinite time
    assert np.allclose(A[[0, 2]] * t[[0, 2]] /
                       np.sqrt(B[[0, 2]] * t[[0, 2]] + C[[0, 2]]), 30.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_psfsource.py::test_solve_time_for_snr_scalar_roundtrips tests/test_psfsource.py::test_solve_time_for_snr_array_and_zero_signal -v`
Expected: FAIL with `ImportError: cannot import name 'solve_time_for_snr'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/wcc_etc/psfsim.py` (just above `def aperture_snr_radial`):

```python
def solve_time_for_snr(snr, A, B, C):
    """
    Solve SNR = A*t / sqrt(B*t + C) for the positive root t.

    A, B, C may be scalars or broadcastable arrays (A = signal rate, B = variance
    rate, C = constant read-noise variance). Returns t in the same shape (a float
    if all inputs are scalar). Entries with A <= 0 return +inf.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    s2 = float(snr) ** 2
    disc = s2 * s2 * B ** 2 + 4.0 * A ** 2 * s2 * C
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (s2 * B + np.sqrt(disc)) / (2.0 * A ** 2)
    t = np.where(A > 0, t, np.inf)
    return t.item() if t.ndim == 0 else t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_psfsource.py -k solve_time_for_snr -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_psfsource.py
git commit -m "Add solve_time_for_snr quadratic helper"
```

---

## Task 2: Refactor `_radial_cumulative` out of `aperture_snr_radial`

Pure refactor so the new inverse can reuse the radial geometry without duplication. No behavior change.

**Files:**
- Modify: `src/wcc_etc/psfsim.py:778-828` (`aperture_snr_radial`)
- Test: existing `tests/test_psfsource.py` (must stay green)

- [ ] **Step 1: Run the existing PSF tests to capture the green baseline**

Run: `python -m pytest tests/test_psfsource.py -q`
Expected: all pass (note the count).

- [ ] **Step 2: Add the geometry helper and rewrite the body to use it**

In `src/wcc_etc/psfsim.py`, add above `aperture_snr_radial`:

```python
def _radial_cumulative(psf_norm, plate_scale_mas):
    """
    Radius-sorted cumulative geometry of a normalized PSF.

    Returns (r_mas, enclosed_fraction, n_pix) as ascending-radius arrays:
    enclosed_fraction is the cumulative PSF sum (psf sums to 1) and n_pix is the
    number of pixels enclosed (1..N).
    """
    psf_norm = np.asarray(psf_norm, dtype=float)
    if psf_norm.ndim != 2 or psf_norm.shape[0] != psf_norm.shape[1]:
        raise ValueError(f"psf_norm must be a square 2D array, got shape {psf_norm.shape}")
    npix = psf_norm.shape[0]
    c = (npix - 1) / 2.0
    yy, xx = np.mgrid[0:npix, 0:npix]
    r_pix = np.sqrt((xx - c) ** 2 + (yy - c) ** 2).ravel()
    order = np.argsort(r_pix, kind="stable")
    r_sorted = r_pix[order]
    enclosed = np.cumsum(psf_norm.ravel()[order])
    n_pix = np.arange(1, r_sorted.size + 1)
    return r_sorted * plate_scale_mas, enclosed, n_pix
```

Then replace the body of `aperture_snr_radial` (the lines that build `r_pix`/`order`/`r_sorted`/`enclosed`/`n_pix`) with:

```python
    r_mas, enclosed, n_pix = _radial_cumulative(psf_norm, plate_scale_mas)

    signal = source_e_total * enclosed
    per_pix_var = diffuse_per_pix + dark_per_pix + read_noise ** 2
    noise = np.sqrt(signal + per_pix_var * n_pix)
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)

    return {"r_mas": r_mas,
            "enclosed_fraction": enclosed,
            "n_pix": n_pix,
            "signal_e": signal,
            "noise_e": noise,
            "snr": snr}
```

(The square-shape validation now lives in `_radial_cumulative`; remove the duplicate check from `aperture_snr_radial`.)

- [ ] **Step 3: Run the existing PSF tests to verify no regression**

Run: `python -m pytest tests/test_psfsource.py -q`
Expected: same pass count as Step 1.

- [ ] **Step 4: Commit**

```bash
git add src/wcc_etc/psfsim.py
git commit -m "Extract _radial_cumulative geometry helper (no behavior change)"
```

---

## Task 3: `aperture_time_for_snr` (PSF-aware per-radius inverse)

**Files:**
- Modify: `src/wcc_etc/psfsim.py` (add after `select_aperture`, ~line 848)
- Test: `tests/test_psfsource.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_psfsource.py`:

```python
def test_aperture_time_for_snr_matches_forward():
    import numpy as np
    from wcc_etc.psfsim import aperture_snr_radial, aperture_time_for_snr
    # simple centered gaussian-ish PSF
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - n // 2) ** 2 + (yy - n // 2) ** 2
    psf = np.exp(-r2 / (2 * 3.0 ** 2))
    psf /= psf.sum()
    plate, src_rate, diff_rate, dark_rate, rn = 50.0, 200.0, 0.5, 0.1, 3.0

    # pick a fixed aperture radius, find the time for SNR=50, then check forward
    res = aperture_time_for_snr(psf, plate, src_rate, diff_rate, dark_rate, rn,
                                n_reads=1, snr=50.0, r_aper_mas=300.0)
    t = res["time_s"]
    prof = aperture_snr_radial(psf, plate, src_rate * t, diff_rate * t,
                               dark_rate * t, rn)
    # SNR at res's radius and that time reproduces the target
    idx = np.searchsorted(prof["r_mas"], res["r_aper_mas"], side="right") - 1
    assert abs(prof["snr"][idx] - 50.0) < 0.05


def test_aperture_time_for_snr_optimize_is_minimum():
    import numpy as np
    from wcc_etc.psfsim import aperture_time_for_snr
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - n // 2) ** 2 + (yy - n // 2) ** 2
    psf = np.exp(-r2 / (2 * 3.0 ** 2)); psf /= psf.sum()
    fixed = aperture_time_for_snr(psf, 50.0, 200.0, 0.5, 0.1, 3.0,
                                  n_reads=1, snr=50.0, r_aper_mas=300.0)
    best = aperture_time_for_snr(psf, 50.0, 200.0, 0.5, 0.1, 3.0,
                                 n_reads=1, snr=50.0, optimize=True)
    assert best["time_s"] <= fixed["time_s"] + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_psfsource.py -k aperture_time_for_snr -v`
Expected: FAIL with `ImportError: cannot import name 'aperture_time_for_snr'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/wcc_etc/psfsim.py` after `select_aperture`:

```python
def aperture_time_for_snr(psf_norm, plate_scale_mas, source_rate_total,
                          diffuse_rate_per_pix, dark_rate_per_pix, read_noise,
                          n_reads=1, snr=None, r_aper_mas=None, ee_frac=None,
                          optimize=False):
    """
    Exposure time (s) to reach `snr` for a rendered PSF, per aperture mode.

    Rates are per second (the time dependence is solved for analytically). The
    read-noise variance is incurred n_reads times. Aperture precedence matches
    select_aperture: optimize (fastest radius) > r_aper_mas > ee_frac.

    Returns {'time_s', 'snr', 'r_aper_mas', 'enclosed_fraction', 'n_pix'}.
    """
    if snr is None:
        raise ValueError("snr is required")
    r_mas, enclosed, n_pix = _radial_cumulative(psf_norm, plate_scale_mas)

    A = source_rate_total * enclosed
    B = A + (diffuse_rate_per_pix + dark_rate_per_pix) * n_pix
    C = n_reads * read_noise ** 2 * n_pix
    t = solve_time_for_snr(snr, A, B, C)              # array over radii

    if optimize:
        idx = int(np.argmin(t))                       # radius reaching snr fastest
    elif r_aper_mas is not None:
        idx = int(np.clip(np.searchsorted(r_mas, r_aper_mas, side="right") - 1,
                          0, r_mas.size - 1))
    elif ee_frac is not None:
        idx = int(np.clip(np.searchsorted(enclosed, ee_frac), 0, enclosed.size - 1))
    else:
        raise ValueError("specify optimize, r_aper_mas, or ee_frac.")

    return {"time_s": float(t[idx]),
            "snr": float(snr),
            "r_aper_mas": float(r_mas[idx]),
            "enclosed_fraction": float(enclosed[idx]),
            "n_pix": int(n_pix[idx])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_psfsource.py -k aperture_time_for_snr -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_psfsource.py
git commit -m "Add aperture_time_for_snr PSF-aware inverse"
```

---

## Task 4: `n_reads` as a Simulation parameter

**Files:**
- Modify: `src/wcc_etc/simulation.py:56` (`_mutable_parameters`), `:58-93` (`__init__`)
- Test: `tests/test_exptime.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_exptime.py`:

```python
import numpy as np
import wcc_etc


def _sim(mag=15):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_n_reads_is_mutable_parameter_and_defaults_to_one():
    sim = _sim()
    assert sim.meta.get("n_reads") == 1
    assert any(k.endswith("n_reads") or k == "n_reads" for k in sim.mutable_parameters)
    sim.update(n_reads=4)
    assert sim.meta["n_reads"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_exptime.py::test_n_reads_is_mutable_parameter_and_defaults_to_one -v`
Expected: FAIL (`sim.meta.get("n_reads")` is None / not mutable).

- [ ] **Step 3: Write minimal implementation**

In `src/wcc_etc/simulation.py`, change line 56:

```python
    _mutable_parameters = ["time", "r_aper_mas", "n_reads"]
```

Add `n_reads=1` to the `__init__` signature (line 58-65), after `r_aper_mas=70,`:

```python
    def __init__(self,
                 telescope,
                 sensor,
                 scene=None,
                 time=90,
                 r_aper_mas=70,
                 n_reads=1,
                 meta={}
                 ):
```

(The existing `locals()` capture at lines 88-90 stores `n_reads` in meta automatically.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_exptime.py::test_n_reads_is_mutable_parameter_and_defaults_to_one -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_exptime.py
git commit -m "Add n_reads as an updatable Simulation parameter"
```

---

## Task 5: Analytic path — `n_reads` + `get_exptime_for_snr`

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_signal_and_variance` (~529), `get_snr` (~757); add `_resolve_n_reads`, `_snr_coefficients`, `get_exptime_for_snr`.
- Test: `tests/test_exptime.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_exptime.py`:

```python
import astropy.units as u


def test_get_snr_n_reads_one_matches_baseline():
    sim = _sim()
    baseline = float(sim.get_snr(60).value)
    assert np.isclose(float(sim.get_snr(60, n_reads=1).value), baseline)


def test_more_reads_lowers_snr_at_fixed_time():
    sim = _sim(mag=19)  # faint -> read-noise matters
    s1 = float(sim.get_snr(30, n_reads=1).value)
    s9 = float(sim.get_snr(30, n_reads=9).value)
    assert s9 < s1


def test_exptime_for_snr_roundtrips():
    sim = _sim()
    for target in (20.0, 100.0):
        t = sim.get_exptime_for_snr(target)
        got = float(sim.get_snr(t).value)
        assert np.isclose(got, target, rtol=1e-3)


def test_exptime_for_snr_roundtrips_with_reads():
    sim = _sim(mag=18)
    t = sim.get_exptime_for_snr(50.0, n_reads=5)
    got = float(sim.get_snr(t, n_reads=5).value)
    assert np.isclose(got, 50.0, rtol=1e-3)


def test_exptime_uses_meta_n_reads_when_unset():
    sim = _sim(mag=18)
    sim.update(n_reads=5)
    t_meta = sim.get_exptime_for_snr(50.0)            # resolves n_reads=5 from meta
    t_explicit = sim.get_exptime_for_snr(50.0, n_reads=5)
    assert np.isclose(t_meta.value, t_explicit.value, rtol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_exptime.py -k "n_reads or exptime or reads" -v`
Expected: FAIL (`get_snr() got an unexpected keyword 'n_reads'` / no `get_exptime_for_snr`).

- [ ] **Step 3: Write minimal implementation**

In `src/wcc_etc/simulation.py`, add two helpers in the internal section (near `_parse_bandpass`, ~line 778):

```python
    def _resolve_n_reads(self, n_reads):
        """n_reads from the argument, else meta['n_reads'], else 1."""
        if n_reads is None:
            n_reads = self._meta.get("n_reads", 1)
        return int(n_reads)

    def _snr_coefficients(self, n_reads=None):
        """
        (A, B, C) floats for SNR(t) = A*t / sqrt(B*t + C), electrons & seconds.
        A = source rate; B = total scene rate + n_pix*dark; C = n_pix*N*RN**2.
        """
        n_reads = self._resolve_n_reads(n_reads)
        count_rates = self.get_countrates(units="e/s", as_dict=True)
        A = count_rates["source"].to(u.electron / u.s).value
        all_rates = float(np.nansum([r.to(u.electron / u.s).value
                                     for r in count_rates.values()]))
        n_pix = self.psf_profile["num_psf_pixels"]
        n_pix = n_pix.value if isinstance(n_pix, u.Quantity) else float(n_pix)
        dark = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        rn = self.sensor.read_noise.to(u.electron / u.pix).value
        B = all_rates + n_pix * dark
        C = n_pix * n_reads * rn ** 2
        return A, B, C
```

Modify `get_signal_and_variance` — change the signature to add `n_reads=None` and resolve it, and multiply the read-noise term by `n_reads`. Replace line 529 and the `detector_variance` line (570):

```python
    def get_signal_and_variance(self, time=None, units="e-", n_reads=None):
```
```python
        n_reads = self._resolve_n_reads(n_reads)
        detector_variance = (dark_signal * u.electron/u.pixel
                             + n_reads * self.sensor.read_noise**2) * self.psf_profile["num_psf_pixels"]
```
(Put the `n_reads = self._resolve_n_reads(n_reads)` line right after the `time` is coerced to a Quantity, before `count_rates` is fetched.)

Modify `get_snr` (line 757):

```python
    def get_snr(self, time=None, n_reads=None):
        signal, variance = self.get_signal_and_variance(time, n_reads=n_reads)
        return signal / np.sqrt(variance)
```

Add `get_exptime_for_snr` after `get_snr`:

```python
    def get_exptime_for_snr(self, snr, n_reads=None):
        """
        Exposure time (seconds, Quantity) to reach a target SNR on the analytic
        (Airy) path. Inverse of get_snr. Returns inf*u.s if the source rate is 0.
        """
        from .psfsim import solve_time_for_snr
        A, B, C = self._snr_coefficients(n_reads=n_reads)
        return solve_time_for_snr(snr, A, B, C) * u.second
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_exptime.py -q`
Expected: all pass.

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m pytest -q`
Expected: all pass (no change to existing counts beyond the new tests).

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_exptime.py
git commit -m "Add n_reads and get_exptime_for_snr on the analytic SNR path"
```

---

## Task 6: PSF-aware path — `n_reads` + `get_image_exptime_for_snr`

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_image_snr` (~669-755); add `get_image_exptime_for_snr`.
- Test: `tests/test_image_exptime.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_exptime.py`:

```python
import numpy as np
import wcc_etc


def _sim(mag=15):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_image_snr_n_reads_one_matches_baseline():
    sim = _sim()
    base = sim.get_image_snr(time=60)["snr"]
    assert np.isclose(sim.get_image_snr(time=60, n_reads=1)["snr"], base)


def test_image_exptime_for_snr_roundtrips_fixed_aperture():
    sim = _sim()
    target = 80.0
    res = sim.get_image_exptime_for_snr(target, r_aper_mas=70)
    t = res["time_s"]
    got = sim.get_image_snr(time=t, r_aper_mas=70)["snr"]
    assert np.isclose(got, target, rtol=2e-3)


def test_image_exptime_matches_analytic_for_infocus_default_aperture():
    # in-focus Airy + default aperture: PSF-aware inverse ~ analytic inverse
    sim = _sim()
    t_img = sim.get_image_exptime_for_snr(50.0)["time_s"]
    t_ana = sim.get_exptime_for_snr(50.0).value
    assert np.isclose(t_img, t_ana, rtol=0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_exptime.py -v`
Expected: FAIL (`get_image_snr() got an unexpected keyword 'n_reads'` / no `get_image_exptime_for_snr`).

- [ ] **Step 3: Write minimal implementation**

In `get_image_snr`, add `n_reads=None` to the signature (line 669-670) and, where `read_noise` is computed (line 739), scale by `sqrt(n_reads)`:

```python
    def get_image_snr(self, time=None, psf=None, r_aper_mas=None, ee_frac=None,
                      optimize=False, jitter_sigma_mas=None, n_reads=None,
                      npix=128, oversample=11):
```
```python
        n_reads = self._resolve_n_reads(n_reads)
        read_noise = self.sensor.read_noise.to(u.electron / u.pix).value * np.sqrt(n_reads)
```
(Insert the `n_reads = self._resolve_n_reads(n_reads)` line right after `time` is coerced to a Quantity.)

Add `get_image_exptime_for_snr` after `get_image_snr`:

```python
    def get_image_exptime_for_snr(self, snr, psf=None, r_aper_mas=None,
                                  ee_frac=None, optimize=False,
                                  jitter_sigma_mas=None, n_reads=None,
                                  npix=128, oversample=11):
        """
        Exposure time (s) to reach a target SNR on the PSF-aware path.

        Inverse of get_image_snr. Aperture precedence: optimize > r_aper_mas >
        ee_frac; if none is given the Simulation's r_aper_mas is used. Returns a
        dict {'time_s', 'snr', 'r_aper_mas', 'enclosed_fraction', 'n_pix'}.
        """
        from .psfsim import ImageSimulator, AiryPSF, aperture_time_for_snr

        n_reads = self._resolve_n_reads(n_reads)
        if psf is None:
            psf = AiryPSF()

        imsim = ImageSimulator(self, npix=npix, oversample=oversample)
        ctx = imsim._context(jitter_sigma_mas=jitter_sigma_mas)
        psf_norm = psf.render(ctx)

        profile = self.psf_profile
        ee_at_aper = profile["ee_at_aper"]
        if ee_at_aper == 0:
            raise ValueError("ee_at_aper is zero; aperture radius is degenerate.")
        num_psf_pixels = profile["num_psf_pixels"]
        n_psf = num_psf_pixels.value if isinstance(num_psf_pixels, u.Quantity) else num_psf_pixels

        count_rates = self.get_countrates(units="e/s", as_dict=True)
        source_rate_total = (count_rates["source"] / ee_at_aper).to(u.electron / u.s).value
        diffuse_rate_per_pix = 0.0
        for name, rate in count_rates.items():
            if name == "source":
                continue
            diffuse_rate_per_pix += (rate / n_psf).to(u.electron / u.s).value
        dark_rate_per_pix = self.sensor.dark_current.to(u.electron / (u.s * u.pix)).value
        read_noise = self.sensor.read_noise.to(u.electron / u.pix).value

        if not optimize and r_aper_mas is None and ee_frac is None:
            r_aper_mas = self._meta.get("r_aper_mas")

        return aperture_time_for_snr(psf_norm, ctx.plate_scale_mas,
                                     source_rate_total, diffuse_rate_per_pix,
                                     dark_rate_per_pix, read_noise,
                                     n_reads=n_reads, snr=snr,
                                     r_aper_mas=r_aper_mas, ee_frac=ee_frac,
                                     optimize=optimize)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_exptime.py -q`
Expected: all pass.

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_image_exptime.py
git commit -m "Add n_reads and get_image_exptime_for_snr on the PSF-aware path"
```

---

## Task 7: Per-frame saturation with `n_reads`

**Files:**
- Modify: `src/wcc_etc/simulation.py` — `get_peak_pixel` (~589-650), `is_saturated` (~652-667)
- Test: `tests/test_saturation_reads.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_saturation_reads.py`:

```python
import numpy as np
import wcc_etc


def _sim(mag):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_peak_pixel_n_reads_one_matches_baseline():
    sim = _sim(12)
    base = sim.get_peak_pixel(60, units="e-").value
    assert np.isclose(sim.get_peak_pixel(60, units="e-", n_reads=1).value, base)


def test_peak_pixel_scales_inverse_with_reads_above_bias():
    # in electrons (no bias): per-frame charge halves when n_reads doubles
    sim = _sim(12)
    p1 = sim.get_peak_pixel(60, units="e-", n_reads=1).value
    p2 = sim.get_peak_pixel(60, units="e-", n_reads=2).value
    assert np.isclose(p2, p1 / 2.0, rtol=1e-6)


def test_more_reads_can_unsaturate_a_bright_star():
    # pick a magnitude that saturates in one read at this time
    sim = _sim(6)
    assert sim.is_saturated(60, n_reads=1)            # one long frame clips
    assert not sim.is_saturated(60, n_reads=100)      # split -> per-frame under full well
```

> If `test_more_reads_can_unsaturate_a_bright_star` does not saturate at N=1, lower the magnitude (brighter) until `is_saturated(60, n_reads=1)` is True, then keep N=100 for the unsaturated case. Adjust the magnitude in `_sim(...)` accordingly and note the chosen value.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_saturation_reads.py -v`
Expected: FAIL (`get_peak_pixel() got an unexpected keyword 'n_reads'`).

- [ ] **Step 3: Write minimal implementation**

In `get_peak_pixel`, add `n_reads=None` to the signature (line 589) and divide the integration time by `n_reads` for the per-frame charge. After `time` is coerced to a Quantity (line 614), insert:

```python
        n_reads = self._resolve_n_reads(n_reads)
        time = time / n_reads          # per-frame integration; saturation is per-frame
```

Signature becomes:

```python
    def get_peak_pixel(self, time=None, units="adu", n_reads=None):
```

The rest of `get_peak_pixel` (source_peak / bkg_peak / dark_peak all use `time`, bias added once for ADU) is unchanged — they now use the per-frame time.

In `is_saturated`, thread `n_reads` (line 652):

```python
    def is_saturated(self, time=None, n_reads=None):
        peak_adu = self.get_peak_pixel(time, units="adu", n_reads=n_reads)
        return peak_adu >= self.sensor.adc_max
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_saturation_reads.py -q`
Expected: all pass. (If the bright-star test needs a brighter magnitude, adjust per the Step 1 note and re-run.)

- [ ] **Step 5: Run the full suite for regressions**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_saturation_reads.py
git commit -m "Make get_peak_pixel/is_saturated per-frame via n_reads"
```

---

## Task 8: Final verification

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: all green, no new warnings beyond the pre-existing deprecation/airy ones.

- [ ] **Step 2: Sanity demo (manual, not committed)**

Run:
```bash
python -c "
import warnings; warnings.simplefilter('ignore')
import wcc_etc
scene = wcc_etc.get_scene(name='G5V', mag=15, background='zodi', bandpass='johnson_r', background_prop={'bandpass':'johnson_r','mag':22.5})
sim = wcc_etc.Simulation.from_sensor_and_scene('sony:r', scene)
t = sim.get_exptime_for_snr(100)
print('t for SNR=100:', t, '-> SNR(t)=', sim.get_snr(t))
print('PSF-aware t:', sim.get_image_exptime_for_snr(100)['time_s'])
sim.update(n_reads=4)
print('with 4 reads, t for SNR=100:', sim.get_exptime_for_snr(100))
"
```
Expected: SNR(t) ≈ 100; PSF-aware t close to analytic; 4-read time longer than 1-read.

---

## Self-review notes (author)

- **Spec coverage:** reads model (Task 5/6 RN×N, Task 7 per-frame `t/N`); shared math + helper (Task 1); analytic inverse (Task 5); PSF-aware inverse incl. optimize-min (Task 3+6); `n_reads` updatable param (Task 4); saturation (Task 7); all spec tests mapped (round-trip, N=1 parity, monotonicity, √N via Task 7 1/N + Task 5 reads, update, cross-check, edge zero-source in Task 1).
- **Naming consistency:** `solve_time_for_snr`, `_radial_cumulative`, `aperture_time_for_snr`, `_resolve_n_reads`, `_snr_coefficients`, `get_exptime_for_snr`, `get_image_exptime_for_snr` used identically across tasks.
- **√N scaling test:** covered implicitly (read-noise-dominated regime) by `test_more_reads_lowers_snr_at_fixed_time` and the per-frame 1/N test; an explicit `t ∝ √N` assertion can be added if desired but is not required for correctness.
