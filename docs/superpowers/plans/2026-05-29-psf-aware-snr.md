# PSF-aware SNR (`get_image_snr`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Simulation.get_image_snr(...)` — a PSF-aware aperture SNR (Airy/defocus/custom) that reproduces the existing `get_snr` for the in-focus case, with fixed/EE-fraction/optimized aperture modes.

**Architecture:** A pure, vectorized aperture-SNR core in `psfsim.py` (radial cumulative profile → SNR vs radius + aperture selector), plus a thin `Simulation.get_image_snr` that renders the PSF (reusing `ImageSimulator`), feeds ETC electron rates into the core, and returns a breakdown dict. `get_snr` is untouched.

**Tech Stack:** Python, numpy, astropy.units. Tests with pytest.

---

## File Structure

- `src/wcc_etc/psfsim.py` — add two pure functions: `aperture_snr_radial(...)` and `select_aperture(...)`.
- `src/wcc_etc/simulation.py` — add `Simulation.get_image_snr(...)` (lazy-imports from `psfsim` to avoid the circular import, since `psfsim` imports `Simulation`).
- `tests/test_image_simulator.py` — unit tests for the two pure helpers.
- `tests/test_simulation.py` — integration + cross-check tests for `get_image_snr` (reuses the existing `_bright_sim(mag, sensor)` helper there).

**Facts the implementer needs (verified against the codebase):**
- `psfsim.py` imports `numpy as np`. `ImageSimulator(simulation, npix=300, oversample=11)`, `ImageSimulator._context(jitter_sigma_mas=None, center=None)` → `DetectorPSFContext` with `.plate_scale_mas`, `.npix`, etc. `AiryPSF`, `DefocusPSF`, `DEFOCUS_2WAVE_PATH` are defined there. A PSF's `.render(ctx)` returns an `(npix, npix)` array summing to 1.
- `simulation.py` imports `numpy as np` and `astropy.units as u`. `Simulation` already uses lazy imports inside methods (`from .airy import ...`, `from .io import ...`).
- `Simulation.get_countrates(units="e/s", as_dict=True)` → dict in electron/s within the aperture; always `"source"`, plus `"background"`/`"host"` if present.
- `Simulation.psf_profile` → `ee_at_aper` (float), `num_psf_pixels` (Quantity; `.value` = pixel count).
- `Sensor.dark_current` (electron/(s·pix)), `read_noise` (electron/pix). `self._meta["r_aper_mas"]` is the default aperture radius in mas.
- `tests/test_simulation.py` already defines `_bright_sim(mag=20, sensor="sony:r")` and imports `wcc_etc`, `numpy as np`, `astropy.units as u`, `pytest`.

**Commit hygiene:** plain messages, NO Co-Authored-By/Claude trailer. Stage only the files named in each task.

---

## Task 1: Pure aperture-SNR core

**Files:**
- Modify: `src/wcc_etc/psfsim.py`
- Test: `tests/test_image_simulator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_image_simulator.py`:

```python
from wcc_etc.psfsim import aperture_snr_radial, select_aperture


def _point_psf(npix=21):
    a = np.zeros((npix, npix)); a[npix // 2, npix // 2] = 1.0
    return a


def _gaussian_psf(npix=41, sigma=3.0):
    c = (npix - 1) / 2
    yy, xx = np.mgrid[0:npix, 0:npix]
    a = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma ** 2)))
    return a / a.sum()


def test_aperture_snr_radial_enclosed_monotonic_to_one():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    enc = prof["enclosed_fraction"]
    assert np.all(np.diff(enc) >= -1e-12)        # monotonic non-decreasing
    assert enc[-1] == pytest.approx(1.0)
    assert prof["n_pix"][0] == 1
    assert prof["n_pix"][-1] == 21 * 21
    assert np.all(np.diff(prof["r_mas"]) >= 0)   # radius sorted ascending


def test_aperture_snr_point_source_noiseless_is_sqrt_signal():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    # all flux in the central pixel: signal=1e4 at every radius, noise=sqrt(1e4)
    assert np.allclose(prof["snr"], np.sqrt(1e4))


def test_aperture_snr_read_noise_penalizes_large_apertures():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=5.0)
    assert prof["snr"][0] > prof["snr"][-1]      # more noise pixels, no more signal


def test_aperture_snr_noise_consistent_with_signal_and_snr():
    prof = aperture_snr_radial(_gaussian_psf(), plate_scale_mas=10.0,
                               source_e_total=1e5, diffuse_per_pix=2.0,
                               dark_per_pix=1.0, read_noise=5.0)
    assert np.allclose(prof["snr"], prof["signal_e"] / prof["noise_e"])


def test_select_aperture_optimize_picks_max_snr():
    prof = aperture_snr_radial(_gaussian_psf(), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=1.0,
                               dark_per_pix=1.0, read_noise=5.0)
    assert select_aperture(prof, optimize=True) == int(np.argmax(prof["snr"]))


def test_select_aperture_ee_frac():
    prof = aperture_snr_radial(_gaussian_psf(41, 3.0), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    idx = select_aperture(prof, ee_frac=0.9)
    assert prof["enclosed_fraction"][idx] >= 0.9


def test_select_aperture_fixed_radius():
    prof = aperture_snr_radial(_point_psf(41), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    idx = select_aperture(prof, r_aper_mas=35.0)   # 3.5 px at 10 mas/pix
    assert prof["r_mas"][idx] <= 35.0


def test_select_aperture_requires_a_mode():
    prof = aperture_snr_radial(_point_psf(21), plate_scale_mas=10.0,
                               source_e_total=1e4, diffuse_per_pix=0.0,
                               dark_per_pix=0.0, read_noise=0.0)
    with pytest.raises(ValueError):
        select_aperture(prof)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_image_simulator.py -k "aperture_snr or select_aperture" -v`
Expected: FAIL with `ImportError` (functions not defined).

- [ ] **Step 3: Implement the two helpers**

Add to `src/wcc_etc/psfsim.py` (a sensible place is just after the `ImageSimulator` class):

```python
def aperture_snr_radial(psf_norm, plate_scale_mas, source_e_total,
                        diffuse_per_pix, dark_per_pix, read_noise):
    """
    SNR as a function of circular-aperture radius for a rendered PSF.

    The PSF (sum=1) sets how much source light falls inside each radius; the
    per-pixel diffuse (sky+host), dark, and read-noise terms set the background
    noise that grows with the number of aperture pixels.

    Parameters
    ----------
    psf_norm : ndarray
        Normalized (sum=1) PSF on the detector grid.
    plate_scale_mas : float
        Detector plate scale, mas/pixel (to report radii in mas).
    source_e_total : float
        Total source electrons (all of the PSF, before aperture clipping).
    diffuse_per_pix, dark_per_pix : float
        Per-pixel sky+host and dark-current electrons.
    read_noise : float
        Read noise (electrons rms per pixel).

    Returns
    -------
    dict of ndarrays, sorted by ascending radius:
        'r_mas', 'enclosed_fraction', 'n_pix', 'signal_e', 'noise_e', 'snr'.
    """
    npix = psf_norm.shape[0]
    c = (npix - 1) / 2.0
    yy, xx = np.mgrid[0:npix, 0:npix]
    r_pix = np.sqrt((xx - c) ** 2 + (yy - c) ** 2).ravel()
    order = np.argsort(r_pix, kind="stable")

    r_sorted = r_pix[order]
    enclosed = np.cumsum(psf_norm.ravel()[order])           # fraction (psf sums to 1)
    n_pix = np.arange(1, r_sorted.size + 1)

    signal = source_e_total * enclosed
    per_pix_var = diffuse_per_pix + dark_per_pix + read_noise ** 2
    noise = np.sqrt(signal + per_pix_var * n_pix)
    snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)

    return {"r_mas": r_sorted * plate_scale_mas,
            "enclosed_fraction": enclosed,
            "n_pix": n_pix,
            "signal_e": signal,
            "noise_e": noise,
            "snr": snr}


def select_aperture(profile, r_aper_mas=None, ee_frac=None, optimize=False):
    """
    Index into an `aperture_snr_radial` profile for the chosen aperture mode.

    Precedence: optimize (max SNR) > explicit r_aper_mas > ee_frac.
    Raises ValueError if no mode is given.
    """
    if optimize:
        return int(np.argmax(profile["snr"]))
    if r_aper_mas is not None:
        r_mas = profile["r_mas"]
        idx = int(np.searchsorted(r_mas, r_aper_mas, side="right") - 1)
        return int(np.clip(idx, 0, r_mas.size - 1))
    if ee_frac is not None:
        enc = profile["enclosed_fraction"]
        idx = int(np.searchsorted(enc, ee_frac))
        return int(np.clip(idx, 0, enc.size - 1))
    raise ValueError("select_aperture: specify optimize, r_aper_mas, or ee_frac.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_image_simulator.py -k "aperture_snr or select_aperture" -v`
Expected: PASS (8 tests).
Then `pytest -q` (no regressions).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py tests/test_image_simulator.py
git commit -m "Add pure aperture-SNR radial profile and aperture selector"
```

---

## Task 2: Simulation.get_image_snr

**Files:**
- Modify: `src/wcc_etc/simulation.py`
- Test: `tests/test_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simulation.py` (reuses the existing `_bright_sim`, `wcc_etc`, `np`, `u`, `pytest`):

```python
def test_get_image_snr_returns_expected_keys():
    sim = _bright_sim(16)
    out = sim.get_image_snr(time=60)
    assert set(out) >= {"snr", "signal_e", "noise_e", "enclosed_fraction", "r_aper_mas", "n_pix"}
    assert 0 < out["enclosed_fraction"] <= 1
    assert out["n_pix"] >= 1
    assert out["snr"] > 0


def test_get_image_snr_matches_get_snr_in_focus():
    sim = _bright_sim(16)
    for t in [30, 300]:
        for m in [16, 20]:
            sim.update(source__mag=m)
            etc = sim.get_snr(t)
            etc = float(etc.value) if hasattr(etc, "value") else float(etc)
            img = sim.get_image_snr(time=t)["snr"]
            assert img == pytest.approx(etc, rel=0.10)


def test_get_image_snr_ee_frac_aperture():
    sim = _bright_sim(16)
    out = sim.get_image_snr(time=60, ee_frac=0.9)
    assert out["enclosed_fraction"] >= 0.9


def test_get_image_snr_optimize_at_least_default():
    sim = _bright_sim(16)
    base = sim.get_image_snr(time=60)["snr"]
    opt = sim.get_image_snr(time=60, optimize=True)["snr"]
    assert opt >= base - 1e-9


def test_get_image_snr_defocus_lower_at_fixed_aperture():
    sim = _bright_sim(16)
    airy = sim.get_image_snr(time=60, r_aper_mas=70)["snr"]
    defo = sim.get_image_snr(time=60, r_aper_mas=70,
                             psf=wcc_etc.DefocusPSF(wcc_etc.DEFOCUS_2WAVE_PATH))["snr"]
    assert defo < airy


def test_get_image_snr_optimize_defocus_uses_larger_radius():
    sim = _bright_sim(16)
    r_airy = sim.get_image_snr(time=60, optimize=True)["r_aper_mas"]
    r_defo = sim.get_image_snr(time=60, optimize=True,
                               psf=wcc_etc.DefocusPSF(wcc_etc.DEFOCUS_2WAVE_PATH))["r_aper_mas"]
    assert r_defo > r_airy


def test_get_image_snr_runs_on_qcmos():
    sim = _bright_sim(16, sensor="qcmos:r")
    assert sim.get_image_snr(time=60)["snr"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulation.py -k get_image_snr -v`
Expected: FAIL with `AttributeError: 'Simulation' object has no attribute 'get_image_snr'`.

- [ ] **Step 3: Implement**

Add this method to the `Simulation` class in `src/wcc_etc/simulation.py`, immediately after `is_saturated`:

```python
    def get_image_snr(self, time=None, psf=None, r_aper_mas=None, ee_frac=None,
                      optimize=False, jitter_sigma_mas=None, npix=128, oversample=11):
        """
        PSF-aware aperture signal-to-noise ratio.

        Unlike get_snr (which assumes the analytic Airy disk), this renders the
        given PSF (default AiryPSF) on the detector grid and computes the SNR for
        a circular aperture. The in-focus default-aperture case reproduces
        get_snr (cross-check). Aperture precedence: optimize > r_aper_mas >
        ee_frac; if none is given, the Simulation's r_aper_mas is used.

        Parameters
        ----------
        time : float or Quantity, optional
            Exposure time (seconds if a bare float). Defaults to meta['time'].
        psf : PSFSource, optional
            PSF model; defaults to AiryPSF().
        r_aper_mas : float, optional
            Fixed aperture radius (mas).
        ee_frac : float, optional
            Aperture enclosing this fraction of the PSF.
        optimize : bool, optional
            If True, use the radius that maximizes SNR.
        jitter_sigma_mas : float, optional
            Override the telescope jitter (mas).
        npix, oversample : int, optional
            Render grid size and oversampling.

        Returns
        -------
        dict
            'snr', 'signal_e', 'noise_e', 'enclosed_fraction', 'r_aper_mas', 'n_pix'.
        """
        from .psfsim import ImageSimulator, AiryPSF, aperture_snr_radial, select_aperture

        if time is None:
            time = self._meta.get("time", None)
        if time is None:
            raise ValueError("no time given, none set to meta")
        if not isinstance(time, u.Quantity):
            time = time * u.second

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
        source_e_total = (count_rates["source"] / ee_at_aper * time).to(u.electron).value

        # per-pixel diffuse electrons from all non-source elements (sky, host)
        diffuse_per_pix = 0.0
        for element_name, rate in count_rates.items():
            if element_name == "source":
                continue
            diffuse_per_pix += (rate * time / n_psf).to(u.electron).value

        dark_per_pix = (self.sensor.dark_current * time).to(u.electron / u.pix).value
        read_noise = self.sensor.read_noise.to(u.electron / u.pix).value

        prof = aperture_snr_radial(psf_norm, ctx.plate_scale_mas, source_e_total,
                                   diffuse_per_pix, dark_per_pix, read_noise)

        # default to the ETC aperture if no mode was requested
        if not optimize and r_aper_mas is None and ee_frac is None:
            r_aper_mas = self._meta.get("r_aper_mas")

        idx = select_aperture(prof, r_aper_mas=r_aper_mas, ee_frac=ee_frac, optimize=optimize)

        return {"snr": float(prof["snr"][idx]),
                "signal_e": float(prof["signal_e"][idx]),
                "noise_e": float(prof["noise_e"][idx]),
                "enclosed_fraction": float(prof["enclosed_fraction"][idx]),
                "r_aper_mas": float(prof["r_mas"][idx]),
                "n_pix": int(prof["n_pix"][idx])}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulation.py -k get_image_snr -v`
Expected: PASS (7 tests). In particular `test_get_image_snr_matches_get_snr_in_focus` must agree within 10%.

If the cross-check fails: do NOT loosen the tolerance blindly. First print `sim.get_image_snr(time=t)` and `sim.get_snr(t)` and compare `enclosed_fraction` vs `ee_at_aper` and `n_pix` vs `num_psf_pixels`. If they are close (within ~15%) and the SNR gap is just over 10%, report the measured numbers — a slightly looser tolerance (rel 0.15) is acceptable; note what you chose and why. If they are far apart, there is a real bug to report (BLOCKED).

Then run `pytest -q` (full suite, no regressions).

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/test_simulation.py
git commit -m "Add Simulation.get_image_snr (PSF-aware aperture SNR)"
```

---

## Self-Review notes

- **Spec coverage:** API + dict return (Task 2); default-aperture/r_aper_mas/ee_frac/optimize precedence (Task 1 `select_aperture` + Task 2 default-fill); cross-check vs `get_snr` (Task 2); defocus lower-at-fixed-aperture + optimize-larger-radius (Task 2); host in noise via the non-source loop (Task 2); lazy import for the circular dependency (Task 2); fast defaults `npix=128` (Task 2). `get_snr` untouched. Aperture optimization included (the hook you asked for). Caching deferred (out of scope).
- **Placeholder scan:** none — full code in every step.
- **Type consistency:** `aperture_snr_radial` returns the dict keys consumed by `select_aperture` and by `get_image_snr`'s return; electron quantities resolve to floats via `.value`; `ctx.plate_scale_mas` matches `DetectorPSFContext`. The non-source diffuse loop mirrors `get_peak_pixel`'s background handling, extended to all non-source elements for the noise term (per spec).
- **Cross-check rationale (for the executor):** at the default aperture, `enclosed_fraction ≈ ee_at_aper` and `n_pix ≈ num_psf_pixels`, so `signal ≈ count_rates["source"]·t` and `noise² ≈ (source+sky)·t + (dark+read²)·num_psf_pixels` — the same terms `get_signal_and_variance` sums. Agreement is limited only by 2-D pixelated render vs the 1-D Airy EE curve.
