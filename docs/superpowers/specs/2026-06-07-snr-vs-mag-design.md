# Fast magnitude sweep on `get_image_snr` (`mags=`)

**Date:** 2026-06-07
**Status:** Approved (brainstorm)

## Problem

A magnitude sweep — SNR as a function of source magnitude at fixed exposure
time — is a common ETC use. The natural way to do it today rebuilds the
`Simulation` (or calls `set_scene`) once per magnitude, which clears
`_image_render_bundle_cache` and re-renders the PSF every iteration. The PSF
render is the expensive step (~90 ms each), so a 200-point sweep pays ~200
renders for no reason.

The render is **magnitude-independent**. Within the cached render bundle, the
only magnitude-dependent quantity is `source_rate_total`, and it scales by the
exact closed form `10**(-0.4 * Δmag)`. So a magnitude sweep can render once and
rescale the source term per magnitude — fast and exact (verified to ≤ 2e-15
relative against a per-magnitude rebuild in a prototype).

## Goals

1. `get_image_snr` accepts an optional `mags=` argument that sweeps the
   **source** magnitude at fixed exposure time, reusing a single PSF render.
2. The result is exact — equal to a per-magnitude `from_sensorfilter` rebuild to
   ~1e-12 relative.
3. Existing call patterns (`mags=None`) are completely untouched.
4. Surgical, minimal edits — ride on the existing cache and aperture helpers.

## Non-goals

- No `mags=` on `get_image_exptime_for_snr` in this change (same refactor is
  possible later; out of scope now).
- No 2-D sweep: `time` and `mags` may not both be arrays (see below).
- No new SNR physics, no redshift/extinction, no host/background variation —
  `mags` varies the point source only; host (if any), zodi/background, dark, and
  read noise are held fixed. This is exactly the regime where the source-flux
  scaling is exact.

## Design

### 1. Signature and behavior

Add one optional kwarg to `get_image_snr`:

```python
def get_image_snr(self, time=None, mags=None, psf=None, r_aper_mas=None,
                  ee_frac=None, optimize=False, jitter_sigma_mas=None,
                  n_reads=None, npix=128, oversample=11):
```

`mags` is `None` (default), a scalar, or a 1-D array of source magnitudes.
Behavior by argument shape:

| `time` | `mags` | result |
|---|---|---|
| scalar | `None` | scalar dict *(unchanged)* |
| array  | `None` | dict of 1-D arrays over time *(unchanged)* |
| scalar | scalar | scalar dict at that magnitude |
| scalar | array  | dict of 1-D arrays over magnitude **(new)** |
| array  | array  | `ValueError` — at most one axis may be an array |

`mags` sweeps the source brightness relative to its set magnitude
`m0 = self.scene.source.mag.value`. Swept values are interpreted in the source's
own magnitude system (the scaling depends only on Δmag, so the system cancels).

### 2. Mechanism

The render bundle is fetched **once** and reused for every magnitude. The only
change to the SNR math is a multiplicative source scale.

Refactor the existing inner helper in `get_image_snr` to take a source scale
(default `1.0`, so every current caller is unaffected):

```python
def _snr_at(t_sec, source_scale=1.0):
    source_e_total = b["source_rate_total"] * source_scale * t_sec
    diffuse_per_pix = b["diffuse_rate_per_pix"] * t_sec      # unchanged
    dark_per_pix = dark_rate_per_pix * t_sec                 # unchanged
    prof = aperture_snr_radial(b["psf_norm"], b["plate_scale_mas"],
                               source_e_total, diffuse_per_pix, dark_per_pix, read_noise)
    idx = select_aperture(prof, r_aper_mas=r_aper_mas, ee_frac=ee_frac, optimize=optimize)
    return { ... }   # unchanged dict assembly
```

The `b = self._image_render_bundle(psf, jitter_sigma_mas, npix, oversample)`
call (the expensive render) happens once, before any sweep loop — that is the
whole speedup.

`optimize` / `r_aper_mas` / `ee_frac` keep working per magnitude:
`aperture_snr_radial` + `select_aperture` re-run cheaply on the cached
`psf_norm`, so each magnitude gets its own correct aperture (and under
`optimize`, the magnitude-dependent optimum).

### 3. Control flow in `get_image_snr`

After the existing `time` normalization and the (unchanged) `b = ...` /
`read_noise` / `dark_rate_per_pix` / default-aperture setup, the magnitude logic
reduces to computing a **source scale** and threading it into the existing
scalar/array-time path. Only the mags-array case (which forces scalar time) is a
genuinely new branch:

```python
# --- resolve the source-flux scale from mags ---
if mags is None:
    source_scale = 1.0                       # existing behavior, unchanged
else:
    if not self.scene.has_source():
        raise ValueError("mags sweep requires a scene with a source")
    m0 = self.scene.source.mag.value
    if np.ndim(mags) > 0:                     # mags is an array
        if not time.isscalar:
            raise ValueError("time and mags cannot both be arrays; "
                             "sweep one axis at a time")
        scales = 10 ** (-0.4 * (np.asarray(mags, dtype=float) - m0))
        t_sec = time.to(u.second).value       # scalar (guarded just above)
        return _assemble_array([_snr_at(t_sec, s) for s in scales])
    source_scale = 10 ** (-0.4 * (float(mags) - m0))   # scalar mags

# --- existing scalar / array-time path, now threading source_scale ---
if time.isscalar:
    return _snr_at(time.to(u.second).value, source_scale)
results = [_snr_at(t, source_scale) for t in time.to(u.second).value]
return _assemble_array(results)
```

This correctly covers all five rows of the behavior table:
- `mags=None`, scalar/array `time` → `source_scale=1.0`, existing path (the only
  change is the now-defaulted `source_scale` argument, a no-op).
- scalar `mags`, scalar `time` → scalar dict at that magnitude.
- scalar `mags`, array `time` → arrays over time at fixed magnitude (loops times,
  threads the single `source_scale`).
- array `mags`, scalar `time` → arrays over magnitude (the new branch).
- array `mags`, array `time` → `ValueError`.

`_assemble_array(results)` is the **same** dict-of-arrays assembly already used
for the array-time path (factor it out from the current inline code so both the
time-array and mags-array branches share it): `snr`, `signal_e`, `noise_e`,
`enclosed_fraction`, `r_aper_mas` as float arrays; `n_pix` as an int array;
length equal to the swept axis.

Edge cases:
- Empty `mags` array → dict of empty arrays (matches empty-time behavior).
- `mags=[m0]` (length-1) → length-1 arrays equal to the scalar result at `m0`.
- `mags=m0` (scalar) → identical to the no-`mags` scalar call.

### 4. Files touched

- `src/wcc_etc/simulation.py` — add `mags=` to `get_image_snr`: the
  `source_scale` parameter on the inner `_snr_at`, the validation guards, the
  reference-magnitude + scale computation, and the mags-array branch. No change
  to `_image_render_bundle`, `get_image_exptime_for_snr`, or any other method.
- `tests/` — new tests (below), in the existing image-SNR test module
  (`tests/test_simulation.py` or wherever `get_image_snr` is currently tested).
- `notebooks_scratch/20260607_from_sensorfilter_cache.ipynb` — add a section
  demonstrating `mags=` (rebuild-vs-`mags=` overlay proving they match, plus a
  one-render timing note). *(Scratch notebook — not part of the package; the
  canonical example may also be promoted into `notebooks/` later.)*
- `project-status` memory — update API summary with the new `mags=` argument.

### 5. New test coverage

1. **Parity (key correctness test):** `get_image_snr(time=t, mags=M)` equals a
   per-magnitude `from_sensorfilter` rebuild loop (`get_image_snr(time=t)` on a
   sim built at each `m`) to `rtol <= 1e-12`, for an array `M` spanning a wide
   magnitude range.
2. **Scalar consistency:** `mags=m0` reproduces the no-`mags` scalar result;
   `mags=[m0]` returns a length-1 array of the same value.
3. **Mutual-exclusion guard:** `time` array **and** `mags` array raises
   `ValueError`.
4. **No-source guard:** `mags=` on a scene with no source raises `ValueError`.
5. **`optimize=True` under sweep:** runs per magnitude; the selected aperture
   radius is non-increasing as the source faints (sanity that the optimum tracks
   magnitude).
6. **Return shape/dtype:** all expected keys present, array values have length
   `len(mags)`, `n_pix` is integer dtype.

### 6. Validation

The prototype (render once, rescale `source_rate_total` by `10**(-0.4 Δmag)`,
re-run `aperture_snr_radial` + `select_aperture`) reproduced a per-magnitude
rebuild at mags 10/15/20/25 to relative difference ≤ 2e-15. The parity test
codifies this at `rtol <= 1e-12`.
