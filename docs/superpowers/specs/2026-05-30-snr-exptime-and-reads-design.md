# SNR ↔ exposure-time inverse, with coadded reads

**Date:** 2026-05-30
**Status:** Design approved, pending spec review

## Problem

The ETC can compute SNR for a given exposure time (`Simulation.get_snr`,
`get_image_snr`) but cannot answer the inverse, "how long to reach a target
SNR?", which is the more common observation-planning question. There is also no
way to model **coadded reads**: stacking N frames incurs the read noise N times,
which matters in the read-noise-dominated (short-exposure / faint) regime.

## Goal

1. **Time-for-SNR inverse** on both SNR paths (analytic `get_snr` and PSF-aware
   `get_image_snr`).
2. A **`n_reads`** parameter (coadded frames; default 1) on all SNR methods and
   the inverses, stored as a settable `Simulation` parameter updatable via
   `update(n_reads=N)`. `n_reads=1` reproduces current behavior exactly.

## Reads model (agreed)

`n_reads = N` coadds N frames spanning the **total** integration time `t`, i.e.
each frame integrates `t/N`. Signal, Poisson, and dark accumulate over the full
`t`; read noise is incurred once per frame:

```
variance = R_scene·t  +  n_pix·(R_dark·t  +  N·RN²)
```

(Equivalently, effective read noise `RN·√N`.)

Saturation, by contrast, is a **per-frame** phenomenon: the ADC clips each
individual readout at `adc_max` and the coadd sums the already-clipped frames,
so only the charge in a single `t/N` frame matters. Consequently more reads →
shorter frames → less saturation (the reason observers split a long integration
into coadds to keep a bright star under full well).

## Shared math

Every case has the form `SNR(t) = A·t / √(B·t + C)`:

| coeff | meaning |
|-------|---------|
| `A` | source electron rate (signal slope), e⁻/s |
| `B` | variance terms ∝ t: total scene rate + `n_pix·R_dark`, e⁻/s |
| `C` | constant read-noise variance: `n_pix · N · RN²`, e⁻² |

Inverse — solve `S²(B·t + C) = A²t²` for the positive root:

```
t = [ S²·B + √( S⁴·B² + 4·A²·S²·C ) ] / (2·A²)
```

A module-level helper `_solve_time_for_snr(snr, A, B, C)` returns `t` (seconds).
If `A <= 0` (no source flux) it returns `inf`. `C >= 0` guarantees a single
positive root.

## Implementation

### Simulation parameter
- Add `n_reads=1` to `Simulation.__init__` (auto-stored in meta via the existing
  `locals()` capture).
- Add `"n_reads"` to `Simulation._mutable_parameters` so `sim.update(n_reads=N)`
  works and is rejected/validated like `time`/`r_aper_mas`.
- Methods take `n_reads=None`; when None they resolve `self._meta.get("n_reads", 1)`
  (mirrors how `time` falls back to `meta['time']`).

### Analytic path
- `get_signal_and_variance(time, units="e-", n_reads=None)` — the detector
  read-noise term becomes `n_pix · N · RN²`. Everything else unchanged.
- `get_snr(time=None, n_reads=None)` — unchanged formula, threads `n_reads`.
- **New** `get_exptime_for_snr(snr, n_reads=None)` → `t` (Quantity, seconds), via
  the helper with `A = R_src`, `B = R_scene_total + n_pix·R_dark`,
  `C = n_pix·N·RN²`.

### PSF-aware path
- `get_image_snr(..., n_reads=None)` — pass effective read noise `RN·√N` into
  `aperture_snr_radial` (it squares the value); one-line change.
- **New** `get_image_exptime_for_snr(snr, psf=None, r_aper_mas=None, ee_frac=None,
  optimize=False, jitter_sigma_mas=None, n_reads=None, npix=128, oversample=11)`:
  - Render the PSF once (geometry is time-independent) and build per-radius
    `(A_r, B_r, C_r)` from the radial profile (cumulative source fraction,
    per-pixel diffuse/dark rates, pixel count, `RN`).
  - Fixed aperture (`r_aper_mas` / `ee_frac` / the Simulation default): solve at
    that radius.
  - `optimize=True`: solve at every radius and return the **minimum** `t` (the
    radius that reaches the target fastest). Still closed-form — no iteration.
  - Returns `{'time_s', 'snr', 'r_aper_mas', 'enclosed_fraction', 'n_pix'}`.

### Saturation (per-frame)
Saturation is evaluated on a single `t/N` frame, not the coadded total:

```
peak_e_per_frame   = (R_src_peak + R_bkg/pix + R_dark) · (t / N)
peak_adu_per_frame = peak_e_per_frame / gain + bias       # bias is per readout
is_saturated       = peak_adu_per_frame >= adc_max
```

- `get_peak_pixel(time=None, units="adu", n_reads=None)` — scale the source /
  background / dark electron terms by `t/N`; bias and `adc_max` are per-frame
  (unchanged). `N=1` is identical to today.
- `is_saturated(time=None, n_reads=None)` — per-frame peak vs `adc_max`.

Saturation stays a separate, explicit check; the SNR / exptime methods do **not**
auto-flag saturation.

### Naming
`get_exptime_for_snr` and `get_image_exptime_for_snr`.

## Testing

- **Round-trip (both paths):** `get_snr(get_exptime_for_snr(S)) ≈ S`; PSF-aware
  `get_image_snr(t)` at the returned fixed-aperture `t` ≈ target S.
- **`n_reads=1` parity:** `get_snr`/`get_image_snr` identical to current behavior
  (regression); `get_signal_and_variance` read term unchanged at N=1.
- **Reads monotonicity:** larger `n_reads` → longer `t` for a fixed target, and
  lower SNR for a fixed `t`.
- **Read-noise-dominated scaling:** in the RN-dominated limit, required `t ∝ √N`.
- **`update(n_reads=N)`:** changes `get_snr`/`get_exptime_for_snr` results; and
  `n_reads` appears in `Simulation.mutable_parameters`.
- **Cross-check:** in-focus Airy, default aperture — analytic inverse ≈ PSF-aware
  inverse (~1%, consistent with the existing forward cross-check).
- **Saturation per-frame:** a peak that saturates at `N=1` becomes unsaturated at
  large `N` (per-frame charge drops ∝ 1/N); `get_peak_pixel` scales ∝ 1/N above
  bias; `N=1` reproduces current `get_peak_pixel`/`is_saturated` exactly.
- **Edge:** zero source rate → `inf`.

## Out of scope (YAGNI)

Per-read time model (we adopted total-time coadds), Fowler/non-destructive
read-noise reduction, auto-flagging saturation inside the SNR/exptime methods,
and dithering between frames.
