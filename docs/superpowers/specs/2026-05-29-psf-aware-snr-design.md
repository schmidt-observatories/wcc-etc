# PSF-aware SNR (`get_image_snr`) — Design

**Date:** 2026-05-29
**Branch:** `psf-aware-snr` (off `psfs`)
**Status:** Approved (pending spec review)

## Goal

Add a PSF-aware signal-to-noise method to the ETC `Simulation` so SNR can be
evaluated for **any** PSF (in-focus Airy, defocus, custom), not just the
analytic Airy disk that `get_snr` assumes. The in-focus case must reproduce the
existing `get_snr`, providing a cross-check between the two paths.

## Background / motivation

`Simulation.get_snr` / `get_signal_and_variance` are **Airy-only**:
`compute_psf_profile` derives `ee_at_aper` from `get_airy_and_ee_curve` (analytic
Airy + jitter), and `num_psf_pixels` is the purely geometric pixel count of the
`r_aper_mas` aperture. Passing a defocus/custom PSF changes nothing in `get_snr`,
so for an extended PSF the current SNR is optimistic (it credits the source with
the compact Airy aperture EE). The new `ImageSimulator`/PSF-source machinery
(branch `psfs`) can render any PSF on the detector grid, which lets us compute a
PSF-correct aperture SNR.

This feature does **not** modify `get_snr`; it adds a parallel method.

## Decisions (from brainstorming)

1. **Scope:** new method only; `get_snr`/`get_signal_and_variance` unchanged.
2. **Aperture:** default to the Simulation's `r_aper_mas` (same aperture as
   `get_snr`, enabling the cross-check); override by explicit radius (`r_aper_mas`),
   by enclosed-energy fraction (`ee_frac`), or by **optimization** (`optimize=True`,
   the radius that maximizes SNR).
3. **Return:** a dict — `snr` plus a breakdown.
4. **Placement:** method on `Simulation`; reuse `ImageSimulator` for PSF rendering
   via a lazy import (avoids the circular import, since `psfsim` imports
   `Simulation`).
5. **Performance:** fast by design — one PSF render per call (tens of ms; defocus
   ~1 ms), aperture selection is vectorized over radii (negligible). Defaults
   chosen for speed.

## API

```python
Simulation.get_image_snr(
    time=None,             # exposure (s if bare float); default meta['time']
    psf=None,              # PSFSource; default AiryPSF()
    r_aper_mas=None,       # explicit aperture radius (mas)
    ee_frac=None,          # OR aperture enclosing this fraction of the PSF
    optimize=False,        # OR search the radius maximizing SNR
    jitter_sigma_mas=None, # default: telescope jitter
    npix=128,              # render grid (contains +2-wave defocus; tunable)
    oversample=11,         # render oversampling (tunable)
) -> dict
```

Returned dict keys:
- `snr` — float
- `signal_e` — source electrons within the aperture
- `noise_e` — total noise (electrons)
- `enclosed_fraction` — fraction of the PSF inside the aperture
- `r_aper_mas` — aperture radius actually used (incl. optimized/EE-derived)
- `n_pix` — detector pixels inside the aperture

**Aperture-mode precedence:** `optimize=True` → optimized radius; else explicit
`r_aper_mas`; else `ee_frac`; else default to the Simulation's `r_aper_mas`.
(Passing more than one of `r_aper_mas`/`ee_frac` with `optimize=False` follows
this precedence; this is documented, not an error.)

## Computation

Mirrors `get_signal_and_variance`'s noise model, but the signal fraction comes
from the **rendered** PSF rather than the analytic Airy curve.

1. Resolve `time` (default `meta['time']`, raise if None, bare float → seconds).
2. Render `psf_norm` (sum = 1) on the detector grid via `ImageSimulator(self,
   npix=npix, oversample=oversample)._context(...)` + `psf.render(ctx)`.
3. **Radial cumulative profile (computed once):** flatten the grid, sort pixels
   by radius from the center; form cumulative arrays
   `enclosed(r) = cumsum(psf_norm_sorted)` and `n_pix(r) = arange(1, N+1)`, with
   the matching sorted radius array `r_sorted` (converted to mas via the plate
   scale).
4. **Per-pixel noise rates** (electrons, from the ETC, same as `get_peak_pixel`):
   - `source_e_total = count_rates["source"] / ee_at_aper * time`
   - diffuse per pixel: for every non-source element,
     `count_rates[elem] / num_psf_pixels * time` (background, host) — summed
     (matches `get_signal_and_variance`'s scene shot-noise accounting)
   - `dark_per_pix = dark_current * time`
   - `read2 = read_noise**2`
5. **Vectorized SNR over all candidate radii:**
   - `signal(r) = source_e_total * enclosed(r)`
   - `noise(r) = sqrt(signal(r) + (diffuse_per_pix + dark_per_pix) * n_pix(r) + read2 * n_pix(r))`
   - `snr(r) = signal(r) / noise(r)`
6. **Select the aperture** per precedence:
   - `optimize=True` → `argmax(snr(r))`
   - explicit `r_aper_mas` → nearest index with `r_sorted <= r_aper_mas`
   - `ee_frac` → first index where `enclosed(r) >= ee_frac`
   - default → the Simulation's `r_aper_mas`
7. Return the dict at the selected radius.

All electron quantities use astropy units internally and are returned as plain
floats (`.value`), consistent with `get_signal_and_variance`.

**Note on host:** host flux is included in the *noise* term here (as
`get_signal_and_variance` does). This is distinct from the saturation
*signal* budget (`get_peak_pixel`), which excludes host by design.

## Structure / circular import

`psfsim` already imports `from .simulation import Simulation`, so `simulation`
cannot import `psfsim` at module load. `get_image_snr` therefore does a **lazy**
`from .psfsim import ImageSimulator, AiryPSF` inside the method body — consistent
with the existing lazy `from .airy import ...` and `from .io import ...` calls in
`Simulation`.

## Performance

Measured on this machine: AiryPSF render `npix=128, oversample=11` ≈ 31 ms;
DefocusPSF ≈ 1 ms. Aperture selection (sort + vectorized SNR) is ≈ 1 ms. A single
`get_image_snr` call is tens of ms; an SNR-vs-magnitude curve re-renders the
(magnitude-independent) PSF each call but stays well under a second for tens of
points. Defaults (`npix=128`) keep it fast while containing the defocus PSFs.

## Testing

**Cross-check (the core purpose):**
- In-focus `AiryPSF`, default aperture: `get_image_snr()['snr']` ≈ `get_snr()`
  within ~10% (residual from 1D-Airy-EE-curve vs 2D-pixelated render and
  continuous vs discrete pixel counts). Verify at two magnitudes and two
  exposure times.
- For that case, `enclosed_fraction` ≈ `ee_at_aper` and `n_pix` ≈
  `num_psf_pixels` (loose tolerance).

**Aperture modes:**
- `ee_frac=0.9` returns an aperture whose `enclosed_fraction` ≈ 0.9.
- `r_aper_mas=<value>` returns `n_pix`/radius consistent with that radius.
- `optimize=True` returns SNR ≥ the SNR at the default aperture for the same PSF
  (optimum is at least as good), and a finite radius.

**Defocus behavior:**
- At the *same fixed* aperture, `DefocusPSF` SNR < `AiryPSF` SNR (light spills out).
- `optimize=True` chooses a larger radius for `DefocusPSF` than for `AiryPSF`.

**Plumbing:**
- Returned dict has all documented keys with sane types/ranges
  (`0 < enclosed_fraction <= 1`, `n_pix >= 1`, `snr > 0`).
- Works built from both `sony:r` and `qcmos:r`.

## Out of scope (YAGNI)

- Changing `get_snr` / making the core ETC PSF-aware (the deeper option 3).
- Polychromatic / band-weighted defocus.
- Caching the rendered PSF across calls (the magnitude-independent render could
  be cached later if curve generation becomes a bottleneck; not needed now).

## Open items to confirm during spec review

1. Cross-check tolerance: ~10% proposed. Tighten if the in-focus paths agree
   more closely once implemented?
