# Structured host background: Sersic-profile galaxies

Date: 2026-09-05
Status: Approved in chat, implementing

## Problem

A scene's `host` element is today either a point (total `mag`, rides the
source PSF) or a *uniform* surface brightness (`surface_brightness=True`,
mag/arcsec², added to every pixel). Real transient hosts are neither: a TDE
sits on the cusp of a bulge, a SN sits 0.5–2″ from the nucleus on a disk. At
the WCC plate scale (16.9 mas/pix) a 1″ host spans ~60 pixels, so the local
host surface brightness under the aperture — and hence the SNR — depends on
the profile shape and the source offset.

wcc-sim already renders Sersic components (`wcc_sim/extended.py`), but it
gets its count rates from wcc-etc's private `_count_rate_components()`. Doing
this in wcc-etc lets wcc-sim depend on wcc-etc for the profile too, not the
other way around. wcc-etc must **not** import wcc-sim.

## Goal

`get_scene(..., host="G5V", host_prop={"mag": 17, "profile": "sersic",
"r_eff": 1.0, "n": 1, "ellip": 0.3, "pa": 45, "dx": 0.8, "dy": 0.0})`
produces a host whose light is a PSF-convolved elliptical Sersic profile,
consistently in `ImageSimulator.simulate`, `get_snr` / `get_image_snr`,
`get_image_exptime_for_snr`, `get_peak_pixel`, and `is_saturated` (one
charge budget — the issue #63 lesson).

## Parameterization

Angular only (no distance / redshift / kpc — decided 2026-09-05), mirroring
wcc-sim, Pandeia and the HST ETC so wcc-sim can adopt it later.

| key | unit | default | meaning |
|---|---|---|---|
| `profile` | str | absent | `"sersic"`. Absent → current behaviour (point or uniform SB). |
| `r_eff` | arcsec | required | half-light radius along the major axis |
| `n` | – | 1.0 | Sersic index (1 disk, 4 de Vaucouleurs); must be > 0 |
| `ellip` | – | 0.0 | 1 − b/a (astropy convention), 0 ≤ ellip < 1 |
| `pa` | deg | 0.0 | major-axis angle, CCW from the +x image axis |
| `dx`, `dy` | arcsec | 0.0 | host-centre offset from the source (+x right, +y up) |

Flux normalization reuses the existing flags:

- `surface_brightness=False` (default): `mag` is the **total** integrated host
  magnitude. Total-to-amplitude is analytic:
  F/I_e = 2π n r_e² (1−ellip) e^{b_n} b_n^{−2n} Γ(2n), b_n = gammaincinv(2n, ½).
  Light falling outside the rendered grid is lost, correctly (no renormalizing
  to the grid sum).
- `surface_brightness=True`: `mag` is μ_e, the surface brightness at r_e in
  mag/arcsec². synphot already returns the per-pixel rate at that SB; the
  profile is then the dimensionless I(r)/I_e.

These are the same two conventions wcc-sim's `SersicComponent` offers
(`total_mag` / `sb_mag_arcsec2`).

## Design

### 1. `scene.py`

- `_PROFILE_PARAMS = {"sersic": ["r_eff", "n", "ellip", "pa", "dx", "dy"]}`;
  `SceneElement.mutable_parameters` appends them when `meta["profile"]` is
  set, so `sim.update(host__r_eff=2.0)` works like `host__teff`.
- `SceneElement.profile` property: `None` or the dict of profile kwargs
  (with defaults filled). Validation (`n > 0`, `r_eff > 0`, `0 ≤ ellip < 1`,
  unknown profile name) happens here, once.
- `get_scene_element` passes profile kwargs through `**kwargs` into meta as it
  already does for `teff`; nothing else to add.

### 2. New `src/wcc_etc/extended.py`

```
sersic_total_over_amplitude(n, r_eff_pix, ellip) -> float
render_sersic(profile, plate_scale_arcsec, npix, oversample, total=True) -> ndarray
```

`render_sersic` evaluates `astropy.modeling.models.Sersic2D` on the
`oversample`× grid (grids are ≤ 300 px, so no memory concern — unlike
wcc-sim's full frames), block-averages to the detector grid, and returns the
per-pixel profile: unit total (analytic) when `total=True`, `I/I_e` when
`False`. Grid centre convention = `ImageSimulator` (source at `(npix−1)/2`);
`dx, dy` shift the profile centre in arcsec.

The PSF convolution lives in `simulation._image_render_bundle` (it needs the
rendered PSF): render the profile on a `2·npix` grid, `fftconvolve(...,
psf_norm, mode="same")`, `center_crop_or_pad` back to `npix`. That removes
the edge-loss bug wcc-sim's review flagged (P5).

### 3. `simulation.py`

- `_count_rate_components` gains `extended`: list of
  `(rate, profile, is_surface_brightness)` for elements with a profile,
  collected *before* the existing SB/point classification (a profiled element
  is neither `diffuse_rate_per_pix` nor `contaminant_rate_total`).
- `_image_render_bundle` adds `extended_rate_image` — a `(npix, npix)`
  e-/s/pix array, sum of the PSF-convolved profiles, cached with the bundle.
  Zeros when no extended element.
- `_per_frame_clean_image_e` adds `b["extended_rate_image"] * tf`.
- `get_image_snr` / `get_image_exptime_for_snr` pass `diffuse_per_pix` as a
  2D array (`scalar sky + extended image`) into the aperture functions.
- `get_snr_airy` / `get_signal_and_variance` (deprecated 1D path): warn once
  that the profile is ignored and treat the element by its existing
  classification. No new physics on the deprecated path.

### 4. `psfsim.py`

- `ImageSimulator.simulate` adds the extended image from a shared
  `Simulation._extended_rate_image(psf_norm, ctx)` helper (also used by
  `_image_render_bundle`), so `simulate` — which supports a sub-pixel
  `center=` the bundle does not — and the SNR path use one implementation.
  `dx, dy` are relative to the source position, i.e. to `center` when given.
- `_radial_cumulative` also returns the radial sort `order`;
  `aperture_snr_radial` / `aperture_time_for_snr` accept `diffuse_per_pix` as
  either a scalar or a 2D image: scalar → `per_pix * n_pix` (unchanged);
  image → `cumsum(image.ravel()[order])`.

### 5. Out of scope

- Distance / redshift / physical sizes (decided: angular only).
- Multiple hosts or a profile on `source` / `background`. Only `host` is
  expected to carry a profile; nothing forbids it elsewhere, but it is not
  tested.
- Reddening, colour gradients, other profiles (Gaussian, exponential are
  Sersic n=0.5/1 anyway).
- Legacy `wcc_etc.py` API.

## Tests

`tests/scene/test_profile.py`
- `profile` absent → `SceneElement.profile is None`; existing tests unchanged.
- defaults filled; invalid `n`, `ellip`, unknown profile raise `ValueError`.
- `host__r_eff` is in `mutable_parameters` and `update` changes it.

`tests/imaging/test_extended.py`
- analytic `sersic_total_over_amplitude` vs a brute-force numeric sum
  (rel 2e-3), for n=1 and n=4.
- total-mag n=1 host with r_eff ≪ grid: `extended_rate_image.sum()` ≈
  `source_rate_total` of a point source at the same mag (rel 1e-2).
- μ_e mode: pixel value at r_e along the major axis ≈ the uniform-SB
  `diffuse_rate_per_pix` for the same `mag` (rel 5e-2, PSF-smeared).
- `dx` shifts the host peak by `dx / plate_scale` pixels.
- `simulate(add_noise=False).image_clean` equals `bkg + dark + source·PSF +
  extended_rate_image·t` (the one-budget test).

`tests/simulation/test_extended_snr.py`
- `get_snr` of a source on a bright nuclear host < same source with `dx=3″`.
- scalar vs 2D `diffuse_per_pix` in `aperture_snr_radial` agree when the
  image is uniform.

## Notebook

`notebooks_scratch/20260905_sersic_host.ipynb` (built with the notebook-demo
skill): gallery of (n, r_eff, ellip) hosts on the WCC grid; nuclear vs
offset transient SNR vs host magnitude; total-mag vs μ_e normalization
check; a `sim.update(host__r_eff=...)` sweep.
