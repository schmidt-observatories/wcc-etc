# ImageSimulator convenience plotting functions — design

**Date:** 2026-06-01
**Branch:** `feature/imagesimulator-plotting`

## Goal

Add convenience plotting helpers for the simulated detector images produced by
`ImageSimulator.simulate()` (a `SimulatedImage`). Four plot types, each available
in both matplotlib and bokeh, where the bokeh variants can render in a notebook
or return embeddable HTML for reuse in the separate Flask web portal.

## Structure

New module `src/wcc_etc/plotting.py` with standalone functions, plus thin
convenience methods on `SimulatedImage` that delegate to them.

- Module-level functions accept **either** a `SimulatedImage` **or** raw arrays
  (so the Flask portal can pass `image_e` / `image_clean` / `saturation_mask`
  / `pixel_scale_mas` directly without constructing a dataclass).
- Backends are **separate functions** (per design decision), one `_mpl` and one
  `_bokeh` per plot:

  ```
  plot_image_mpl(...)             plot_image_bokeh(..., return_=...)
  plot_image_row_mpl(...)         plot_image_row_bokeh(..., return_=...)
  plot_radial_mpl(...)            plot_radial_bokeh(..., return_=...)
  plot_encircled_energy_mpl(...)  plot_encircled_energy_bokeh(..., return_=...)
  ```

- Convenience methods on `SimulatedImage`: `.plot_image(backend='mpl', ...)`,
  `.plot_image_row(...)`, `.plot_radial(...)`, `.plot_encircled_energy(...)`.
  Each dispatches to the matching `_mpl`/`_bokeh` function based on `backend=`
  (`'mpl'` default). This preserves the separate-functions core while giving one
  ergonomic entry point per plot for notebook users.

### Input resolution helper

A small internal helper resolves the `(image_e, image_clean, saturation_mask,
pixel_scale_mas)` tuple from whatever was passed (a `SimulatedImage` or explicit
arrays), so every function shares one input contract.

### Backend return contract

- **matplotlib** functions return `(fig, ax)` (or `(fig, axes)` for the row),
  accepting an optional `ax=`/`axes=` to draw into an existing figure.
- **bokeh** functions take `return_`:
  - `return_='obj'` (default): return the bokeh figure / layout object
    (caller calls `show()` in a notebook).
  - `return_='html'`: return a standalone HTML `str` (via `bokeh.embed.file_html`).
  - `return_='components'`: return the `(script, div)` tuple
    (via `bokeh.embed.components`) for embedding in a Flask/Jinja template.

## The four plots

### (a) `plot_image` — single panel
- `aspect='equal'`, `origin='lower'`.
- `noise=True|False` selects `image_e` (noisy) vs `image_clean` (noiseless).
- `show_saturation=True|False` overlays the saturation mask as a translucent
  layer (default red) only where `saturation_mask` is `True`.
- Shared kwargs: `stretch='log'|'hist'|'linear'`, `cmap`, `vmin`, `vmax`,
  `colorbar`, `title`, and a `units='pix'|'mas'` axis toggle (mas via
  `pixel_scale_mas`, centered on the grid center).

### (b) `plot_image_row` — three panels in a row
Three panels, equal aspect, **sharing the same color scale** (computed from the
noisy image so all three are directly comparable):
1. **PSF + noise** (`image_e`)
2. **PSF, no noise** (`image_clean`)
3. **Saturation mask** (boolean mask shown on its own)

Same shared display kwargs as (a). Panels 1 and 2 share `vmin/vmax`; panel 3 is
a binary mask render.

### (c) `plot_radial` — radial profile
- Azimuthally-averaged radial profile via the existing `radial_data`.
- Defaults to `image_clean` (the PSF); `noise=True` uses `image_e`.
- `units='mas'` (default, from `pixel_scale_mas`) or `units='pix'`.
- Optional HWHM marker (reusing `calc_hwhm` from this module's neighbors).

### (d) `plot_encircled_energy` — EE curve
- EE vs radius via the existing `airy.psf_to_encircled_energy(psf2d, px_x_mas,
  px_y_mas)`, using `image_clean` by default.
- Normalized to 1 at large radius.
- `units='mas'|'pix'`.
- Optional marker at a target EE fraction (e.g. `ee_target=0.8`) drawing the
  enclosing radius.

## Cleanup carried along

The existing `FitsImg.plot` / `get_radial_profile` emit `print()` debug lines
("hist stretch", "Calculating radial data", etc.). The new functions emit no
stray prints. Shared stretch/normalize logic (the `astropy.visualization`
`ImageNormalize` + stretch selection) is factored into one small helper used by
the new matplotlib image functions so stretch behavior is consistent. We do not
refactor `FitsImg` itself beyond what is needed.

## Testing (TDD)

No pixel-perfect image comparisons. Tests assert:
- mpl functions return a `matplotlib.figure.Figure` and `Axes`; `plot_image_row_mpl`
  yields exactly 3 axes.
- bokeh `return_='obj'` returns a bokeh figure/layout; `return_='html'` returns a
  `str` containing `<script`; `return_='components'` returns a 2-tuple of `str`.
- saturation overlay marks only pixels where `saturation_mask` is `True`
  (verified via the masked-array / alpha layer, not rendered pixels).
- `plot_radial` / `plot_encircled_energy` return monotonic-where-expected arrays
  (EE non-decreasing; EE → ~1 at large radius).
- `units='mas'` vs `'pix'` scales the x-axis by `pixel_scale_mas`.
- Convenience methods on `SimulatedImage` dispatch to the right backend.

## Out of scope (YAGNI)

- No new plot types beyond the four.
- No refactor of `FitsImg`/`get_radial_profile` internals beyond removing reliance
  on their prints (we simply don't call into them where prints would fire).
- No interactive bokeh tools/callbacks beyond default pan/zoom/hover.
- No saving-to-file convenience (caller handles `fig.savefig` / `file_html`).
