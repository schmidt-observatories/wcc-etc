# Mark 90% encircled energy by default

**Date:** 2026-06-07
**Branch:** `ee-default-90`
**Status:** Approved

## Goal

The encircled-energy (EE) plot should mark the 90% EE radius by default, so
`SimulatedImage.plot_encircled_energy()` shows it with no arguments. Surgical
default change only.

## Background / what already exists

The plotting layer (`src/wcc_etc/plotting.py`) and the `SimulatedImage`
convenience methods already cover most of the original request:

- `plot_radial_mpl` / `img.plot_radial()` — azimuthally-averaged radial profile.
  Uses the **noiseless** image by default (`noise=False`); `noise=True` uses the
  noisy image. Marks **HWHM** with a dashed line + legend (`show_hwhm=True`
  default). No change needed.
- `plot_encircled_energy_mpl` / `plot_encircled_energy_bokeh` /
  `img.plot_encircled_energy()` — EE curve (normalized to 1), noiseless by
  default. Marks a target-EE radius when `ee_target` is set, but **defaults to
  `ee_target=None`** (no marker). This is the only gap.

Both EE functions already implement the marker identically (gated on
`ee_target is not None`): mpl draws an `axvline` at the enclosing radius + an
`axhline` at the target with a legend; bokeh adds two `Span`s. So only the
default value must change — no new marker logic.

## Design (Option A — chosen)

Set the default in the plotting functions so the convenience method and direct
calls agree:

1. `plot_encircled_energy_mpl`: signature default `ee_target=None` → `ee_target=0.9`.
2. `plot_encircled_energy_bokeh`: signature default `ee_target=None` → `ee_target=0.9`.
3. `ee_target=None` continues to disable the marker; any float overrides 0.9.

No marker-drawing code changes — only the default argument value and docstrings.

### Why Option A over a convenience-method-only default

A single, consistent default (raw function and `SimulatedImage` method agree) is
least surprising for an ETC, where 90% EE is a standard reference. `ee_target=None`
still yields the unmarked curve, so nothing is lost.

## Docstrings

- `plot_encircled_energy_mpl` / `plot_encircled_energy_bokeh`: note that
  `ee_target` defaults to 0.9 (marks the 90% EE radius) and that `None` disables
  the marker.
- `SimulatedImage.plot_encircled_energy`: note it marks 90% EE by default; pass
  `ee_target=None` to disable or another fraction to override.

## Testing

In `tests/test_plotting.py`:

1. Default mpl call marks 90% EE: `plot_encircled_energy_mpl(s, units="pix")`
   produces a legend (not None) and a horizontal line at `y == 0.9` among the
   axes artifacts — i.e. the marker is drawn without passing `ee_target`.
2. `ee_target=None` draws no marker: `ax.get_legend()` is None and no `y=0.9`
   horizontal line is present.
3. Explicit `ee_target=0.8` still works (existing test
   `test_plot_ee_mpl_target_marker_returns_radius` covers this).
4. Bokeh default call (`plot_encircled_energy_bokeh(s, return_="obj")`) returns a
   valid object without error (existing test covers the no-arg call; confirm it
   still passes with the new default).

Existing default-call tests (`test_plot_ee_mpl_monotonic_to_one`,
`test_plot_ee_mpl_pix_vs_mas_scaling`, the `SimulatedImage` delegation test) do
not assert the absence of a marker, so they remain green.

## Out of scope

- No change to the radial plot (already noiseless-default, noise toggle, HWHM).
- No new combined/overlay plotting functions.
- No change to the marker's visual style or the EE computation.

## Follow-ups (per feature workflow, after merge)

- If a demo notebook shows the EE plot, regenerate it so the 90% marker appears.
- Update `project-status` memory if the EE default is worth recording.
