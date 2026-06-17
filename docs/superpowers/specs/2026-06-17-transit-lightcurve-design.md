# Transit Light-Curve Simulation — Design

**Date:** 2026-06-17
**Branch:** `transit-lightcurve`
**Status:** Approved (design), pending spec review

## Goal

Add the ability to simulate and predict exoplanet transit light curves as the
WCC would observe them. The transit shape comes from the `batman` package; the
photometric noise comes from the existing WCC ETC. Output is a synthetic
light curve (clean model + noisy realization) that can be plotted in both
matplotlib and bokeh.

The feature is designed so that future time-series scenarios (moon transits,
Cepheid pulsations, etc.) slot in as sibling models behind one common
interface, with no changes to the simulator or plotting code.

## Why it lives in `wcc-etc`

The value of the feature is the coupling to the ETC: a transit is a
time-varying source brightness, and the ETC already knows how to turn a source
+ exposure into a photometric SNR (`get_image_snr`). The light-curve simulator
turns that SNR into a realistic per-point photometric error. Without the ETC
bridge this would be a generic batman wrapper with no reason to live here; with
it, it is a natural extension of the existing simulation stack.

## Architecture

One new module: `src/wcc_etc/lightcurve.py`. It follows the pluggable-source
pattern already used in `psfsim.py` (`AiryPSF` / `DefocusPSF` / `CustomPSF`
sharing one interface).

### `FluxModel` (abstract base)

The extension hook. Single responsibility: map time → normalized relative flux.

```python
class FluxModel:
    def relative_flux(self, time):
        """Return relative flux (ndarray), normalized to 1.0 out of event."""
        raise NotImplementedError
```

Future `MoonTransitModel`, `CepheidModel`, etc. subclass this. Nothing else in
the module needs to know which concrete model it holds.

### `TransitModel(FluxModel)`

Thin wrapper over `batman`.

- Constructor takes transit parameters: `t0`, `per`, `rp` (Rp/R*), `a` (a/R*),
  `inc` (deg), `ecc`, `w`, `limb_dark` (e.g. `"quadratic"`), `u` (coeff list).
- `batman` is imported **lazily inside the class** (not at module top), so it is
  an optional dependency. A clear `ImportError` with install hint
  (`pip install wcc-etc[lightcurve]`) is raised if missing.
- `relative_flux(time)` builds a `batman.TransitParams`, constructs a
  `batman.TransitModel(params, time)`, and returns `m.light_curve(params)`.

### `LightCurveSimulator`

The ETC bridge.

- Constructed from a `Simulation` and a `FluxModel`:
  `LightCurveSimulator(sim, model)`.
- `simulate(time, exptime, *, r_aper_mas=None, ee_frac=None, psf=None,
  jitter_sigma_mas=None, npix=128, seed=None) -> LightCurve`.

### `LightCurve` (result object)

Lightweight container with fields:
`time`, `flux` (noisy), `flux_clean` (model), `flux_err` (scalar σ),
`exptime`, `snr`. Carries a dispatching `.plot(backend='mpl'|'bokeh', **kw)`.

## Data flow (`simulate`)

1. **One photometric measurement at baseline brightness.** Call
   `sim.get_image_snr(time=exptime, n_reads=1, r_aper_mas=..., ee_frac=...,
   psf=..., jitter_sigma_mas=..., npix=...)` once. This is the single-frame
   aperture photometry that establishes the SNR for one exposure of length
   `exptime`. `n_reads=1` → each light-curve point is effectively a single read.
2. **Photometric precision.** `sigma = 1.0 / snr` in relative-flux units.
   **Held constant** across all points (transit depths are small, so SNR is
   ~constant; no per-point rescaling by the dimmed flux). This is a deliberate
   simplifying assumption.
3. **Clean model.** `flux_clean = model.relative_flux(time)`.
4. **Noisy realization.** `flux = flux_clean + rng.normal(0, sigma, time.shape)`,
   using a seedable `np.random.default_rng(seed)`.
5. **Return** a `LightCurve(time, flux, flux_clean, flux_err=sigma, exptime, snr)`.

## Plotting

Follows `plotting.py` conventions (mpl + bokeh variants, `set_wcc_style()`,
Title Case labels, `_resolve_inputs`-style flexibility):

- `plot_lightcurve_mpl(source_or_arrays, *, show_noise=True, show_model=True,
  ax=None, ...) -> (fig, ax)`
- `plot_lightcurve_bokeh(..., return_='obj'|'html'|'components')`

Each can render: the **clean model** line, the **noisy points** with error bars,
or **both** overlaid (default). Accepts a `LightCurve` (positional) or raw
arrays (`time=`, `flux=`, `flux_clean=`, `flux_err=`). `LightCurve.plot()`
dispatches to these.

## Public API / exports

Add to `wcc_etc/__init__.py`:
`FluxModel`, `TransitModel`, `LightCurveSimulator`, `LightCurve`,
`plot_lightcurve_mpl`, `plot_lightcurve_bokeh`.

## Dependencies

`batman-package` added under `[project.optional-dependencies]` as the
`lightcurve` extra (lazy-imported). Core ETC install and CI remain
batman-free; batman-dependent tests use `pytest.importorskip("batman")`.

## Testing (TDD)

- `TransitModel.relative_flux` matches a direct `batman` call for the same
  params (exact agreement).
- Mid-transit depth ≈ `rp**2` for a central, uniform-ish transit (within LD
  tolerance); out-of-transit flux == 1.0.
- σ wiring: with a stubbed/mock `Simulation` returning a known SNR,
  `flux_err == 1/snr` and noise is reproducible for a fixed `seed`.
- Noise statistics: residual std of `flux - flux_clean` ≈ σ over many points.
- Plotting smoke tests: mpl returns `(fig, ax)`; bokeh `components` returns
  `(script, div)`; `show_noise` / `show_model` toggles produce the expected
  artists.
- batman-dependent tests guarded by `pytest.importorskip("batman")`.

## Out of scope (YAGNI)

- Per-point σ rescaling by instantaneous brightness.
- Full image render + photometry per epoch (the analytic single-frame SNR is
  the chosen path).
- Moon transits, Cepheids, eclipsing binaries — supported by the `FluxModel`
  interface but not implemented now.
- Correlated/red noise, systematics, transit-timing variations.
- Fitting / inversion (this is forward simulation only).

## Extensibility summary

Adding a new scenario later = one new `FluxModel` subclass implementing
`relative_flux(time)`. `LightCurveSimulator`, `LightCurve`, and the plotting
functions work on it unchanged.
