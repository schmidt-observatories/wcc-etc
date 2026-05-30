# Parametric source spectra for the Scene system

**Date:** 2026-05-30
**Status:** Design approved, pending spec review

## Problem

`wcc-etc` sources can currently only be built from a Pickles spectral-type name,
a named astro file, a path to a spectrum FITS file, or a pre-built synphot
`SourceSpectrum`. There is no way to specify a *parametric* source. The legacy
helper `get_wcc_snr_and_simulation` advertises `source_type='blackbody'` but
raises `ValueError("Blackbody not implemented yet")`, and a `get_blackbody_flux`
prototype exists but is not wired into the `Scene` system.

## Goal

Let users build four parametric source spectra directly through the existing
scene API using a reserved spectrum name plus keyword arguments:

```python
scene = get_scene(name='blackbody', teff=5777, mag=15, bandpass='johnson_r')
scene = get_scene(name='flat',      mag=18)
scene = get_scene(name='powerlaw',  alpha=-1.0, mag=18)
scene = get_scene(name='emission',
                  lines=[{'wave': 6563, 'flux': 1e-15, 'fwhm': 3},
                         {'wave': 6583, 'flux': 4e-16, 'fwhm': 3}],
                  mag=None)
```

No new classes. Each type is a synphot model wrapped in a `SourceSpectrum`.

## Design

### 1. Spectrum-builder registry (`scene.py`)

Reserved spectrum names map to module-level builder functions. Each builder
receives the element's `meta` dict and returns a synphot `SourceSpectrum`.

```python
_SPECTRUM_BUILDERS = {"blackbody": _build_blackbody,
                      "flat":      _build_flat,
                      "powerlaw":  _build_powerlaw,
                      "emission":  _build_emission}
```

| Name | synphot model | meta params | Defaults |
|------|---------------|-------------|----------|
| `blackbody` | `BlackBodyNorm1D(temperature=teff)` | `teff` (K) | required; raise if missing |
| `flat` | `ConstFlux1D` | `flat_unit` ∈ {`fnu`, `flam`} | `fnu` (AB-flat) |
| `powerlaw` | `PowerLawFlux1D(x_0=lambda_ref, alpha=alpha)` | `alpha`, `lambda_ref` | `lambda_ref=5500` Å |
| `emission` | sum of `GaussianFlux1D(total_flux, mean, fwhm)` | `lines=[{wave, flux, fwhm}]` | `fwhm=2` Å per line if omitted |

Conventions to document in the builders:
- **powerlaw:** normalized shape is `F_lambda ∝ (lambda / lambda_ref) ** alpha`.
  Note synphot's `PowerLawFlux1D` uses `(x / x_0) ** (-alpha)`; the builder
  negates so the public `alpha` means the F_lambda exponent.
- **flat:** `fnu` builds a constant-`F_nu` spectrum (flat in AB); `flam` builds
  constant `F_lambda`. Absolute amplitude is arbitrary because the spectrum is
  magnitude-normalized downstream.
- **emission:** each line's `flux` is the integrated line flux in erg/s/cm².
  Lines are summed. No continuum (out of scope).

### 2. Hook into `SceneElement.set_spectrum` (`scene.py:207`)

Add a branch *before* the existing file / spectral-type lookup:

```python
if type(spec_or_file) is str:
    if spec_or_file.lower() in _SPECTRUM_BUILDERS:
        spectrum = _SPECTRUM_BUILDERS[spec_or_file.lower()](self.meta)
    elif not os.path.isfile(spec_or_file):
        ... # existing get_any_astro_name path
    else:
        ... # existing file path
elif isinstance(spec_or_file, SourceSpectrum) or spec_or_file is None:
    spectrum = spec_or_file
```

`set_spectrum` runs after `super().__init__(meta | input_parameters)`, so
`self.meta` already holds `teff` / `alpha` / `lines` when the builder is called.
All existing input types (file path, SpT name, `SourceSpectrum`, `None`) are
unchanged.

### 3. Normalization rule in `get_spectrum` (`scene.py:285`)

Change the normalize step so that a `None` magnitude skips normalization:

```python
if apply_mag and self.mag is not None:
    spectrum = self.spectrum.normalize(mag, band=self.band)
else:
    spectrum = self.spectrum
```

This is what lets absolute-flux emission-line sources pass through un-normalized.
It is backward-compatible: every existing source carries a `mag`, so behavior is
unchanged for them.

### 4. Wire up the legacy helper (`wcc_etc.py:910`)

Replace the `source_type=='blackbody'` `ValueError` with a real call:

```python
elif source_type == 'blackbody':
    scene = get_scene('blackbody', mag=mag, teff=teff, host=None,
                      background="zodi", bandpass=source_bandpass,
                      background_prop={"bandpass": bg_bandpass,
                                       'mag': bg_surface_brightness})
```

## Testing

One test per source type plus the `mag=None` path:

- **blackbody:** observe the normalized spectrum through its band → recovers the
  input magnitude (~0.01 mag); color is consistent with `get_blackbody_flux`.
- **flat:** AB-flat round-trips to a constant `F_nu` and the correct observed mag.
- **powerlaw:** flux ratio at two wavelengths equals `(lambda1 / lambda2) ** alpha`.
- **emission:** integrating a line over wavelength recovers the input `flux`;
  centroid sits at `wave`; produces a sensible countrate through `sony:halpha`.
- **mag=None:** `get_spectrum` returns the un-normalized flux.

## Out of scope (YAGNI)

Continuum-under-emission-lines, redshift, interstellar extinction, and arbitrary
multi-component composite sources. Each can be added later as an additional
builder without changing this architecture.
