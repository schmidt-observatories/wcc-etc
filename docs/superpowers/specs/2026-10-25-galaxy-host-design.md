# Galaxy hosts at redshift (design)

Port of slicersim's galaxy capability (commits 4079f43, 75c4124, 1522f7d by
Mickael Rigault) to wcc-etc. Agreed in conversation on 2026-10-25.

## Intent

The transient-in-a-host picture of the slicersim README: a SN at redshift z in
an NGC host moved to that z, with the galaxy's own spectrum and absolute
magnitude, so one `redshift` drives the spectrum, the brightness and the
angular size of both source and host consistently.

## Goals

- A scene element can take an NGC galaxy as its spatial model
  (`profile="ngc0628"`) next to the existing `sersic`. The r-band thumbnail
  from `lazuli_astrosim_data` is projected to the element's `redshift` at fixed
  physical size (angular-diameter-distance ratio), with an arcsec offset
  (`dx`, `dy`) and a rotation (`pa`, degrees CCW). It goes through the existing
  extended-source path (`Simulation.extended_rate_image`), so it is
  PSF-convolved and appears in image SNR, peak pixel and the simulated frame.
- A scene element can take an NGC galaxy as its spectrum (`host="ngc0628"`),
  the galaxy's own template.
- `redshift` on an element stretches its spectrum by 1+z (synphot `z`,
  wavelength only). It applies to the source too (the bundled hsiao07 SN Ia
  template).
- `magabs` on an element is a rest-frame absolute magnitude in the element's
  `bandpass`/`magsys`, used when `mag` is None: the rest-frame template is
  scaled to `magabs` at 10 pc, dimmed by `distmod(z)` and by `1+z` (per-Å flux
  density), exactly as slicersim's `generic_get_flux`. Planck18 is the default
  cosmology.
- The Sersic host accepts `r_kpc` (proper kpc at `redshift`) as an alternative
  to `r_eff`.
- `get_scene(..., redshift=z)` sets the redshift of both source and host unless
  `host_prop` gives its own.
- A bare NGC name as `host` means image and spectrum of that galaxy, with
  defaults `magabs=-21`, `bandpass="sdss_r"`, `magsys="abmag"` (slicersim's).
- Plots: `SceneElement.show_image(plate_scale_arcsec, npix, ...)` renders the
  spatial model alone (log stretch), and `Simulation.plot_host(time, ...)` puts
  the simulated frame next to the PSF-convolved host alone.
- `lazuli-astrosim-data` becomes a hard dependency (PyPI 0.1.0, 47 MB).
- README section, tutorial notebook `08_galaxy_host.ipynb`, changelog entry.

## Non-goals

- Blackbody or other parametric host spectra at redshift beyond what the
  existing builders already do (they get the 1+z stretch for free).
- K-corrections other than the slicersim convention above.
- A `DefocusPSF`-style render cache for galaxy images.

## Constraints

- No leading underscores on new function or method names.
- No numpy-2-only or py311-breaking constructs; CI runs numpy 2 / py3.11.
- Tests follow `tests/` subdirectory conventions; one assert per test.
- Numerical agreement with slicersim for the projection routine (same
  algorithm, ported with an explicit output `center`), and the same
  magabs-to-flux formula.

## Builder's calls

- `redshift=None` on an element means no shift (z=0), so existing scenes are
  unchanged.
- An NGC spatial model cannot be a surface-brightness element; `profile`
  raises.
- `redshift` must exceed the galaxy's own redshift for an NGC image
  (`galaxy.galaxy_pixel_scale` raises otherwise), as in slicersim.
