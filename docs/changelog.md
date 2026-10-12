# Changelog

## Unreleased

- The user guide is rewritten for a first-time reader: the SNR page states the noise
  model and shows the source-limited and background-limited regimes, the saturation and
  scenes pages gain figures drawn with the new plot methods, and change-history asides are
  gone from the prose.
- The tutorial notebooks are committed with their outputs, so the figures and printed
  numbers are visible on GitHub without running them; the pre-commit hook that stripped
  outputs is gone. `CITATION.cff` is removed until there is something to cite.
- `wcc_etc.io` exposes `SENSORFILTER_FOCUS`, `SENSORFILTER_IMPLEMENTED` and `KIND_NAMES`
  as public names (in `__all__`, documented). The underscore spellings remain as aliases
  so wcc-sim keeps importing until it moves to the public ones.
- Simulation-level plot methods, each taking `ax=None` and returning `(fig, ax)`:
  `Simulation.plot_snr` (alias `plot`), `plot_snr_vs_mag`, `plot_sweep`,
  `plot_peak_pixel` (with the `saturation_e` ceiling and onset marker),
  `plot_exptime_for_snr`, `plot_snr_vs_aperture`, and `ImageSimulator.plot_psf`.
  `Sensor.plot`, `SceneElement.plot` and `SimulatedImage.plot` alias the existing
  `show` / `plot_image`; `Sensor.show` and `SceneElement.show` now return `(fig, ax)`
  like the rest (they returned the bare figure). The image plotters accept
  `units="arcsec"` alongside `mas` and `pix`, and `stretch="log"` is now a matplotlib
  `LogNorm` (clipped, so read-noise-negative pixels keep the floor colour) whose
  floor sits `dyn_range` (default 1e4) below the peak; `vmin=` / `vmax=` fold into
  it. `plotting` declares `__all__`. The tutorials use the new
  methods instead of hand-rolled matplotlib,
  and their text now matches the code: saturation is `sensor.saturation_e`
  (the full well on the Sony, the ADC on the qCMOS in high gain), the H-alpha labels
  are `zwo:halpha*`, and `update()` with an unknown key raises. The installation,
  concepts, configuration and API-reference pages are brought up to date.
- The two Huygens defocus `.txt` exports are removed from the repository; the FITS
  products are the only bundled form (`DEFOCUS_*_TXT_PATH` is gone). Tests that read
  a raw export use a small synthetic one written to `tmp_path`.
- One saturation ceiling: `Sensor.saturation_e` = min(full well, (adc_max - bias) * gain),
  with `Sensor.saturation_limit` reporting which binds (`"well"` / `"adc"`).
  `Simulation.is_saturated`, `psfsim.saturation_mask_from_image_e` and
  `ImageSimulator.simulate` all compare electrons to it. Previously `is_saturated`
  tested only the ADC clip, so the Sony IMX455 reported unsaturated between the
  ~16.3 ke- well and the ~17.1 ke- ADC ceiling, and the image mask ignored bias (#64).
- Packaging: `MANIFEST.in` is an explicit allowlist and `package-data` names only the two
  bundled defocus PSF FITS, so a build no longer depends on what is in the developer's
  working tree (188 MB from a dev tree before). The unused `data/flats/` (36 MB) and the
  Pickles `dat_uvi/` atlas are removed; `hsiao07.dat` is now the phase-0 slice (2 columns,
  130 kB instead of 8 MB) and loads with `get_scene("hsiao07", ...)` instead of raising.
  Wheel and sdist are 8 MB; CI fails a build over 20 MB or carrying local products (#62, #101).
- `DATA_SOURCES.md` lists every bundled dataset with its citation or origin;
  `radial_data.py` carries its attribution (#101).
- Release hygiene (#101): the CALSPEC Vega spectrum and the Johnson V curve are bundled
  (`data/astrophysics/alpha_lyr_stis_011.fits`, `data/throughput/other_filters/johnson_v_004_syn.fits`)
  instead of downloaded from STScI at first use, so zero points cannot drift and no
  network is needed; `io.PACKAGE_PATH` uses the public `importlib.resources` API; the
  unused `flask` dependency is gone and the others carry minimum versions; the README
  clone link is HTTPS (#4) and a "Known limitations" section lists #98, #99, #100, #104;
  the stale mkdocs files and `docs/superpowers/` are removed; this file is the one
  changelog (the Sphinx page includes it).
- Input guards for inputs that used to give a plausible wrong answer (#102):
  `Simulation.update()` raises `KeyError` on an unknown or ambiguous key instead of
  warning and ignoring it; a file spectrum that does not cover the bandpass warns with
  the uncovered Angstrom range (the edge value is still extrapolated); an `sdss_*`
  normalization band without an explicit `magsys` warns that SDSS magnitudes are AB;
  `Sensor(qe=...)` other than 1 raises (QE lives in the throughput file);
  `parse_and_interpolate` warns when the ZWO gain / read-noise / dark-current tables
  are clamped outside their range, and a zwo config without `sensor_temp` raises a
  clear `ValueError` instead of an `AttributeError`.

- Sersic hosts: the 5x5 pixels around the cusp are now integrated exactly (adaptive
  quadrature split at the cusp) instead of sampled on the `oversample` sub-grid, which
  overshot the analytic total by up to 3x (n=4) and 4 orders of magnitude (n=8) for
  hosts with r_eff below a pixel. On-grid totals are now within ~1e-4 of analytic for
  n in [0.5, 8] and r_eff >= 0.18 pixel, and never exceed 1 (#84).
- One centre convention for PSFs and hosts: a render with `center=None` puts the source
  on the integer pixel `(npix-1)//2` (`psfsim.grid_center`), on odd and even grids;
  `recenter` shifts from the PSF's actual centre, so `center=(cx, cy)` is honoured to
  sub-pixel precision for Airy and custom PSFs alike; and `extended_rate_image` places
  the host by convolution with the source PSF, so a zero-offset host sits on the source
  (it was half a pixel off on even grids, including the default 128-pixel SNR grid) (#84).

- Sersic-profile hosts (structured background):
  `host_prop={"profile": "sersic", "r_eff": 1.0, "n": 1, "ellip": 0.3, "pa": 45, "dx": 0.8, "dy": 0}`
  with angular parameters (arcsec / degrees). `mag` is the total host magnitude, or
  μ_e (mag/arcsec² at r_eff) with `surface_brightness=True`. The profile is rendered and
  PSF-convolved once and shared by `ImageSimulator.simulate`, `get_snr`,
  `get_image_exptime_for_snr`, `get_peak_pixel` and `is_saturated`. New module
  `wcc_etc.extended`; the deprecated analytic Airy path warns that it ignores profiles.

### Earlier in this cycle


- The qCMOS (HWK4123) now carries its two gain modes. `qcmos.toml` has a
  `[sensor.gain_modes.<mode>]` table per mode and a `gain_mode` default of
  `"high"` (32x: 7.42 ADU/e, 0.25 e- read noise); `"low"` (1x: 0.242 ADU/e,
  2.33 e- read noise) is selected with `gain_mode="low"` on
  `from_sensorfilter`,
  `from_sensor_and_scene` or
  `from_name`. The 12-bit ADC is real: in high gain it
  clips at 552 e-, far below the 7500 e- full well (was 7000), so faint-source
  "saturation" there is physical; low gain is well-limited. Gain is stored in
  e-/ADU with its provenance stated; the old `gain = 0.112` was a single
  unlabelled value (#94, #61).
- Both sensor configs carry an informational `min_frame_time` (s): 0.25 for
  a full-frame IMX455 read, 0.008 for the qCMOS at 125 fps. Nothing enforces
  it yet.
- Defocused filters (`zwo:r+1`, `zwo:bb2`, ...) no longer use the 70 mas
  Airy-core aperture by default, which held only 2 to 8 percent of the flux and
  under-reported SNR by 3 to 7x. With no aperture argument the aperture now
  follows the PSF: 70 mas for the Airy PSF, the SNR-optimized radius otherwise.
  `Simulation(r_aper_mas=...)` still pins a fixed radius. The default
  `r_aper_mas` is `None`; `sim.meta` only carries it when set (#95).
- `get_scene("emission", mag=..., lines=...)` now raises instead of silently
  renormalizing the absolute line fluxes to the magnitude (a 400x error for a
  1e-15 erg/s/cm^2 line). `get_scene_element("emission", ...)` defaults to
  `mag=None`, and `surface_brightness=True` with `mag=None` scales the
  per-arcsec^2 spectrum by the pixel area instead of crashing (#96).
- The x2 read-noise margin on the Sony/ZWO path is now the config key
  `read_noise_margin` (2.0 in `zwo.toml` and `qcmos.toml`, whose
  `read_noise` is back to the measured 0.28 e-) instead of a hidden factor
  in `Sensor.from_config`. Results are unchanged. `get_snr` reports the
  effective per-pixel read noise as `read_noise_e` (#97).

### H-alpha filters

- Three H-alpha bands on the Sony (IMX455) sensors now have EOL telescope + WCC
  throughput curves and are usable with
  `from_sensorfilter`: `zwo:halpha2` (2 nm,
  position 10), `zwo:halpha6` (6 nm, position 9, formerly the N-II
  placeholder) and `zwo:halpha20` (20 nm, position 21, formerly the O-III
  placeholder). The `nii` / `oiii` keys are removed; `hbeta` / `heii`
  remain unimplemented placeholders.

### Plotting and PSF introspection

New public API, each replacing something the docs or a notebook previously had
to hand-roll:

- `default_psf` — the PSF used when `psf=` is
  omitted, replacing reads of the private `_default_psf`. Falls back to a
  diffraction-limited `AiryPSF`. Every internal call site,
  including `polychromatic_psf`, now goes through it.
- `render_psf` — the bare normalized PSF on the
  detector grid, replacing manual construction of the private render context.
  It builds that context the same way `simulate` does, so it honours the
  source-weighted effective wavelength from issue #65.
- `plate_scale_mas` and
  `default_psf`.
- `plot_bandpass_mpl` and `show` — filter throughput
  curves.
- `label=` and `**kwargs` passthrough on `plot_radial_mpl`,
  `plot_encircled_energy_mpl` and `plot_bandpass_mpl`, so overlaid curves
  can carry a legend without reaching into `ax.lines`.
- `wave=` and `flux_unit=` on
  `get_spectrum` and
  `show`, so a source can be plotted over a
  chosen grid in F_lambda rather than only PHOTLAM.

### Behaviour change

- `simulate` now defaults to `default_psf`
  instead of always `AiryPSF`. For a simulator built by
  `from_sensorfilter` with a defocused label, `simulate()` without `psf=`
  now renders that filter's defocus PSF, matching what `get_image_snr` has
  always done. Every other construction path is unaffected. Pass `psf=`
  explicitly to override.

- **Host and diffuse elements are classified by** `surface_brightness`
  **, not by element name** (issue #63). A host given as an ordinary
  integrated magnitude used to have its *total* count rate added to
  **every** detector pixel, overstating its shot-noise contribution by
  `n_pix / enclosed_fraction` — 64.6x for the representative mag 20 source /
  mag 16 host scene (38.6 M e- of host charge instead of 597 k e-). Such an
  element is now treated as unresolved and co-located with the source and is
  rendered through the same PSF; only `surface_brightness=True` elements are
  per-pixel. See the user guide (host-spatial-treatment).

  This also unifies the charge budget: `get_peak_pixel`,
  `is_saturated`, the
  `get_image_snr` saturation check and
  `ImageSimulator.simulate` previously disagreed about whether host charge
  counted, so identical scenes gave different saturation answers depending on
  which API you called. They now all see the same electrons, and
  `get_peak_pixel` includes an unresolved host (it no longer drops it).

- **Source-weighted PSF wavelength** (issue #65, option **A**). The PSF is now
  rendered at the photon-weighted effective wavelength of the bandpass times
  the source SED — `effective_wavelength` — instead
  of the filter pivot wavelength, which is the same for every source. Through
  the WCC broad band this moves the render wavelength from 582.1 nm to 540.7 nm
  (O5V) or 716.5 nm (M5V) and corrects the modelled central-pixel fraction by
  −9 % to +36 %; the pivot-only model predicted premature saturation for red
  sources. New `spectral` module with
  `effective_wavelength` and
  `photon_weighted_subbands`.
- **Polychromatic PSFs**: `PolychromaticPSF` coadds a base PSF
  rendered at `n_sub` equal-photon-weight sub-bands;
  `polychromatic_psf` builds one for a simulation's
  band and source.
- **Defocus products carry their optics.** The bundled Zemax Huygens defocus
  PSFs now ship as FITS with `PIXSCALE` / `WAVELEN` / `FNUM` /
  `DEFOCUSW` headers (regenerate with
  `huygens_txt_to_fits`), and
  `DEFOCUS_1WAVE_PATH` / `DEFOCUS_2WAVE_PATH` point at them.
  `DefocusPSF` and `CustomPSF` gained a
  `wavelength_scaling` argument (`"despace"` — the default — `"waves"`,
  or `"none"`) and now **raise** instead of silently ignoring
  `ctx.wavelength_m` and `ctx.fnum` when they carry no reference metadata.
  See the user guide (resampled-psf-scaling).

  *Migration*: a `CustomPSF` built from a bare array needs
  either `ref_wavelength_m` and `ref_fnum`, or
  `wavelength_scaling="none"`.

## v0.1.0

Initial public release. Highlights:

- **Scene abstraction**: `Scene` and
  `SceneElement` manage source, host, and background
  elements, built through `get_scene`.
- **Selectable magnitude system**: `magsys="vegamag"` (default) or
  `magsys="abmag"` (case-insensitive) on
  `SceneElement` and `get_scene`. Note
  that magnitudes without an explicit `magsys` — including the built-in
  `zodi` background (`mag=22.5`) — are Vega magnitudes. The AB offset is
  negligible in Johnson *V* (~0.002 mag) but grows toward the red (~0.26 mag
  in *R*, ~1.9 mag in *K*).
- **Separate** `Sensor` **and** `Telescope`
  **objects**, with sensor `bit_depth` / `bias_level` / `adc_max`
  properties.
- **2-D image SNR by default**: `get_snr` delegates
  to `get_image_snr` and returns a **dict** (index
  `["snr"]`). The analytic Airy form remains as `get_snr_airy`
  (deprecated) for cross-checks.
- **Exposure-time inversion**:
  `get_exptime_for_snr`.
- **PSF simulator and PSF-aware SNR**: `AiryPSF`,
  `DefocusPSF`, `CustomPSF`,
  `ImageSimulator`, and aperture optimization in
  `get_image_snr`.
- **Saturation flagging**: `get_peak_pixel` and
  `is_saturated`.
- **Parametric source spectra** in `get_scene`: `blackbody`,
  `flat` (`F_\nu`/`F_\lambda`), `powerlaw`, and `emission`;
  shape parameters are rebuildable via `update`.
- **Background spectrum support** and a selection of filters.
- **Convenience plotting** (`plotting`): matplotlib and bokeh
  variants of image, image-row, radial-profile, and encircled-energy plots,
  plus a shared house style (`set_wcc_style` / `WCC_STYLE`) and dispatch
  methods on `SimulatedImage`.
