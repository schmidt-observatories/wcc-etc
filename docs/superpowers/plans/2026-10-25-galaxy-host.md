# Galaxy Host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A SN at redshift z in an NGC host moved to that z, with the galaxy's own spectrum and absolute magnitude, rendered through the existing extended-source path.

**Architecture:** A new `wcc_etc.galaxy` module holds the data access, the ported projection routine and the cosmology. `SceneElement` gains `redshift` and `magabs` (spectral side) and an `ngc` spatial model next to `sersic` (dispatched by a new `extended.render_profile`). Plots reuse `plotting.plot_image_mpl`.

**Tech Stack:** numpy, astropy (cosmology Planck18, Sersic2D), synphot (`z`), `lazuli_astrosim_data` (new hard dependency).

**Spec:** `docs/superpowers/specs/2026-10-25-galaxy-host-design.md`

## Global Constraints

- No leading underscores on new function/method names.
- CI is numpy 2 / py3.11: no `np.trapz`, keep scalars scalar.
- Tests in domain subdirs, one assert per test, build scenes via `tests/helpers.py`.
- `lazuli-astrosim-data>=0.1.0` in `[project.dependencies]`.

## Review Focus

- `redshift` below the galaxy's own (0.0022 for NGC 0628): `galaxy_pixel_scale` raises ValueError (Task 1 test).
- `magabs` with a Vega `magsys`: the rest-frame magnitude uses the Vega spectrum (Task 3 test).
- A galaxy whose projected image spills past the grid: flux fraction < 1, never renormalised (Task 1 test).
- `profile="ngc0628"` on a `surface_brightness=True` element: `profile` raises (Task 4 test).
- `update(redshift=...)` rebuilds the spectrum and invalidates the magabs/normalised caches (Task 3 test).

---

### Task 1: `wcc_etc.galaxy` — data access, cosmology, projection

**Files:** Create `src/wcc_etc/galaxy.py`; Test `tests/scene/test_galaxy.py`.

**Produces:**
- `COSMOLOGY` (Planck18)
- `galaxy_filepath(name, which="spectrum") -> Path` (wraps `lazuli_astrosim_data.io.get_galaxy_filepath`)
- `load_galaxy_spectrum(name) -> SourceSpectrum` (lru_cache; Empirical1D from WAVELENGTH/FLUX, flam)
- `GalaxyImage` dataclass: `image` (sum 1), `pixel_scale` (arcsec), `redshift`, `center` (x, y 0-based); `load_galaxy_image(name) -> GalaxyImage` (lru_cache; nan→0, clip ≥ 0)
- `galaxy_pixel_scale(image_pixel_scale, galaxy_redshift, redshift, cosmology=COSMOLOGY) -> float`, raises ValueError if `redshift <= galaxy_redshift`
- `project_image(image, center, scale, shape, out_center, position=(0, 0), rotation=0, subsampling=8) -> ndarray` — slicersim `_image_projection` with the output centre explicit
- `render_galaxy(name, redshift, plate_scale_arcsec, npix, dx=0.0, dy=0.0, pa=0.0, center=None, cosmology=COSMOLOGY) -> ndarray (npix, npix)` flux fraction; `center` default `((npix-1)//2,)*2`; `dx`,`dy` arcsec; `pa` degrees CCW.

- [ ] Tests (class per function): spectrum loads with flam unit and >2000 points; image sums to 1; `galaxy_pixel_scale` at z=0.0022→0.9 equals `0.858981 * D_A(0.0022)/D_A(0.9)`; raises for z=0.001; `project_image` conserves flux on a big grid (`sum` within 1e-6 of 1 for an identity scale), a 90° rotation equals `np.rot90` of the unrotated projection (symmetric synthetic image), and a far-off `position` yields < 1 total; `render_galaxy` at dx=dy=0 peaks within 1 px of `grid_center` for NGC 0628 at z=0.9 and dx=0.5" moves the centroid by `0.5/plate` px.
- [ ] Implement; `pytest tests/scene/test_galaxy.py -q`; commit `feat: galaxy module (NGC data, projection, cosmology)`.

### Task 2: `extended.render_profile` dispatch + `extended_rate_image`

**Files:** Modify `src/wcc_etc/extended.py`, `src/wcc_etc/simulation.py:910-943`; Test `tests/imaging/test_galaxy_host.py`.

**Produces:** `render_profile(profile, plate_scale_arcsec, npix, oversample=11, center=None, total=True) -> ndarray`: `profile["model"] == "sersic"` → `render_sersic`; `"ngc"` → `galaxy.render_galaxy(profile["name"], profile["redshift"], ...)`. `extended_rate_image` calls `render_profile`.

- [ ] Tests: `render_profile` with a sersic dict equals `render_sersic` (array_equal); with an ngc dict equals `render_galaxy`; a scene with an NGC host gives `extended_rate_image` summing to ≈ host rate × on-grid fraction (`isclose`, rtol 1e-6) once Task 4 lands (write it then).
- [ ] Commit `feat: render_profile dispatch for sersic and ngc spatial models`.

### Task 3: `SceneElement` redshift + magabs

**Files:** Modify `src/wcc_etc/scene.py` (`SceneElement.__init__`, `_mutable_parameters`, `set_spectrum`, `update`, `_parse_mag_`, new `absolute_to_observed_mag`); Test `tests/scene/test_redshift.py`.

**Produces:**
- `SceneElement(spectrum, mag, magsys=None, bandpass="johnson_v", surface_brightness=False, redshift=None, magabs=None, meta={})`.
- `set_spectrum` sets `spectrum.z = redshift` (0 if None). NGC names load via `galaxy.load_galaxy_spectrum`.
- `update(redshift=...)` rebuilds the spectrum (redshift is a shape parameter).
- `absolute_to_observed_mag(magabs) -> Quantity`: rest_mag = effstim of the z=0 template in `band`/`magsys`; ratio = 10^(-0.4(magabs + distmod(z) - rest_mag))/(1+z); observed = effstim(template_z * ratio); single-entry cache keyed on (magabs, z, band, spectrum).
- `_parse_mag_`: `mag is None and magabs is not None` → `absolute_to_observed_mag(magabs)`.

- [ ] Tests: `redshift=0.9` scales `spectrum.waveset.max()` by 1.9; `update(redshift=0.5)` changes it; `magabs` reproduces slicersim: for a magabs=-19.3 hsiao07 at z=0.9 in sdss_r/AB, the rest-frame absmag recovered as in slicersim's `_rest_frame_absmag` equals -19.3 within 1e-3; Vega magsys path runs and differs from AB; `get_spectrum()` is cached across calls and misses after `update(magabs=-20)`.
- [ ] Commit `feat: redshift and absolute magnitude on scene elements`.

### Task 4: NGC spatial model on `SceneElement` + `get_scene` shortcuts

**Files:** Modify `src/wcc_etc/scene.py` (`_PROFILE_PARAMS`, `profile`, `get_scene_element`, `get_scene`); Test `tests/scene/test_galaxy_scene.py`.

**Produces:**
- `_PROFILE_PARAMS = {"sersic": ["r_eff", "r_kpc", *SERSIC_DEFAULTS], "ngc": ["dx", "dy", "pa"]}`; `profile` returns `{"model": "sersic", ...}` (r_eff from r_kpc × `arcsec_per_kpc_proper(z)` when r_eff is None) or `{"model": "ngc", "name": "ngc0628", "redshift": z, "dx", "dy", "pa"}`; NGC + surface_brightness → ValueError; NGC without redshift → ValueError.
- `get_scene_element`: an NGC `name` defaults `{"mag": None, "magabs": -21, "bandpass": "sdss_r", "magsys": "abmag", "profile": name}`.
- `get_scene(..., redshift=z)`: `host_prop.setdefault("redshift", z)`.

- [ ] Tests: sersic `r_kpc=5, redshift=0.9` → `profile["r_eff"] == 5*arcsec_per_kpc_proper(0.9)`; ngc profile dict; the two ValueErrors; `get_scene("hsiao07", mag=None, magabs=-19.3, redshift=0.9, bandpass="sdss_r", magsys="abmag", host="ngc0628")` has host redshift 0.9 and host magabs -21; Task 2's `extended_rate_image` test.
- [ ] Commit `feat: NGC galaxy hosts and r_kpc Sersic sizes in scenes`.

### Task 5: Plots

**Files:** Modify `src/wcc_etc/scene.py` (`SceneElement.show_image`), `src/wcc_etc/plotting.py` (`plot_host`), `src/wcc_etc/simulation.py` (`Simulation.plot_host`); Test `tests/plotting/test_host_plots.py`.

**Produces:** `SceneElement.show_image(plate_scale_arcsec, npix=256, ax=None, **kw) -> (fig, ax)` via `render_profile` + `plot_image_mpl(stretch="log")`; `plotting.plot_host(sim, time, *, psf=None, seed=0, npix=None, units="mas", axes=None, **kw) -> (fig, axes)`: left `ImageSimulator(sim, npix).simulate(time, psf, seed)` noisy frame, right `extended_rate_image * time`.

- [ ] Tests (Agg): both return Axes; `plot_host` axes length 2; an element without a profile raises ValueError in `show_image`.
- [ ] Commit `feat: host image plots`.

### Task 6: Dependency, docs, notebook

**Files:** Modify `pyproject.toml`, `README.md`, `docs/changelog.md`, `docs/sphinx/tutorials.rst`; Create `notebooks/08_galaxy_host.ipynb`.

- [ ] Add `"lazuli-astrosim-data>=0.1.0"` to dependencies; README "A supernova in its host galaxy" section mirroring slicersim's; changelog "Unreleased" entry; notebook built with the notebook-demo skill (SN Ia at z=0.9 in NGC 0628: host image, simulated frame, SNR with/without host).
- [ ] `pytest -q`; ruff F + format; commit `docs: galaxy host tutorial, README and changelog`; push; PR.
