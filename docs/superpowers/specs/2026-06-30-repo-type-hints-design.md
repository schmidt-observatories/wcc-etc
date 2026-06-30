# Repository-wide type hints (public-API-first)

**Date:** 2026-06-30
**Status:** Approved (design)
**Author:** Gummi + Claude

## Goal

Add inline type annotations across the `wcc_etc` package, prioritising the
public API surface, ahead of making the ETC public. Ship the package as
PEP 561-typed so downstream users and IDEs pick up the annotations. Wire up
`mypy` as an informational (non-blocking) CI check.

Type hints serve three purposes here: readability for new public users, IDE
autocomplete/inline docs, and catching real type errors via mypy — without
gating merges on incomplete third-party stubs.

## Decisions (locked)

- **Scope:** Public API first, then expand. Annotate the user-facing surface
  fully and well now; internal/private helpers only where a hint is obviously
  cheap and helpful.
- **Method:** Hand-written, module-by-module annotations (not auto-inferred,
  not `.pyi` stubs). Domain knowledge produces correct hints; auto-inference
  yields `Any`/wrong types for the astropy/synphot-heavy code.
- **Enforcement:** `mypy`, lenient config, **non-blocking** in CI.
- **Distribution:** Ship a `py.typed` marker (PEP 561).
- **Legacy excluded:** `src/wcc_etc/wcc_etc.py` (the deprecated legacy
  interface) is **not** annotated and is excluded from mypy.
- **Delivery:** Three PRs, one per phase.

## Scope definition

"Public API" = everything re-exported by `src/wcc_etc/__init__.py`
(`__all__` plus the explicit `from .module import ...` lines), and the public
classes / functions / methods of the modules those names live in.

### In scope, by phase

**Phase 1 — leaf modules** (few/no internal dependencies; types reused widely):
- `meta.py` — `_MetaHolder_` base class
- `telescope.py` — `Telescope`
- `sensor.py` — `Sensor`
- `astro.py` — `get_moon_magnitude`
- `utils.py` — `parse_element`, `parse_and_interpolate`, `list_of_quantity_to_array`
- `io.py` — public funcs: `read_config`, `get_sensor_config`,
  `get_pickles_spectrum_filename`, `resolve_bandpass`, `expand_path`,
  `get_any_astro_name`

**Phase 2 — core:**
- `scene.py` — `Scene`, `SceneElement`, `get_scene`, `get_scene_from_file`,
  `get_scene_element`, `broadcast_mapping`
- `psfsim.py` — `PSFSource`, `AiryPSF`, `DefocusPSF`, `CustomPSF`,
  `_ResampledPSF`, `SimulatedImage`, `ImageSimulator`, plus public functions
  (`normalize_psf`, `center_crop_or_pad`, `recenter`,
  `saturation_mask_from_image_e`, `howell_center`, `psf_center`, `apply_jitter`,
  `aperture_snr_radial`, `select_aperture`, `aperture_time_for_snr`,
  `solve_time_for_snr`, `calc_hwzm`, `calc_hwhm`). `FitsImg`/`FitsImgList`
  public methods annotated pragmatically.
- `simulation.py` — `Simulation` + public methods;
  `calculate_bg_normalization_magnitude`

**Phase 3 — presentation / domain:**
- `plotting.py` — all public `plot_*` functions, `set_wcc_style`
- `lightcurve.py` — `LightCurve`, `LightCurveSimulator`, `FluxModel`, `TransitModel`
- `airy.py` — public functions
- `radial_data.py` — `radial_data`, `calc_ee`

### Out of scope

- `src/wcc_etc/wcc_etc.py` — deprecated legacy interface. Left unannotated and
  excluded from mypy.
- Private (`_underscore`) helpers — annotated only when a hint is trivially
  cheap and clarifying; not required.

## Type conventions

- Add `from __future__ import annotations` at the top of every touched module.
  Annotations become lazy strings: no runtime evaluation, no runtime cost, no
  circular-import risk, and `Quantity`/`NDArray` subscripts are never evaluated
  at import time.
- Vocabulary:
  - astropy quantities → `astropy.units.Quantity`
  - numpy arrays → `numpy.typing.NDArray[np.float64]` for image/PSF/float data;
    `np.ndarray` where dtype genuinely varies
  - synphot objects → their real classes (`SourceSpectrum`, `SpectralElement`,
    `Observation`, …)
  - matplotlib → `matplotlib.axes.Axes`, `matplotlib.figure.Figure`
  - bokeh → real classes where stubs allow, else `Any`
  - pandas → `pd.DataFrame`, `pd.Series`
  - file paths → `str | os.PathLike[str]`
  - config dicts → `dict[str, Any]`
  - optionals → `X | None`
  - classmethod factories (`from_config`, `from_name`, …) → `Self`
    (`from typing import Self`, available in 3.11)
- **Pragmatic, not dogmatic:** where astropy/synphot internals get ugly,
  `Any` is acceptable rather than fighting incomplete upstream stubs. Correct
  and readable beats maximally precise.

## Tooling changes

### mypy config (`pyproject.toml`)

```toml
[tool.mypy]
python_version = "3.11"
files = ["src/wcc_etc"]
ignore_missing_imports = true
disallow_untyped_defs = false
check_untyped_defs = false
no_implicit_optional = true
warn_redundant_casts = true
exclude = ['src/wcc_etc/wcc_etc\.py$']
```

Effect: mypy checks what we annotate, stays silent about what we don't
(unannotated defs are not errors), never trips over third-party stub gaps, and
skips the legacy module entirely.

### Optional dependency

Add a `typecheck` extra under `[project.optional-dependencies]`:

```toml
typecheck = ["mypy"]
```

### py.typed (PEP 561)

- Add empty file `src/wcc_etc/py.typed`.
- Register it in `[tool.setuptools.package-data]` so it ships in the wheel
  (the section already exists and `include-package-data = true` is set).

### CI (`.github/workflows/tests.yml`)

Add a `mypy src/wcc_etc` step to the existing Tests job with
`continue-on-error: true` (informational, never blocks a merge). The existing
pytest gate is untouched. Install via `pip install -e .[typecheck]` (or
`pip install mypy`).

## Verification

- **Per module:** `mypy` reports no new errors for that file; the matching
  `tests/<domain>/` subdir passes.
- **End of each phase:** full `pytest -q` (the 552-test suite) stays green —
  this is the primary guard against runtime breakage introduced by annotations
  (forward refs, subscripting, import order). Run from repo root.
- **Packaging:** `python -c "import wcc_etc"` succeeds; a built wheel includes
  `wcc_etc/py.typed`.
- **CI note:** CI runs numpy 2.x / Python 3.11 (local is often numpy 1.26 /
  py313). Keep annotation-adjacent edits numpy-2 safe (no `np.trapz`, keep
  `dr`/`arange` args scalar) per the repo's CLAUDE.md.

## Delivery plan

Three PRs, each landable independently:

1. **PR 1 — Phase 1 + tooling.** Leaf modules annotated, plus mypy config,
   `py.typed`, `typecheck` extra, and the non-blocking CI step. (Tooling lands
   here so later phases get checked.)
2. **PR 2 — Phase 2.** Core modules (`scene`, `psfsim`, `simulation`).
3. **PR 3 — Phase 3.** Presentation/domain (`plotting`, `lightcurve`, `airy`,
   `radial_data`).

Each PR: annotate → `mypy` clean on touched files → full `pytest -q` green →
push branch → open PR → watch the `test` check before merge.

## Non-goals

- Annotating the legacy `wcc_etc.py`.
- A blocking/strict mypy gate (can tighten later once coverage is broad).
- Refactoring runtime behavior. This is annotations + tooling only; no logic
  changes beyond what an annotation strictly requires.
