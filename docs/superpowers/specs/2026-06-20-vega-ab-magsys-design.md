# Vega / AB magnitude system support (modern scene path)

Date: 2026-06-20
Status: Draft for review

## Problem

Magnitudes in the modern ETC path (`scene.py` + `simulation.py`) are effectively
AB-only, and the `magsys` parameter that `SceneElement` already exposes is broken
for Vega in two independent ways:

1. `_parse_mag_` (scene.py:~603) resolves the magsys string with
   `getattr(u, magsys)` where `u` is `astropy.units`. `astropy.units` has **no**
   `VEGAMAG` (it lives in `synphot.units`, imported as `su`). So
   `magsys="VEGAMAG"` raises `AttributeError` before any normalization happens.
2. `get_spectrum` (scene.py:~506) calls `self.spectrum.normalize(mag, band=...)`
   without a `vegaspec=` argument, which synphot **requires** to normalize in
   `VEGAMAG`. So even if (1) were fixed, Vega normalization would still fail.

The legacy `wcc_etc.py` interface already supports both systems and is **out of
scope**.

## Goal

Make the magnitude system selectable end-to-end in the modern path via a
`magsys` string accepting `"abmag"` and `"vegamag"` (case-insensitive), with
**`vegamag` as the new default everywhere** — source, host, and the built-in
zodi/background element. Add careful tests verifying the expected, quantitative
flux differences between the two systems.

### Science note (intentional behavior change)

Flipping the default from AB to Vega re-interprets every magnitude that does not
explicitly set `magsys`, **including the built-in zodi/background default**
(`mag=22.5` in `get_scene_element`). Per decision, the background also defaults to
`vegamag` — i.e. 22.5 is now read as Vega mag/arcsec². Existing callers/notebooks
that relied on the implicit AB default will change numerically unless they pass
`magsys="abmag"`. This is accepted and called out here so it is not a surprise.

## Design

All changes are in `src/wcc_etc/scene.py` plus one deletion in
`src/wcc_etc/simulation.py` and new tests in `tests/test_scene.py`.

### 1. Magsys string resolution (`scene.py`)

Add a single canonical resolver used by `_parse_mag_`:

- A small case-insensitive mapping: `{"abmag": u.ABmag, "vegamag": su.VEGAMAG}`.
- Unit objects passed directly (anything with `is_equivalent`) pass through
  unchanged, preserving the current escape hatch.
- An unrecognized string raises a clear `ValueError` listing the valid options
  (`"abmag"`, `"vegamag"`), rather than the current opaque `AttributeError`.

`_parse_mag_` uses this resolver instead of `getattr(u, magsys)`.

### 2. Default magsys → `vegamag`

- `SceneElement.__init__` default changes `magsys="ABmag"` → `magsys="vegamag"`.
  Docstring updated to state the default and the accepted values.
- `get_scene_element` builds its `default_config` without an explicit `magsys`
  key, so both the zodi/background config and the generic source config inherit
  the new `SceneElement` default (`vegamag`). No per-element override is added —
  this is what makes the default uniform across source/host/background.

### 3. Vega-aware normalization (`scene.py`)

- Add a module-level, lazily-cached `_get_vega()` wrapping
  `SourceSpectrum.from_vega()` (the same call the legacy path already uses) so the
  reference spectrum is loaded at most once per process.
- In `get_spectrum`, when the resolved magnitude's unit is `su.VEGAMAG`, pass
  `vegaspec=_get_vega()` to `normalize`. For AB (and any non-Vega unit), behavior
  is unchanged (no `vegaspec`). Detection is by comparing the mag `Quantity`'s
  unit to `su.VEGAMAG`.

`get_observation` needs no change — it delegates to `get_spectrum`.

### 4. Remove dead AB-hardcoded code (`simulation.py`)

`Simulation._get_spectrum_observation(spectrum, abmag, bandpass)` (simulation.py:~538)
hardcodes `abmag * u.ABmag` but has **zero callers** (the live count-rate path is
`_count_rate_components → scene.get_observation → SceneElement.get_spectrum`).
Delete it so the only normalization path is the magsys-aware one in `scene.py`.

## What needs no change

- `get_scene` / `get_scene_from_file` / `get_scene_element` already forward
  `magsys` through `**kwargs` → `SceneElement.from_config`. Once the bugs above
  are fixed, `get_scene(name, mag, magsys="abmag")` works without signature
  changes.
- `get_image_snr`'s `mags` sweep is a **relative** flux scale,
  `10**(-0.4*(mags - m0))`; the magnitude system cancels, so the sweep inherits
  the source's `magsys` automatically. No change, but the docstring will note
  that swept `mags` are in the source's own system.

## Testing (`tests/test_scene.py`)

Tests must verify *quantitative* AB↔Vega differences, not just "it runs":

1. **Default is vegamag.** `SceneElement(spectrum, mag=15)` with no `magsys` →
   `element.mag.unit is su.VEGAMAG`. `get_scene(...).source.mag.unit` likewise,
   and the zodi/background element's `mag.unit is su.VEGAMAG`.

2. **Alias / case-insensitivity.** `"abmag"`, `"ABMAG"`, `"AbMag"` → `u.ABmag`;
   `"vegamag"`, `"VEGAMAG"` → `su.VEGAMAG`; a bogus string raises `ValueError`.

3. **Cross-system flux offset (the careful one).** For a fixed band B and a fixed
   spectrum S, normalize S to the *same numeric* magnitude `m` in both systems and
   measure the in-band flux (via a synphot `Observation` effstim) of each. The
   ratio `flux_vega / flux_ab` must equal `10**(-0.4 * offset_B)` to tight
   tolerance, where `offset_B` (band B's AB−Vega offset) is computed in-test
   directly from synphot's own Vega spectrum — no hardcoded magic constants.
   Run this for **Johnson V** and a redder synphot-bundled band (e.g.
   **johnson_k**, which `SpectralElement.from_filter` supports — 2MASS is not
   bundled) to demonstrate the offset is band-dependent and larger in the IR.

4. **Round-trip self-consistency.** A source built with `magsys="vegamag"`,
   `mag=m0`, whose synthetic Vega magnitude (effstim in vegamag through the same
   band) recovers `m0` within tolerance; same for `magsys="abmag"`.

5. **AB path unchanged.** An explicit `magsys="abmag"` source produces the same
   normalized flux as the pre-change AB default (guard against regressions in the
   AB branch).

Existing `tests/test_scene.py` / `tests/test_snr.py` assertions that assume the
old AB default will be updated to pass `magsys="abmag"` (or have expected values
recomputed) as part of this work — these updates will be enumerated in the
implementation plan after auditing which tests depend on the default.

## Out of scope

- Legacy `wcc_etc.py` (already supports both systems).
- Loose aliases beyond `abmag`/`vegamag` (e.g. `"vega"`, `"AB"`).
- Module-level magsys configuration or automatic per-band Vega selection.
