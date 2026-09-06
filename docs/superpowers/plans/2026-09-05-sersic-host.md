# Sersic-Profile Host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scene `host` can carry a Sersic surface-brightness profile (angular parameters) that is rendered, PSF-convolved, and counted consistently by every image / SNR / saturation API.

**Architecture:** A new standalone module `wcc_etc/extended.py` renders a normalized Sersic profile on the detector grid. `SceneElement` gains a `profile` property (validated kwargs from meta). `Simulation._count_rate_components` classifies profiled elements as `extended`; one helper `Simulation._extended_rate_image` turns them into a PSF-convolved e-/s/pix image that `ImageSimulator.simulate`, `_per_frame_clean_image_e`, `get_peak_pixel`, `get_image_snr` and `get_image_exptime_for_snr` all add. The aperture functions accept a 2D diffuse image.

**Tech Stack:** numpy, scipy (`gammaincinv`, `gamma`, `fftconvolve`), `astropy.modeling.models.Sersic2D`, pytest. No new dependencies. No `wcc_sim` or `astropylib` imports.

**Spec:** `docs/superpowers/specs/2026-09-05-sersic-host-design.md`

## Global Constraints

- No leading underscores on **new** function names (CLAUDE.md): the new Simulation
  helper is `extended_rate_image` (the spec's draft name `_extended_rate_image` is
  superseded); `diffuse_enclosed`, `render_sersic`, `sersic_total_over_amplitude`
  likewise. Existing `_count_rate_components` etc. keep their names.
- Never `import astropylib` or `import wcc_sim` from `src/wcc_etc/`.
- CI is numpy 2.x / Python 3.11: keep `arange`/reshape args scalar ints; no `np.trapz`.
- Tests: one assert per test, subdirectory packages, build scenes via `tests/helpers.py`.
- Ponytail mode: minimum code that works; mark cut corners with `# ponytail:`.
- Run `pytest -q` from the repo root before every commit; the full suite must stay green.

---

### Task 1: `extended.py` — analytic normalization and grid render

**Files:**
- Create: `src/wcc_etc/extended.py`
- Test: `tests/imaging/test_extended.py`

**Interfaces:**
- Produces:
  - `SERSIC_DEFAULTS: dict = {"n": 1.0, "ellip": 0.0, "pa": 0.0, "dx": 0.0, "dy": 0.0}`
  - `sersic_total_over_amplitude(n: float, r_eff_pix: float, ellip: float = 0.0) -> float`
  - `render_sersic(profile: dict, plate_scale_arcsec: float, npix: int, oversample: int = 11, center: tuple | None = None, total: bool = True) -> np.ndarray` (shape `(npix, npix)`)

- [ ] **Step 1: Write the failing tests**

```python
"""Sersic host rendering: analytic normalization and grid placement."""

import numpy as np
import pytest
from astropy.modeling.models import Sersic2D

from wcc_etc.extended import render_sersic, sersic_total_over_amplitude

PLATE = 0.016869  # arcsec/pix, sony/zwo IMX455 on the WCC


class TestTotalOverAmplitude:
    """The analytic F/I_e matches a brute-force numeric integral."""

    @pytest.mark.parametrize("n", [1.0, 4.0])
    def test_matches_numeric_integral(self, n):
        """2 pi n r_e^2 (1-e) e^bn bn^-2n Gamma(2n) == sum of Sersic2D on a fine grid."""
        r_eff, ellip = 20.0, 0.3
        x = np.arange(-1500, 1501, dtype=float)
        num = Sersic2D(amplitude=1.0, r_eff=r_eff, n=n, ellip=ellip)(x[None, :], x[:, None]).sum()
        assert sersic_total_over_amplitude(n, r_eff, ellip) == pytest.approx(num, rel=2e-3)


class TestRenderSersic:
    """render_sersic places a normalized profile on the detector grid."""

    def test_total_mode_sums_to_one_when_contained(self):
        """A compact n=1 profile well inside the grid carries (almost) unit flux."""
        img = render_sersic({"r_eff": 0.1}, PLATE, 128)
        assert img.sum() == pytest.approx(1.0, rel=1e-2)

    def test_total_mode_loses_light_off_grid(self):
        """A profile larger than the grid keeps less than unit flux — no renormalizing."""
        img = render_sersic({"r_eff": 5.0}, PLATE, 128)
        assert img.sum() < 0.5

    def test_sb_mode_is_one_at_r_eff(self):
        """total=False returns I/I_e: the pixel at r_eff along +x reads ~1."""
        img = render_sersic({"r_eff": 0.5, "n": 1.0}, PLATE, 128, total=False)
        c = (128 - 1) / 2
        assert img[int(round(c)), int(round(c + 0.5 / PLATE))] == pytest.approx(1.0, rel=5e-2)

    def test_dx_shifts_peak(self):
        """dx (arcsec) moves the brightest pixel by dx / plate_scale pixels in x."""
        img = render_sersic({"r_eff": 0.1, "dx": 0.3}, PLATE, 128)
        _, ix = np.unravel_index(np.argmax(img), img.shape)
        assert ix == pytest.approx(63.5 + 0.3 / PLATE, abs=1.0)

    def test_center_moves_profile(self):
        """center=(cx, cy) is the source position the profile is attached to."""
        img = render_sersic({"r_eff": 0.1}, PLATE, 128, center=(40.0, 90.0))
        iy, ix = np.unravel_index(np.argmax(img), img.shape)
        assert (ix, iy) == (40, 90)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/imaging/test_extended.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'wcc_etc.extended'`

- [ ] **Step 3: Write the module**

```python
"""Extended (spatially resolved) scene elements: Sersic-profile hosts.

Angular parameterization only — ``r_eff``, ``dx``, ``dy`` in arcsec, ``pa`` in
degrees CCW from +x — the convention shared with wcc-sim's SersicComponent,
Pandeia and the HST ETC. Distance / redshift are deliberately not modelled.
"""

import numpy as np
from astropy.modeling.models import Sersic2D
from scipy.special import gamma, gammaincinv

__all__ = ["SERSIC_DEFAULTS", "sersic_total_over_amplitude", "render_sersic"]

SERSIC_DEFAULTS = {"n": 1.0, "ellip": 0.0, "pa": 0.0, "dx": 0.0, "dy": 0.0}


def sersic_total_over_amplitude(n, r_eff_pix, ellip=0.0):
    """F_total / I_e for a Sersic2D profile with ``r_eff`` in pixels (analytic).

    Sersic2D's ``amplitude`` is the surface brightness at r_eff per pixel area;
    integrating over the plane gives 2 pi n r_eff^2 (1-ellip) e^bn bn^(-2n) Gamma(2n).
    """
    bn = float(gammaincinv(2.0 * n, 0.5))
    return float(
        2.0 * np.pi * n * r_eff_pix**2 * (1.0 - ellip)
        * np.exp(bn) * bn ** (-2.0 * n) * gamma(2.0 * n)
    )


def render_sersic(profile, plate_scale_arcsec, npix, oversample=11, center=None, total=True):
    """Sersic profile on an (npix, npix) detector grid, attached to ``center``.

    ``profile`` holds ``r_eff`` (arcsec, required) and any of SERSIC_DEFAULTS.
    ``center`` is the source position (cx, cy) the ``dx``/``dy`` offset is
    measured from; default is the grid centre, as in ImageSimulator.
    Returns the per-pixel profile: unit total flux (analytic, so light off the
    grid is lost, not renormalized) when ``total`` is True, else I/I_e.
    """
    p = SERSIC_DEFAULTS | profile
    if center is None:
        center = ((npix - 1) / 2.0, (npix - 1) / 2.0)
    r_eff_pix = p["r_eff"] / plate_scale_arcsec
    amp = 1.0 / sersic_total_over_amplitude(p["n"], r_eff_pix, p["ellip"]) if total else 1.0
    model = Sersic2D(
        amplitude=amp, r_eff=r_eff_pix, n=p["n"],
        x_0=center[0] + p["dx"] / plate_scale_arcsec,
        y_0=center[1] + p["dy"] / plate_scale_arcsec,
        ellip=p["ellip"], theta=np.radians(p["pa"]),
    )
    xx = np.arange(npix, dtype=float)
    img = model(xx[None, :], xx[:, None])  # pixel centres, native grid

    # A Sersic cusp changes across a pixel; the rest of the profile does not.
    # Re-evaluate a core box on the oversampled sub-grid and pixel-average it.
    # ponytail: box half-width capped at 64 px (wcc-sim's choice); beyond that
    # the profile is smooth at pixel scale. Raise the cap if r_eff < 3 px ever
    # matters at large npix.
    half = int(np.clip(np.ceil(2.0 * r_eff_pix), 8, 64))
    x0, y0 = int(round(model.x_0.value)), int(round(model.y_0.value))
    xlo, xhi = max(x0 - half, 0), min(x0 + half + 1, npix)
    ylo, yhi = max(y0 - half, 0), min(y0 + half + 1, npix)
    if xlo < xhi and ylo < yhi:
        off = (np.arange(oversample) + 0.5) / oversample - 0.5
        fx = (np.arange(xlo, xhi, dtype=float)[:, None] + off).ravel()
        fy = (np.arange(ylo, yhi, dtype=float)[:, None] + off).ravel()
        fine = model(fx[None, :], fy[:, None])
        img[ylo:yhi, xlo:xhi] = fine.reshape(
            yhi - ylo, oversample, xhi - xlo, oversample
        ).mean(axis=(1, 3))
    return img
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/imaging/test_extended.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/extended.py tests/imaging/test_extended.py
git commit -m "feat: Sersic profile renderer (wcc_etc.extended)"
```

---

### Task 2: `SceneElement.profile` and mutable profile parameters

**Files:**
- Modify: `src/wcc_etc/scene.py` (imports ~line 16; `_SPECTRUM_PARAMS` ~line 221; `SceneElement.mutable_parameters` ~line 779; properties block ~line 814)
- Test: `tests/scene/test_profile.py`

**Interfaces:**
- Consumes: `wcc_etc.extended.SERSIC_DEFAULTS`
- Produces:
  - `scene._PROFILE_PARAMS = {"sersic": ["r_eff", "n", "ellip", "pa", "dx", "dy"]}`
  - `SceneElement.profile -> dict | None` — defaults filled, validated; `None` when meta has no `"profile"`.
  - `Scene.call_down("profile", as_dict=True)` therefore works with no further change.

- [ ] **Step 1: Write the failing tests**

```python
"""SceneElement.profile: spatial profile kwargs carried in meta."""

import pytest

import wcc_etc
from tests.helpers import make_scene


def sersic_scene(**profile):
    """Standard scene with a mag-17 G5V host carrying a sersic profile."""
    return make_scene(
        host="G5V",
        host_prop={"mag": 17, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 1.0} | profile,
    )


class TestProfileProperty:
    def test_no_profile_is_none(self):
        """A plain element has profile None (point or uniform SB, unchanged behaviour)."""
        assert make_scene().source.profile is None

    def test_defaults_filled(self):
        """Only r_eff was given: n, ellip, pa, dx, dy come from SERSIC_DEFAULTS."""
        assert sersic_scene().host.profile == {
            "r_eff": 1.0, "n": 1.0, "ellip": 0.0, "pa": 0.0, "dx": 0.0, "dy": 0.0,
        }

    def test_given_values_kept(self):
        """Explicit kwargs override the defaults."""
        assert sersic_scene(n=4, ellip=0.3, pa=45, dx=0.8).host.profile["n"] == 4

    @pytest.mark.parametrize("bad", [{"n": 0}, {"ellip": 1.0}, {"r_eff": -1}])
    def test_invalid_values_raise(self, bad):
        """n > 0, 0 <= ellip < 1, r_eff > 0 are enforced."""
        with pytest.raises(ValueError):
            sersic_scene(**bad).host.profile

    def test_unknown_profile_raises(self):
        """Only 'sersic' is a known profile."""
        el = wcc_etc.get_scene_element("G5V", mag=17, profile="gaussian", r_eff=1.0)
        with pytest.raises(ValueError):
            el.profile

    def test_missing_r_eff_raises(self):
        """A sersic profile without r_eff is an error."""
        el = wcc_etc.get_scene_element("G5V", mag=17, profile="sersic")
        with pytest.raises(ValueError):
            el.profile


class TestMutableProfile:
    def test_r_eff_is_mutable(self):
        """Profile shape parameters are updatable like blackbody teff."""
        assert "r_eff" in sersic_scene().host.mutable_parameters

    def test_update_changes_profile(self):
        """scene.update(host__r_eff=...) flows into profile."""
        scene = sersic_scene()
        scene.update(host__r_eff=2.5)
        assert scene.host.profile["r_eff"] == 2.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scene/test_profile.py -q`
Expected: FAIL with `AttributeError: 'SceneElement' object has no attribute 'profile'`

- [ ] **Step 3: Implement**

In `src/wcc_etc/scene.py`, after `from .meta import _MetaHolder_` add:

```python
from .extended import SERSIC_DEFAULTS
```

After the `_SPECTRUM_PARAMS` dict add:

```python
# Spatial-profile shape parameters, mutable like the spectrum shape parameters.
_PROFILE_PARAMS = {"sersic": ["r_eff", *SERSIC_DEFAULTS]}
```

In `SceneElement.mutable_parameters`, after the `params += _SPECTRUM_PARAMS...` line add:

```python
        params += _PROFILE_PARAMS.get(self.meta.get("profile"), [])
```

After the `mag_is_surface_brightness` property add:

```python
    @property
    def profile(self):
        """
        Spatial profile parameters (defaults filled), or None for a point /
        uniform-surface-brightness element. Only ``"sersic"`` is supported;
        ``r_eff`` (arcsec) is required, see :mod:`wcc_etc.extended`.
        """
        name = self.meta.get("profile")
        if name is None:
            return None
        if name not in _PROFILE_PARAMS:
            raise ValueError(f"unknown profile {name!r}; supported: {list(_PROFILE_PARAMS)}")
        if self.meta.get("r_eff") is None:
            raise ValueError("a 'sersic' profile requires r_eff (arcsec)")
        p = SERSIC_DEFAULTS | {k: self.meta[k] for k in _PROFILE_PARAMS[name] if k in self.meta}
        if not (p["r_eff"] > 0 and p["n"] > 0 and 0 <= p["ellip"] < 1):
            raise ValueError(
                f"invalid sersic profile: need r_eff > 0, n > 0, 0 <= ellip < 1; got {p}"
            )
        return p
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scene -q`
Expected: all pass (new 9 + existing)

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/scene.py tests/scene/test_profile.py
git commit -m "feat: SceneElement.profile with mutable sersic parameters"
```

---

### Task 3: `Simulation` — extended components and the shared rate image

**Files:**
- Modify: `src/wcc_etc/simulation.py` (`_count_rate_components` ~634–705; `_image_render_bundle` ~885–924; `_per_frame_clean_image_e` ~926–945; `get_peak_pixel` ~786–839; `_countrates_in_aperture` ~573)
- Test: `tests/simulation/test_extended.py`

**Interfaces:**
- Consumes: `SceneElement.profile`, `wcc_etc.extended.render_sersic`, `psfsim.center_crop_or_pad`
- Produces:
  - `_count_rate_components()["extended"]: list[tuple[float, dict, bool]]` — `(rate, profile, is_surface_brightness)`; `rate` is total e-/s when `is_surface_brightness` is False, else e-/s/pix at μ_e.
  - `Simulation.extended_rate_image(psf_norm, ctx, comps=None) -> np.ndarray` `(ctx.npix, ctx.npix)` e-/s/pix, PSF-convolved.
  - `_image_render_bundle(...)["extended_rate_image"]` — same array, cached.

- [ ] **Step 1: Write the failing tests**

```python
"""Extended (Sersic) host: classification and the shared rate image."""

import numpy as np
import pytest

import wcc_etc
from tests.helpers import make_scene, make_simulation

HOST = {"mag": 17, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 0.1}


def sersic_sim(source_mag=20, **host):
    """G5V source on sony:r with a compact (0.1") G5V Sersic host."""
    scene = make_scene(mag=source_mag, host="G5V", host_prop=HOST | host)
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


@pytest.fixture
def ext_sim():
    return sersic_sim()


class TestClassification:
    def test_profiled_host_is_extended(self, ext_sim):
        """A host with a profile lands in 'extended', not diffuse or contaminant."""
        assert len(ext_sim._count_rate_components()["extended"]) == 1

    def test_profiled_host_is_not_a_contaminant(self, ext_sim):
        """It must not also ride the source PSF (double counting)."""
        assert ext_sim._count_rate_components()["contaminant_rate_total"] == 0

    def test_profiled_sb_host_is_not_diffuse(self):
        """A mu_e (surface_brightness=True) host is extended, not uniform diffuse."""
        sim = sersic_sim(surface_brightness=True, mag=22)
        assert sim._count_rate_components()["diffuse_rate_per_pix"] == 0

    def test_no_profile_gives_empty_list(self):
        """Scenes without a profile are unchanged."""
        assert make_simulation()._count_rate_components()["extended"] == []


class TestExtendedRateImage:
    def test_total_mag_conserves_flux(self, ext_sim):
        """A contained total-mag host image sums to the point-source rate at that mag."""
        b = ext_sim._image_render_bundle(ext_sim.default_psf, None, 128, 11)
        ref = make_simulation(mag=17)._count_rate_components()["source_rate_total"]
        assert b["extended_rate_image"].sum() == pytest.approx(ref, rel=1e-2)

    def test_sb_mode_matches_uniform_sb_at_r_eff(self):
        """mu_e mode: the pixel at r_eff reads the uniform-SB per-pixel rate."""
        sim = sersic_sim(surface_brightness=True, mag=22, r_eff=0.5, n=1.0)
        b = sim._image_render_bundle(sim.default_psf, None, 128, 11)
        uniform = make_scene(host="G5V", host_prop={"mag": 22, "bandpass": "johnson_r", "surface_brightness": True})
        ref = wcc_etc.Simulation.from_sensor_and_scene("sony:r", uniform)._count_rate_components()["diffuse_rate_per_pix"]
        c = int(round((128 - 1) / 2))
        r_pix = int(round(0.5 / (b["plate_scale_mas"] / 1000.0)))
        assert b["extended_rate_image"][c, c + r_pix] == pytest.approx(ref, rel=5e-2)

    def test_dx_shifts_peak(self):
        """dx moves the host peak by dx / plate_scale pixels."""
        sim = sersic_sim(dx=0.3)
        b = sim._image_render_bundle(sim.default_psf, None, 128, 11)
        _, ix = np.unravel_index(np.argmax(b["extended_rate_image"]), (128, 128))
        assert ix == pytest.approx(63.5 + 0.3 / (b["plate_scale_mas"] / 1000.0), abs=1.5)

    def test_zero_image_without_profile(self):
        """No extended element -> an all-zero image of the right shape."""
        sim = make_simulation()
        b = sim._image_render_bundle(sim.default_psf, None, 64, 11)
        assert b["extended_rate_image"].shape == (64, 64) and not b["extended_rate_image"].any()


class TestOneBudget:
    def test_per_frame_image_includes_host(self, ext_sim):
        """_per_frame_clean_image_e carries the extended charge."""
        b = ext_sim._image_render_bundle(ext_sim.default_psf, None, 128, 11)
        hostless = make_simulation(mag=20)
        bh = hostless._image_render_bundle(hostless.default_psf, None, 128, 11)
        diff = ext_sim._per_frame_clean_image_e(b, 10.0) - hostless._per_frame_clean_image_e(bh, 10.0)
        assert diff.sum() == pytest.approx(10.0 * b["extended_rate_image"].sum(), rel=1e-6)

    def test_peak_pixel_is_image_max(self, ext_sim):
        """get_peak_pixel equals the brightest pixel of the per-frame clean image."""
        b = ext_sim._image_render_bundle(ext_sim.default_psf, None, 128, 11)
        expect = ext_sim._per_frame_clean_image_e(b, 10.0).max()
        assert ext_sim.get_peak_pixel(10.0, units="e-").value == pytest.approx(expect, rel=1e-9)


class TestDeprecatedPathWarns:
    def test_airy_path_warns_profile_ignored(self, ext_sim):
        """The analytic Airy path cannot place a profile; it says so."""
        with pytest.warns(UserWarning, match="profile"):
            ext_sim._countrates_in_aperture()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/simulation/test_extended.py -q`
Expected: FAIL with `KeyError: 'extended'`

- [ ] **Step 3: Implement in `simulation.py`**

Add to the module imports: `from scipy.signal import fftconvolve`.

In `_count_rate_components`, after `sb_flags = ...` add `profiles = self.scene.call_down("profile", as_dict=True)` and `extended = []`; inside the loop, right after computing `rate`, insert before the `if sb_flags...` branch:

```python
            if profiles.get(name) is not None:
                # Spatially resolved element: rendered by extended_rate_image.
                # rate is total e-/s (mag) or e-/s/pix at mu_e (surface brightness).
                extended.append((rate, profiles[name], bool(sb_flags.get(name, False))))
                continue
```

Add `"extended": extended,` to the returned dict and document it in the docstring:

```
            ``extended`` : list of (rate, profile, is_surface_brightness) for
                elements with a spatial profile (e.g. a Sersic host). See
                extended_rate_image.
```

Add the helper right before `_image_render_bundle`:

```python
    def extended_rate_image(self, psf_norm, ctx, comps=None):
        """PSF-convolved e-/s/pix image of the profiled (extended) elements.

        Profiles are rendered on a grid padded by ``npix`` (even) and cropped
        back, so light from just outside the detector grid still convolves
        inward — no edge loss. ``dx``/``dy`` are measured from ``ctx.center``
        (the source). Zeros when the scene has no extended element.
        """
        from .extended import render_sersic
        from .psfsim import center_crop_or_pad

        if comps is None:
            comps = self._count_rate_components()
        npix = ctx.npix
        image = np.zeros((npix, npix))
        if not comps["extended"]:
            return image
        pad = 2 * (npix // 2)
        big = npix + pad
        cx, cy = ctx.center if ctx.center is not None else ((npix - 1) / 2.0,) * 2
        center = (cx + pad / 2, cy + pad / 2)
        for rate, profile, is_sb in comps["extended"]:
            image_big = rate * render_sersic(
                profile, ctx.plate_scale_mas / 1000.0, big, ctx.oversample,
                center=center, total=not is_sb,
            )
            image += center_crop_or_pad(fftconvolve(image_big, psf_norm, mode="same"), npix)
        return image
```

In `_image_render_bundle`, after `comps = self._count_rate_components()` add
`"extended_rate_image": self.extended_rate_image(psf_norm, ctx, comps),` to the bundle dict.

In `_per_frame_clean_image_e`, add `+ b["extended_rate_image"] * tf` to the returned sum and mention it in the docstring ("source + unresolved contaminants (both through the PSF) + extended profiles + background + diffuse + dark").

In `get_peak_pixel`, replace the `peak_rate = (...)` expression and its comment with:

```python
        # Brightest pixel of the shared per-frame budget (source+contaminant PSF,
        # extended profiles, sky, diffuse, dark) at unit time — one image, so an
        # offset host and the source peak are not naively summed. Linear in time.
        peak_rate = float(self._per_frame_clean_image_e(b, 1.0).max())
```

and delete the now-unused `dark_rate_per_pix` lines in that method. Update the docstring formula to "Peak = max over the per-frame clean image (see _per_frame_clean_image_e)".

In `_countrates_in_aperture`, after `if scene is None: scene = self.scene` add:

```python
        if any(p is not None for p in scene.call_down("profile")):
            warnings.warn(
                "spatial profiles are ignored on the analytic Airy path; "
                "use get_snr / get_image_snr for a Sersic host.",
                stacklevel=2,
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/simulation tests/imaging -q`
Expected: all pass. (`test_peak_pixel*` existing tests must still pass — peak for a point source is unchanged because `_per_frame_clean_image_e` and the old formula agree when everything is co-located.)

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/simulation.py tests/simulation/test_extended.py
git commit -m "feat: classify profiled hosts as extended; shared PSF-convolved rate image"
```

---

### Task 4: `psfsim.py` — simulate() and 2D diffuse in the aperture functions

**Files:**
- Modify: `src/wcc_etc/psfsim.py` (`ImageSimulator.simulate` ~867–957; `_radial_cumulative` ~1646; `aperture_snr_radial` ~1670; `aperture_time_for_snr` ~1749)
- Modify: `src/wcc_etc/simulation.py` (`get_image_snr._snr_at` ~1058–1066; `get_image_exptime_for_snr` ~1190–1193)
- Modify: `tests/psf/test_psfsource.py:334` (unpack 4 values)
- Test: `tests/simulation/test_extended_snr.py`, add to `tests/imaging/test_extended.py`

**Interfaces:**
- Consumes: `Simulation.extended_rate_image`, bundle key `extended_rate_image`
- Produces:
  - `_radial_cumulative(psf_norm, plate_scale_mas) -> (r_mas, enclosed, n_pix, order)`
  - `diffuse_enclosed(diffuse_per_pix, order, n_pix) -> np.ndarray`
  - `aperture_snr_radial(..., diffuse_per_pix: float | ndarray, ...)` and `aperture_time_for_snr(..., diffuse_rate_per_pix: float | ndarray, ...)` accept a 2D image.

- [ ] **Step 1: Write the failing tests**

Append to `tests/imaging/test_extended.py`:

```python
from wcc_etc.psfsim import ImageSimulator, aperture_snr_radial
from tests.helpers import make_scene


class TestSimulateIncludesHost:
    def test_clean_image_carries_extended_charge(self):
        """simulate(add_noise=False) adds extended_rate_image * t to the clean image."""
        host = {"mag": 17, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 0.1}
        imsim = ImageSimulator.from_sensor_and_scene("sony:r", make_scene(mag=20, host="G5V", host_prop=host), npix=128)
        hostless = ImageSimulator.from_sensor_and_scene("sony:r", make_scene(mag=20), npix=128)
        diff = imsim.simulate(10.0, add_noise=False).image_clean - hostless.simulate(10.0, add_noise=False).image_clean
        b = imsim.sim._image_render_bundle(imsim.default_psf, None, 128, 11)
        assert np.allclose(diff, 10.0 * b["extended_rate_image"], rtol=1e-6, atol=1e-9)


class TestDiffuseImage:
    def test_uniform_image_matches_scalar(self):
        """A constant 2D diffuse image reproduces the scalar per-pixel result."""
        psf = np.zeros((31, 31)); psf[15, 15] = 1.0
        scalar = aperture_snr_radial(psf, 17.0, 1e4, 3.0, 0.1, 2.0)["snr"]
        image = aperture_snr_radial(psf, 17.0, 1e4, np.full((31, 31), 3.0), 0.1, 2.0)["snr"]
        assert np.allclose(scalar, image)
```

Create `tests/simulation/test_extended_snr.py`:

```python
"""SNR with a Sersic host: nuclear vs offset transients."""

import pytest

import wcc_etc
from tests.helpers import make_scene

HOST = {"mag": 15, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 0.3, "n": 4.0}


def snr_with_host(**host):
    scene = make_scene(mag=21, host="G5V", host_prop=HOST | host)
    sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
    return sim.get_snr(time=60.0, warn=False)["snr"]


class TestHostPosition:
    def test_nuclear_transient_has_lower_snr_than_offset(self):
        """A source on the bulge cusp sees more host shot noise than one 1.5" out."""
        assert snr_with_host(dx=0.0) < snr_with_host(dx=1.5)

    def test_far_offset_host_approaches_hostless(self):
        """With the host 2" away (beyond the grid) the SNR is the hostless SNR."""
        scene = make_scene(mag=21)
        hostless = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene).get_snr(time=60.0, warn=False)["snr"]
        assert snr_with_host(dx=2.0, r_eff=0.05, n=1.0) == pytest.approx(hostless, rel=1e-2)


class TestExptimeInverse:
    def test_exptime_inverts_snr_with_host(self):
        """get_image_exptime_for_snr(get_snr(t)) returns t with an extended host."""
        scene = make_scene(mag=21, host="G5V", host_prop=HOST)
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        snr = sim.get_snr(time=60.0, warn=False)["snr"]
        assert sim.get_image_exptime_for_snr(snr, warn=False)["time_s"] == pytest.approx(60.0, rel=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/imaging/test_extended.py tests/simulation/test_extended_snr.py -q`
Expected: `TestSimulateIncludesHost` FAILS (diff is all zero); `TestDiffuseImage` FAILS (shape broadcast error); `test_nuclear_transient...` FAILS (equal SNRs).

- [ ] **Step 3: Implement**

`psfsim.py`, `_radial_cumulative`: change the final line to `return r_sorted * plate_scale_mas, enclosed, n_pix, order` and the docstring to "Returns (r_mas, enclosed_fraction, n_pix, order)". Add after it:

```python
def diffuse_enclosed(diffuse_per_pix, order, n_pix):
    """Enclosed diffuse charge per radius: a scalar is uniform (per_pix * n_pix);
    a 2D image (e.g. a Sersic host) is summed in the same radial order."""
    d = np.asarray(diffuse_per_pix, dtype=float)
    if d.ndim == 0:
        return d * n_pix
    return np.cumsum(d.ravel()[order])
```

`aperture_snr_radial`: unpack `r_mas, enclosed, n_pix, order = ...`; replace the `per_pix_var` / `noise` lines with

```python
    diffuse = diffuse_enclosed(diffuse_per_pix, order, n_pix)
    noise = np.sqrt(signal + contaminant + diffuse + (dark_per_pix + read_noise**2) * n_pix)
```

and document `diffuse_per_pix : float or ndarray` ("per-pixel diffuse electrons; a 2D image on the PSF grid for a spatially varying host").

`aperture_time_for_snr`: unpack 4 values; `B = A + contaminant_rate_total * enclosed + diffuse_enclosed(diffuse_rate_per_pix, order, n_pix) + dark_rate_per_pix * n_pix`.

`ImageSimulator.simulate`: after `diffuse_per_pix = ...` add
`extended_e = sim.extended_rate_image(psf_norm, ctx, comps) * t_s` and use
`image_clean = source_image + bkg_per_pix + extended_e + dark_per_pix`.

`simulation.py` `get_image_snr._snr_at`: `diffuse_per_pix = (b["diffuse_rate_per_pix"] + b["background_rate_per_pix"] + b["extended_rate_image"]) * t_sec` (now a 2D array; comment: "plus the PSF-convolved extended-host image").
`get_image_exptime_for_snr`: `diffuse_rate_per_pix = b["diffuse_rate_per_pix"] + b["background_rate_per_pix"] + b["extended_rate_image"]`.

`tests/psf/test_psfsource.py:334`: `r_mas, enclosed, n_pix, _ = _radial_cumulative(psf, plate)`.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/wcc_etc/psfsim.py src/wcc_etc/simulation.py tests
git commit -m "feat: extended host in simulate() and 2D diffuse in aperture SNR/exptime"
```

---

### Task 5: Docs — changelog, API export, README snippet

**Files:**
- Modify: `docs/changelog.md`, `src/wcc_etc/__init__.py` (export `extended`), `README.md` (one example under the scene section, if one exists — otherwise skip)

- [ ] **Step 1: Changelog**

Add under a new `## Unreleased` heading at the top of `docs/changelog.md`:

```
- Sersic-profile hosts: `host_prop={"profile": "sersic", "r_eff": 1.0, "n": 1, "ellip": 0.3, "pa": 45, "dx": 0.8, "dy": 0}`
  (angular parameters; `mag` is the total host magnitude, or μ_e with `surface_brightness=True`).
  Rendered and PSF-convolved once, shared by `simulate`, `get_snr`, `get_image_exptime_for_snr`,
  `get_peak_pixel` and `is_saturated`. New module `wcc_etc.extended`.
```

- [ ] **Step 2: Export** — in `src/wcc_etc/__init__.py` add `from . import extended  # noqa: F401` alongside the other module imports (check the existing style first and match it).

- [ ] **Step 3: Run `pytest -q`, commit**

```bash
git add docs/changelog.md src/wcc_etc/__init__.py README.md
git commit -m "docs: changelog and export for Sersic hosts"
```

---

### Task 6: Demo notebook

**Files:**
- Create: `notebooks_scratch/_build_20260905_sersic_host.py` (builder), `notebooks_scratch/20260905_sersic_host.ipynb` (output)

Use the `notebook-demo` skill (builder script → execute → verify). Content, `plt.style.use('gks')`:

1. Intro markdown: what a Sersic profile is, the parameter table from the spec, the two normalization modes, typical values (n 1–4, r_eff 0.3–10″, μ_e 20–24).
2. Gallery: `ImageSimulator.from_sensor_and_scene("zwo:r", scene, npix=256)` for hosts `(n, r_eff, ellip, pa)` ∈ {(1, 0.5, 0, 0), (4, 0.5, 0, 0), (1, 1.0, 0.5, 45), (4, 0.3, 0.3, 120)} with a mag-21 source at the centre; show `simulate(60, add_noise=False).image_clean` with a log stretch, and one noisy realization.
3. Normalization check: total-mag host image sum vs point-source rate at the same mag (print both); μ_e host pixel at r_eff vs uniform-SB rate.
4. Science plot: SNR(60 s) of a mag-21 transient vs host total magnitude 14–20, for dx = 0 (nuclear, TDE-like), 0.5″, 1.5″ (SN-like), n=4 bulge r_eff=0.5″; hostless SNR as a dashed reference. Use `sim.update(host__mag=m)` and `sim.update(host__dx=d)`.
5. Sweep: `sim.update(host__r_eff=r)` for r in [0.2, 0.5, 1, 2]″ at fixed total mag — SNR vs r_eff, showing that a more compact host hurts a nuclear transient more.
6. Sanity asserts in code cells (flux conservation to 1%, nuclear SNR < offset SNR).

- [ ] **Step: Build, execute, verify outputs, commit**

```bash
git add notebooks_scratch/_build_20260905_sersic_host.py notebooks_scratch/20260905_sersic_host.ipynb
git commit -m "docs: scratch notebook demonstrating Sersic hosts"
```

---

### Task 7: PR

- [ ] `pytest -q` green; `ruff check src/wcc_etc/extended.py` clean.
- [ ] Push `feat/sersic-host`, open PR against `main` titled "feat: Sersic-profile host galaxies (structured background)" with the spec summary, the parameter table, and a note that the API mirrors wcc-sim's `SersicComponent` so wcc-sim can later depend on `wcc_etc.extended`. Watch the `test` check (numpy 2.x / py3.11).
