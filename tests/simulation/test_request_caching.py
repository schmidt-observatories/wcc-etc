"""The per-request caches (airy renders, bandpass load, normalised spectrum)
must not change any number: compare one full ETC request computed with every
cache bypassed against the same request served warm from the caches."""

import numpy as np
import pytest

from tests.helpers import make_scene
from wcc_etc import airy, io
from wcc_etc.psfsim import ImageSimulator
from wcc_etc.scene import SceneElement
from wcc_etc.simulation import Simulation

TIME_S = 100


def plain(x):
    """Quantity/array/scalar -> ndarray (units stripped) for exact comparison."""
    return np.asarray(getattr(x, "value", x))


def run_request(sensorfilter):
    """One WCC ETC web request; every numerical output, flattened to a dict."""
    sim = Simulation.from_sensorfilter(sensorfilter, make_scene())
    out = {}
    out |= {f"snr.{k}": plain(v) for k, v in sim.get_snr(TIME_S).items()}
    out |= {f"psf_profile.{k}": plain(v) for k, v in sim.psf_profile.items()}
    out |= {f"image_snr.{k}": plain(v) for k, v in sim.get_image_snr(TIME_S).items()}
    poly = sim.get_image_snr(TIME_S, psf=sim.polychromatic_psf())
    out |= {f"poly_snr.{k}": plain(v) for k, v in poly.items()}
    img = ImageSimulator(sim).simulate(TIME_S, seed=0)
    out |= {
        f"simulate.{k}": v for k, v in vars(img).items() if isinstance(v, np.ndarray)
    }
    psf, pscale = airy.render_detector_psf(
        sim.effective_wavelength.to("m").value,
        sim.telescope.f_num,
        sim.telescope.diameter_primary.value,
        sim.sensor.pixel_size.value,
        jitter_sigma_mas=sim.telescope.jitter_sigma.to("mas").value,
        n_pixels=256,
        oversample=5,
    )
    out["profile_curve.psf"] = psf
    out["profile_curve.pscale"] = np.asarray(pscale)
    return out


def bypass_caches(monkeypatch):
    """Route every cached path through its uncached implementation."""
    monkeypatch.setattr(
        airy, "render_detector_psf_cached", airy.render_detector_psf_cached.__wrapped__
    )
    monkeypatch.setattr(
        airy, "airy_ee_curve_cached", airy.airy_ee_curve_cached.__wrapped__
    )
    monkeypatch.setattr(io, "load_bandpass", io.load_bandpass.__wrapped__)
    normalize = SceneElement.normalized_spectrum

    def uncached(self, mag):
        self._normalized_cache = None
        return normalize(self, mag)

    monkeypatch.setattr(SceneElement, "normalized_spectrum", uncached)


SENSORFILTERS = ["qcmos:bb", "zwo:r+1"]  # 0wave (Airy) and 1wave (defocused)


@pytest.fixture(scope="module", params=SENSORFILTERS)
def before_after(request):
    """(uncached, warm-cached) results of the same request."""
    mp = pytest.MonkeyPatch()
    bypass_caches(mp)
    try:
        before = run_request(request.param)
    finally:
        mp.undo()
    run_request(request.param)  # populate the caches
    after = run_request(request.param)  # served from them
    return before, after


KEYS = sorted(run_request(SENSORFILTERS[0]))


@pytest.mark.parametrize("key", KEYS)
def test_request_is_bit_identical(before_after, key):
    """Every output of a cached request equals the uncached computation exactly."""
    before, after = before_after
    assert np.array_equal(before[key], after[key])


class TestCachedArraysAreCopies:
    def test_render_detector_psf_returns_a_copy(self):
        """Mutating a rendered PSF must not corrupt the cached one."""
        kw = dict(wavelength=0.6e-6, fnum=15, D=3, pixel_size=3.76, n_pixels=21)
        psf, _ = airy.render_detector_psf(**kw)
        psf[:] = 0.0
        assert airy.render_detector_psf(**kw)[0].sum() > 0

    def test_ee_curve_returns_a_copy(self):
        """Mutating an EE curve must not corrupt the cached one."""
        kw = dict(wavelength=0.6e-6, r_aper_mas=70, grid_size=256)
        ee = airy.get_airy_and_ee_curve(**kw)[2]
        ee[:] = 0.0
        assert airy.get_airy_and_ee_curve(**kw)[2][-1] == 1.0


class TestCacheHits:
    def test_resolve_bandpass_returns_one_object_per_name(self):
        """The loaded SpectralElement is shared, so band identity is stable."""
        assert io.resolve_bandpass("johnson_r") is io.resolve_bandpass("johnson_r")

    def test_normalised_spectrum_is_reused(self):
        """A second get_spectrum with unchanged mag/band returns the cached object."""
        element = make_scene().source
        assert element.get_spectrum() is element.get_spectrum()

    def test_normalised_spectrum_invalidates_on_mag_change(self):
        """Changing the magnitude must miss the cache."""
        element = make_scene().source
        first = element.get_spectrum()
        element.update(mag=16)
        assert element.get_spectrum() is not first
