"""Tests for the public PSF-rendering surface of ImageSimulator.

`render_psf` returns the bare normalized PSF on the detector grid, and
`default_psf` / `plate_scale_mas` expose what `simulate()` and the plotting
helpers need without reaching into the private render context.
"""

import numpy as np
import pytest

from tests.helpers import make_scene
from wcc_etc.psfsim import DEFOCUS_2WAVE_PATH, AiryPSF, DefocusPSF, ImageSimulator


@pytest.fixture
def defocus_imsim():
    """An ImageSimulator built from a 2-wave defocused sensorfilter label."""
    return ImageSimulator.from_sensorfilter("zwo:bb2", make_scene(), npix=128)


class TestRenderPsf:
    """render_psf produces a normalized PSF image on the detector grid."""

    def test_returns_the_grid_shape(self, imsim):
        """The render matches the simulator's npix."""
        assert imsim.render_psf(AiryPSF()).shape == (128, 128)

    def test_normalized_to_unit_sum(self, imsim):
        """The PSF is normalized so its pixels sum to 1."""
        assert imsim.render_psf(AiryPSF()).sum() == pytest.approx(1.0)

    def test_npix_overrides_the_grid_size(self, imsim):
        """npix= re-sizes this render only."""
        assert imsim.render_psf(AiryPSF(), npix=64).shape == (64, 64)

    def test_npix_override_does_not_mutate_the_simulator(self, imsim):
        """The npix override is per call, not sticky."""
        imsim.render_psf(AiryPSF(), npix=64)
        assert imsim.npix == 128

    def test_zero_jitter_gives_a_sharper_peak(self, imsim):
        """Jitter blurs the PSF, so no jitter concentrates more light in the peak."""
        sharp = imsim.render_psf(AiryPSF(), jitter_sigma_mas=0.0).max()
        blurred = imsim.render_psf(AiryPSF(), jitter_sigma_mas=50.0).max()
        assert sharp > blurred

    def test_defocus_spreads_more_than_airy(self, imsim):
        """A defocused PSF has a lower peak fraction than the diffraction limit."""
        airy = imsim.render_psf(AiryPSF(), jitter_sigma_mas=0.0).max()
        defocus = imsim.render_psf(
            DefocusPSF(DEFOCUS_2WAVE_PATH), jitter_sigma_mas=0.0
        ).max()
        assert defocus < airy

    def test_defaults_to_the_simulations_default_psf(self, defocus_imsim):
        """With no psf= the render uses default_psf, not a bare Airy disk."""
        implicit = defocus_imsim.render_psf(jitter_sigma_mas=0.0)
        explicit = defocus_imsim.render_psf(
            DefocusPSF(DEFOCUS_2WAVE_PATH), jitter_sigma_mas=0.0
        )
        np.testing.assert_allclose(implicit, explicit)


class TestPlateScale:
    """plate_scale_mas exposes the detector plate scale for plotting in mas."""

    def test_is_positive(self, imsim):
        """The plate scale is a positive number of mas per pixel."""
        assert imsim.plate_scale_mas > 0

    def test_matches_the_simulated_image(self, imsim):
        """It agrees with the pixel scale carried on a SimulatedImage."""
        res = imsim.simulate(time=10, add_noise=False)
        assert imsim.plate_scale_mas == pytest.approx(res.pixel_scale_mas)


class TestDefaultPsf:
    """default_psf is the PSF used when no psf= argument is given."""

    def test_is_airy_without_a_focus_level(self, imsim):
        """A plain sensor+scene build is diffraction limited."""
        assert isinstance(imsim.default_psf, AiryPSF)

    def test_is_defocus_for_a_defocused_label(self, defocus_imsim):
        """A 2-wave label resolves to a DefocusPSF."""
        assert isinstance(defocus_imsim.default_psf, DefocusPSF)

    def test_simulate_uses_it_when_no_psf_is_given(self, defocus_imsim):
        """simulate() picks up the focus-level PSF automatically."""
        res = defocus_imsim.simulate(time=5, add_noise=False)
        assert isinstance(res.psf, DefocusPSF)

    def test_simulate_still_honors_an_explicit_psf(self, defocus_imsim):
        """An explicit psf= overrides the default."""
        res = defocus_imsim.simulate(time=5, psf=AiryPSF(), add_noise=False)
        assert isinstance(res.psf, AiryPSF)
