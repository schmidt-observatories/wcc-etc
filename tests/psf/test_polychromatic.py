"""Tests for PolychromaticPSF, the photon-weighted multi-wavelength PSF coadd."""

import astropy.units as u
import numpy as np
import pytest

from wcc_etc.psfsim import AiryPSF, DetectorPSFContext, PolychromaticPSF


def make_ctx(npix=65, wavelength_m=600e-9):
    """A detector context with the Lazuli optics and no jitter."""
    return DetectorPSFContext(
        npix=npix,
        pixel_size_um=3.76,
        plate_scale_mas=16.9,
        wavelength_m=wavelength_m,
        diameter_m=3.065,
        fnum=15.0,
        jitter_sigma_mas=0.0,
        oversample=11,
    )


class TestPolychromaticConstruction:
    def test_rejects_mismatched_lengths(self):
        """One weight per wavelength is required."""
        with pytest.raises(ValueError, match="same length"):
            PolychromaticPSF([500e-9, 600e-9], [1.0])

    def test_rejects_empty_input(self):
        """At least one wavelength is required."""
        with pytest.raises(ValueError, match="at least one"):
            PolychromaticPSF([], [])

    def test_rejects_non_positive_weight_sum(self):
        """Weights that do not carry any flux cannot be normalized."""
        with pytest.raises(ValueError, match="positive"):
            PolychromaticPSF([500e-9, 600e-9], [0.0, 0.0])

    def test_normalizes_weights(self):
        """Weights are stored normalized to sum 1."""
        psf = PolychromaticPSF([500e-9, 600e-9], [2.0, 2.0])
        assert psf.weights.sum() == pytest.approx(1.0)

    def test_accepts_wavelength_quantities(self):
        """Wavelengths may be given as a Quantity in any length unit."""
        psf = PolychromaticPSF([500, 600] * u.nm, [0.5, 0.5])
        assert psf.wavelengths_m[0] == pytest.approx(500e-9)

    def test_defaults_to_an_airy_base(self):
        """Without an explicit base the coadd renders Airy PSFs."""
        assert isinstance(PolychromaticPSF([600e-9], [1.0]).base, AiryPSF)


class TestPolychromaticRender:
    def test_render_shape(self):
        """The coadd lands on the requested detector grid."""
        psf = PolychromaticPSF([500e-9, 600e-9, 700e-9], [1, 1, 1])
        assert psf.render(make_ctx()).shape == (65, 65)

    def test_render_normalized(self):
        """The coadd sums to 1 like any other rendered PSF."""
        psf = PolychromaticPSF([500e-9, 600e-9, 700e-9], [1, 1, 1])
        assert psf.render(make_ctx()).sum() == pytest.approx(1.0, abs=1e-6)

    def test_single_wavelength_matches_the_monochromatic_render(self):
        """One sub-band is exactly the monochromatic PSF at that wavelength."""
        poly = PolychromaticPSF([700e-9], [1.0]).render(make_ctx())
        assert np.allclose(poly, AiryPSF().render(make_ctx(wavelength_m=700e-9)))

    def test_ignores_the_context_wavelength(self):
        """The coadd's own wavelengths win over ctx.wavelength_m."""
        a = PolychromaticPSF([700e-9], [1.0]).render(make_ctx(wavelength_m=400e-9))
        b = PolychromaticPSF([700e-9], [1.0]).render(make_ctx(wavelength_m=900e-9))
        assert np.allclose(a, b)

    def test_redder_coadd_is_wider(self):
        """A red-weighted coadd puts less flux on the brightest pixel."""
        blue = PolychromaticPSF([450e-9, 500e-9, 550e-9], [1, 1, 1]).render(make_ctx())
        red = PolychromaticPSF([700e-9, 750e-9, 800e-9], [1, 1, 1]).render(make_ctx())
        assert red.max() < blue.max()

    def test_coadd_is_bracketed_by_its_endpoints(self):
        """The coadd's peak lies between the reddest and bluest monochromatic peaks."""
        wavelengths = [500e-9, 600e-9, 700e-9]
        poly = PolychromaticPSF(wavelengths, [1, 1, 1]).render(make_ctx()).max()
        peaks = [AiryPSF().render(make_ctx(wavelength_m=w)).max() for w in wavelengths]
        assert min(peaks) < poly < max(peaks)


class TestPolychromaticFromBandpass:
    def test_builds_requested_number_of_subbands(self, g5v_sim):
        """from_bandpass honours n_sub."""
        psf = PolychromaticPSF.from_bandpass(
            g5v_sim.sensor.bandpass,
            g5v_sim.scene.source.get_spectrum(apply_mag=False),
            n_sub=5,
        )
        assert len(psf.wavelengths_m) == 5

    def test_single_subband_sits_at_the_effective_wavelength(self, g5v_sim):
        """n_sub=1 reproduces the monochromatic effective-wavelength case."""
        psf = PolychromaticPSF.from_bandpass(
            g5v_sim.sensor.bandpass,
            g5v_sim.scene.source.get_spectrum(apply_mag=False),
            n_sub=1,
        )
        expected = g5v_sim.effective_wavelength.to_value(u.m)
        assert psf.wavelengths_m[0] == pytest.approx(expected)


class TestSimulationPolychromaticPSF:
    def test_returns_a_polychromatic_psf(self, g5v_sim):
        """Simulation.polychromatic_psf hands back a PolychromaticPSF."""
        assert isinstance(g5v_sim.polychromatic_psf(), PolychromaticPSF)

    def test_uses_the_source_spectrum(self, g5v_sim, m5v_sim):
        """A redder source shifts the sub-band wavelengths redward."""
        assert (
            m5v_sim.polychromatic_psf().wavelengths_m.mean()
            > g5v_sim.polychromatic_psf().wavelengths_m.mean()
        )

    def test_stays_close_to_the_monochromatic_peak_fraction(self, g5v_sim):
        """The coadd is a refinement, not a different answer, at the same lambda_eff."""
        mono = g5v_sim.peak_pixel_fraction()
        poly = g5v_sim.peak_pixel_fraction(psf=g5v_sim.polychromatic_psf())
        assert poly == pytest.approx(mono, rel=0.05)

    def test_red_source_coadd_has_a_lower_peak_fraction(self, g5v_sim, m5v_sim):
        """The colour ordering survives the polychromatic treatment."""
        assert m5v_sim.peak_pixel_fraction(
            psf=m5v_sim.polychromatic_psf()
        ) < g5v_sim.peak_pixel_fraction(psf=g5v_sim.polychromatic_psf())

    def test_cache_key_distinguishes_sources(self, g5v_sim, m5v_sim):
        """Different sources give different cache keys, so renders are not shared."""
        assert (
            g5v_sim.polychromatic_psf().cache_key()
            != m5v_sim.polychromatic_psf().cache_key()
        )
