"""Tests for the source-weighted spectral quantities in wcc_etc.spectral."""

import astropy.units as u
import numpy as np
import pytest
from synphot import SourceSpectrum, SpectralElement
from synphot.models import BlackBodyNorm1D, Box1D, ConstFlux1D

from wcc_etc.spectral import (
    effective_wavelength,
    photon_weighted_subbands,
    photon_weights,
)


@pytest.fixture
def box_band():
    """A 400-800 nm boxcar bandpass."""
    return SpectralElement(Box1D, amplitude=1.0, x_0=600 * u.nm, width=400 * u.nm)


@pytest.fixture
def flat_photon_spectrum():
    """A spectrum that is flat in photons per unit wavelength."""
    return SourceSpectrum(ConstFlux1D, amplitude=1 * u.photon / (u.s * u.cm**2 * u.AA))


class TestPhotonWeights:
    def test_returns_matching_lengths(self, box_band, flat_photon_spectrum):
        """The wavelength grid and the weights have the same length."""
        wave, weight = photon_weights(box_band, flat_photon_spectrum)
        assert len(wave) == len(weight)

    def test_weights_are_non_negative(self, box_band, flat_photon_spectrum):
        """Photon weights are never negative."""
        _, weight = photon_weights(box_band, flat_photon_spectrum)
        assert weight.min() >= 0.0

    def test_weights_vanish_outside_the_band(self, box_band, flat_photon_spectrum):
        """No weight lands outside the boxcar throughput."""
        wave, weight = photon_weights(box_band, flat_photon_spectrum)
        outside = (wave < 399 * u.nm) | (wave > 801 * u.nm)
        assert np.all(weight[outside] == 0.0)

    def test_no_spectrum_gives_throughput_only_weights(self, box_band):
        """With spectrum=None the weights are the bare throughput."""
        wave, weight = photon_weights(box_band, None)
        assert np.allclose(weight, box_band(wave).value)

    def test_raises_when_no_overlap(self, flat_photon_spectrum):
        """A bandpass with no positive throughput cannot be weighted."""
        empty = SpectralElement(Box1D, amplitude=0.0, x_0=600 * u.nm, width=400 * u.nm)
        with pytest.raises(ValueError, match="no positive photon flux"):
            photon_weights(empty, flat_photon_spectrum)


class TestEffectiveWavelength:
    def test_flat_photon_spectrum_gives_band_center(
        self, box_band, flat_photon_spectrum
    ):
        """A flat photon spectrum through a symmetric boxcar peaks at the center."""
        lam = effective_wavelength(box_band, flat_photon_spectrum)
        assert lam.to_value(u.nm) == pytest.approx(600.0, abs=1.0)

    def test_returns_a_wavelength_quantity(self, box_band, flat_photon_spectrum):
        """The result carries wavelength units."""
        lam = effective_wavelength(box_band, flat_photon_spectrum)
        assert lam.unit.physical_type == "length"

    def test_red_source_shifts_effective_wavelength_redward(self, box_band):
        """A cool blackbody pulls the effective wavelength above the band center."""
        cool = SourceSpectrum(BlackBodyNorm1D, temperature=3000 * u.K)
        assert effective_wavelength(box_band, cool) > 600 * u.nm

    def test_blue_source_shifts_effective_wavelength_blueward(self, box_band):
        """A hot blackbody pulls the effective wavelength below the band center."""
        hot = SourceSpectrum(BlackBodyNorm1D, temperature=40000 * u.K)
        assert effective_wavelength(box_band, hot) < 600 * u.nm

    def test_matches_synphot_effective_wavelength(self, box_band):
        """Agrees with synphot's Observation.effective_wavelength.

        synphot's ``efflerg`` mode integrates ``F_lambda T lambda**2`` over
        ``F_lambda T lambda``; since ``F_photon ~ F_lambda * lambda`` that is the
        same photon-weighted mean this module computes. (Its ``efflphot`` mode is
        a different, energy-weighted quantity.)
        """
        from synphot import Observation

        spec = SourceSpectrum(BlackBodyNorm1D, temperature=5000 * u.K)
        ours = effective_wavelength(box_band, spec).to_value(u.AA)
        theirs = (
            Observation(spec, box_band, force="extrap")
            .effective_wavelength(binned=False, mode="efflerg")
            .to_value(u.AA)
        )
        assert ours == pytest.approx(theirs, rel=1e-4)


class TestPhotonWeightedSubbands:
    def test_returns_requested_number_of_samples(self, box_band, flat_photon_spectrum):
        """n_sub sub-bands give n_sub render wavelengths."""
        wave, _ = photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=7)
        assert len(wave) == 7

    def test_weights_sum_to_one(self, box_band, flat_photon_spectrum):
        """The coadd weights are normalized."""
        _, weight = photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=7)
        assert weight.sum() == pytest.approx(1.0)

    def test_wavelengths_are_increasing(self, box_band, flat_photon_spectrum):
        """Sub-band wavelengths come back sorted."""
        wave, _ = photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=9)
        assert np.all(np.diff(wave.to_value(u.nm)) > 0)

    def test_single_subband_reduces_to_effective_wavelength(
        self, box_band, flat_photon_spectrum
    ):
        """n_sub=1 is exactly the monochromatic effective-wavelength case."""
        wave, _ = photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=1)
        expected = effective_wavelength(box_band, flat_photon_spectrum)
        assert wave[0].to_value(u.nm) == pytest.approx(expected.to_value(u.nm))

    def test_single_subband_carries_all_the_weight(
        self, box_band, flat_photon_spectrum
    ):
        """n_sub=1 puts unit weight on the one sample."""
        _, weight = photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=1)
        assert weight[0] == pytest.approx(1.0)

    def test_subbands_span_the_band(self, box_band, flat_photon_spectrum):
        """Sub-band centroids stay inside the throughput support."""
        wave, _ = photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=9)
        nm = wave.to_value(u.nm)
        assert nm.min() > 400 and nm.max() < 800

    def test_weighted_mean_recovers_effective_wavelength(
        self, box_band, flat_photon_spectrum
    ):
        """The weighted mean of the samples is the effective wavelength."""
        wave, weight = photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=9)
        expected = effective_wavelength(box_band, flat_photon_spectrum)
        got = np.sum(weight * wave.to_value(u.nm))
        assert got == pytest.approx(expected.to_value(u.nm), rel=1e-6)

    def test_rejects_non_positive_n_sub(self, box_band, flat_photon_spectrum):
        """n_sub must be at least 1."""
        with pytest.raises(ValueError, match="n_sub"):
            photon_weighted_subbands(box_band, flat_photon_spectrum, n_sub=0)
