"""Tests for n_reads, get_exptime_for_snr, and related analytic exposure-time methods."""

import numpy as np
import pytest

from tests.helpers import make_simulation


class TestNReads:
    def test_n_reads_default_is_one(self, sim):
        assert sim.meta.get("n_reads") == 1

    def test_n_reads_is_mutable_parameter(self, sim):
        assert any(
            k.endswith("n_reads") or k == "n_reads" for k in sim.mutable_parameters
        )

    def test_n_reads_update(self, sim):
        sim.update(n_reads=4)
        assert sim.meta["n_reads"] == 4

    def test_n_reads_one_matches_baseline(self, sim):
        baseline = sim.get_snr(60)["snr"]
        assert np.isclose(sim.get_snr(60, n_reads=1)["snr"], baseline)

    def test_more_reads_lowers_snr_faint_source(self):
        sim = make_simulation(mag=19)
        s1 = sim.get_snr(30, n_reads=1)["snr"]
        s9 = sim.get_snr(30, n_reads=9)["snr"]
        assert s9 < s1

    def test_n_reads_zero_raises(self, sim):
        with pytest.raises(ValueError):
            sim.get_snr(60, n_reads=0)

    def test_n_reads_negative_raises(self, sim):
        with pytest.raises(ValueError):
            sim.get_exptime_for_snr(50, n_reads=-1)


class TestExptimeForSnr:
    def test_roundtrips_snr(self, sim):
        for target in (20.0, 100.0):
            t = sim.get_exptime_for_snr(target)
            with pytest.warns(DeprecationWarning):
                got = float(sim.get_snr_airy(t).value)
            assert np.isclose(got, target, rtol=1e-3)

    def test_roundtrips_with_reads(self):
        sim = make_simulation(mag=18)
        t = sim.get_exptime_for_snr(50.0, n_reads=5)
        with pytest.warns(DeprecationWarning):
            got = float(sim.get_snr_airy(t, n_reads=5).value)
        assert np.isclose(got, 50.0, rtol=1e-3)

    def test_uses_meta_n_reads_when_not_given(self):
        sim = make_simulation(mag=18)
        sim.update(n_reads=5)
        t_meta = sim.get_exptime_for_snr(50.0)
        t_explicit = sim.get_exptime_for_snr(50.0, n_reads=5)
        assert np.isclose(t_meta.value, t_explicit.value, rtol=1e-9)
