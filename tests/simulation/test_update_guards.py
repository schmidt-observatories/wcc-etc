"""Simulation.update() rejects unknown and ambiguous keys instead of warning (#102)."""

import pytest


class TestUpdateKeyGuards:
    def test_unknown_key_raises(self, sim):
        """A key matching no mutable parameter raises KeyError."""
        with pytest.raises(KeyError, match="no entries found"):
            sim.update(not_a_parameter=1)

    def test_ambiguous_key_raises(self, sim):
        """A partial key matching several parameters raises KeyError."""
        with pytest.raises(KeyError, match="several entries found"):
            sim.update(mag=18)

    def test_ambiguous_key_lists_candidates(self, sim):
        """The ambiguous-key error names the candidate full keys."""
        with pytest.raises(KeyError, match="scene__source__mag"):
            sim.update(mag=18)

    def test_ambiguous_key_leaves_source_mag_unchanged(self, sim):
        """A rejected update must not change anything."""
        before = sim.scene.source.meta["mag"]
        with pytest.raises(KeyError):
            sim.update(mag=18)
        assert sim.scene.source.meta["mag"] == before
