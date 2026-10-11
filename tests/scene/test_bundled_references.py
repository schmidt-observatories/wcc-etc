"""Vega and Johnson V come from package data, not from STScI at runtime (#101)."""

import pytest
import synphot

import wcc_etc
import wcc_etc.scene as scene_mod
from wcc_etc.io import resolve_bandpass


@pytest.fixture
def no_network(monkeypatch):
    """Make synphot's remote loaders fail loudly if anything still calls them."""

    def boom(*args, **kwargs):
        raise AssertionError("remote synphot loader called")

    monkeypatch.setattr(synphot.SourceSpectrum, "from_vega", boom)
    monkeypatch.setattr(synphot.SpectralElement, "from_filter", boom)
    monkeypatch.setattr(scene_mod, "_VEGA", None)


class TestBundledReferences:
    def test_vega_loads_offline(self, no_network):
        assert scene_mod._get_vega()(5500).value > 0

    def test_johnson_v_loads_offline(self, no_network):
        assert resolve_bandpass("johnson_v")(5500).value > 0.5

    def test_vegamag_scene_builds_offline(self, no_network):
        scene = wcc_etc.get_scene("G2V", mag=15, host=None, bandpass="johnson_v")
        assert scene.source.get_observation().countrate(area=1).value > 0

