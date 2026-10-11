"""A file spectrum that does not cover the bandpass warns instead of silently extrapolating (#102)."""

import warnings

import pytest

from wcc_etc.scene import get_scene_from_file
from wcc_etc.sensor import Sensor


@pytest.fixture(scope="module")
def bb_band():
    return Sensor.from_name("sony:bb").bandpass


def flat_scene(tmp_path, wmin, wmax):
    specfile = tmp_path / "flat.txt"
    specfile.write_text(f"{wmin} 1e-16\n{wmax} 1e-16\n")
    return get_scene_from_file(str(specfile), mag=18, background=None)


class TestSpectrumCoverage:
    def test_short_spectrum_warns(self, tmp_path, bb_band):
        """4000-7000 A does not cover the sony:bb band (~3700-9100 A)."""
        scene = flat_scene(tmp_path, 4000, 7000)
        with pytest.warns(UserWarning, match="does not cover the bandpass"):
            scene.source.get_observation(bb_band)

    def test_warning_names_the_uncovered_range(self, tmp_path, bb_band):
        """The red gap 7000-9100 A is spelled out in Angstrom."""
        scene = flat_scene(tmp_path, 4000, 7000)
        with pytest.warns(UserWarning, match=r"7000-9100 A"):
            scene.source.get_observation(bb_band)

    def test_warning_says_edge_value_is_extrapolated(self, tmp_path, bb_band):
        """The user is told what happens in the gap."""
        scene = flat_scene(tmp_path, 4000, 7000)
        with pytest.warns(UserWarning, match="extrapolated"):
            scene.source.get_observation(bb_band)

    def test_covering_spectrum_does_not_warn(self, tmp_path, bb_band):
        """A spectrum spanning the whole band is silent."""
        scene = flat_scene(tmp_path, 3000, 10000)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scene.source.get_observation(bb_band)

    def test_observation_is_still_built(self, tmp_path, bb_band):
        """force='extrap' is kept: behaviour is unchanged apart from the warning."""
        scene = flat_scene(tmp_path, 4000, 7000)
        with pytest.warns(UserWarning):
            obs = scene.source.get_observation(bb_band)
        assert obs is not None
