"""SceneElement.profile: spatial profile kwargs carried in meta."""

import pytest

import wcc_etc
from tests.helpers import make_scene


def sersic_scene(**profile):
    """Standard scene with a mag-17 G5V host carrying a sersic profile."""
    host_prop = {"mag": 17, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 1.0}
    return make_scene(host="G5V", host_prop=host_prop | profile)


class TestProfileProperty:
    def test_no_profile_is_none(self):
        """A plain element has profile None (point or uniform SB, unchanged)."""
        assert make_scene().source.profile is None

    def test_defaults_filled(self):
        """Only r_eff was given: n, ellip, pa, dx, dy come from SERSIC_DEFAULTS."""
        assert sersic_scene().host.profile == {
            "r_eff": 1.0,
            "n": 1.0,
            "ellip": 0.0,
            "pa": 0.0,
            "dx": 0.0,
            "dy": 0.0,
        }

    def test_given_values_kept(self):
        """Explicit kwargs override the defaults."""
        assert sersic_scene(n=4, ellip=0.3, pa=45, dx=0.8).host.profile["n"] == 4

    @pytest.mark.parametrize(
        "bad",
        [
            {"n": 0},
            {"ellip": 1.0},
            {"r_eff": -1},
            {"n": float("nan")},
            {"dx": float("inf")},
            {"r_eff": float("nan")},
        ],
    )
    def test_invalid_values_raise(self, bad):
        """n > 0, 0 <= ellip < 1, r_eff > 0 and finite values are enforced (#86)."""
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
