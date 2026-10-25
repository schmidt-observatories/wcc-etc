"""NGC galaxy hosts and redshift shortcuts at the scene level."""

import numpy as np
import pytest
from astropy.cosmology import Planck18

from wcc_etc import get_scene
from wcc_etc.scene import get_scene_element

Z = 0.9


def ngc_host(**kw):
    return get_scene_element("ngc0628", redshift=Z, **kw)


class TestSersicKpc:
    def test_r_kpc_converts_at_redshift(self):
        """r_kpc is turned into r_eff (arcsec) with arcsec_per_kpc_proper(z)."""
        el = get_scene_element("G5V", mag=16, profile="sersic", r_kpc=5, redshift=Z)
        assert np.isclose(
            el.profile["r_eff"], 5 * Planck18.arcsec_per_kpc_proper(Z).value
        )

    def test_r_eff_wins_over_r_kpc(self):
        """An explicit r_eff is used as is."""
        el = get_scene_element(
            "G5V", mag=16, profile="sersic", r_kpc=5, r_eff=0.3, redshift=Z
        )
        assert el.profile["r_eff"] == 0.3

    def test_r_kpc_needs_redshift(self):
        """r_kpc without a redshift cannot be sized."""
        with pytest.raises(ValueError):
            get_scene_element("G5V", mag=16, profile="sersic", r_kpc=5).profile

    def test_sersic_profile_carries_model(self):
        """The profile dict says which renderer to use."""
        el = get_scene_element("G5V", mag=16, profile="sersic", r_eff=0.3)
        assert el.profile["model"] == "sersic"


class TestNgcProfile:
    def test_profile_dict(self):
        """An NGC name as profile gives the ngc spatial model at the redshift."""
        el = ngc_host(dx=0.3, pa=45)
        assert el.profile == {
            "model": "ngc",
            "name": "ngc0628",
            "redshift": Z,
            "dx": 0.3,
            "dy": 0.0,
            "pa": 45,
        }

    def test_needs_redshift(self):
        """An NGC image cannot be placed without a redshift."""
        with pytest.raises(ValueError):
            get_scene_element("ngc0628").profile

    def test_rejects_surface_brightness(self):
        """An NGC image is total-flux normalised; mag/arcsec^2 makes no sense."""
        with pytest.raises(ValueError):
            ngc_host(surface_brightness=True).profile

    def test_name_is_case_insensitive(self):
        """NGC0628 and ngc0628 are the same galaxy."""
        assert get_scene_element("NGC0628", redshift=Z).profile["name"] == "ngc0628"

    def test_offsets_are_mutable(self):
        """dx, dy, pa can be updated on an NGC host."""
        el = ngc_host()
        el.update(dx=0.5)
        assert el.profile["dx"] == 0.5


class TestNgcDefaults:
    def test_spectrum_is_the_galaxy_template(self):
        """A bare NGC name uses the galaxy's own spectrum."""
        assert ngc_host().meta["spectrum"] == "ngc0628"

    def test_default_absolute_magnitude(self):
        """slicersim's defaults: magabs=-21 in sdss_r (AB)."""
        el = ngc_host()
        assert (el.meta["magabs"], el.meta["bandpass"], el.meta["magsys"]) == (
            -21,
            "sdss_r",
            "abmag",
        )

    def test_observed_mag_overrides_default(self):
        """Giving mag drops the default magabs."""
        assert ngc_host(mag=18).mag.value == 18


class TestSceneRedshift:
    def test_redshift_reaches_the_host(self):
        """get_scene(redshift=z) sets the host redshift too."""
        scene = get_scene(
            "hsiao07",
            mag=None,
            magabs=-19.3,
            redshift=Z,
            bandpass="sdss_r",
            magsys="abmag",
            host="ngc0628",
        )
        assert scene.host.meta["redshift"] == Z

    def test_host_prop_redshift_wins(self):
        """A redshift in host_prop is kept."""
        scene = get_scene(
            "hsiao07", mag=20, redshift=Z, host="ngc0628", host_prop={"redshift": 0.5}
        )
        assert scene.host.meta["redshift"] == 0.5

    def test_extended_rate_image_sums_to_host_rate(self):
        """The PSF-convolved NGC host image carries the host rate times its on-grid fraction."""
        from wcc_etc import Simulation
        from wcc_etc.psfsim import ImageSimulator

        scene = get_scene(
            "hsiao07",
            mag=None,
            magabs=-19.3,
            redshift=Z,
            bandpass="sdss_r",
            magsys="abmag",
            host="ngc0628",
            background=None,
        )
        sim = Simulation.from_sensorfilter("zwo:r", scene)
        b = sim._image_render_bundle(sim.default_psf, None, 128, 5)
        host_rate = sim._count_rate_components()["extended"][0][0]
        from wcc_etc.extended import render_profile

        on_grid = render_profile(
            scene.host.profile, b["plate_scale_mas"] / 1000, 128
        ).sum()
        assert np.isclose(
            b["extended_rate_image"].sum(), host_rate * on_grid, rtol=0.05
        )
