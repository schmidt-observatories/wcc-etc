"""Extended (Sersic) host: classification and the shared rate image."""

import numpy as np
import pytest

import wcc_etc
from tests.helpers import make_scene, make_simulation

HOST = {"mag": 17, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 0.1}


def sersic_sim(source_mag=20, **host):
    """G5V source on sony:r with a compact (0.1") G5V Sersic host."""
    scene = make_scene(mag=source_mag, host="G5V", host_prop=HOST | host)
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def uniform_sb_rate(mag):
    """Per-pixel rate of a uniform surface-brightness host at `mag` mag/arcsec^2."""
    prop = {"mag": mag, "bandpass": "johnson_r", "surface_brightness": True}
    scene = make_scene(host="G5V", host_prop=prop)
    sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
    return sim._count_rate_components()["diffuse_rate_per_pix"]


@pytest.fixture
def ext_sim():
    return sersic_sim()


class TestClassification:
    def test_profiled_host_is_extended(self, ext_sim):
        """A host with a profile lands in 'extended', not diffuse or contaminant."""
        assert len(ext_sim._count_rate_components()["extended"]) == 1

    def test_profiled_host_is_not_a_contaminant(self, ext_sim):
        """It must not also ride the source PSF (double counting)."""
        assert ext_sim._count_rate_components()["contaminant_rate_total"] == 0

    def test_profiled_sb_host_is_not_diffuse(self):
        """A mu_e (surface_brightness=True) host is extended, not uniform diffuse."""
        sim = sersic_sim(surface_brightness=True, mag=22)
        assert sim._count_rate_components()["diffuse_rate_per_pix"] == 0

    def test_no_profile_gives_empty_list(self):
        """Scenes without a profile are unchanged."""
        assert make_simulation()._count_rate_components()["extended"] == []


class TestExtendedRateImage:
    def test_total_mag_conserves_flux(self, ext_sim):
        """A contained total-mag host image sums to the point-source rate at its mag."""
        b = ext_sim._image_render_bundle(ext_sim.default_psf, None, 128, 11)
        ref = make_simulation(mag=17)._count_rate_components()["source_rate_total"]
        assert b["extended_rate_image"].sum() == pytest.approx(ref, rel=1e-2)

    def test_sb_mode_matches_uniform_sb_at_r_eff(self):
        """mu_e mode: the pixel at r_eff reads the uniform-SB per-pixel rate."""
        sim = sersic_sim(surface_brightness=True, mag=22, r_eff=0.5, n=1.0)
        b = sim._image_render_bundle(sim.default_psf, None, 128, 11)
        c = (128 - 1) // 2  # psfsim.grid_center: where a center=None render sits
        r_pix = int(round(0.5 / (b["plate_scale_mas"] / 1000.0)))
        assert b["extended_rate_image"][c, c + r_pix] == pytest.approx(
            uniform_sb_rate(22), rel=5e-2
        )

    def test_dx_shifts_peak(self):
        """dx moves the host peak by dx / plate_scale pixels."""
        sim = sersic_sim(dx=0.3)
        b = sim._image_render_bundle(sim.default_psf, None, 128, 11)
        _, ix = np.unravel_index(np.argmax(b["extended_rate_image"]), (128, 128))
        assert ix == pytest.approx(63 + 0.3 / (b["plate_scale_mas"] / 1000.0), abs=0.5)

    def test_zero_image_without_profile(self):
        """No extended element -> an all-zero image of the right shape."""
        sim = make_simulation()
        b = sim._image_render_bundle(sim.default_psf, None, 64, 11)
        img = b["extended_rate_image"]
        assert img.shape == (64, 64) and not img.any()


class TestOneBudget:
    def test_per_frame_image_includes_host(self, ext_sim):
        """_per_frame_clean_image_e carries the extended charge."""
        b = ext_sim._image_render_bundle(ext_sim.default_psf, None, 128, 11)
        hostless = make_simulation(mag=20)
        bh = hostless._image_render_bundle(hostless.default_psf, None, 128, 11)
        diff = ext_sim._per_frame_clean_image_e(
            b, 10.0
        ) - hostless._per_frame_clean_image_e(bh, 10.0)
        assert diff.sum() == pytest.approx(
            10.0 * b["extended_rate_image"].sum(), rel=1e-6
        )

    def test_peak_pixel_is_image_max(self, ext_sim):
        """get_peak_pixel equals the brightest pixel of the per-frame clean image."""
        b = ext_sim._image_render_bundle(ext_sim.default_psf, None, 128, 11)
        expect = ext_sim._per_frame_clean_image_e(b, 10.0).max()
        peak = ext_sim.get_peak_pixel(10.0, units="e-").value
        assert peak == pytest.approx(expect, rel=1e-9)


class TestDeprecatedPathWarns:
    def test_airy_path_warns_profile_ignored(self, ext_sim):
        """The analytic Airy path cannot place a profile; it says so."""
        with pytest.warns(UserWarning, match="profile"):
            ext_sim._countrates_in_aperture()
