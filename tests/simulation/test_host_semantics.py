"""Host semantics: elements are classified by ``surface_brightness``, not by name.

Issue #63. A host given as an ordinary magnitude (``surface_brightness=False``)
is a *total* count rate. It used to be added to every detector pixel, which
overcounted its shot-noise contribution by ``n_pix / enclosed_fraction`` — 64.6x
for the representative mag 20 source / mag 16 host scene. It is now treated as
unresolved and co-located with the source, so it rides the same PSF.
"""

import numpy as np
import pytest

import wcc_etc
from wcc_etc.psfsim import AiryPSF, aperture_snr_radial

TIME_S = 15.06
EE_FRAC = 0.882


def host_scene(host_mag=16, surface_brightness=False, source_mag=20):
    """Representative scene: a mag-20 source with a mag-16 host and zodi sky."""
    return wcc_etc.get_scene(
        "G5V",
        mag=source_mag,
        host={
            "name": "G5V",
            "mag": host_mag,
            "surface_brightness": surface_brightness,
        },
        background="zodi",
    )


@pytest.fixture
def host_sim():
    """Simulation of the representative unresolved-host scene."""
    return wcc_etc.Simulation.from_sensor_and_scene("sony:bb", host_scene())


@pytest.fixture
def hostless_sim():
    """The same scene with no host, for differencing the host's contribution."""
    scene = wcc_etc.get_scene("G5V", mag=20, host=None, background="zodi")
    return wcc_etc.Simulation.from_sensor_and_scene("sony:bb", scene)


class TestElementClassification:
    """`_count_rate_components` dispatches on surface_brightness, not on name."""

    def test_unresolved_host_is_a_contaminant(self, host_sim):
        """A surface_brightness=False host carries a total rate, so it is a contaminant."""
        assert host_sim._count_rate_components()["contaminant_rate_total"] > 0

    def test_unresolved_host_is_not_diffuse(self, host_sim):
        """The #63 bug: that total must not land in the per-pixel diffuse term."""
        assert host_sim._count_rate_components()["diffuse_rate_per_pix"] == 0

    def test_resolved_host_is_diffuse(self):
        """A surface_brightness=True host is a per-pixel rate, like the sky."""
        sim = wcc_etc.Simulation.from_sensor_and_scene(
            "sony:bb", host_scene(host_mag=22, surface_brightness=True)
        )
        assert sim._count_rate_components()["diffuse_rate_per_pix"] > 0

    def test_resolved_host_is_not_a_contaminant(self):
        """A resolved host is already per-pixel, so nothing rides the PSF."""
        sim = wcc_etc.Simulation.from_sensor_and_scene(
            "sony:bb", host_scene(host_mag=22, surface_brightness=True)
        )
        assert sim._count_rate_components()["contaminant_rate_total"] == 0

    def test_sky_background_stays_per_pixel(self, host_sim):
        """Zodi is surface_brightness=True and must remain a per-pixel rate."""
        assert host_sim._count_rate_components()["background_rate_per_pix"] > 0

    def test_source_is_unaffected_by_the_host(self, host_sim, hostless_sim):
        """Reclassifying the host must not change the source count rate."""
        assert host_sim._count_rate_components()["source_rate_total"] == pytest.approx(
            hostless_sim._count_rate_components()["source_rate_total"]
        )


class TestUnresolvedHostShotNoise:
    """The host's in-aperture charge follows the PSF, not the pixel count."""

    def test_host_variance_equals_total_times_enclosed_fraction(
        self, host_sim, hostless_sim
    ):
        """Extra aperture variance from the host is host_rate * t * enclosed_fraction."""
        with_host = host_sim.get_image_snr(TIME_S, psf=AiryPSF(), ee_frac=EE_FRAC)
        without = hostless_sim.get_image_snr(TIME_S, psf=AiryPSF(), ee_frac=EE_FRAC)
        host_rate = host_sim._count_rate_components()["contaminant_rate_total"]

        added_variance = with_host["noise_e"] ** 2 - without["noise_e"] ** 2
        expected = host_rate * TIME_S * with_host["enclosed_fraction"]
        assert added_variance == pytest.approx(expected, rel=1e-6)

    def test_host_charge_matches_the_issue_63_figure(self, host_sim):
        """The representative scene gives ~597k e-, not the ~38.6M of the old code."""
        r = host_sim.get_image_snr(TIME_S, psf=AiryPSF(), ee_frac=EE_FRAC)
        host_rate = host_sim._count_rate_components()["contaminant_rate_total"]
        host_charge = host_rate * TIME_S * r["enclosed_fraction"]
        assert host_charge == pytest.approx(597_361, rel=0.02)

    def test_host_charge_is_far_below_the_per_pixel_treatment(self, host_sim):
        """Guards the regression direction: per-pixel replication is ~60x larger."""
        r = host_sim.get_image_snr(TIME_S, psf=AiryPSF(), ee_frac=EE_FRAC)
        assert r["n_pix"] / r["enclosed_fraction"] > 50

    def test_host_still_lowers_snr(self, host_sim, hostless_sim):
        """The host is a real noise source — it must not vanish from the budget."""
        with_host = host_sim.get_image_snr(TIME_S, psf=AiryPSF(), ee_frac=EE_FRAC)
        without = hostless_sim.get_image_snr(TIME_S, psf=AiryPSF(), ee_frac=EE_FRAC)
        assert with_host["snr"] < without["snr"]


class TestApertureSnrRadialContaminant:
    """The `contaminant_e_total` term on the SNR helper."""

    @staticmethod
    def psf():
        """A tiny normalized PSF: one bright centre pixel plus a faint halo."""
        p = np.full((5, 5), 1.0)
        p[2, 2] = 20.0
        return p / p.sum()

    def test_contaminant_adds_variance_scaled_by_enclosed_fraction(self):
        """Added variance at each radius is contaminant_e_total * enclosed_fraction."""
        base = aperture_snr_radial(self.psf(), 10.0, 1000.0, 0.0, 0.0, 0.0)
        with_c = aperture_snr_radial(
            self.psf(), 10.0, 1000.0, 0.0, 0.0, 0.0, contaminant_e_total=500.0
        )
        added = with_c["noise_e"] ** 2 - base["noise_e"] ** 2
        assert added == pytest.approx(500.0 * base["enclosed_fraction"])

    def test_contaminant_does_not_change_the_signal(self):
        """A contaminant is noise only — it never counts as source signal."""
        base = aperture_snr_radial(self.psf(), 10.0, 1000.0, 0.0, 0.0, 0.0)
        with_c = aperture_snr_radial(
            self.psf(), 10.0, 1000.0, 0.0, 0.0, 0.0, contaminant_e_total=500.0
        )
        assert with_c["signal_e"] == pytest.approx(base["signal_e"])

    def test_zero_contaminant_is_the_default(self):
        """Omitting the term must reproduce the previous behaviour exactly."""
        a = aperture_snr_radial(self.psf(), 10.0, 1000.0, 2.0, 1.0, 3.0)
        b = aperture_snr_radial(
            self.psf(), 10.0, 1000.0, 2.0, 1.0, 3.0, contaminant_e_total=0.0
        )
        assert a["snr"] == pytest.approx(b["snr"])


class TestSharedChargeBudget:
    """Every saturation API must see the same electrons (issue #63, item 4)."""

    def test_peak_pixel_matches_the_rendered_image(self, host_sim):
        """get_peak_pixel equals the brightest pixel of the noiseless render."""
        imsim = wcc_etc.ImageSimulator(host_sim, npix=128, oversample=11)
        img = imsim.simulate(time=TIME_S, psf=AiryPSF(), add_noise=False)
        peak = host_sim.get_peak_pixel(
            TIME_S, units="e-", psf=AiryPSF(), npix=128, oversample=11
        )
        assert float(peak.value) == pytest.approx(img.image_clean.max(), rel=1e-6)

    def test_peak_pixel_includes_the_unresolved_host(self, host_sim, hostless_sim):
        """A mag-16 host dominates the core, so it must raise the peak pixel."""
        kw = dict(units="e-", psf=AiryPSF(), npix=128, oversample=11)
        assert host_sim.get_peak_pixel(TIME_S, **kw) > hostless_sim.get_peak_pixel(
            TIME_S, **kw
        )
