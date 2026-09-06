"""SNR with a Sersic host: nuclear vs offset transients."""

import pytest

import wcc_etc
from tests.helpers import make_scene

HOST = {"mag": 15, "bandpass": "johnson_r", "profile": "sersic", "r_eff": 0.3, "n": 4.0}


def host_sim(**host):
    """Mag-21 G5V transient on sony:r with a bright n=4 bulge host."""
    scene = make_scene(mag=21, host="G5V", host_prop=HOST | host)
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def snr_with_host(**host):
    return host_sim(**host).get_snr(time=60.0, warn=False)["snr"]


class TestHostPosition:
    def test_nuclear_transient_has_lower_snr_than_offset(self):
        """A source on the bulge cusp sees more host shot noise than one 1.5" out."""
        assert snr_with_host(dx=0.0) < snr_with_host(dx=1.5)

    def test_far_offset_host_approaches_hostless(self):
        """With a compact host 2" away (beyond the grid) the SNR is the hostless SNR."""
        hostless = wcc_etc.Simulation.from_sensor_and_scene(
            "sony:r", make_scene(mag=21)
        )
        ref = hostless.get_snr(time=60.0, warn=False)["snr"]
        assert snr_with_host(dx=2.0, r_eff=0.05, n=1.0) == pytest.approx(ref, rel=1e-2)


class TestExptimeInverse:
    def test_exptime_inverts_snr_with_host(self):
        """get_image_exptime_for_snr(get_snr(t)) returns t with an extended host."""
        sim = host_sim()
        snr = sim.get_snr(time=60.0, warn=False)["snr"]
        t = sim.get_image_exptime_for_snr(snr, warn=False)["time_s"]
        assert t == pytest.approx(60.0, rel=1e-3)
