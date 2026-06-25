"""Tests for the sensorfilter registry, focus-level dispatch, and ImageSimulator.from_sensorfilter."""

import pytest
import wcc_etc
from wcc_etc.io import (
    SENSORS,
    _SENSORFILTER_FOCUS,
    _SENSORFILTER_IMPLEMENTED,
    sensor_info,
)
from wcc_etc.simulation import Simulation, _psf_from_focus_level
from wcc_etc.psfsim import AiryPSF, DefocusPSF, ImageSimulator
from tests.helpers import make_scene


class TestSensorRegistry:
    def test_zwo_no_legacy_defocus(self):
        assert "r_defocus" not in SENSORS["zwo"] and "bb_defocus" not in SENSORS["zwo"]

    def test_zwo_has_r_plus1(self):
        assert "r+1" in SENSORS["zwo"]

    def test_zwo_has_r_minus1(self):
        assert "r-1" in SENSORS["zwo"]

    def test_zwo_has_bb2(self):
        assert "bb2" in SENSORS["zwo"]

    def test_zwo_has_hbeta(self):
        assert "hbeta" in SENSORS["zwo"]

    def test_zwo_r_variants_share_throughput(self):
        assert SENSORS["zwo"]["r+1"] == SENSORS["zwo"]["r"]
        assert SENSORS["zwo"]["r-1"] == SENSORS["zwo"]["r"]

    def test_zwo_bb2_shares_throughput(self):
        assert SENSORS["zwo"]["bb2"] == SENSORS["zwo"]["bb"]

    def test_zwo_hbeta_is_none(self):
        assert SENSORS["zwo"]["hbeta"] is None

    def test_sensorfilter_focus_covers_all_sensor_info_labels(self):
        all_sf = {entry["sensorfilter"] for entry in sensor_info.values()}
        assert all_sf == set(_SENSORFILTER_FOCUS.keys())

    def test_zwo_r_is_0wave(self):
        assert _SENSORFILTER_FOCUS["zwo:r"] == "0wave"

    def test_qcmos_bb_is_0wave(self):
        assert _SENSORFILTER_FOCUS["qcmos:bb"] == "0wave"

    def test_zwo_r_plus1_is_1wave(self):
        assert _SENSORFILTER_FOCUS["zwo:r+1"] == "1wave"

    def test_zwo_r_minus1_is_1wave(self):
        assert _SENSORFILTER_FOCUS["zwo:r-1"] == "1wave"

    def test_sensorfilter_focus_2wave(self):
        assert _SENSORFILTER_FOCUS["zwo:bb2"] == "2wave"

    def test_psf_from_focus_level_0wave(self):
        assert isinstance(_psf_from_focus_level("0wave"), AiryPSF)

    def test_psf_from_focus_level_1wave(self):
        assert isinstance(_psf_from_focus_level("1wave"), DefocusPSF)

    def test_psf_from_focus_level_2wave(self):
        assert isinstance(_psf_from_focus_level("2wave"), DefocusPSF)

    def test_psf_from_focus_level_unknown(self):
        with pytest.raises(ValueError, match="Unknown focus_level"):
            _psf_from_focus_level("3wave")

    def test_narrowband_filters_not_implemented(self):
        for sf in ("zwo:nii", "zwo:halpha", "zwo:hbeta", "zwo:heii", "zwo:oiii"):
            assert _SENSORFILTER_IMPLEMENTED[sf] is False

    def test_standard_filters_are_implemented(self):
        for sf in ("zwo:r", "zwo:r+1", "zwo:bb2", "qcmos:bb"):
            assert _SENSORFILTER_IMPLEMENTED[sf] is True


class TestFromSensorfilter:
    def test_default_psf_is_none_before_from_sensorfilter(self):
        sim = Simulation(telescope=None, sensor=None, scene=None)
        assert sim._default_psf is None

    def test_0wave_uses_airy(self):
        assert isinstance(
            Simulation.from_sensorfilter("zwo:r", make_scene())._default_psf, AiryPSF
        )

    def test_1wave_uses_defocus(self):
        assert isinstance(
            Simulation.from_sensorfilter("zwo:r+1", make_scene())._default_psf,
            DefocusPSF,
        )

    def test_2wave_uses_defocus(self):
        assert isinstance(
            Simulation.from_sensorfilter("zwo:bb2", make_scene())._default_psf,
            DefocusPSF,
        )

    def test_qcmos_in_focus(self):
        assert isinstance(
            Simulation.from_sensorfilter("qcmos:bb", make_scene())._default_psf,
            AiryPSF,
        )

    def test_not_implemented_raises(self):
        for sf in ("zwo:nii", "zwo:halpha", "zwo:hbeta", "zwo:heii", "zwo:oiii"):
            with pytest.raises(NotImplementedError, match="not yet implemented"):
                Simulation.from_sensorfilter(sf, make_scene())

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sensorfilter"):
            Simulation.from_sensorfilter("zwo:nonexistent", make_scene())

    def test_builds_working_sim(self):
        sim = Simulation.from_sensorfilter("zwo:r", make_scene())
        assert sim.get_snr(60)["snr"] > 0

    def test_defocused_snr_differs_from_infocus(self):
        scene = make_scene()
        snr_inf = Simulation.from_sensorfilter("zwo:r", scene).get_image_snr(60)["snr"]
        snr_def = Simulation.from_sensorfilter("zwo:r+1", scene).get_image_snr(60)[
            "snr"
        ]
        assert abs(snr_inf - snr_def) > 0.01

    def test_psf_override_works(self):
        scene = make_scene()
        sim_def = Simulation.from_sensorfilter("zwo:r+1", scene)
        sim_ref = Simulation.from_sensorfilter("zwo:r", scene)
        snr_override = sim_def.get_image_snr(60, psf=AiryPSF())["snr"]
        snr_ref = sim_ref.get_image_snr(60)["snr"]
        assert abs(snr_override - snr_ref) < 0.01

    def test_exptime_differs_for_defocused(self):
        scene = make_scene()
        t_inf = Simulation.from_sensorfilter("zwo:r", scene).get_image_exptime_for_snr(
            10
        )["time_s"]
        t_def = Simulation.from_sensorfilter(
            "zwo:r+1", scene
        ).get_image_exptime_for_snr(10)["time_s"]
        assert abs(t_inf - t_def) > 0.01

    def test_from_sensor_and_scene_default_psf_is_none(self):
        scene = make_scene()
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        assert sim._default_psf is None

    def test_from_sensor_and_scene_fallback_to_airy(self):
        scene = make_scene()
        sim = wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)
        result = sim.get_image_snr(60)
        result_explicit = sim.get_image_snr(60, psf=AiryPSF())
        assert abs(result["snr"] - result_explicit["snr"]) < 1e-6


class TestImageSimulatorFromSensorfilter:
    def test_0wave_uses_airy(self):
        imsim = ImageSimulator.from_sensorfilter("zwo:r", make_scene())
        assert isinstance(imsim.sim._default_psf, AiryPSF)

    def test_1wave_uses_defocus(self):
        imsim = ImageSimulator.from_sensorfilter("zwo:r+1", make_scene())
        assert isinstance(imsim.sim._default_psf, DefocusPSF)

    def test_2wave_uses_defocus(self):
        imsim = ImageSimulator.from_sensorfilter("zwo:bb2", make_scene())
        assert isinstance(imsim.sim._default_psf, DefocusPSF)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sensorfilter"):
            ImageSimulator.from_sensorfilter("zwo:nonexistent", make_scene())

    def test_passes_npix(self):
        imsim = ImageSimulator.from_sensorfilter(
            "zwo:r", make_scene(), npix=128, oversample=5
        )
        assert imsim.npix == 128

    def test_passes_oversample(self):
        imsim = ImageSimulator.from_sensorfilter(
            "zwo:r", make_scene(), npix=128, oversample=5
        )
        assert imsim.oversample == 5

    def test_builds_working_imsim_returns_result(self):
        imsim = ImageSimulator.from_sensorfilter("zwo:r", make_scene(), npix=64)
        result = imsim.simulate(60, add_noise=False)
        assert result is not None

    def test_builds_working_imsim_has_positive_signal(self):
        imsim = ImageSimulator.from_sensorfilter("zwo:r", make_scene(), npix=64)
        result = imsim.simulate(60, add_noise=False)
        assert result.image_e.max() > 0
