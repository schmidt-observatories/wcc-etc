import pytest
import wcc_etc
from wcc_etc.io import SENSORS, _SENSORFILTER_FOCUS, sensor_info
from wcc_etc.simulation import Simulation, _psf_from_focus_level
from wcc_etc.psfsim import AiryPSF, DefocusPSF


def test_sensors_zwo_no_legacy_defocus():
    assert "r_defocus" not in SENSORS["zwo"]
    assert "bb_defocus" not in SENSORS["zwo"]


def test_sensors_zwo_has_new_bands():
    assert "r+1" in SENSORS["zwo"]
    assert "r-1" in SENSORS["zwo"]
    assert "bb2" in SENSORS["zwo"]
    assert "hbeta" in SENSORS["zwo"]


def test_sensors_zwo_r_variants_share_throughput():
    assert SENSORS["zwo"]["r+1"] == SENSORS["zwo"]["r"]
    assert SENSORS["zwo"]["r-1"] == SENSORS["zwo"]["r"]


def test_sensors_zwo_bb2_shares_throughput():
    assert SENSORS["zwo"]["bb2"] == SENSORS["zwo"]["bb"]


def test_sensors_zwo_hbeta_is_none():
    assert SENSORS["zwo"]["hbeta"] is None


def test_sensorfilter_focus_covers_all_sensor_info_labels():
    all_sf = {entry["sensorfilter"] for entry in sensor_info.values()}
    assert all_sf == set(_SENSORFILTER_FOCUS.keys())


def test_sensorfilter_focus_0wave_in_focus():
    assert _SENSORFILTER_FOCUS["zwo:r"] == "0wave"
    assert _SENSORFILTER_FOCUS["qcmos:bb"] == "0wave"


def test_sensorfilter_focus_1wave():
    assert _SENSORFILTER_FOCUS["zwo:r+1"] == "1wave"
    assert _SENSORFILTER_FOCUS["zwo:r-1"] == "1wave"


def test_sensorfilter_focus_2wave():
    assert _SENSORFILTER_FOCUS["zwo:bb2"] == "2wave"


def test_psf_from_focus_level_0wave():
    psf = _psf_from_focus_level("0wave")
    assert isinstance(psf, AiryPSF)


def test_psf_from_focus_level_1wave():
    psf = _psf_from_focus_level("1wave")
    assert isinstance(psf, DefocusPSF)


def test_psf_from_focus_level_2wave():
    psf = _psf_from_focus_level("2wave")
    assert isinstance(psf, DefocusPSF)


def test_psf_from_focus_level_unknown():
    with pytest.raises(ValueError, match="Unknown focus_level"):
        _psf_from_focus_level("3wave")


def test_simulation_default_psf_is_none_by_default():
    sim = Simulation(telescope=None, sensor=None, scene=None)
    assert sim._default_psf is None


def _make_scene():
    return wcc_etc.get_scene(name="G5V", mag=15, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})


def test_from_sensorfilter_0wave_uses_airy():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:r", scene)
    assert isinstance(sim._default_psf, AiryPSF)


def test_from_sensorfilter_1wave_uses_defocus():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:r+1", scene)
    assert isinstance(sim._default_psf, DefocusPSF)


def test_from_sensorfilter_2wave_uses_defocus():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:bb2", scene)
    assert isinstance(sim._default_psf, DefocusPSF)


def test_from_sensorfilter_qcmos():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("qcmos:bb", scene)
    assert isinstance(sim._default_psf, AiryPSF)


def test_from_sensorfilter_unknown_raises():
    scene = _make_scene()
    with pytest.raises(ValueError, match="Unknown sensorfilter"):
        Simulation.from_sensorfilter("zwo:nonexistent", scene)


def test_from_sensorfilter_builds_working_sim():
    scene = _make_scene()
    sim = Simulation.from_sensorfilter("zwo:r", scene)
    # basic sanity: can compute an analytic SNR
    snr = sim.get_snr(60)
    assert snr.value > 0
