import pytest
from wcc_etc.io import SENSORS, _SENSORFILTER_FOCUS, sensor_info


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
