import numpy as np
import pytest

import wcc_etc


def _sim(mag=15):
    scene = wcc_etc.get_scene(
        name="G5V",
        mag=mag,
        background="zodi",
        bandpass="johnson_r",
        background_prop={"bandpass": "johnson_r", "mag": 22.5},
    )
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_n_reads_is_mutable_parameter_and_defaults_to_one():
    sim = _sim()
    assert sim.meta.get("n_reads") == 1
    assert any(k.endswith("n_reads") or k == "n_reads" for k in sim.mutable_parameters)
    sim.update(n_reads=4)
    assert sim.meta["n_reads"] == 4


def test_get_snr_n_reads_one_matches_baseline():
    sim = _sim()
    baseline = sim.get_snr(60)["snr"]
    assert np.isclose(sim.get_snr(60, n_reads=1)["snr"], baseline)


def test_more_reads_lowers_snr_at_fixed_time():
    sim = _sim(mag=19)  # faint -> read-noise matters
    s1 = sim.get_snr(30, n_reads=1)["snr"]
    s9 = sim.get_snr(30, n_reads=9)["snr"]
    assert s9 < s1


def test_exptime_for_snr_roundtrips():
    # get_exptime_for_snr is the analytic path; verify roundtrip with get_snr_airy
    sim = _sim()
    for target in (20.0, 100.0):
        t = sim.get_exptime_for_snr(target)
        with pytest.warns(DeprecationWarning):
            got = float(sim.get_snr_airy(t).value)
        assert np.isclose(got, target, rtol=1e-3)


def test_exptime_for_snr_roundtrips_with_reads():
    # get_exptime_for_snr is the analytic path; verify roundtrip with get_snr_airy
    sim = _sim(mag=18)
    t = sim.get_exptime_for_snr(50.0, n_reads=5)
    with pytest.warns(DeprecationWarning):
        got = float(sim.get_snr_airy(t, n_reads=5).value)
    assert np.isclose(got, 50.0, rtol=1e-3)


def test_exptime_uses_meta_n_reads_when_unset():
    sim = _sim(mag=18)
    sim.update(n_reads=5)
    t_meta = sim.get_exptime_for_snr(50.0)  # resolves n_reads=5 from meta
    t_explicit = sim.get_exptime_for_snr(50.0, n_reads=5)
    assert np.isclose(t_meta.value, t_explicit.value, rtol=1e-9)


def test_n_reads_below_one_raises():
    import pytest

    sim = _sim()
    with pytest.raises(ValueError):
        sim.get_snr(60, n_reads=0)
    with pytest.raises(ValueError):
        sim.get_exptime_for_snr(50, n_reads=-1)
