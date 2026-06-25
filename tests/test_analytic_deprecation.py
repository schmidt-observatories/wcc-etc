import warnings

import pytest

import wcc_etc


def _sim():
    scene = wcc_etc.get_scene(
        name="G2V",
        mag=15.0,
        background="zodi",
        bandpass="johnson_v",
        background_prop={"bandpass": "johnson_v", "mag": 22.5},
    )
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.get_countrates(units="e/s"),
        lambda s: s.compute_psf_profile(),
        lambda s: s.get_signal_and_variance(1.0),
        lambda s: s.get_exptime_for_snr(50.0),
    ],
)
def test_analytic_methods_warn(call):
    s = _sim()
    with pytest.warns(DeprecationWarning):
        call(s)


def test_peak_pixel_fraction_retired():
    s = _sim()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = s.compute_psf_profile()
    assert "peak_pixel_fraction" not in prof
