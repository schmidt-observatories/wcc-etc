import numpy as np
import wcc_etc


def _sim(mag):
    scene = wcc_etc.get_scene(name="G5V", mag=mag, background="zodi",
                              bandpass="johnson_r",
                              background_prop={"bandpass": "johnson_r", "mag": 22.5})
    return wcc_etc.Simulation.from_sensor_and_scene("sony:r", scene)


def test_peak_pixel_n_reads_one_matches_baseline():
    sim = _sim(12)
    base = sim.get_peak_pixel(60, units="e-").value
    assert np.isclose(sim.get_peak_pixel(60, units="e-", n_reads=1).value, base)


def test_peak_pixel_scales_inverse_with_reads_above_bias():
    # in electrons (no bias): per-frame charge halves when n_reads doubles
    sim = _sim(12)
    p1 = sim.get_peak_pixel(60, units="e-", n_reads=1).value
    p2 = sim.get_peak_pixel(60, units="e-", n_reads=2).value
    assert np.isclose(p2, p1 / 2.0, rtol=1e-6)


def test_more_reads_can_unsaturate_a_bright_star():
    # mag=17: saturates in a single 60-s frame; splits to under full-well at n_reads=100
    sim = _sim(17)
    assert sim.is_saturated(60, n_reads=1)            # one long frame clips
    assert not sim.is_saturated(60, n_reads=100)      # split -> per-frame under full well
