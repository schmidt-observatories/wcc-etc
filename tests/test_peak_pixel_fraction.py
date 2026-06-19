import pytest
import wcc_etc


def _scene(mag=15.0):
    return wcc_etc.get_scene(name='G2V', mag=mag, background='zodi',
                             bandpass='johnson_v',
                             background_prop={'bandpass': 'johnson_v', 'mag': 22.5})


def test_peak_pixel_fraction_in_unit_interval_and_matches_render():
    sim = wcc_etc.Simulation.from_sensor_and_scene('sony:r', _scene())
    frac = sim.peak_pixel_fraction()
    assert 0.0 < frac <= 1.0
    b = sim._image_render_bundle(wcc_etc.AiryPSF(), None, 128, 11)
    assert frac == pytest.approx(float(b["psf_norm"].max()), rel=1e-9)


def test_defocus_peak_fraction_below_in_focus():
    sim_def = wcc_etc.Simulation.from_sensorfilter('zwo:bb2', _scene())  # 2-wave defocus
    sim_foc = wcc_etc.Simulation.from_sensorfilter('zwo:r', _scene())    # in-focus
    assert sim_def.peak_pixel_fraction() < 0.5 * sim_foc.peak_pixel_fraction()
