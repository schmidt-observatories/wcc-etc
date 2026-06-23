"""Top-level shared fixtures and factory helpers for the wcc_etc test suite."""

import pytest
import wcc_etc


# ---------------------------------------------------------------------------
# Module-level factory helpers
# Used directly by tests that need parametric control (different mags, sensors)
# ---------------------------------------------------------------------------


def make_scene(
    name="G5V", mag=15, bandpass="johnson_r", background="zodi", background_mag=22.5
):
    """Build a standard scene with a zodi background."""
    return wcc_etc.get_scene(
        name=name,
        mag=mag,
        background=background,
        bandpass=bandpass,
        background_prop={"bandpass": bandpass, "mag": background_mag},
    )


def make_simulation(mag=15, sensor="sony:r", name="G5V", bandpass="johnson_r"):
    """Build a Simulation from a standard zodi-background scene."""
    scene = make_scene(name=name, mag=mag, bandpass=bandpass)
    return wcc_etc.Simulation.from_sensor_and_scene(sensor, scene)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def g5v_scene():
    """G5V star, mag=15, johnson_r band, zodi background."""
    return make_scene()


@pytest.fixture
def g2v_scene():
    """G2V star, mag=15, johnson_v band, zodi background."""
    return make_scene(name="G2V", bandpass="johnson_v")


@pytest.fixture
def std_sim():
    """G5V mag=15 simulation on sony:r — the most-common sim in the suite."""
    return make_simulation()


@pytest.fixture
def g2v_sim():
    """G2V mag=15 simulation on sony:r — used in analytic/count-rate tests."""
    return make_simulation(name="G2V", bandpass="johnson_v")
