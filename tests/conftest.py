"""Top-level shared fixtures for the wcc_etc test suite."""

import pytest

from tests.helpers import make_scene, make_simulation


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
