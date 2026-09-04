"""Top-level shared fixtures for the wcc_etc test suite."""

import pytest

from tests.helpers import make_broadband_simulation, make_scene, make_simulation


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


@pytest.fixture
def o5v_sim():
    """O5V mag=12 broad-band simulation — the bluest source-weighting case."""
    return make_broadband_simulation(name="O5V")


@pytest.fixture
def g5v_sim():
    """G5V mag=12 broad-band simulation — a solar-type source-weighting case."""
    return make_broadband_simulation(name="G5V")


@pytest.fixture
def m5v_sim():
    """M5V mag=12 broad-band simulation — the reddest source-weighting case."""
    return make_broadband_simulation(name="M5V")
