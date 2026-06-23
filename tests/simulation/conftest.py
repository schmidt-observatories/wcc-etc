"""Simulation-level fixtures."""

import pytest
from tests.conftest import make_simulation


@pytest.fixture
def bright_sim():
    """G5V mag=16 on sony:r — bright enough for SNR tests without saturation."""
    return make_simulation(mag=16)


@pytest.fixture
def faint_sim():
    """G5V mag=25.4 on sony:r — faint, reference SNR target."""
    return make_simulation(mag=25)
