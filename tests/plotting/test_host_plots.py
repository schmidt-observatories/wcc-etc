"""Host image plots: SceneElement.show_image and Simulation.plot_host."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import wcc_etc
from wcc_etc import get_scene, plotting
from tests.helpers import make_host_scene

PLATE = 0.016869


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sn_in_host():
    return make_host_scene(host_prop={"dx": 0.3})


@pytest.fixture
def sim(sn_in_host):
    return wcc_etc.Simulation.from_sensorfilter("zwo:r", sn_in_host)


class TestShowImage:
    def test_returns_axes(self, sn_in_host):
        """show_image draws the host alone and returns (fig, ax)."""
        fig, ax = sn_in_host.host.show_image(PLATE, npix=64)
        assert isinstance(ax, Axes)

    def test_sersic_host_too(self):
        """A Sersic host renders through the same method."""
        scene = get_scene(
            "G5V",
            mag=20,
            host="G5V",
            host_prop={"mag": 16, "profile": "sersic", "r_eff": 0.3},
        )
        assert isinstance(scene.host.show_image(PLATE, npix=64)[1], Axes)

    def test_point_source_raises(self, sn_in_host):
        """An element without a spatial profile has no image to show."""
        with pytest.raises(ValueError):
            sn_in_host.source.show_image(PLATE)


class TestPlotHost:
    def test_two_panels(self, sim):
        """plot_host returns the frame and the host-alone panels."""
        fig, axes = plotting.plot_host(sim, 60, npix=64, oversample=3)
        assert len(axes) == 2

    def test_returns_figure(self, sim):
        """The first return value is the figure."""
        assert isinstance(plotting.plot_host(sim, 60, npix=64, oversample=3)[0], Figure)

    def test_simulation_method(self, sim):
        """Simulation.plot_host is the same plot."""
        assert len(sim.plot_host(60, npix=64, oversample=3)[1]) == 2

    def test_title_kwarg_wins(self, sim):
        """A caller's title replaces the default on both panels."""
        axes = sim.plot_host(60, npix=64, oversample=3, title="x")[1]
        assert [ax.get_title() for ax in axes] == ["x", "x"]
