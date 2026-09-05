"""Tests for the filter-throughput plotter and Sensor.show."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest

from wcc_etc import plotting
from wcc_etc.sensor import Sensor


@pytest.fixture
def sensor():
    """A Sony r-band sensor with a real throughput curve."""
    return Sensor.from_kind_and_band("sony", "r")


class TestResolveBandpassCurve:
    """resolve_bandpass_curve accepts a SpectralElement or anything carrying one."""

    def test_accepts_a_spectral_element(self, sensor):
        """A bare synphot bandpass resolves to a (wave, throughput) pair."""
        wave, _throughput = plotting.resolve_bandpass_curve(sensor.bandpass)
        assert wave.size > 0

    def test_accepts_an_object_with_a_bandpass_attribute(self, sensor):
        """A Sensor resolves via its .bandpass attribute."""
        wave, _throughput = plotting.resolve_bandpass_curve(sensor)
        assert wave.size > 0

    def test_returns_the_same_curve_either_way(self, sensor):
        """Passing the Sensor matches passing its bandpass."""
        _w1, t1 = plotting.resolve_bandpass_curve(sensor)
        _w2, t2 = plotting.resolve_bandpass_curve(sensor.bandpass)
        np.testing.assert_allclose(t1, t2)

    def test_wave_override_sets_the_sampling_grid(self, sensor):
        """An explicit wave array in Angstrom is used verbatim."""
        grid = np.arange(4000.0, 8000.0, 100.0)
        wave, _throughput = plotting.resolve_bandpass_curve(sensor, wave=grid)
        np.testing.assert_allclose(wave, grid)

    def test_throughput_matches_the_wavelength_grid(self, sensor):
        """Throughput is returned with one value per sampled wavelength."""
        wave, throughput = plotting.resolve_bandpass_curve(sensor)
        assert throughput.shape == wave.shape


class TestPlotBandpassMpl:
    """plot_bandpass_mpl draws a throughput curve onto a matplotlib axis."""

    def test_returns_a_figure(self, sensor):
        """The first return value is a matplotlib Figure."""
        fig, _ax, _curve = plotting.plot_bandpass_mpl(sensor.bandpass)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_an_axes(self, sensor):
        """The second return value is a matplotlib Axes."""
        _fig, ax, _curve = plotting.plot_bandpass_mpl(sensor.bandpass)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_draws_onto_a_supplied_axis(self, sensor):
        """Passing ax= adds one line to that axis rather than making a figure."""
        _fig, ax = matplotlib.pyplot.subplots()
        plotting.plot_bandpass_mpl(sensor.bandpass, ax=ax)
        assert len(ax.lines) == 1

    def test_overlays_several_bandpasses_on_one_axis(self, sensor):
        """Repeated calls with the same ax accumulate lines."""
        _fig, ax = matplotlib.pyplot.subplots()
        plotting.plot_bandpass_mpl(sensor.bandpass, ax=ax)
        plotting.plot_bandpass_mpl(sensor.bandpass, ax=ax)
        assert len(ax.lines) == 2

    def test_label_reaches_the_line(self, sensor):
        """label= is forwarded to ax.plot so a legend can pick it up."""
        _fig, ax, _curve = plotting.plot_bandpass_mpl(sensor.bandpass, label="sony:r")
        assert ax.lines[0].get_label() == "sony:r"

    def test_extra_kwargs_reach_the_line(self, sensor):
        """Unrecognized keyword arguments are forwarded to ax.plot."""
        _fig, ax, _curve = plotting.plot_bandpass_mpl(sensor.bandpass, ls="--")
        assert ax.lines[0].get_linestyle() == "--"

    def test_ylabel_names_the_quantity(self, sensor):
        """The y axis is labelled Throughput."""
        _fig, ax, _curve = plotting.plot_bandpass_mpl(sensor.bandpass)
        assert ax.get_ylabel() == "Throughput"


class TestSensorShow:
    """Sensor.show plots the sensor's own filter curve."""

    def test_returns_a_figure(self, sensor):
        """show() returns the matplotlib Figure, like SceneElement.show."""
        assert isinstance(sensor.show(), matplotlib.figure.Figure)

    def test_draws_onto_a_supplied_axis(self, sensor):
        """Passing ax= draws the curve into the caller's axis."""
        _fig, ax = matplotlib.pyplot.subplots()
        sensor.show(ax=ax)
        assert len(ax.lines) == 1

    def test_label_reaches_the_line(self, sensor):
        """label= is forwarded through to ax.plot."""
        _fig, ax = matplotlib.pyplot.subplots()
        sensor.show(ax=ax, label="sony:r")
        assert ax.lines[0].get_label() == "sony:r"
