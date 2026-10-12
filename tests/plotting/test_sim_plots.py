"""Tests for the Simulation / ImageSimulator plotting methods and .plot() aliases."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

import wcc_etc
from tests.helpers import make_scene, make_simulation

TIMES = np.array([10.0, 30.0, 60.0])
MAGS = np.array([14.0, 16.0, 18.0])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sim():
    return make_simulation()


@pytest.fixture
def imsim():
    return wcc_etc.ImageSimulator.from_sensor_and_scene("sony:r", make_scene(), npix=64)


def single_point_lines(ax):
    """Lines carrying exactly one (x, y) point: the marker overlays."""
    return [ln for ln in ax.lines if len(ln.get_xdata()) == 1]


def hlines(ax):
    """Lines with two identical y values: axhline markers."""
    return [
        ln
        for ln in ax.lines
        if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
    ]


def vlines(ax):
    """Lines with two identical x values: axvline markers."""
    return [
        ln
        for ln in ax.lines
        if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
    ]


class TestPlotSnr:
    def test_returns_axes(self, sim):
        """plot_snr returns (fig, ax)."""
        _fig, ax = sim.plot_snr(TIMES)
        assert isinstance(ax, Axes)

    def test_curve_matches_get_snr(self, sim):
        """The plotted y data is get_snr(time)['snr']."""
        _fig, ax = sim.plot_snr(TIMES)
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), sim.get_snr(TIMES, warn=False)["snr"]
        )

    def test_draws_on_given_ax(self, sim):
        """ax= draws onto the supplied axis."""
        _fig, ax = plt.subplots()
        assert sim.plot_snr(TIMES, ax=ax)[1] is ax

    def test_label_reaches_line(self, sim):
        """Extra keywords go to ax.plot."""
        _fig, ax = sim.plot_snr(TIMES, label="G5V")
        assert ax.lines[0].get_label() == "G5V"

    def test_plot_alias(self, sim):
        """sim.plot is the SNR-vs-time plot."""
        _fig, ax = sim.plot(TIMES)
        assert ax.get_ylabel() == "SNR"


class TestPlotSnrVsMag:
    def test_curve_matches_mags_sweep(self, sim):
        """The plotted y data is get_image_snr(time, mags=)['snr']."""
        _fig, ax = sim.plot_snr_vs_mag(MAGS, time=60)
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(),
            sim.get_image_snr(time=60, mags=MAGS, warn=False)["snr"],
        )

    def test_xdata_is_mags(self, sim):
        """The x data is the magnitude grid."""
        _fig, ax = sim.plot_snr_vs_mag(MAGS, time=60)
        np.testing.assert_allclose(ax.lines[0].get_xdata(), MAGS)


class TestPlotSweep:
    def test_curve_matches_update_loop(self, sim):
        """plot_sweep reproduces an update()/get_snr loop over the key."""
        values = [12.0, 15.0, 18.0]
        expected = []
        for v in values:
            sim.update(source__mag=v)
            expected.append(sim.get_snr(60, warn=False)["snr"])
        sim.update(source__mag=15)
        _fig, ax = sim.plot_sweep("source__mag", values, time=60)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)

    def test_restores_parameter_after_sweep(self, sim):
        """The swept parameter is put back when the sweep ends."""
        sim.plot_sweep("source__mag", [12.0, 18.0], time=60)
        assert sim.get_parameter("source__mag")[0] == 15

    def test_restores_parameter_when_a_point_raises(self, sim):
        """A failing point still puts the parameter back."""
        with pytest.raises(Exception):
            sim.plot_sweep("source__mag", [12.0, "bad"], time=60)
        assert sim.get_parameter("source__mag")[0] == 15

    def test_xlabel_names_key(self, sim):
        """The x axis is labelled with the swept key."""
        _fig, ax = sim.plot_sweep("source__mag", [12.0, 18.0], time=60)
        assert ax.get_xlabel() == "source__mag"

    def test_top_level_key(self, sim):
        """A Simulation-level key such as n_reads sweeps too."""
        _fig, ax = sim.plot_sweep("n_reads", [1, 4], time=60)
        assert len(ax.lines[0].get_xdata()) == 2


class TestPlotPeakPixel:
    def test_curve_matches_get_peak_pixel_adu(self, sim):
        """The plotted y data is get_peak_pixel(time, units='adu')."""
        _fig, ax = sim.plot_peak_pixel(TIMES)
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), sim.get_peak_pixel(TIMES, units="adu").value
        )

    def test_ceiling_line_at_saturation_in_adu(self, sim):
        """The dashed ceiling is saturation_e converted to ADU."""
        _fig, ax = sim.plot_peak_pixel(TIMES)
        ceiling_adu = (
            (sim.sensor.saturation_e / sim.sensor.gain).to("ct") + sim.sensor.bias_level
        ).value
        assert hlines(ax)[0].get_ydata()[0] == pytest.approx(ceiling_adu)

    def test_ceiling_line_in_electrons(self, sim):
        """units='e-' puts the ceiling at saturation_e itself."""
        _fig, ax = sim.plot_peak_pixel(TIMES, units="e-")
        assert hlines(ax)[0].get_ydata()[0] == pytest.approx(
            sim.sensor.saturation_e.value
        )

    def test_onset_marker_at_first_saturated_time(self):
        """A vertical line marks the first time is_saturated is True."""
        bright = make_simulation(mag=10)
        times = np.logspace(-2, 1, 20)
        _fig, ax = bright.plot_peak_pixel(times)
        expected = times[bright.is_saturated(times)][0]
        assert vlines(ax)[0].get_xdata()[0] == pytest.approx(expected)

    def test_no_onset_marker_when_unsaturated(self):
        """No vertical line when nothing saturates."""
        faint = make_simulation(mag=20)
        _fig, ax = faint.plot_peak_pixel(TIMES)
        assert vlines(ax) == []

    def test_show_ceiling_false(self, sim):
        """show_ceiling=False draws only the curve."""
        _fig, ax = sim.plot_peak_pixel(TIMES, show_ceiling=False)
        assert hlines(ax) == []


class TestPlotExptimeForSnr:
    def test_curve_matches_inverse(self, sim):
        """The plotted y data is get_image_exptime_for_snr(s)['time_s']."""
        targets = [10.0, 30.0]
        _fig, ax = sim.plot_exptime_for_snr(targets)
        expected = [sim.get_image_exptime_for_snr(s)["time_s"] for s in targets]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)

    def test_xdata_is_targets(self, sim):
        """The x data is the target-SNR grid."""
        _fig, ax = sim.plot_exptime_for_snr([10.0, 30.0])
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [10.0, 30.0])


class TestPlotSnrVsAperture:
    RADII = np.array([50.0, 100.0, 200.0])

    def test_curve_matches_fixed_apertures(self, sim):
        """The plotted y data is get_image_snr at each fixed r_aper_mas."""
        _fig, ax = sim.plot_snr_vs_aperture(self.RADII, time=60)
        expected = [
            sim.get_image_snr(time=60, r_aper_mas=r, warn=False)["snr"]
            for r in self.RADII
        ]
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)

    def test_marks_default_aperture(self, sim):
        """A single-point marker sits at the default aperture's (r, snr)."""
        _fig, ax = sim.plot_snr_vs_aperture(self.RADII, time=60)
        default = sim.get_image_snr(time=60, warn=False)
        xs = [ln.get_xdata()[0] for ln in single_point_lines(ax)]
        assert any(x == pytest.approx(default["r_aper_mas"]) for x in xs)

    def test_marks_optimized_aperture(self, sim):
        """A single-point marker sits at the optimized aperture's (r, snr)."""
        _fig, ax = sim.plot_snr_vs_aperture(self.RADII, time=60)
        best = sim.get_image_snr(time=60, optimize=True, warn=False)
        xs = [ln.get_xdata()[0] for ln in single_point_lines(ax)]
        assert any(x == pytest.approx(best["r_aper_mas"]) for x in xs)

    def test_show_markers_false(self, sim):
        """show_markers=False draws only the curve."""
        _fig, ax = sim.plot_snr_vs_aperture(self.RADII, time=60, show_markers=False)
        assert single_point_lines(ax) == []


class TestPlotPsf:
    def test_draws_one_image(self, imsim):
        """plot_psf puts one image on the axis."""
        _fig, ax = imsim.plot_psf()
        assert len(ax.images) == 1

    def test_image_is_the_rendered_psf(self, imsim):
        """The image data is render_psf() with the same arguments."""
        _fig, ax = imsim.plot_psf(jitter_sigma_mas=0.0)
        np.testing.assert_allclose(
            np.asarray(ax.images[0].get_array()),
            imsim.render_psf(jitter_sigma_mas=0.0),
        )

    def test_axes_in_mas_by_default(self, imsim):
        """The axes are labelled in milliarcseconds."""
        _fig, ax = imsim.plot_psf()
        assert ax.get_xlabel() == "X [mas]"

    def test_draws_on_given_ax(self, imsim):
        """ax= draws onto the supplied axis."""
        _fig, ax = plt.subplots()
        assert imsim.plot_psf(ax=ax)[1] is ax


class TestPlotAliases:
    def test_sensor_plot_is_show(self):
        assert wcc_etc.Sensor.plot is wcc_etc.Sensor.show

    def test_scene_element_plot_is_show(self):
        from wcc_etc.scene import SceneElement

        assert SceneElement.plot is SceneElement.show

    def test_simulated_image_plot_is_plot_image(self):
        assert wcc_etc.SimulatedImage.plot is wcc_etc.SimulatedImage.plot_image

    def test_simulation_plot_is_plot_snr(self):
        assert wcc_etc.Simulation.plot is wcc_etc.Simulation.plot_snr


class TestLegendDedup:
    def test_overlaid_ceilings_give_one_legend_entry(self, sim):
        """Three plot_peak_pixel calls on one axis list the ceiling once."""
        _fig, ax = plt.subplots()
        for mag in (12.0, 15.0, 18.0):
            sim.update(source__mag=mag)
            sim.plot_peak_pixel(TIMES, ax=ax, label=f"r = {mag:.0f}")
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert sum(lbl.startswith("saturation") for lbl in labels) == 1
