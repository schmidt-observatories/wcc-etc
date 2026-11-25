"""Grating design numbers against the WCC grating trade memo (2026-11-16)."""

import numpy as np
import pytest

from tests.helpers import make_scene
from wcc_etc.grating import (
    GRATING_HALVES,
    GratingSimulation,
    edge_coating,
    staircase_efficiency,
)


def log_mean_efficiency(half, lo, hi, order):
    wave = np.exp(np.linspace(np.log(lo), np.log(hi), 500))
    return staircase_efficiency(wave, order, GRATING_HALVES[half]["blaze_nm"]).mean()


class TestStaircaseEfficiency:
    def test_blue_first_order_mean(self):
        """Blue half: mean first-order efficiency 0.845 over 350-700 nm."""
        assert log_mean_efficiency("blue", 350, 700, 1) == pytest.approx(
            0.845, abs=0.005
        )

    def test_red_first_order_mean(self):
        """Red half: mean first-order efficiency 0.892 over 600-1100 nm."""
        assert log_mean_efficiency("red", 600, 1100, 1) == pytest.approx(
            0.892, abs=0.005
        )

    def test_blue_zeroth_order_mean(self):
        """Blue half: the zeroth order keeps 7.0% of the light."""
        assert log_mean_efficiency("blue", 350, 700, 0) == pytest.approx(
            0.070, abs=0.005
        )

    def test_peaks_at_blaze(self):
        """At the blaze wavelength a 16-level staircase gives sinc^2(1/16)."""
        assert staircase_efficiency(430.0, 1, 430.0) == pytest.approx(0.987, abs=0.001)

    def test_orders_sum_to_one(self):
        """The efficiencies of all orders sum to one (lossless scalar theory)."""
        orders = np.arange(-200, 201)
        assert staircase_efficiency(500.0, orders, 430.0).sum() == pytest.approx(
            1, abs=5e-3
        )


class TestEdgeCoating:
    def test_blue_passes_in_band(self):
        """The short-pass transmits its peak inside 350-700 nm."""
        assert edge_coating(500.0, "blue") == pytest.approx(0.95, abs=1e-3)

    def test_blue_blocks_red(self):
        """The short-pass blocks light beyond its 700 nm edge."""
        assert edge_coating(800.0, "blue") < 1e-6

    def test_red_blocks_blue(self):
        """The long-pass blocks light below its 600 nm edge."""
        assert edge_coating(500.0, "red") < 1e-6


class TestGeometry:
    def test_imx_dispersion(self, blue):
        """IMX455: 0.0822 px/nm at 4.414 lines/mm and 70 mm."""
        assert blue.dispersion_px_per_nm == pytest.approx(0.0822, abs=2e-4)

    def test_imx_first_order_start(self, blue):
        """IMX455: 350 nm falls 28.8 px from the zeroth order."""
        assert blue.offset_px(350.0) == pytest.approx(28.8, abs=0.1)

    def test_hwk_dispersion_scales_with_pixel(self, blue, blue_hwk):
        """The same grating gives fewer pixels per nm on the larger HWK pixels."""
        ratio = blue_hwk.dispersion_px_per_nm / blue.dispersion_px_per_nm
        assert ratio == pytest.approx(3.76 / 4.6, rel=1e-6)

    def test_resolving_power(self, blue):
        """R = 20 for the f/15 footprint at 70 mm."""
        assert blue.resolving_power == pytest.approx(20, abs=0.1)

    def test_unknown_half_raises(self):
        """Only the blue and red halves exist."""
        with pytest.raises(ValueError):
            GratingSimulation.from_sensor_and_scene("zwo", make_scene(), half="green")
