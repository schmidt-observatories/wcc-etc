"""Airy + FRED scattered-light PSFs (:mod:`wcc_etc.scatter_psf`)."""

import os

import numpy as np
import pytest
from astropy.io import fits

from wcc_etc import scatter_psf as sp

#: The 24 MB FRED map is a local data file, not shipped in the wheel. Only the
#: tests that check its *specific* numbers need it; everything else -- including
#: the stamp-size invariance -- runs against `synthetic_fgd` so CI exercises it.
requires_fgd = pytest.mark.skipif(
    not os.path.exists(sp.DEFAULT_SCATTER_FGD),
    reason="the FRED .fgd is a large local data file, not shipped in the wheel",
)

#: Reference values, cross-checked against notebooks_scratch/verify/scatter_verify.png
#: (Scott's "Comparison of PSFs"): Airy peak 1.712e4 W/mm^2 against a 0.7% halo
#: peaking at 2.084e-5 W/mm^2.
EXPECTED_CONTRAST = 8.213e8
EXPECTED_CROSSOVER_MM = 5.01
DESIRED_POWER = 0.007


@pytest.fixture(scope="module")
def fgd():
    """``(header, data, x, y)`` for the packaged FRED map, parsed once."""
    from wcc_etc.scatter_help import read_fgd

    header, data = read_fgd(sp.DEFAULT_SCATTER_FGD)
    x, y = sp.fgd_grid(header, data)
    return header, data, x, y


@pytest.fixture(scope="module")
def synthetic_fgd():
    """A FRED-like halo on the real map's footprint, with no data file.

    Broad, smooth and centrally peaked, on 2 mm cells over the same
    670 x 280 mm focal plane, with a peak chosen to land the Airy/halo crossover
    in the same few-mm range as the real map. Lets CI exercise the stamp-size
    invariance, which is a property of the algorithm rather than of one .fgd.
    """
    nx, ny = 335, 140
    x = -335.0 + (np.arange(nx) + 0.5) * (670.0 / nx)
    y = -140.0 + (np.arange(ny) + 0.5) * (280.0 / ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    data = 1.65e-05 / (1.0 + (xx**2 + yy**2) / 400.0) ** 0.9
    header = {
        "A_AXIS_MIN": -335,
        "A_AXIS_MAX": 335,
        "B_AXIS_MIN": -140,
        "B_AXIS_MAX": 140,
    }
    return header, data, x, y


@pytest.fixture(scope="module")
def gaussian():
    """A synthetic map with a known analytic integral."""
    g = np.linspace(-10.0, 10.0, 201)
    xx, yy = np.meshgrid(g, g, indexing="xy")
    return g, np.exp(-(xx**2 + yy**2) / 8.0) / (2 * np.pi * 4.0)


@requires_fgd
class TestFgdGrid:
    """FRED's own axis convention."""

    def test_grid_reproduces_header_integrated_power(self, fgd):
        """Cell-centre spacing recovers the header's stated integrated power."""
        header, data, x, y = fgd
        dx, dy = np.mean(np.diff(x)), np.mean(np.diff(y))
        assert float((data * dx * dy).sum()) == pytest.approx(
            sp.fgd_integrated_power(sp.DEFAULT_SCATTER_FGD), rel=1e-9
        )

    def test_cell_edges_span_the_stated_axis_range(self, fgd):
        """Cell edges, not cell centres, land on A_AXIS_MIN/MAX."""
        header, data, x, _ = fgd
        edges, _ = sp.cell_edges(x)
        assert edges[0] == pytest.approx(float(header["A_AXIS_MIN"]))


class TestResampleConservesFlux:
    """resample_scatter must not create or destroy power."""

    @pytest.mark.parametrize(
        "n_out,half",
        [(41, 10.0), (801, 10.0), (401, 2.5)],
        ids=["downsample", "upsample", "crop"],
    )
    @pytest.mark.parametrize("smooth", [False, True], ids=["exact", "smooth"])
    def test_stamp_sums_to_the_exact_enclosed_power(
        self, gaussian, n_out, half, smooth
    ):
        """Delivered array sums to the area-overlap integral of its footprint."""
        g, m = gaussian
        out = np.linspace(-half, half, n_out)
        stamp, p_exact, _ = sp.resample_scatter(
            m, g, g, out, out, smooth=smooth, dtype=np.float64
        )
        assert stamp.sum() == pytest.approx(p_exact, rel=1e-12)

    def test_exact_rebin_reproduces_the_source_integral(self, gaussian):
        """Full-extent rebin recovers the source grid's own Riemann sum."""
        g, m = gaussian
        out = np.linspace(-10.0, 10.0, 41)
        stamp, _, p_full = sp.resample_scatter(
            m, g, g, out, out, smooth=False, dtype=np.float64
        )
        assert stamp.sum() == pytest.approx(p_full, rel=1e-13)

    def test_smooth_and_exact_carry_the_same_power(self, gaussian):
        """The two paths differ in shape, never in total."""
        g, m = gaussian
        out = np.linspace(-2.5, 2.5, 401)
        a, _, _ = sp.resample_scatter(m, g, g, out, out, smooth=False, dtype=np.float64)
        b, _, _ = sp.resample_scatter(m, g, g, out, out, smooth=True, dtype=np.float64)
        assert a.sum() == pytest.approx(b.sum(), rel=1e-12)


class TestAiryIrradiance:
    """The closed-form Airy used to verify the pipeline."""

    def test_peak_matches_the_analytic_expression(self):
        """Peak is P*A/(lambda^2 f^2) with A = pi D^2 / 4."""
        d, fnum, lam, power = 3.065, 15.0, 450e-9, 1.0 - DESIRED_POWER
        expected = power * (np.pi * d**2 / 4) / (lam**2 * (fnum * d) ** 2) / 1e6
        assert sp.airy_irradiance(0.0, d, fnum, lam, power=power) == pytest.approx(
            expected, rel=1e-12
        )

    def test_first_null_falls_at_1p22_lambda_f_over_d(self):
        """First zero sits at 1.22 lambda F# in the focal plane."""
        d, fnum, lam = 3.065, 15.0, 450e-9
        r_null_mm = 1.22 * lam * fnum * 1e3
        near = sp.airy_irradiance(r_null_mm, d, fnum, lam)
        assert near < 1e-6 * sp.airy_irradiance(0.0, d, fnum, lam)


@requires_fgd
class TestVerifyFigureIsReproduced:
    """The numbers in notebooks_scratch/verify/scatter_verify.png."""

    @pytest.fixture(scope="class")
    def measured(self, fgd):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, _, out = sp.plot_scatter_verify(fgd, desired_power=DESIRED_POWER)
        plt.close("all")
        return out

    def test_contrast_is_about_1e9(self, measured):
        """Airy peak sits ~1e9 above the scatter peak."""
        assert measured["contrast"] == pytest.approx(EXPECTED_CONTRAST, rel=0.02)

    def test_crossover_is_about_5mm(self, measured):
        """Fringed Airy stops exceeding the halo near 5 mm."""
        assert measured["crossover_fringed_mm"] == pytest.approx(
            EXPECTED_CROSSOVER_MM, rel=0.02
        )


class TestContrastIsIndependentOfStampSize:
    """The invariant that fixes desired_power to the whole FRED map.

    Normalizing the halo against the power inside the output window instead
    would scale it by 1/scatter_enclosed_frac, moving both the contrast and the
    crossover with the extent. These tests are what forbid that.
    """

    #: 1001 spans +/-1.9 mm, 3001 +/-5.6 mm, 5001 +/-9.4 mm on IMX455 pixels.
    EXTENTS = [1001, 3001, 5001]
    #: Only stamps that actually reach past 5 mm can contain the crossover.
    CROSSOVER_EXTENTS = [3001, 5001]

    @pytest.fixture(scope="class")
    def psfs(self, synthetic_fgd):
        return {
            n: sp.make_total_psf(
                scatter_data=synthetic_fgd,
                sensor="imx455",
                extent=n,
                desired_power=DESIRED_POWER,
                keep_components=True,
                verbose=False,
            )
            for n in self.EXTENTS
        }

    @pytest.fixture(scope="class")
    def expected_crossover(self, synthetic_fgd):
        """Crossover read straight off the source map: extent-independent by
        construction, so it is the right thing for every stamp to reproduce."""
        _, data, x, y = synthetic_fgd
        dx, dy = np.mean(np.diff(x)), np.mean(np.diff(y))
        row = np.asarray(data[data.shape[0] // 2], dtype=float)
        halo = row * DESIRED_POWER / float((data * dx * dy).sum())
        keep = x >= 0
        return self.crossover(x[keep], halo[keep])

    @pytest.fixture(scope="class")
    def expected_halo_peak(self, synthetic_fgd):
        """dp * (map peak / map integral): the extent-independent prediction."""
        _, data, x, y = synthetic_fgd
        dx, dy = np.mean(np.diff(x)), np.mean(np.diff(y))
        return DESIRED_POWER * float(data.max()) / float((data * dx * dy).sum())

    @staticmethod
    def crossover(r_halo, halo, rmax=5.6):
        """Largest radius at which the fringed Airy still exceeds the halo."""
        r = np.linspace(1e-4, rmax, 120001)
        d = sp.airy_irradiance(
            r, 3.065, 15.0, 450e-9, power=1.0 - DESIRED_POWER
        ) - np.interp(r, r_halo, halo)
        i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        return float(r[i[-1]]) if len(i) else float("nan")

    @staticmethod
    def halo_profile(psf):
        """Halo irradiance [W/mm^2] along the +x half-row, and its radii [mm].

        Only the positive half: abs(x_mm) is V-shaped, and interpolating on it
        silently mixes the left and right sides where the map is asymmetric.
        """
        xm = psf.x_mm
        keep = xm >= 0
        row = np.asarray(psf.scatter[psf.shape[0] // 2], dtype=float)[keep]
        return xm[keep], row * DESIRED_POWER * psf.irradiance_scale

    def halo_peak(self, psf):
        return float(psf.scatter.max()) * psf.irradiance_scale * DESIRED_POWER

    def test_halo_peak_is_the_same_at_every_extent(self, psfs):
        """The invariant, stated directly: peaks agree across all stamp sizes."""
        peaks = [self.halo_peak(psfs[n]) for n in self.EXTENTS]
        assert max(peaks) / min(peaks) - 1 < 0.005

    def test_halo_peak_matches_the_source_map(self, psfs, expected_halo_peak):
        """And they sit at dp * (map peak / map integral), not somewhere else."""
        peaks = [self.halo_peak(psfs[n]) for n in self.EXTENTS]
        assert np.mean(peaks) == pytest.approx(expected_halo_peak, rel=0.01)

    def test_contrast_is_the_same_at_every_extent(self, psfs):
        """Airy peak over halo peak does not drift with the stamp."""
        airy_peak = sp.airy_irradiance(
            0.0, 3.065, 15.0, 450e-9, power=1.0 - DESIRED_POWER
        )
        contrasts = [airy_peak / self.halo_peak(psfs[n]) for n in self.EXTENTS]
        assert max(contrasts) / min(contrasts) - 1 < 0.005

    def test_crossover_is_the_same_at_every_extent(self, psfs):
        """Airy-vs-halo crossover does not move with the stamp.

        Measured against the halo's real radial profile, not its peak.
        """
        xs = [
            self.crossover(*self.halo_profile(psfs[n])) for n in self.CROSSOVER_EXTENTS
        ]
        assert max(xs) / min(xs) - 1 < 0.01

    def test_crossover_matches_the_source_map(self, psfs, expected_crossover):
        """And it lands where the source map alone says it should."""
        got = self.crossover(*self.halo_profile(psfs[self.CROSSOVER_EXTENTS[-1]]))
        assert got == pytest.approx(expected_crossover, rel=0.03)

    @pytest.mark.parametrize("extent", EXTENTS[1:])
    def test_halo_profile_matches_the_smallest_stamp(self, psfs, extent):
        """The root invariant: identical halo irradiance over shared radii.

        The default smooth path rescales each window to its own exact enclosed
        integral, which leaves a ~0.2% residual; see the exact-rebin test below
        for the zero-residual case.
        """
        r_ref, ref = self.halo_profile(psfs[self.EXTENTS[0]])
        r_big, big = self.halo_profile(psfs[extent])
        sampled = np.interp(r_ref, r_big, big)
        assert np.max(np.abs(sampled / ref - 1)) < 0.01

    def test_exact_rebin_halo_is_bit_for_bit_stamp_independent(self, synthetic_fgd):
        """With smooth=False the halo carries no extent dependence at all."""
        small, big = (
            sp.make_total_psf(
                scatter_data=synthetic_fgd,
                sensor="imx455",
                extent=n,
                desired_power=DESIRED_POWER,
                smooth=False,
                keep_components=True,
                verbose=False,
            )
            for n in (1001, 3001)
        )
        r_ref, ref = self.halo_profile(small)
        r_big, big_prof = self.halo_profile(big)
        assert np.max(np.abs(np.interp(r_ref, r_big, big_prof) / ref - 1)) == 0.0

    def test_enclosed_fraction_does_grow_with_the_stamp(self, psfs):
        """Sanity: a bigger stamp catches more of the halo, as it must."""
        fracs = [psfs[n].scatter_enclosed_frac for n in self.EXTENTS]
        assert fracs == sorted(fracs)


class TestMakeTotalPsf:
    """End-to-end behaviour of the public entry point."""

    @pytest.fixture(scope="class")
    def psf(self, synthetic_fgd):
        return sp.make_total_psf(
            scatter_data=synthetic_fgd,
            sensor="hwk4123",
            extent=1001,
            desired_power=DESIRED_POWER,
            keep_components=True,
            verbose=False,
        )

    def test_output_sums_to_one(self, psf):
        """renormalize=True leaves a unit-sum PSF."""
        assert psf.data.sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-6)

    def test_scatter_in_array_is_desired_power_times_enclosed(self, psf):
        """The stamp carries the enclosed share, and says so."""
        expected = (
            DESIRED_POWER
            * psf.scatter_enclosed_frac
            / (
                (1 - DESIRED_POWER) * psf.core_enclosed_frac
                + DESIRED_POWER * psf.scatter_enclosed_frac
            )
        )
        assert psf.scatter_in_array == pytest.approx(expected, rel=1e-6)

    def test_core_fills_the_grid_with_no_square_edge(self, psf):
        """The Airy is rendered to the corners, not zero-padded from a window."""
        assert float(psf.core[0, 0]) > 0.0

    def test_plate_scale_matches_the_package_formula(self, psf):
        """One plate scale shared with wcc_etc.airy."""
        from wcc_etc import airy

        assert psf.pixel_scale_mas == pytest.approx(
            airy.calc_plate_scale_from_flength(15.0 * 3.065, psf.pixel_size_um) * 1000.0
        )

    def test_size_guard_refuses_an_oversized_grid(self, synthetic_fgd):
        """max_size_mb fires before anything is allocated."""
        with pytest.raises(ValueError, match="max_size_mb"):
            sp.make_total_psf(
                scatter_data=synthetic_fgd,
                sensor="imx455",
                extent="full",
                max_size_mb=500,
                verbose=False,
            )

    def test_fits_round_trips_bit_for_bit(self, psf, tmp_path):
        """A saved PSF reads back identical."""
        path = tmp_path / "psf.fits"
        psf.save(path, overwrite=True)
        assert np.array_equal(fits.getdata(path), psf.data)

    def test_lossless_compression_round_trips_bit_for_bit(self, psf, tmp_path):
        """GZIP_2 at quantize_level=0 is lossless."""
        path = tmp_path / "psf_c.fits"
        psf.save(path, overwrite=True, compress="GZIP_2")
        assert np.array_equal(
            np.asarray(fits.getdata(path), dtype=np.float32), psf.data
        )

    def test_report_writes_a_pdf(self, psf, tmp_path):
        """report= produces a real, multi-page PDF."""
        path = tmp_path / "report.pdf"
        sp.write_psf_report(psf, path)
        head = path.read_bytes()[:5]
        assert head == b"%PDF-"

    def test_report_is_emitted_by_make_total_psf(self, synthetic_fgd, tmp_path):
        """The report= argument is wired through the entry point."""
        path = tmp_path / "r.pdf"
        sp.make_total_psf(
            scatter_data=synthetic_fgd,
            sensor="hwk4123",
            extent=401,
            desired_power=DESIRED_POWER,
            report=str(path),
            verbose=False,
        )
        assert path.exists() and path.stat().st_size > 1000

    def test_report_call_is_valid_python(self, psf):
        """The REPRODUCE block can be pasted back into a session."""
        import ast

        ast.parse(psf.settings["call"].replace("np.float32", "float"))

    def test_report_call_names_the_scatter_file(self, synthetic_fgd, tmp_path):
        """The source .fgd is named even when the map is passed in pre-parsed."""
        p = sp.make_total_psf(
            scatter_file="/some/where/my_run.fgd",
            scatter_data=synthetic_fgd,
            sensor="hwk4123",
            extent=401,
            desired_power=DESIRED_POWER,
            verbose=False,
        )
        assert p.settings["scatter_file"] == "my_run.fgd"
        assert "my_run.fgd" in p.settings["call"]

    def test_report_flags_a_pre_parsed_map(self, psf):
        """...and says so, rather than implying it was read from disk."""
        assert psf.settings["scatter_preparsed"] is True

    def test_report_records_the_settings_it_was_built_with(self, psf):
        """settings carries the call, so the report is not guessing."""
        assert psf.settings["inner_npix"] == 201

    def test_diagnostics_report_the_contrast(self, psf):
        """report_diagnostics exposes the Airy-to-halo contrast."""
        assert sp.report_diagnostics(psf)["contrast"] > 0

    def test_crossover_is_nan_when_it_falls_outside_the_stamp(self, synthetic_fgd):
        """A stamp too small to contain the crossover must not report its edge."""
        small = sp.make_total_psf(
            scatter_data=synthetic_fgd,
            sensor="imx455",
            extent=1501,
            desired_power=DESIRED_POWER,
            keep_components=True,
            verbose=False,
        )
        assert np.isnan(sp.report_diagnostics(small)["crossover_fringed_mm"])

    def test_rice_at_quantize_zero_is_refused(self, psf, tmp_path):
        """RICE_1 with quantize_level=0 destroys float data; save() blocks it."""
        with pytest.raises(ValueError, match="lossless"):
            psf.save(tmp_path / "bad.fits", compress="RICE_1", quantize_level=0.0)


class TestExtents:
    """Named extents resolve to the documented pixel counts."""

    def test_full_is_the_minimal_non_clipping_stamp(self):
        """(2Nx-1, 2Ny-1) reaches every chip pixel from any star position."""
        geom = sp.get_sensor_geometry("imx455")
        pitch = geom["pixel_size"] / 1000.0
        nx, ny = sp.resolve_extent("chip", None, geom, pitch)
        assert sp.resolve_extent("full", None, geom, pitch) == (2 * nx - 1, 2 * ny - 1)

    def test_diagonal_side_equals_the_chip_diagonal(self):
        """extent='diagonal' is a square whose side is the chip diagonal."""
        geom = sp.get_sensor_geometry("imx455")
        pitch = geom["pixel_size"] / 1000.0
        nx, _ = sp.resolve_extent("diagonal", None, geom, pitch)
        diag = np.hypot(geom["width_mm"], geom["height_mm"])
        assert nx * pitch == pytest.approx(diag, rel=1e-3)

    def test_extents_are_forced_odd(self):
        """An odd grid puts a pixel on the PSF peak."""
        geom = sp.get_sensor_geometry("hwk4123")
        assert sp.resolve_extent(400, None, geom, 0.0046) == (401, 401)


@requires_fgd
class TestRealMapInvariance:
    """The same invariance, against the real FRED map and Scott's numbers.

    Skipped without the .fgd; TestContrastIsIndependentOfStampSize covers the
    same property on a synthetic map so CI still exercises it.
    """

    EXTENTS = [1001, 3001, 5001]

    @pytest.fixture(scope="class")
    def psfs(self, fgd):
        return {
            n: sp.make_total_psf(
                scatter_data=fgd,
                sensor="imx455",
                extent=n,
                desired_power=DESIRED_POWER,
                keep_components=True,
                verbose=False,
            )
            for n in self.EXTENTS
        }

    @pytest.mark.parametrize("extent", EXTENTS)
    def test_scatter_peak_is_the_fred_value(self, psfs, extent):
        """2.084e-5 W/mm^2 at 0.7%, whatever the stamp."""
        psf = psfs[extent]
        peak = float(psf.scatter.max()) * psf.irradiance_scale * DESIRED_POWER
        assert peak == pytest.approx(2.0842e-05, rel=0.02)

    @pytest.mark.parametrize("extent", EXTENTS)
    def test_contrast_is_the_verify_figure_value(self, psfs, extent):
        """Scott's ~1e9, whatever the stamp."""
        psf = psfs[extent]
        airy_peak = sp.airy_irradiance(
            0.0, 3.065, 15.0, 450e-9, power=1.0 - DESIRED_POWER
        )
        scatter_peak = float(psf.scatter.max()) * psf.irradiance_scale * DESIRED_POWER
        assert airy_peak / scatter_peak == pytest.approx(EXPECTED_CONTRAST, rel=0.02)

    @pytest.mark.parametrize("extent", [3001, 5001])
    def test_crossover_is_the_verify_figure_value(self, psfs, extent):
        """Scott's ~5 mm, whatever the stamp."""
        psf = psfs[extent]
        xm = psfs[extent].x_mm
        keep = xm >= 0
        row = np.asarray(psf.scatter[psf.shape[0] // 2], dtype=float)[keep]
        halo = row * DESIRED_POWER * psf.irradiance_scale
        assert TestContrastIsIndependentOfStampSize.crossover(
            xm[keep], halo
        ) == pytest.approx(EXPECTED_CROSSOVER_MM, rel=0.03)
