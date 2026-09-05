"""Tests for plotting utilities: image, radial, EE, Bokeh, style."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.figure

from wcc_etc import plotting
from wcc_etc.psfsim import AiryPSF, SimulatedImage


def _make_simimg(npix=2):
    image_e = np.array([[100.0, 200.0], [300.0, 9e9]])
    clean = np.array([[100.0, 200.0], [300.0, 400.0]])
    sat = np.array([[False, False], [False, True]])
    return SimulatedImage(
        image_e=image_e,
        image_clean=clean,
        saturation_mask=sat,
        gain=2.0,
        bias_level=100.0,
        npix=npix,
        pixel_scale_mas=20.0,
        psf=AiryPSF(),
    )


def _gaussian_simimg(npix=41, sigma=4.0, scale=20.0):
    c = (npix - 1) / 2
    yy, xx = np.mgrid[0:npix, 0:npix]
    g = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2)))
    return SimulatedImage(
        image_e=g.copy(),
        image_clean=g.copy(),
        saturation_mask=np.zeros_like(g, bool),
        gain=1.0,
        bias_level=0.0,
        npix=npix,
        pixel_scale_mas=scale,
        psf=AiryPSF(),
    )


def _ee_hline_at(ax, y, tol=1e-9):
    for line in ax.get_lines():
        yd = line.get_ydata()
        if len(yd) == 2 and abs(yd[0] - yd[1]) < tol and abs(yd[0] - y) < tol:
            return True
    return False


class TestImagePlotting:
    def test_resolve_inputs_image_e(self):
        s = _make_simimg()
        ie, ic, sat, ps = plotting._resolve_inputs(s)
        assert np.array_equal(ie, s.image_e)

    def test_resolve_inputs_image_clean(self):
        s = _make_simimg()
        ie, ic, sat, ps = plotting._resolve_inputs(s)
        assert np.array_equal(ic, s.image_clean)

    def test_resolve_inputs_saturation_mask(self):
        s = _make_simimg()
        ie, ic, sat, ps = plotting._resolve_inputs(s)
        assert np.array_equal(sat, s.saturation_mask)

    def test_resolve_inputs_pixel_scale(self):
        s = _make_simimg()
        ie, ic, sat, ps = plotting._resolve_inputs(s)
        assert ps == 20.0

    def test_resolve_inputs_from_arrays_pixel_scale(self):
        ie, ic, sat, ps = plotting._resolve_inputs(
            image_e=np.zeros((2, 2)),
            image_clean=np.ones((2, 2)),
            saturation_mask=np.zeros((2, 2), bool),
            pixel_scale_mas=5.0,
        )
        assert ps == 5.0

    def test_resolve_inputs_from_arrays_image_clean(self):
        ie, ic, sat, ps = plotting._resolve_inputs(
            image_e=np.zeros((2, 2)),
            image_clean=np.ones((2, 2)),
            saturation_mask=np.zeros((2, 2), bool),
            pixel_scale_mas=5.0,
        )
        assert np.array_equal(ic, np.ones((2, 2)))

    def test_saturation_overlay_masks_only_unsaturated(self):
        mask = np.array([[False, True], [True, False]])
        ov = plotting._saturation_overlay(mask)
        assert np.array_equal(ov.mask, ~mask)

    def test_image_extent_pix_returns_none(self):
        assert plotting._image_extent(4, 4, 10.0, "pix") is None

    def test_image_extent_mas_returns_list(self):
        ext = plotting._image_extent(4, 4, 10.0, "mas")
        assert ext == [-20.0, 20.0, -20.0, 20.0]

    def test_make_norm_linear_returns_none(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert plotting._make_norm(data, "linear") is None

    def test_make_norm_log_returns_image_normalize(self):
        from astropy.visualization.mpl_normalize import ImageNormalize

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(plotting._make_norm(data, "log"), ImageNormalize)

    def test_make_norm_hist_returns_image_normalize(self):
        from astropy.visualization.mpl_normalize import ImageNormalize

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(plotting._make_norm(data, "hist"), ImageNormalize)

    def test_plot_image_mpl_returns_figure(self):
        fig, ax = plotting.plot_image_mpl(_make_simimg(), stretch="linear")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_image_mpl_returns_axes(self):
        fig, ax = plotting.plot_image_mpl(_make_simimg(), stretch="linear")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_noise_selects_image_e(self):
        s = _make_simimg()
        fig, ax = plotting.plot_image_mpl(s, noise=True, stretch="linear")
        assert np.array_equal(ax.get_images()[0].get_array().data, s.image_e)

    def test_no_noise_selects_image_clean(self):
        s = _make_simimg()
        fig, ax = plotting.plot_image_mpl(s, noise=False, stretch="linear")
        assert np.array_equal(ax.get_images()[0].get_array().data, s.image_clean)

    def test_saturation_overlay_adds_second_image(self):
        s = _make_simimg()
        fig, ax = plotting.plot_image_mpl(s, show_saturation=True, stretch="linear")
        assert len(ax.get_images()) == 2

    def test_saturation_overlay_correct_mask(self):
        s = _make_simimg()
        fig, ax = plotting.plot_image_mpl(s, show_saturation=True, stretch="linear")
        overlay = ax.get_images()[1].get_array()
        assert np.array_equal(np.ma.getmaskarray(overlay), ~s.saturation_mask)

    def test_equal_aspect(self):
        s = _make_simimg()
        fig, ax = plotting.plot_image_mpl(s, stretch="linear")
        assert ax.get_aspect() in (1.0, "equal")

    def test_accepts_raw_arrays(self):
        fig, ax = plotting.plot_image_mpl(
            image_e=np.ones((4, 4)),
            image_clean=np.zeros((4, 4)),
            saturation_mask=np.zeros((4, 4), bool),
            pixel_scale_mas=10.0,
            stretch="linear",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_row_mpl_has_three_axes(self):
        s = _make_simimg()
        fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
        assert len(axes) == 3

    def test_row_mpl_axes_equal_aspect(self):
        s = _make_simimg()
        fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
        assert all(ax.get_aspect() in (1.0, "equal") for ax in axes)

    def test_row_mpl_first_panel_is_image_e(self):
        s = _make_simimg()
        fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
        assert np.array_equal(axes[0].get_images()[0].get_array().data, s.image_e)

    def test_row_mpl_second_panel_is_image_clean(self):
        s = _make_simimg()
        fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
        assert np.array_equal(axes[1].get_images()[0].get_array().data, s.image_clean)

    def test_row_mpl_third_panel_is_saturation_mask(self):
        s = _make_simimg()
        fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
        assert np.array_equal(
            np.asarray(axes[2].get_images()[0].get_array()).astype(bool),
            s.saturation_mask,
        )

    def test_row_mpl_shared_color_scale(self):
        s = _make_simimg()
        fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
        im0, im1 = axes[0].get_images()[0], axes[1].get_images()[0]
        assert im0.get_clim() == im1.get_clim()


class TestRadialPlotting:
    def test_radial_mpl_returns_figure(self):
        s = _gaussian_simimg()
        fig, ax, (r, prof) = plotting.plot_radial_mpl(s, units="pix")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_radial_profile_is_decreasing(self):
        s = _gaussian_simimg()
        fig, ax, (r, prof) = plotting.plot_radial_mpl(s, units="pix")
        assert prof[0] > prof[-1]

    def test_radial_mpl_units_scale_x_axis(self):
        s = _gaussian_simimg(scale=20.0)
        _, _, (r_pix, _) = plotting.plot_radial_mpl(s, units="pix")
        _, _, (r_mas, _) = plotting.plot_radial_mpl(s, units="mas")
        assert np.allclose(r_mas, r_pix * 20.0)

    def test_ee_mpl_is_monotonic(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(s, units="pix")
        assert np.all(np.diff(ee) >= -1e-9)

    def test_ee_mpl_reaches_one(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(s, units="pix")
        assert ee[-1] == pytest.approx(1.0, abs=1e-6)

    def test_ee_mpl_target_marker_returns_radius(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(
            s, units="mas", ee_target=0.8
        )
        idx = np.searchsorted(ee, 0.8)
        assert 0 < idx < len(r)

    def test_ee_mpl_marks_90_percent_by_default(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(s, units="pix")
        assert ax.get_legend() is not None

    def test_ee_mpl_90pct_hline_present(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(s, units="pix")
        assert _ee_hline_at(ax, 0.9)

    def test_ee_mpl_none_disables_legend(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(
            s, units="pix", ee_target=None
        )
        assert ax.get_legend() is None

    def test_ee_mpl_none_disables_hline(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(
            s, units="pix", ee_target=None
        )
        assert not _ee_hline_at(ax, 0.9)


class TestBokehPlotting:
    def test_plot_image_bokeh_obj_returns_plot(self):
        from bokeh.models import Plot

        s = _gaussian_simimg()
        assert isinstance(plotting.plot_image_bokeh(s, return_="obj"), Plot)

    def test_plot_image_bokeh_html_returns_string(self):
        s = _gaussian_simimg()
        html = plotting.plot_image_bokeh(s, return_="html")
        assert isinstance(html, str)

    def test_plot_image_bokeh_html_contains_script(self):
        s = _gaussian_simimg()
        html = plotting.plot_image_bokeh(s, return_="html")
        assert "<script" in html

    def test_plot_image_bokeh_components_returns_tuple(self):
        s = _gaussian_simimg()
        comp = plotting.plot_image_bokeh(s, return_="components")
        assert isinstance(comp, tuple) and len(comp) == 2

    def test_plot_image_bokeh_bad_return_raises(self):
        with pytest.raises(ValueError):
            plotting.plot_image_bokeh(_gaussian_simimg(), return_="nope")

    def test_plot_image_bokeh_accepts_raw_arrays(self):
        from bokeh.models import Plot

        obj = plotting.plot_image_bokeh(
            image_e=np.ones((8, 8)),
            image_clean=np.zeros((8, 8)),
            saturation_mask=np.zeros((8, 8), bool),
            pixel_scale_mas=10.0,
            return_="obj",
        )
        assert isinstance(obj, Plot)

    def test_plot_image_row_bokeh_obj_returns_layoutdom(self):
        from bokeh.models import LayoutDOM

        s = _gaussian_simimg()
        assert isinstance(plotting.plot_image_row_bokeh(s, return_="obj"), LayoutDOM)

    def test_plot_image_row_bokeh_html_is_string(self):
        s = _gaussian_simimg()
        html = plotting.plot_image_row_bokeh(s, return_="html")
        assert isinstance(html, str) and "<script" in html

    def test_plot_image_row_bokeh_components_are_strings(self):
        s = _gaussian_simimg()
        script, div = plotting.plot_image_row_bokeh(s, return_="components")
        assert isinstance(script, str) and isinstance(div, str)

    def test_plot_radial_bokeh_returns_plot(self):
        from bokeh.models import Plot

        s = _gaussian_simimg()
        assert isinstance(plotting.plot_radial_bokeh(s, return_="obj"), Plot)

    def test_plot_radial_bokeh_components_are_strings(self):
        s = _gaussian_simimg()
        script, div = plotting.plot_radial_bokeh(s, return_="components")
        assert isinstance(script, str) and isinstance(div, str)

    def test_plot_ee_bokeh_returns_plot(self):
        from bokeh.models import Plot

        s = _gaussian_simimg()
        assert isinstance(plotting.plot_encircled_energy_bokeh(s, return_="obj"), Plot)

    def test_plot_ee_bokeh_html_is_string(self):
        s = _gaussian_simimg()
        html = plotting.plot_encircled_energy_bokeh(s, return_="html")
        assert isinstance(html, str) and "<script" in html


class TestStyle:
    def test_simimg_plot_image_mpl(self):
        s = _gaussian_simimg()
        fig, ax = s.plot_image(backend="mpl", stretch="linear")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_simimg_plot_image_row_mpl(self):
        s = _gaussian_simimg()
        fig2, axes = s.plot_image_row(backend="mpl", stretch="linear")
        assert len(axes) == 3

    def test_simimg_plot_radial_mpl(self):
        s = _gaussian_simimg()
        fig3, ax3, (r, prof) = s.plot_radial(backend="mpl", units="pix")
        assert len(r) == len(prof)

    def test_simimg_plot_ee_mpl(self):
        s = _gaussian_simimg()
        fig4, ax4, (re, ee) = s.plot_encircled_energy(backend="mpl", units="pix")
        assert ee[-1] == pytest.approx(1.0, abs=1e-6)

    def test_simimg_plot_image_bokeh(self):
        from bokeh.models import Plot

        s = _gaussian_simimg()
        assert isinstance(s.plot_image(backend="bokeh", return_="obj"), Plot)

    def test_simimg_plot_radial_bokeh(self):
        from bokeh.models import Plot

        s = _gaussian_simimg()
        assert isinstance(s.plot_radial(backend="bokeh", return_="obj"), Plot)

    def test_simimg_bad_backend_raises(self):
        with pytest.raises(ValueError):
            _gaussian_simimg().plot_image(backend="nope")

    def test_simimg_plot_ee_has_legend(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = s.plot_encircled_energy(backend="mpl", units="pix")
        assert ax.get_legend() is not None

    def test_simimg_plot_ee_has_90pct_hline(self):
        s = _gaussian_simimg()
        fig, ax, (r, ee) = s.plot_encircled_energy(backend="mpl", units="pix")
        assert _ee_hline_at(ax, 0.9)

    def test_plotting_functions_exported(self):
        import wcc_etc

        for name in [
            "plot_image_mpl",
            "plot_image_bokeh",
            "plot_image_row_mpl",
            "plot_image_row_bokeh",
            "plot_radial_mpl",
            "plot_radial_bokeh",
            "plot_encircled_energy_mpl",
            "plot_encircled_energy_bokeh",
        ]:
            assert hasattr(wcc_etc, name), f"{name} not exported from wcc_etc"

    @pytest.fixture(autouse=True)
    def _restore_rcparams(self):
        import matplotlib as mpl

        saved = {k: mpl.rcParams[k] for k in plotting.WCC_STYLE}
        mpl.rcParams["axes.formatter.useoffset"] = True
        yield
        mpl.rcParams.update(saved)

    def test_wcc_style_mathtext_fontset(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["mathtext.fontset"] == "stix"

    def test_wcc_style_font_family(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["font.family"] == ["STIXGeneral"]

    def test_wcc_style_no_offset(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["axes.formatter.useoffset"] is False

    def test_wcc_style_figure_dpi(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["figure.dpi"] == 150

    def test_wcc_style_axes_labelsize(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["axes.labelsize"] == 13

    def test_wcc_style_xtick_minor_visible(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["xtick.minor.visible"] is True

    def test_wcc_style_ytick_minor_visible(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["ytick.minor.visible"] is True

    def test_wcc_style_xtick_direction(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["xtick.direction"] == "in"

    def test_wcc_style_ytick_direction(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["ytick.direction"] == "in"

    def test_wcc_style_axes_grid(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["axes.grid"] is True

    def test_wcc_style_grid_alpha(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["grid.alpha"] == 0.3

    def test_wcc_style_grid_linewidth(self):
        import matplotlib as mpl

        import wcc_etc

        wcc_etc.set_wcc_style()
        assert mpl.rcParams["grid.linewidth"] == 0.3

    def test_plot_image_disables_x_grid(self):
        import matplotlib as mpl

        s = _make_simimg()
        saved = mpl.rcParams["axes.grid"]
        try:
            mpl.rcParams["axes.grid"] = True
            fig, ax = plotting.plot_image_mpl(s, stretch="linear")
            assert all(not gl.get_visible() for gl in ax.get_xgridlines())
        finally:
            mpl.rcParams["axes.grid"] = saved

    def test_plot_image_disables_y_grid(self):
        import matplotlib as mpl

        s = _make_simimg()
        saved = mpl.rcParams["axes.grid"]
        try:
            mpl.rcParams["axes.grid"] = True
            fig, ax = plotting.plot_image_mpl(s, stretch="linear")
            assert all(not gl.get_visible() for gl in ax.get_ygridlines())
        finally:
            mpl.rcParams["axes.grid"] = saved

    def test_plot_image_row_disables_x_grid(self):
        import matplotlib as mpl

        s = _make_simimg()
        saved = mpl.rcParams["axes.grid"]
        try:
            mpl.rcParams["axes.grid"] = True
            fig2, axes = plotting.plot_image_row_mpl(s, stretch="linear")
            assert all(not gl.get_visible() for a in axes for gl in a.get_xgridlines())
        finally:
            mpl.rcParams["axes.grid"] = saved

    def test_radial_has_gridlines(self):
        s = _gaussian_simimg()
        fig, ax, _ = plotting.plot_radial_mpl(s, units="pix")
        assert any(gl.get_visible() for gl in ax.get_xgridlines())

    def test_ee_has_gridlines(self):
        s = _gaussian_simimg()
        fig2, ax2, _ = plotting.plot_encircled_energy_mpl(s, units="pix")
        assert any(gl.get_visible() for gl in ax2.get_xgridlines())


class TestOverlayLabels:
    """plot_radial_mpl / plot_encircled_energy_mpl forward label= and **kwargs to
    ax.plot, so several profiles can be overlaid on one axis with a legend."""

    def test_radial_label_reaches_the_line(self):
        """label= lands on the radial profile line."""
        s = _gaussian_simimg()
        _fig, ax, _curve = plotting.plot_radial_mpl(s, units="pix", label="in-focus")
        assert ax.lines[0].get_label() == "in-focus"

    def test_ee_label_reaches_the_line(self):
        """label= lands on the encircled-energy line."""
        s = _gaussian_simimg()
        _fig, ax, _curve = plotting.plot_encircled_energy_mpl(
            s, units="pix", label="in-focus"
        )
        assert ax.lines[0].get_label() == "in-focus"

    def test_radial_extra_kwargs_reach_the_line(self):
        """Unrecognized keyword arguments are forwarded to ax.plot."""
        s = _gaussian_simimg()
        _fig, ax, _curve = plotting.plot_radial_mpl(s, units="pix", ls="--")
        assert ax.lines[0].get_linestyle() == "--"

    def test_ee_extra_kwargs_reach_the_line(self):
        """Unrecognized keyword arguments are forwarded to ax.plot."""
        s = _gaussian_simimg()
        _fig, ax, _curve = plotting.plot_encircled_energy_mpl(
            s, units="pix", ls="--", ee_target=None
        )
        assert ax.lines[0].get_linestyle() == "--"

    def test_radial_overlays_accumulate_on_one_axis(self):
        """Two labelled radial profiles share an axis and both carry their label."""
        s = _gaussian_simimg()
        _fig, ax, _curve = plotting.plot_radial_mpl(
            s, units="pix", label="a", show_hwhm=False
        )
        plotting.plot_radial_mpl(s, units="pix", label="b", show_hwhm=False, ax=ax)
        assert [line.get_label() for line in ax.lines] == ["a", "b"]

    def test_ee_overlays_accumulate_on_one_axis(self):
        """Two labelled EE curves share an axis and both carry their label."""
        s = _gaussian_simimg()
        _fig, ax, _curve = plotting.plot_encircled_energy_mpl(
            s, units="pix", label="a", ee_target=None
        )
        plotting.plot_encircled_energy_mpl(
            s, units="pix", label="b", ee_target=None, ax=ax
        )
        assert [line.get_label() for line in ax.lines] == ["a", "b"]

    def test_radial_label_defaults_to_unset(self):
        """With no label the line keeps matplotlib's auto label."""
        s = _gaussian_simimg()
        _fig, ax, _curve = plotting.plot_radial_mpl(s, units="pix", show_hwhm=False)
        assert ax.lines[0].get_label().startswith("_")
