# tests/test_plotting.py
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from wcc_etc.psfsim import SimulatedImage, AiryPSF
from wcc_etc import plotting


def _make_simimg(npix=2):
    image_e = np.array([[100.0, 200.0], [300.0, 9e9]])
    clean = np.array([[100.0, 200.0], [300.0, 400.0]])
    sat = np.array([[False, False], [False, True]])
    return SimulatedImage(image_e=image_e, image_clean=clean, saturation_mask=sat,
                          gain=2.0, bias_level=100.0, npix=npix,
                          pixel_scale_mas=20.0, psf=AiryPSF())


def test_resolve_inputs_from_simimg():
    s = _make_simimg()
    ie, ic, sat, ps = plotting._resolve_inputs(s)
    assert np.array_equal(ie, s.image_e)
    assert np.array_equal(ic, s.image_clean)
    assert np.array_equal(sat, s.saturation_mask)
    assert ps == 20.0


def test_resolve_inputs_from_arrays():
    ie, ic, sat, ps = plotting._resolve_inputs(
        image_e=np.zeros((2, 2)), image_clean=np.ones((2, 2)),
        saturation_mask=np.zeros((2, 2), bool), pixel_scale_mas=5.0)
    assert ps == 5.0
    assert np.array_equal(ic, np.ones((2, 2)))


def test_saturation_overlay_masks_only_unsaturated():
    mask = np.array([[False, True], [True, False]])
    ov = plotting._saturation_overlay(mask)
    assert np.array_equal(ov.mask, ~mask)


def test_image_extent_mas_vs_pix():
    assert plotting._image_extent(4, 4, 10.0, "pix") is None
    ext = plotting._image_extent(4, 4, 10.0, "mas")
    assert ext == [-20.0, 20.0, -20.0, 20.0]


def test_make_norm_returns_expected_types():
    from astropy.visualization.mpl_normalize import ImageNormalize
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert plotting._make_norm(data, "linear") is None
    assert isinstance(plotting._make_norm(data, "log"), ImageNormalize)
    assert isinstance(plotting._make_norm(data, "hist"), ImageNormalize)


import matplotlib.figure
import matplotlib.axes


def test_plot_image_mpl_returns_fig_ax():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, stretch="linear")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_image_mpl_noise_selects_image_e():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, noise=True, stretch="linear")
    main = ax.get_images()[0]
    assert np.array_equal(main.get_array().data, s.image_e)


def test_plot_image_mpl_no_noise_selects_image_clean():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, noise=False, stretch="linear")
    main = ax.get_images()[0]
    assert np.array_equal(main.get_array().data, s.image_clean)


def test_plot_image_mpl_saturation_overlay_adds_second_image():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, show_saturation=True, stretch="linear")
    assert len(ax.get_images()) == 2
    overlay = ax.get_images()[1].get_array()
    assert np.array_equal(np.ma.getmaskarray(overlay), ~s.saturation_mask)


def test_plot_image_mpl_equal_aspect():
    s = _make_simimg()
    fig, ax = plotting.plot_image_mpl(s, stretch="linear")
    assert ax.get_aspect() in (1.0, "equal")


def test_plot_image_mpl_accepts_raw_arrays():
    fig, ax = plotting.plot_image_mpl(
        image_e=np.ones((4, 4)), image_clean=np.zeros((4, 4)),
        saturation_mask=np.zeros((4, 4), bool), pixel_scale_mas=10.0,
        stretch="linear")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_image_row_mpl_three_axes():
    s = _make_simimg()
    fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
    assert len(axes) == 3
    for ax in axes:
        assert ax.get_aspect() in (1.0, "equal")


def test_plot_image_row_mpl_panel_data():
    s = _make_simimg()
    fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
    assert np.array_equal(axes[0].get_images()[0].get_array().data, s.image_e)
    assert np.array_equal(axes[1].get_images()[0].get_array().data, s.image_clean)
    assert np.array_equal(
        np.asarray(axes[2].get_images()[0].get_array()).astype(bool),
        s.saturation_mask)


def test_plot_image_row_mpl_shared_color_scale():
    s = _make_simimg()
    fig, axes = plotting.plot_image_row_mpl(s, stretch="linear")
    im0, im1 = axes[0].get_images()[0], axes[1].get_images()[0]
    assert im0.get_clim() == im1.get_clim()


def _gaussian_simimg(npix=41, sigma=4.0, scale=20.0):
    c = (npix - 1) / 2
    yy, xx = np.mgrid[0:npix, 0:npix]
    g = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma ** 2)))
    return SimulatedImage(image_e=g.copy(), image_clean=g.copy(),
                          saturation_mask=np.zeros_like(g, bool),
                          gain=1.0, bias_level=0.0, npix=npix,
                          pixel_scale_mas=scale, psf=AiryPSF())


def test_plot_radial_mpl_returns_fig_ax_and_decreasing():
    s = _gaussian_simimg()
    fig, ax, (r, prof) = plotting.plot_radial_mpl(s, units="pix")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert prof[0] > prof[-1]


def test_plot_radial_mpl_units_scale_x_axis():
    s = _gaussian_simimg(scale=20.0)
    _, _, (r_pix, _) = plotting.plot_radial_mpl(s, units="pix")
    _, _, (r_mas, _) = plotting.plot_radial_mpl(s, units="mas")
    assert np.allclose(r_mas, r_pix * 20.0)


def test_plot_ee_mpl_monotonic_to_one():
    s = _gaussian_simimg()
    fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(s, units="pix")
    assert np.all(np.diff(ee) >= -1e-9)
    assert ee[-1] == pytest.approx(1.0, abs=1e-6)


def test_plot_ee_mpl_target_marker_returns_radius():
    s = _gaussian_simimg()
    fig, ax, (r, ee) = plotting.plot_encircled_energy_mpl(
        s, units="mas", ee_target=0.8)
    idx = np.searchsorted(ee, 0.8)
    assert 0 < idx < len(r)


from bokeh.models import Plot


def test_finish_bokeh_obj_html_components():
    s = _gaussian_simimg()
    obj = plotting.plot_image_bokeh(s, return_="obj")
    assert isinstance(obj, Plot)

    html = plotting.plot_image_bokeh(s, return_="html")
    assert isinstance(html, str) and "<script" in html

    comp = plotting.plot_image_bokeh(s, return_="components")
    assert isinstance(comp, tuple) and len(comp) == 2
    assert all(isinstance(x, str) for x in comp)


def test_plot_image_bokeh_bad_return_raises():
    s = _gaussian_simimg()
    with pytest.raises(ValueError):
        plotting.plot_image_bokeh(s, return_="nope")


def test_plot_image_bokeh_accepts_raw_arrays():
    obj = plotting.plot_image_bokeh(
        image_e=np.ones((8, 8)), image_clean=np.zeros((8, 8)),
        saturation_mask=np.zeros((8, 8), bool), pixel_scale_mas=10.0,
        return_="obj")
    assert isinstance(obj, Plot)


from bokeh.models import LayoutDOM


def test_plot_image_row_bokeh_obj_is_layout():
    s = _gaussian_simimg()
    obj = plotting.plot_image_row_bokeh(s, return_="obj")
    assert isinstance(obj, LayoutDOM)


def test_plot_image_row_bokeh_html_and_components():
    s = _gaussian_simimg()
    html = plotting.plot_image_row_bokeh(s, return_="html")
    assert isinstance(html, str) and "<script" in html
    script, div = plotting.plot_image_row_bokeh(s, return_="components")
    assert isinstance(script, str) and isinstance(div, str)


def test_plot_radial_bokeh_obj_and_components():
    s = _gaussian_simimg()
    assert isinstance(plotting.plot_radial_bokeh(s, return_="obj"), Plot)
    script, div = plotting.plot_radial_bokeh(s, return_="components")
    assert isinstance(script, str) and isinstance(div, str)


def test_plot_ee_bokeh_obj_and_html():
    s = _gaussian_simimg()
    assert isinstance(plotting.plot_encircled_energy_bokeh(s, return_="obj"), Plot)
    html = plotting.plot_encircled_energy_bokeh(s, return_="html")
    assert isinstance(html, str) and "<script" in html


def test_simimg_method_dispatch_mpl():
    s = _gaussian_simimg()
    fig, ax = s.plot_image(backend="mpl", stretch="linear")
    assert isinstance(fig, matplotlib.figure.Figure)
    fig2, axes = s.plot_image_row(backend="mpl", stretch="linear")
    assert len(axes) == 3
    fig3, ax3, (r, prof) = s.plot_radial(backend="mpl", units="pix")
    assert len(r) == len(prof)
    fig4, ax4, (re, ee) = s.plot_encircled_energy(backend="mpl", units="pix")
    assert ee[-1] == pytest.approx(1.0, abs=1e-6)


def test_simimg_method_dispatch_bokeh():
    s = _gaussian_simimg()
    assert isinstance(s.plot_image(backend="bokeh", return_="obj"), Plot)
    assert isinstance(s.plot_radial(backend="bokeh", return_="obj"), Plot)


def test_simimg_method_bad_backend_raises():
    s = _gaussian_simimg()
    with pytest.raises(ValueError):
        s.plot_image(backend="nope")


def test_plotting_functions_exported():
    import wcc_etc
    for name in ["plot_image_mpl", "plot_image_bokeh", "plot_image_row_mpl",
                 "plot_image_row_bokeh", "plot_radial_mpl", "plot_radial_bokeh",
                 "plot_encircled_energy_mpl", "plot_encircled_energy_bokeh"]:
        assert hasattr(wcc_etc, name), f"{name} not exported from wcc_etc"


def test_set_wcc_style_updates_rcparams():
    import matplotlib as mpl
    import wcc_etc
    saved = {k: mpl.rcParams[k] for k in plotting.WCC_STYLE}
    try:
        mpl.rcParams["axes.formatter.useoffset"] = True
        wcc_etc.set_wcc_style()
        assert mpl.rcParams["mathtext.fontset"] == "stix"
        assert mpl.rcParams["font.family"] == ["STIXGeneral"]
        assert mpl.rcParams["axes.formatter.useoffset"] is False
        assert mpl.rcParams["figure.dpi"] == 150
        assert mpl.rcParams["axes.labelsize"] == 13
        assert mpl.rcParams["xtick.minor.visible"] is True
        assert mpl.rcParams["ytick.minor.visible"] is True
        assert mpl.rcParams["xtick.direction"] == "in"
        assert mpl.rcParams["ytick.direction"] == "in"
        assert mpl.rcParams["axes.grid"] is True
        assert mpl.rcParams["grid.alpha"] == 0.3
        assert mpl.rcParams["grid.linewidth"] == 0.3
    finally:
        mpl.rcParams.update(saved)


def test_image_plots_disable_grid_even_when_global_grid_on():
    import matplotlib as mpl
    s = _make_simimg()
    saved = mpl.rcParams["axes.grid"]
    try:
        mpl.rcParams["axes.grid"] = True  # global grid on (as set_wcc_style does)
        fig, ax = plotting.plot_image_mpl(s, stretch="linear")
        assert all(not gl.get_visible() for gl in ax.get_xgridlines())
        assert all(not gl.get_visible() for gl in ax.get_ygridlines())
        fig2, axes = plotting.plot_image_row_mpl(s, stretch="linear")
        for a in axes:
            assert all(not gl.get_visible() for gl in a.get_xgridlines())
    finally:
        mpl.rcParams["axes.grid"] = saved


def test_oned_plots_have_faint_gridlines():
    s = _gaussian_simimg()
    fig, ax, _ = plotting.plot_radial_mpl(s, units="pix")
    fig2, ax2, _ = plotting.plot_encircled_energy_mpl(s, units="pix")
    assert any(gl.get_visible() for gl in ax.get_xgridlines())
    assert any(gl.get_visible() for gl in ax2.get_xgridlines())
