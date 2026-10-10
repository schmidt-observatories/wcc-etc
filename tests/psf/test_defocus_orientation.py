"""The Huygens products are oriented like FITS and rendered about their chief ray.

wcc-sim issue #18: the Zemax export is written top row first and puts the chief
ray at sample (129, 129) 1-based, so a renderer that assumes a centred array
mirrors the PSF and lands every defocused star ~0.7 px (15 mas) off its
position; and ``zoom`` without ``grid_mode`` scales by (nz-1)/(n-1).
"""

import numpy as np
import pytest
from astropy.io import fits

from wcc_etc.psfsim import (
    DEFOCUS_1WAVE_PATH,
    DEFOCUS_2WAVE_PATH,
    DEFOCUS_2WAVE_TXT_PATH,
    CustomPSF,
    DefocusPSF,
    DetectorPSFContext,
    grid_center,
    load_huygens_psf,
    load_psf_fits,
    parse_huygens_header,
)


def _ctx(npix, pixel_size_um=3.76, center=None):
    return DetectorPSFContext(
        npix=npix, pixel_size_um=pixel_size_um, plate_scale_mas=16.87,
        wavelength_m=500e-9, diameter_m=3.0, fnum=15.0, jitter_sigma_mas=0.0,
        center=center,
    )


def _blob(x0, y0, n=40, sigma=2.5):
    """A Gaussian at (x0, y0): a delta point-sampled by ``zoom`` has no centroid."""
    yy, xx = np.indices((n, n))
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))


def _centroid(img):
    i = np.arange(img.shape[0])
    return (img.sum(axis=0) @ i) / img.sum(), (img.sum(axis=1) @ i) / img.sum()


def test_huygens_rows_are_flipped_so_index_increases_with_y():
    with open(DEFOCUS_2WAVE_TXT_PATH, encoding="utf-16") as fh:
        rows = [
            ln for ln in fh.read().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    last_text_row = np.array([float(x) for x in rows[-1].split()])
    data = load_huygens_psf(DEFOCUS_2WAVE_TXT_PATH)
    assert np.array_equal(data[0], last_text_row)


def test_header_chief_ray_is_zero_based_center_point():
    assert parse_huygens_header(DEFOCUS_2WAVE_TXT_PATH)["chief_ray_pix"] == (128, 128)


@pytest.mark.parametrize("path", [DEFOCUS_1WAVE_PATH, DEFOCUS_2WAVE_PATH])
def test_bundled_products_record_the_chief_ray(path):
    hdr = fits.getheader(path)
    assert (hdr["CRPIX1"], hdr["CRPIX2"]) == (129, 129)
    assert load_psf_fits(path)[1]["chief_ray_pix"] == (128.0, 128.0)


def test_bundled_product_matches_the_flipped_export():
    data, _ = load_psf_fits(DEFOCUS_2WAVE_PATH)
    flipped = load_huygens_psf(DEFOCUS_2WAVE_TXT_PATH).astype(np.float32)
    assert np.allclose(data, flipped)


@pytest.mark.parametrize("npix", [64, 65])
@pytest.mark.parametrize("zoom", [1.0, 0.5, 1.5])
def test_chief_ray_lands_on_grid_center(npix, zoom):
    """A delta at an off-centre chief ray renders centred, whatever the grid parity or scale."""
    arr = _blob(27, 13)
    src = CustomPSF(arr, src_um_per_pix=zoom, wavelength_scaling="none",
                    chief_ray_pix=(27, 13))
    psf = src.render(_ctx(npix, pixel_size_um=1.0))
    assert _centroid(psf) == pytest.approx(grid_center(npix), abs=0.02)


def test_chief_ray_lands_on_requested_center():
    arr = _blob(27, 13)
    src = CustomPSF(arr, src_um_per_pix=1.0, wavelength_scaling="none",
                    chief_ray_pix=(27, 13))
    psf = src.render(_ctx(64, pixel_size_um=1.0, center=(40.0, 32.25)))
    assert _centroid(psf) == pytest.approx((40.0, 32.25), abs=0.02)


def test_zoom_scales_the_grid_not_the_nodes():
    """Two blobs 20 samples apart render 10 px apart at zoom 0.5, not 20*31/63."""
    arr = _blob(20, 32, n=64, sigma=2.0) + _blob(40, 32, n=64, sigma=2.0)
    src = CustomPSF(arr, src_um_per_pix=0.5, wavelength_scaling="none")
    psf = src.render(_ctx(64, pixel_size_um=1.0))
    prof = psf.sum(axis=0)
    i = np.arange(64)
    left = (prof[:32] @ i[:32]) / prof[:32].sum()
    right = (prof[32:] @ i[32:]) / prof[32:].sum()
    assert right - left == pytest.approx(10.0, abs=0.01)


def test_defocus_product_renders_about_its_chief_ray():
    """On the detector grid the chief ray is at the centre; the photocentre sits
    +Y of it by Zemax's own 'Centroid offset' (0.73 um = 0.18 samples for 2 waves)."""
    psf = DefocusPSF(DEFOCUS_2WAVE_PATH).render(_ctx(301))
    cx, cy = _centroid(psf)
    gx, gy = grid_center(301)
    assert cx - gx == pytest.approx(0.0, abs=0.02)
    assert cy - gy == pytest.approx(0.18 * 4.0 / 3.76, abs=0.03)
