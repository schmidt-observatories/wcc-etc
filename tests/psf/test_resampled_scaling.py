"""Tests that a resampled (defocus/custom) PSF uses the optics it is handed.

Before issue #65, _ResampledPSF.render ignored ctx.wavelength_m and ctx.fnum
entirely, so a bundled 500 nm f/15 defocus array was rendered identically no
matter what optics or source colour it was asked about.
"""

import numpy as np
import pytest

from wcc_etc.psfsim import (
    DEFOCUS_1WAVE_PATH,
    DEFOCUS_1WAVE_TXT_PATH,
    DEFOCUS_2WAVE_PATH,
    DEFOCUS_REF_FNUM,
    DEFOCUS_REF_WAVELENGTH_M,
    CustomPSF,
    DefocusPSF,
    DetectorPSFContext,
    load_psf_fits,
    parse_huygens_header,
)


def make_ctx(wavelength_m=500e-9, fnum=15.0, npix=200):
    """A detector context at the defocus products' reference optics by default."""
    return DetectorPSFContext(
        npix=npix,
        pixel_size_um=3.76,
        plate_scale_mas=16.9,
        wavelength_m=wavelength_m,
        diameter_m=3.065,
        fnum=fnum,
        jitter_sigma_mas=0.0,
        oversample=11,
    )


def second_moment(psf):
    """Flux-weighted radial second moment, in pixels — a scale-free width proxy."""
    npix = psf.shape[0]
    y, x = np.mgrid[0:npix, 0:npix]
    cy = np.sum(psf * y) / psf.sum()
    cx = np.sum(psf * x) / psf.sum()
    return np.sqrt(np.sum(psf * ((x - cx) ** 2 + (y - cy) ** 2)) / psf.sum())


class TestDefocusProductMetadata:
    def test_fits_product_exists(self):
        """The canonical defocus product is the FITS file."""
        assert DEFOCUS_1WAVE_PATH.endswith(".fits")

    def test_fits_carries_the_sampling(self):
        """PIXSCALE records the Zemax data spacing."""
        _, meta = load_psf_fits(DEFOCUS_1WAVE_PATH)
        assert meta["src_um_per_pix"] == pytest.approx(4.0)

    def test_fits_carries_the_reference_wavelength(self):
        """WAVELEN records the 500 nm the Zemax run was computed at."""
        _, meta = load_psf_fits(DEFOCUS_1WAVE_PATH)
        assert meta["ref_wavelength_m"] == pytest.approx(DEFOCUS_REF_WAVELENGTH_M)

    def test_fits_carries_the_reference_fnum(self):
        """FNUM records the f/15 design the run was computed for."""
        _, meta = load_psf_fits(DEFOCUS_1WAVE_PATH)
        assert meta["ref_fnum"] == pytest.approx(DEFOCUS_REF_FNUM)

    def test_fits_carries_the_defocus(self):
        """DEFOCUSW records how much defocus the product models."""
        _, meta = load_psf_fits(DEFOCUS_1WAVE_PATH)
        assert meta["defocus_waves"] == pytest.approx(1.0)

    def test_two_wave_product_carries_its_own_defocus(self):
        """The 2-wave product is distinguishable from the 1-wave one by header."""
        _, meta = load_psf_fits(DEFOCUS_2WAVE_PATH)
        assert meta["defocus_waves"] == pytest.approx(2.0)

    def test_fits_matches_the_txt_export_shape(self):
        """The FITS product is a faithful conversion of the Zemax export."""
        data, _ = load_psf_fits(DEFOCUS_1WAVE_PATH)
        assert (
            data.shape
            == (parse_huygens_header(DEFOCUS_1WAVE_TXT_PATH)["grid_size"],) * 2
        )


class TestDefocusPSFMetadata:
    def test_reads_sampling_from_the_product(self):
        """DefocusPSF no longer needs the 4 um spacing passed in by hand."""
        assert DefocusPSF(DEFOCUS_1WAVE_PATH).src_um_per_pix == pytest.approx(4.0)

    def test_reads_reference_wavelength_from_the_product(self):
        """The reference wavelength comes off the header."""
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
        assert psf.ref_wavelength_m == pytest.approx(DEFOCUS_REF_WAVELENGTH_M)

    def test_reads_reference_fnum_from_the_product(self):
        """The reference f-number comes off the header."""
        assert DefocusPSF(DEFOCUS_1WAVE_PATH).ref_fnum == pytest.approx(
            DEFOCUS_REF_FNUM
        )

    def test_still_loads_the_raw_zemax_export(self):
        """The .txt exports remain loadable, with the sampling from their header."""
        assert DefocusPSF(DEFOCUS_1WAVE_TXT_PATH).src_um_per_pix == pytest.approx(4.0)

    def test_raw_export_cannot_be_scaled(self):
        """A .txt export carries no reference optics, so rendering it must object."""
        with pytest.raises(ValueError, match="ref_wavelength_m"):
            DefocusPSF(DEFOCUS_1WAVE_TXT_PATH).render(make_ctx())

    def test_raw_export_renders_when_told_the_optics(self):
        """Supplying the missing reference optics makes a .txt export renderable."""
        psf = DefocusPSF(
            DEFOCUS_1WAVE_TXT_PATH,
            ref_wavelength_m=DEFOCUS_REF_WAVELENGTH_M,
            ref_fnum=DEFOCUS_REF_FNUM,
        )
        assert psf.render(make_ctx()).sum() == pytest.approx(1.0, abs=1e-6)

    def test_cache_key_distinguishes_scaling_modes(self):
        """Two scaling modes must not share a cached render."""
        assert (
            DefocusPSF(DEFOCUS_1WAVE_PATH).cache_key()
            != DefocusPSF(DEFOCUS_1WAVE_PATH, wavelength_scaling="waves").cache_key()
        )


class TestDespaceScaling:
    def test_wavelength_does_not_change_a_fixed_despace_blur(self):
        """A fixed focus error blurs by dz/F#, which carries no wavelength."""
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
        blue = psf.render(make_ctx(wavelength_m=450e-9))
        red = psf.render(make_ctx(wavelength_m=800e-9))
        assert np.allclose(blue, red)

    def test_faster_beam_gives_a_larger_blur(self):
        """Shrinking F# at fixed dz widens the blur (dz/F#)."""
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
        slow = psf.render(make_ctx(fnum=15.0))
        fast = psf.render(make_ctx(fnum=7.5))
        assert second_moment(fast) > second_moment(slow)

    def test_reference_optics_render_at_native_sampling(self):
        """At the product's own f-number nothing is rescaled."""
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH)
        native = CustomPSF(
            *load_psf_fits(DEFOCUS_1WAVE_PATH)[:1],
            src_um_per_pix=4.0,
            wavelength_scaling="none",
        ).render(make_ctx())
        assert np.allclose(psf.render(make_ctx(fnum=DEFOCUS_REF_FNUM)), native)


class TestWavesScaling:
    def test_redder_light_gives_a_larger_blur(self):
        """At a constant wavefront error in waves the blur scales as lambda."""
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH, wavelength_scaling="waves")
        blue = psf.render(make_ctx(wavelength_m=450e-9))
        red = psf.render(make_ctx(wavelength_m=800e-9))
        assert second_moment(red) > second_moment(blue)

    def test_reference_wavelength_matches_despace_at_reference_optics(self):
        """The two laws agree where the product was computed."""
        ctx = make_ctx(wavelength_m=DEFOCUS_REF_WAVELENGTH_M, fnum=DEFOCUS_REF_FNUM)
        despace = DefocusPSF(DEFOCUS_1WAVE_PATH).render(ctx)
        waves = DefocusPSF(DEFOCUS_1WAVE_PATH, wavelength_scaling="waves").render(ctx)
        assert np.allclose(despace, waves)

    def test_scales_linearly_with_wavelength(self):
        """Doubling lambda doubles the blur scale, so the second moment doubles."""
        psf = DefocusPSF(DEFOCUS_1WAVE_PATH, wavelength_scaling="waves")
        narrow = second_moment(psf.render(make_ctx(wavelength_m=400e-9, npix=400)))
        wide = second_moment(psf.render(make_ctx(wavelength_m=800e-9, npix=400)))
        assert wide / narrow == pytest.approx(2.0, rel=0.05)


class TestMissingMetadata:
    def test_raises_without_reference_metadata(self):
        """A bare array cannot be scaled, and must not be silently unscaled."""
        arr = np.zeros((51, 51))
        arr[25, 25] = 1.0
        with pytest.raises(ValueError, match="ref_wavelength_m"):
            CustomPSF(arr, src_um_per_pix=3.76).render(make_ctx())

    def test_explicit_none_scaling_is_allowed(self):
        """Opting out explicitly renders at native sampling without complaint."""
        arr = np.zeros((51, 51))
        arr[25, 25] = 1.0
        psf = CustomPSF(arr, src_um_per_pix=3.76, wavelength_scaling="none")
        assert psf.render(make_ctx()).sum() == pytest.approx(1.0, abs=1e-6)

    def test_supplied_metadata_enables_scaling(self):
        """A custom PSF that declares its optics gets scaled like a defocus product."""
        arr = np.zeros((101, 101))
        arr[40:61, 40:61] = 1.0
        psf = CustomPSF(
            arr,
            src_um_per_pix=3.76,
            ref_wavelength_m=500e-9,
            ref_fnum=15.0,
            wavelength_scaling="waves",
        )
        blue = second_moment(psf.render(make_ctx(wavelength_m=450e-9)))
        red = second_moment(psf.render(make_ctx(wavelength_m=800e-9)))
        assert red > blue

    def test_rejects_an_unknown_scaling_mode(self):
        """Typos in the scaling mode fail loudly at construction."""
        with pytest.raises(ValueError, match="wavelength_scaling"):
            CustomPSF(np.ones((5, 5)), src_um_per_pix=3.76, wavelength_scaling="magic")
