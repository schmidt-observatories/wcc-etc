"""wcc_etc.galaxy: NGC data access, cosmological pixel scale, image projection."""

import numpy as np
import pytest
from astropy.cosmology import Planck18

from wcc_etc import galaxy
from wcc_etc.psfsim import grid_center

NGC = "ngc0628"
NGC_Z = 0.0022
NGC_PIXSCALE = 0.858981
PLATE = 0.016869  # arcsec/pix


def centroid(img):
    yy, xx = np.indices(img.shape)
    return (xx * img).sum() / img.sum(), (yy * img).sum() / img.sum()


class TestGalaxyData:
    def test_spectrum_is_flam(self):
        """The galaxy template loads as a synphot spectrum in erg/s/cm^2/A."""
        spec = galaxy.load_galaxy_spectrum(NGC)
        assert spec(5000.0).unit.is_equivalent("photlam")

    def test_spectrum_is_cached(self):
        """The same name returns the same loaded object."""
        assert galaxy.load_galaxy_spectrum(NGC) is galaxy.load_galaxy_spectrum(NGC)

    def test_image_sums_to_one(self):
        """The thumbnail is flux-normalised."""
        assert np.isclose(galaxy.load_galaxy_image(NGC).image.sum(), 1.0)

    def test_image_header_values(self):
        """Pixel scale, redshift and 0-based centre come from the FITS header."""
        img = galaxy.load_galaxy_image(NGC)
        assert (img.pixel_scale, img.redshift, img.center) == (
            NGC_PIXSCALE,
            NGC_Z,
            (359.5, 360.5),
        )


class TestGalaxyPixelScale:
    def test_angular_diameter_ratio(self):
        """Moving the galaxy to z shrinks its pixel scale by D_A(z_gal)/D_A(z)."""
        expected = NGC_PIXSCALE * (
            Planck18.angular_diameter_distance(NGC_Z)
            / Planck18.angular_diameter_distance(0.9)
        )
        assert np.isclose(galaxy.galaxy_pixel_scale(NGC_PIXSCALE, NGC_Z, 0.9), expected)

    def test_rejects_redshift_below_galaxy(self):
        """A galaxy cannot be moved closer than it already is."""
        with pytest.raises(ValueError):
            galaxy.galaxy_pixel_scale(NGC_PIXSCALE, NGC_Z, 0.001)


@pytest.fixture
def blob():
    """A symmetric synthetic galaxy: a 41x41 Gaussian, sum 1, centred on (20, 20)."""
    yy, xx = np.indices((41, 41))
    img = np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 4.0**2))
    return img / img.sum()


class TestProjectImage:
    def test_conserves_flux_on_a_large_grid(self, blob):
        """At scale 1 on a grid that holds the whole blob, nothing is lost."""
        out = galaxy.project_image(blob, (20, 20), 1.0, (101, 101), (50, 50))
        assert np.isclose(out.sum(), 1.0, atol=1e-6)

    def test_rotation_matches_rot90(self, blob):
        """A 90 degree CCW rotation (origin lower) is np.rot90(..., -1) in array terms."""
        blob = blob.copy()
        blob[20, 30] += 0.5  # break the symmetry
        base = galaxy.project_image(blob, (20, 20), 1.0, (101, 101), (50, 50))
        rot = galaxy.project_image(
            blob, (20, 20), 1.0, (101, 101), (50, 50), rotation=90
        )
        assert np.allclose(rot, np.rot90(base, -1), atol=1e-9)

    def test_offset_moves_the_centroid(self, blob):
        """position=(dx, dy) output pixels shifts the centroid by that much."""
        out = galaxy.project_image(
            blob, (20, 20), 1.0, (101, 101), (50, 50), position=(7, -3)
        )
        assert np.allclose(centroid(out), (57, 47), atol=1e-6)

    def test_light_off_the_grid_is_lost(self, blob):
        """The output is not renormalised when the galaxy spills off the grid."""
        out = galaxy.project_image(blob, (20, 20), 1.0, (21, 21), (10, 10))
        assert out.sum() < 1.0


class TestRenderGalaxy:
    def test_centred_on_grid_center(self):
        """dx=dy=0 puts the galaxy centre on psfsim.grid_center."""
        img = galaxy.render_galaxy(NGC, 0.9, PLATE, 256)
        assert np.allclose(centroid(img), grid_center(256), atol=1.0)

    def test_offset_in_arcsec(self):
        """dx=0.5 arcsec moves the centroid by 0.5/plate pixels."""
        base = centroid(galaxy.render_galaxy(NGC, 0.9, PLATE, 256))
        moved = centroid(galaxy.render_galaxy(NGC, 0.9, PLATE, 256, dx=0.5))
        assert np.isclose(moved[0] - base[0], 0.5 / PLATE, atol=0.05)

    def test_flux_fraction_at_most_one(self):
        """The rendered image is a flux fraction: it never exceeds 1."""
        assert galaxy.render_galaxy(NGC, 0.9, PLATE, 256).sum() <= 1.0 + 1e-9
