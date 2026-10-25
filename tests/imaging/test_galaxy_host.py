"""render_profile dispatch: sersic and ngc spatial models on the detector grid."""

import numpy as np
import pytest

from wcc_etc import galaxy
from wcc_etc.extended import render_profile, render_sersic

PLATE = 0.016869  # arcsec/pix


class TestRenderProfile:
    def test_sersic_dispatches_to_render_sersic(self):
        """A sersic profile dict renders exactly as render_sersic does."""
        profile = {"model": "sersic", "r_eff": 0.3, "n": 1.0}
        assert np.array_equal(
            render_profile(profile, PLATE, 64, 5), render_sersic(profile, PLATE, 64, 5)
        )

    def test_ngc_dispatches_to_render_galaxy(self):
        """An ngc profile dict renders exactly as render_galaxy does."""
        profile = {
            "model": "ngc",
            "name": "ngc0628",
            "redshift": 0.9,
            "dx": 0.2,
            "dy": 0.0,
            "pa": 30.0,
        }
        assert np.array_equal(
            render_profile(profile, PLATE, 128),
            galaxy.render_galaxy("ngc0628", 0.9, PLATE, 128, dx=0.2, dy=0.0, pa=30.0),
        )

    def test_ngc_ignores_total_flag(self):
        """An NGC image is always a flux fraction; total=False changes nothing."""
        profile = {"model": "ngc", "name": "ngc0628", "redshift": 0.9}
        assert np.array_equal(
            render_profile(profile, PLATE, 64, total=False),
            render_profile(profile, PLATE, 64, total=True),
        )

    def test_unknown_model_raises(self):
        """A profile dict with an unknown model is an error."""
        with pytest.raises(ValueError):
            render_profile({"model": "gaussian"}, PLATE, 64)
