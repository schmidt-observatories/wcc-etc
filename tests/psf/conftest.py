"""Fixtures for the PSF tests."""

import numpy as np
import pytest

HUYGENS_GRID = 32
#: 1-based ``Center point`` written into the synthetic export.
HUYGENS_CENTER_POINT = (17, 17)


def synthetic_huygens_psf():
    """A Gaussian off the grid centre, so a row flip or a mirror is detectable."""
    yy, xx = np.indices((HUYGENS_GRID, HUYGENS_GRID))
    return np.exp(-((xx - 16) ** 2 + (yy - 20) ** 2) / (2 * 2.0**2))


@pytest.fixture
def huygens_export(tmp_path):
    """A small Huygens PSF export in Zemax's own layout, written to ``tmp_path``.

    UTF-16, ``#`` header lines, then tab-separated relative intensities with the
    top row (largest +Y) first.
    """
    psf = synthetic_huygens_psf()
    header = [
        "#Defocus_waves: 1",
        "#0.4300 to 0.8300 µm at 0.0000, 0.0000 (deg).",
        "#Data spacing is 4.000 µm.",
        f"#Data area is {4.0 * HUYGENS_GRID:.3f} by {4.0 * HUYGENS_GRID:.3f} µm.",
        "#Strehl ratio: 0.011",
        f"#Pupil grid size: {HUYGENS_GRID} by {HUYGENS_GRID}",
        f"#Image grid size: {HUYGENS_GRID} by {HUYGENS_GRID}",
        "#Center point is: {}, {}".format(*HUYGENS_CENTER_POINT),
        "#",
        "#Values are relative intensity.",
        "#",
    ]
    rows = ["\t".join(f"{v:.4E}" for v in row) for row in psf[::-1]]
    path = tmp_path / "synthetic_Huygens-PSF-Data_Linear.txt"
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-16")
    return str(path)
