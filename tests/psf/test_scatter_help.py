"""FRED ``.fgd`` reading (:mod:`wcc_etc.scatter_help`)."""

import os

import numpy as np
import pytest

from wcc_etc import scatter_help
from wcc_etc.scatter_psf import DEFAULT_SCATTER_FGD

FAKE_FGD = """FRED_DATA_FILE
FILETYPE= Grid2D
DATAUNITS= "Watts/mm^2"
A_AXIS_MIN= -2
A_AXIS_MAX= 2
A_AXIS_UNITS= mm
B_AXIS_MIN= -1
B_AXIS_MAX= 1
B_AXIS_UNITS= mm
BeginData
 1.0 2.0 3.0 4.0
 5.0 9.0 7.0 8.0
"""


@pytest.fixture
def fake_fgd(tmp_path):
    path = tmp_path / "fake.fgd"
    path.write_text(FAKE_FGD)
    return str(path)


class TestReadFgd:
    """The header/data split at the BeginData token."""

    def test_parses_the_data_block(self, fake_fgd):
        """Everything after BeginData becomes the array."""
        _, data = scatter_help.read_fgd(fake_fgd)
        assert data.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 9.0, 7.0, 8.0]]

    def test_parses_key_equals_value_header_lines(self, fake_fgd):
        """`key= value` lines land in the header dict."""
        header, _ = scatter_help.read_fgd(fake_fgd)
        assert header["A_AXIS_MIN"] == "-2"

    def test_returns_empty_array_when_there_is_no_data(self, tmp_path):
        """A header-only file gives an empty array rather than raising."""
        path = tmp_path / "empty.fgd"
        path.write_text("FILETYPE= Grid2D\nBeginData\n")
        _, data = scatter_help.read_fgd(str(path))
        assert data.size == 0


class TestFredResult:
    """The convenience wrapper, and its line cuts."""

    def test_desired_power_rescales_the_integral(self, fake_fgd):
        """desired_power sets the integrated power in watts."""
        fred = scatter_help.FredResult(filename=fake_fgd, desired_power=0.25)
        assert fred.current_power == pytest.approx(0.25)

    def test_line_cuts_have_the_grid_dimensions(self, fake_fgd):
        """X cut spans the columns, Y cut the rows."""
        fred = scatter_help.FredResult(filename=fake_fgd)
        assert (fred.x_cut.size, fred.y_cut.size) == (4, 2)

    def test_x_cut_runs_through_the_centroid_row(self, fake_fgd):
        """The X cut is the row containing the brightest structure."""
        fred = scatter_help.FredResult(filename=fake_fgd)
        assert fred.x_cut.tolist() == [5.0, 9.0, 7.0, 8.0]


class TestNoAstropylibDependency:
    """wcc_etc must import in an environment without astropylib."""

    def test_scatter_help_does_not_import_astropylib(self):
        """The module never pulls in the non-installable local package."""
        import wcc_etc.scatter_help as mod

        src = open(mod.__file__, encoding="utf-8").read()
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "astropylib" not in code


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_SCATTER_FGD),
    reason="the FRED .fgd is a large local data file, not shipped in the wheel",
)
class TestRealFgd:
    """Against the packaged Lazuli scatter map."""

    def test_shape_matches_the_declared_axis_dimensions(self):
        """A_AXIS_DIM x B_AXIS_DIM agrees with the parsed array."""
        header, data = scatter_help.read_fgd(DEFAULT_SCATTER_FGD)
        assert data.shape == (int(header["B_AXIS_DIM"]), int(header["A_AXIS_DIM"]))

    def test_map_is_everywhere_positive(self):
        """No hole values or negative irradiance survive the parse."""
        _, data = scatter_help.read_fgd(DEFAULT_SCATTER_FGD)
        assert np.all(data > 0)
