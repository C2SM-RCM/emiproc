# %%
import pytest
import geopandas as gpd
from emiproc.grids import RegularGrid
from emiproc.tests_utils.test_grids import regular_grid


# %%
def test_creation():
    RegularGrid(
        xmin=-1, xmax=5, ymin=-2, ymax=3, nx=10, ny=15, name="Test Regular Grid"
    )


def test_creation_rounding():

    kwargs = dict(
        xmin=-0.1,
        xmax=0.5,
        ymin=3.3,
        ymax=3.5,
        dx=0.2,
        dy=0.1,
        name="Test rounding grid",
    )
    grid = RegularGrid(**kwargs)

    # This was failing before the rounding was internally implemented
    assert grid.ny == 2
    assert grid.nx == 3


def test_creation_rounding_int():
    """Test that the grid is rounded to the nearest integer when dx and dy are integers"""

    kwargs = dict(
        xmin=1,
        xmax=5,
        ymin=2,
        ymax=5,
        dx=2,
        dy=1,
        name="Test rounding grid",
    )
    grid = RegularGrid(**kwargs)

    assert grid.ny == 3
    assert grid.nx == 2


def test_creation_n_d():
    RegularGrid(
        xmin=-1, ymin=-2, dx=0.1, dy=0.2, nx=10, ny=15, name="Test Regular Grid"
    )


def test_creation_max_d():
    RegularGrid(
        xmin=-1, ymin=-2, xmax=5, ymax=3, dx=0.1, dy=0.2, name="Test Regular Grid"
    )


def test_creation_fails_max_d_n():
    pytest.raises(
        ValueError,
        RegularGrid,
        xmin=-1,
        ymin=-2,
        xmax=5,
        ymax=3,
        dx=0.1,
        dy=0.2,
        nx=10,
        ny=15,
        name="Test Regular Grid",
    )


def test_creation_fails_not_enough():
    pytest.raises(
        ValueError,
        RegularGrid,
        xmin=-1,
        ymin=-2,
        xmax=5,
        ymax=3,
        name="Test Regular Grid",
    )
    pytest.raises(
        ValueError,
        RegularGrid,
        xmin=-1,
        ymin=-2,
        dx=0.1,
        dy=0.2,
        name="Test Regular Grid",
    )
    pytest.raises(
        ValueError,
        RegularGrid,
        xmin=-1,
        ymin=-2,
        nx=10,
        ny=15,
        name="Test Regular Grid",
    )


def test_polylist():
    regular_grid.cells_as_polylist

    # Test how we iterate over the cells in the polygon list


def test_area():
    regular_grid.cell_areas


def test_shape():
    nx, ny = regular_grid.shape

    assert nx == regular_grid.nx
    assert ny == regular_grid.ny
