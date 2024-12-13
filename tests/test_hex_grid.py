# %%
import pytest
import geopandas as gpd
from emiproc.grids import HexGrid
from emiproc.tests_utils.test_grids import hex_grid


def test_creation():
    HexGrid(xmin=-1, xmax=5, ymin=-2, ymax=3, nx=10, ny=15, name="Test Regular Grid")


def test_creation_orientation():
    HexGrid(
        xmin=-1,
        xmax=5,
        ymin=-2,
        ymax=3,
        nx=10,
        ny=15,
        name="Test Regular Grid",
        oriented_north=False,
    )


def test_creation_n_d():
    HexGrid(xmin=-1, ymin=-2, spacing=0.1, nx=10, ny=15, name="Test Regular Grid")


def test_creation_max_d():
    HexGrid(xmin=-1, ymin=-2, xmax=5, ymax=3, spacing=0.5, name="Test Regular Grid")


def test_creation_fails_max_d_n():
    pytest.raises(
        ValueError,
        HexGrid,
        xmin=-1,
        ymin=-2,
        xmax=5,
        ymax=3,
        spacing=0.2,
        nx=10,
        ny=15,
        name="Test Regular Grid",
    )


def test_creation_fails_not_enough():
    pytest.raises(
        ValueError,
        HexGrid,
        xmin=-1,
        ymin=-2,
        xmax=5,
        ymax=3,
        name="Test Regular Grid",
    )
    pytest.raises(
        ValueError,
        HexGrid,
        xmin=-1,
        ymin=-2,
        spacing=0.1,
        name="Test Hex  Grid",
    )
    pytest.raises(
        ValueError,
        HexGrid,
        xmin=-1,
        ymin=-2,
        nx=10,
        ny=15,
        name="Test Hex  Grid",
    )


def test_polylist():
    hex_grid.cells_as_polylist

    # Test how we iterate over the cells in the polygon list


def test_area():
    hex_grid.cell_areas


def test_shape():
    nx, ny = hex_grid.shape

    assert nx == hex_grid.nx
    assert ny == hex_grid.ny
