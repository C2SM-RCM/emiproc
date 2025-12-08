import pytest
import geopandas as gpd
import pandas as pd
from emiproc.grids import RegularGrid
from emiproc.tests_utils.test_grids import gpd_grid


def test_polylist():
    polylist = gpd_grid.cells_as_polylist
    assert len(polylist) == gpd_grid.nx * gpd_grid.ny


def test_shape():
    nx, ny = gpd_grid.shape

    assert nx == gpd_grid.nx
    assert ny == 1


def test_centers():
    centers = gpd_grid.centers

    assert len(centers) == gpd_grid.nx * gpd_grid.ny
    assert isinstance(centers, gpd.GeoSeries)

    # Can get the x and y
    x = centers.x
    y = centers.y
    assert isinstance(x, pd.Series)
    assert isinstance(y, pd.Series)

    # Get centroid with gpd method
    center_gpd = gpd_grid.gdf.centroid
    x_gpd, y_gpd = center_gpd.x, center_gpd.y

    # Check that the centers are in the right order

    pd.testing.assert_series_equal(x, x_gpd)
    pd.testing.assert_series_equal(y, y_gpd)
