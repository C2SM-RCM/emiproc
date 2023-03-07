#%%
import geopandas as gpd
from emiproc.grids import RegularGrid
from emiproc.tests_utils.test_grids import regular_grid

#%%
def test_creation():
    RegularGrid(-1, 5, -2, 3, 10, 15, name='Test Regular Grid')

def test_polylist():
    regular_grid.cells_as_polylist

    # Test how we iterate over the cells in the polygon list

def test_area():

    regular_grid.cell_areas

def test_shape():
    nx, ny = regular_grid.shape

    assert nx == regular_grid.nx 
    assert ny == regular_grid.ny
