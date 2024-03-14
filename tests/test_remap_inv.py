import pytest 
from emiproc.inventories import  EmissionInfo
from emiproc.regrid import remap_inventory
from emiproc.tests_utils.test_inventories import inv_with_pnt_sources
from emiproc.tests_utils.test_grids import regular_grid, gpd_grid

from emiproc.inventories.utils import get_total_emissions
from emiproc.utilities import total_emissions_almost_equal




def test_remap():

    grid = regular_grid
    grid.crs = None
    remaped_inv = remap_inventory(inv_with_pnt_sources, grid)

    # Check the grid size
    assert remaped_inv.gdf.shape[0] == len(grid.gdf)
    assert len(remaped_inv.gdfs) == 0, "No point sources should be left"

    # Check the total emissions dictionaries are the same (grid is larger than inventory)
    total_inv = get_total_emissions(inv_with_pnt_sources)
    total_remapped = get_total_emissions(remaped_inv)
    assert total_emissions_almost_equal(total_inv, total_remapped)


def test_remap_keep_shapes():


    grid = regular_grid
    grid.crs = None
    remaped_inv = remap_inventory(inv_with_pnt_sources, grid, keep_gdfs=True)

    # Check the grid size
    assert remaped_inv.gdf.shape[0] == len(grid.gdf)
    # Check the point sources
    # The dataframes should be the same 
    assert list(remaped_inv.gdfs.keys()) == list(inv_with_pnt_sources.gdfs.keys())
    for key in remaped_inv.gdfs.keys():
        assert remaped_inv.gdfs[key].shape == inv_with_pnt_sources.gdfs[key].shape
        assert list(remaped_inv.gdfs[key].columns) == list(inv_with_pnt_sources.gdfs[key].columns)
        assert remaped_inv.gdfs[key].geometry.equals(inv_with_pnt_sources.gdfs[key].geometry)
        # Check all values 
        for col in remaped_inv.gdfs[key].columns:
            assert remaped_inv.gdfs[key][col].equals(inv_with_pnt_sources.gdfs[key][col])

    # Check the total emissions dictionaries are the same (grid is larger than inventory)
    total_inv = get_total_emissions(inv_with_pnt_sources)
    total_remapped = get_total_emissions(remaped_inv)
    assert total_emissions_almost_equal(total_inv, total_remapped)




def test_remap_different_grids():
    for grid in [regular_grid, gpd_grid]:
        try:
            grid.crs = None
            remaped_inv = remap_inventory(inv_with_pnt_sources, grid=grid)
        except Exception as e:
            raise AssertionError(f"Remapping failed for grid {grid.name}") from e

