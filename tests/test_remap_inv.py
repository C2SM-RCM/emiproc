import pytest 
from emiproc.inventories import  EmissionInfo, Inventory
from emiproc.regrid import remap_inventory
from emiproc.tests_utils.test_inventories import inv_with_pnt_sources, inv_with_gdfs_bad_indexes, inv
from emiproc.tests_utils.test_grids import regular_grid, gpd_grid
from emiproc.tests_utils.temporal_profiles import three_composite_profiles, indexes_inv_catsubcell

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



def test_remap_with_gdf_wrong_indices():
    """Test that the remap_inventory function works also if the indices in the gdfs are bad."""

    inv = inv_with_gdfs_bad_indexes

    regular_grid.crs = None
    remapped = remap_inventory(inv, regular_grid)


def test_remap_inv_with_profiles():

    this_inv = inv.copy()

    this_inv.set_profiles(three_composite_profiles, indexes_inv_catsubcell)

    remapped_inv = remap_inventory(this_inv, gpd_grid)

    assert len(remapped_inv.t_profiles_groups) > 1, "There should be more than one profiles"
    assert remapped_inv.t_profiles_indexes.dims == indexes_inv_catsubcell.dims