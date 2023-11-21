import geopandas as gpd
import pytest
import xarray as xr

from emiproc.grids import Grid
from emiproc.inventories import Inventory
from emiproc.profiles.vertical_profiles import VerticalProfiles
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources


@pytest.mark.parametrize(
    "inv",
    (
        inv,
        inv_with_pnt_sources,
    ),
)
def test_correct_fromat(inv: Inventory):
    """Test that the inventory is in the correct format"""

    assert hasattr(inv, "grid")
    if inv.grid is not None:
        assert isinstance(inv.grid, Grid)

    # Gdf or gdfs must be specified
    assert not (inv.gdf is None and inv.gdfs is None)
    if inv.gdf is not None:
        assert isinstance(inv.gdf, gpd.GeoDataFrame)
    if inv.gdfs is not None:
        assert isinstance(inv.gdfs, dict)
        for gdf in inv.gdfs.values():
            assert isinstance(gdf, gpd.GeoDataFrame)

    if hasattr(inv, "v_profiles") and inv.v_profiles is not None:
        assert isinstance(inv.v_profiles, VerticalProfiles)
        assert hasattr(inv, "v_profiles_indexes")
        assert isinstance(inv.v_profiles_indexes, xr.DataArray)
