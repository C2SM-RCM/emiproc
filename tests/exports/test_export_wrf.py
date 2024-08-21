from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from emiproc.exports.wrf import WRF_Grid, export_wrf_hourly_emissions
from emiproc.inventories import Inventory
from emiproc.tests_utils import TEST_OUTPUTS_DIR, TESTFILES_DIR, temporal_profiles

test_grid_filepath = Path(TESTFILES_DIR / "wrf" / "wrfinput_for_testing.nc")


def test_grid_file_is_there():
    assert test_grid_filepath.is_file()


def test_load_grid():
    grid = WRF_Grid(test_grid_filepath)

    assert grid is not None
    assert isinstance(grid, WRF_Grid)

    assert grid.shape == (3, 6)


def get_inventory():
    grid = WRF_Grid(test_grid_filepath)
    gdf = grid.gdf.copy()

    inv = Inventory.from_gdf(
        gpd.GeoDataFrame(
            {
                ("CO2", "test"): np.ones(len(gdf)),
                ("CO2", "test2"): np.arange(len(gdf)),
            },
            geometry=gdf.geometry,
        )
    )
    inv.grid = grid

    return inv


def test_inventory_export():
    inv = get_inventory()
    inv.set_profiles(
        [[temporal_profiles.daily_test_profile]],
        indexes=xr.DataArray([0], coords={"substance": ["CO2"]}),
    )

    output_dir = TEST_OUTPUTS_DIR / "wrf"
    output_dir.mkdir(exist_ok=True, parents=True)

    export_wrf_hourly_emissions(inv, inv.grid, ("2018-01-01", "2018-01-02"), output_dir)
