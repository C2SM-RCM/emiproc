import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
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


def get_test_file_path() -> tuple[Path, Path]:
    output_first_file = TEST_OUTPUTS_DIR / "wrf" / "wrfchemi_d01_2018-01-01_00:00:00"
    output_last_file = TEST_OUTPUTS_DIR / "wrf" / "wrfchemi_d01_2018-01-01_23:00:00"

    # Windows has no `:` in filenames
    if os.name == "nt":
        replace = lambda x: x.parent / x.name.replace(":", "-")
        output_first_file = replace(output_first_file)
        output_last_file = replace(output_last_file)

    return output_first_file, output_last_file


@pytest.mark.incremental
class TestOutputs:

    def test_inventory_export(self):
        inv = get_inventory()
        inv.set_profiles(
            [[temporal_profiles.daily_test_profile]],
            indexes=xr.DataArray([0], coords={"substance": ["CO2"]}),
        )

        output_dir = TEST_OUTPUTS_DIR / "wrf"
        output_dir.mkdir(exist_ok=True, parents=True)

        export_wrf_hourly_emissions(
            inv, inv.grid, ("2018-01-01", "2018-01-02"), output_dir
        )

    def test_output_file(self):
        output_first_file, output_last_file = get_test_file_path()

        assert output_first_file.is_file()
        assert output_last_file.is_file()

    def test_file_content(self):
        file_path, _ = get_test_file_path()

        ds = xr.open_dataset(file_path)

        assert ds is not None
        assert isinstance(ds, xr.Dataset)

        assert "E_test_CO2" in ds
        assert "E_test2_CO2" in ds

        assert "Times" in ds
        assert ds["Times"].shape == (1,)
        assert ds["Times"].values[0] == b"2018-01-01_00:00:00"

        # Requires many attributes, just test one or two randomly
        assert len(ds.attrs) > 20
        assert "GFDDA_INTERVAL_M" in ds.attrs
        assert "DX" in ds.attrs
        assert "emiproc" in ds.attrs
