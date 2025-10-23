from datetime import datetime

import pytest
import xarray as xr

from emiproc import FILES_DIR
from emiproc.exports.hourly import export_hourly_emissions
from emiproc.regrid import remap_inventory
from emiproc.tests_utils.exports import test_nc_metadata
from emiproc.tests_utils.temporal_profiles import (
    get_random_profiles,
    indexes_inv_catsub,
)
from emiproc.tests_utils.test_grids import regular_grid
from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources

# output path and filename
output_dir = FILES_DIR / "test/exports/test_hourly"
output_dir.mkdir(parents=True, exist_ok=True)


def test_no_profiles():
    """Test that if you have no profile, you get an error."""
    with pytest.raises(ValueError):
        export_hourly_emissions(
            inv=inv,
            path=output_dir,
            netcdf_attributes=test_nc_metadata,
            start_time=datetime(2018, 1, 1),
            end_time=datetime(2018, 1, 2),
        )


def test_export_simple():
    # Export the temporal profiles of TNO
    output_files = [
        output_dir / stem for stem in ["20180101T000000Z.nc", "20180101T010000Z.nc"]
    ]
    for file in output_files:
        file.unlink(missing_ok=True)
    inv_profiled = inv.copy()
    inv_profiled.set_profiles(
        get_random_profiles(indexes_inv_catsub.max().values + 1),
        indexes=indexes_inv_catsub,
    )
    export_hourly_emissions(
        inv=inv_profiled,
        path=output_dir,
        netcdf_attributes=test_nc_metadata,
        start_time=datetime(2018, 1, 1),
        end_time=datetime(2018, 1, 2),
    )

    # Check that the file is there
    for file in output_files:
        assert file.exists(), f"File {file} does not exist"


def test_export_monthly():
    # Export the temporal profiles of TNO
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        monthly_dir / stem for stem in ["20180101T000000Z.nc", "20180601T000000Z.nc"]
    ]
    inv_profiled = inv.copy()
    inv_profiled.set_profiles(
        get_random_profiles(indexes_inv_catsub.max().values + 1),
        indexes=indexes_inv_catsub,
    )
    export_hourly_emissions(
        inv=inv_profiled,
        path=monthly_dir,
        netcdf_attributes=test_nc_metadata,
        start_time=datetime(2018, 1, 1),
        end_time=datetime(2018, 6, 1),
        freq="MS",
    )

    # Check that the file is there
    for file in output_files:
        assert file.exists(), f"File {file} does not exist"


def test_with_regular_grid():

    # Convert to the correct CRS
    raster_inv = inv_with_pnt_sources.copy()
    raster_inv.set_crs(regular_grid.crs)
    raster_inv = remap_inventory(
        inv=raster_inv,
        grid=regular_grid,
    )
    # Export the temporal profiles of TNO
    regular_output_dir = output_dir / "regular"
    regular_output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        regular_output_dir / stem
        for stem in ["20180101T000000Z.nc", "20180101T010000Z.nc"]
    ]
    for file in output_files:
        file.unlink(missing_ok=True)
    raster_inv.set_profiles(
        get_random_profiles(indexes_inv_catsub.max().values + 1),
        indexes=indexes_inv_catsub,
    )
    export_hourly_emissions(
        inv=raster_inv,
        path=regular_output_dir,
        netcdf_attributes=test_nc_metadata,
        start_time=datetime(2018, 1, 1),
        end_time=datetime(2018, 1, 2),
    )

    # Check that the file is there
    for file in output_files:
        assert file.exists(), f"File {file} does not exist"
        ds = xr.open_dataset(file)
        assert "cell_area" in ds.variables.keys(), "cell_area not in dataset"
