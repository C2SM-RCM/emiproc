from datetime import date

import xarray as xr
import numpy as np

from emiproc.tests_utils.test_grids import regular_grid
from emiproc.tests_utils.test_inventories import inv_with_pnt_sources
from emiproc.tests_utils import TEST_OUTPUTS_DIR

from emiproc.exports.rasters import export_raster_netcdf
from emiproc.utilities import Units

# Convert to the correct CRS
raster_inv = inv_with_pnt_sources.copy()
raster_inv.set_crs(regular_grid.crs)


def test_base_function():
    """Simply test that the function works with defaults"""

    export_raster_netcdf(
        raster_inv,
        TEST_OUTPUTS_DIR / "test_raster.nc",
        regular_grid,
        netcdf_attributes={},
    )


def test_group_categories():
    """Simply test that the function works with defaults"""

    export_raster_netcdf(
        raster_inv,
        TEST_OUTPUTS_DIR / "test_raster.nc",
        regular_grid,
        netcdf_attributes={},
        group_categories=True,
    )


def test_unit():

    export_raster_netcdf(
        raster_inv,
        TEST_OUTPUTS_DIR / "test_raster_kg_per_m2_per_s.nc",
        regular_grid,
        netcdf_attributes={},
        unit=Units.KG_PER_M2_PER_S,
    )


def test_with_year():

    raster_inv_with_year = raster_inv.copy()
    raster_inv_with_year.year = 2025

    file = TEST_OUTPUTS_DIR / "test_raster_with_year.nc"

    export_raster_netcdf(
        raster_inv_with_year,
        file,
        regular_grid,
    )

    # Check that the year is in the file
    with xr.open_dataset(file) as ds:

        assert ds.attrs["year"] == 2025
        assert "time" in ds.dims
        assert "time" in ds.coords

        # Value assigned should be middle of the year
        np.testing.assert_array_equal(
            ds["time"].values, np.array([date(2025, 7, 1)], dtype="datetime64[D]")
        )
