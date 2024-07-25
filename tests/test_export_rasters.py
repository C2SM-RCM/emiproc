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