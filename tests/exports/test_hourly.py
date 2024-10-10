from datetime import datetime

import pytest

from emiproc import FILES_DIR
from emiproc.exports.hourly import export_hourly_emissions
from emiproc.tests_utils.exports import test_nc_metadata
from emiproc.tests_utils.temporal_profiles import (
    get_random_profiles,
    indexes_inv_catsub,
)
from emiproc.tests_utils.test_inventories import inv

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
    assert (output_dir / "20180101T000000Z.nc").exists()
    # Check until the last
    assert (output_dir / "20180101T230000Z.nc").exists()
