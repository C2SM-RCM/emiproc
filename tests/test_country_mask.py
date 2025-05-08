import numpy as np
import xarray as xr

import pytest

from emiproc.utilities import get_country_mask
from emiproc.grids import RegularGrid

from emiproc.tests_utils.test_grids import regular_grid_africa

from emiproc import FILES_DIR


def test_create_simple_mask():
    arr = get_country_mask(
        regular_grid_africa,
        resolution="110m",
    )
    # check that there are some countries in there
    # Not just -1 values
    assert len(np.unique(arr)) > 1
    # Make sure GMB was not found
    assert "GMB" not in np.unique(arr)
    # MAke sure Mauritania was found
    assert "MRT" in np.unique(arr)


def test_with_fractions():
    da = get_country_mask(
        regular_grid_africa,
        return_fractions=True,
    )
    total_fractions = da.sum(dim="country")
    # Check that sum of fractions is never more than 1
    assert (
        (total_fractions.values <= 1) | (np.isclose(total_fractions.values, 1.0))
    ).all()
    assert (total_fractions.values >= 0).all()

    # Full in MRT
    assert da.sel(cell=78, country="MRT").values == 1.0
    # THis is shared between SEN and ocean
    assert da.sel(cell=26, country="SEN").values > 0.01
    assert da.sel(cell=26, country="SEN").values < 0.5
    assert total_fractions.sel(cell=26).values < 0.5
    # This is just ocean
    assert (total_fractions.sel(cell=0).values == 0).all()


@pytest.mark.parametrize(
    "expected_weight_file, kwargs",
    [
        (FILES_DIR / "test" / "test_country_mask.npy", {}),
        (FILES_DIR / "test" / "test_country_mask.nc", {"return_fractions": True}),
    ],
)
def test_save_mask(expected_weight_file, kwargs):
    weigth_file = expected_weight_file
    if weigth_file.exists():
        # Clean the test
        weigth_file.unlink()
    out1 = get_country_mask(
        regular_grid_africa,
        weight_filepath=weigth_file,
        **kwargs,
    )
    assert weigth_file.exists()
    # Call again to make sure it is loaded
    out2 = get_country_mask(
        regular_grid_africa,
        weight_filepath=weigth_file,
        **kwargs,
    )

    # Check that the two are the same
    if isinstance(out1, xr.DataArray):
        xr.testing.assert_equal(out1, out2)
    elif isinstance(out1, np.ndarray):
        np.testing.assert_array_equal(out1, out2)
    else:
        raise TypeError(f"Unexpected type {type(out1)}")


if __name__ == "__main__":
    pytest.main([__file__])
