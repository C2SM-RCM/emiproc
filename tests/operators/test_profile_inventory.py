"""Tests for profiles operators on inventories."""

import pytest
from emiproc.inventories import Inventory
from emiproc.inventories.utils import interpolate_temporal_profiles, country_to_cells


from emiproc.tests_utils.african_case import (
    african_inv_with_tprofiles_2d,
    african_inv_with_tprofiles,
)


def inventory_parametrize(func):
    """Decorator to parametrize tests with different inventories."""
    return pytest.mark.parametrize(
        "inv",
        [
            african_inv_with_tprofiles_2d,
            african_inv_with_tprofiles,
        ],
    )(func)


@inventory_parametrize
def test_countries_to_cells(inv: Inventory):
    """Test the countries to cells operator."""
    inv_countries_to_cells = country_to_cells(inv)
    dims = inv_countries_to_cells.t_profiles_indexes.dims
    assert "country" not in dims
    assert "cell" in dims


@inventory_parametrize
def test_interpolate_temporal_profiles(inv: Inventory):
    """Test the interpolation of temporal profiles."""
    # Year is required for interpolation
    if inv.year is None:
        inv.year = 2020
    inv_interpolated = interpolate_temporal_profiles(inv)
    assert inv_interpolated.t_profiles_groups is not None
    assert inv_interpolated.t_profiles_indexes is not None


if __name__ == "__main__":
    pytest.main([__file__])
