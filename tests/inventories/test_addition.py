"""Test the addition of two inventories."""

from __future__ import annotations

import pytest
import pandas as pd

from emiproc.inventories.utils import add_inventories, gdf_to_gdfs, scale_inventory
from emiproc.profiles.temporal.profiles import HourOfYearProfile, WeeklyProfile
from emiproc.tests_utils import temporal_profiles, test_inventories


def _clean_total(df: pd.DataFrame, like: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean the total emissions dataframe for comparison."""
    if like is not None:
        df = df.reindex_like(like)
    return df.fillna(0)


def test_self_addition():
    """Test the addition of an inventory with itself."""

    inv = test_inventories.inv_with_pnt_sources
    inv_added = add_inventories(inv, inv)

    assert inv_added.total_emissions.loc[(inv.substances, inv.categories)].equals(
        scale_inventory(inv, 2).total_emissions.loc[(inv.substances, inv.categories)]
    )


def test_addition():
    """Test the addition of two inventories."""

    inv1 = test_inventories.inv_with_pnt_sources
    inv2 = test_inventories.inv

    inv_added = add_inventories(inv1, inv2)

    # Here we need to fill the nan values with 0 to be able to compare the dataframes
    tot_added = _clean_total(inv_added.total_emissions)
    # Also expand the dataframes to have the same columns and rows
    tot1 = _clean_total(inv1.total_emissions, like=tot_added)
    tot2 = _clean_total(inv2.total_emissions, like=tot_added)

    assert tot_added.equals(tot1 + tot2)


def test_cannot_add_different_grid():
    """Test that we can only add inventories on the same grid."""

    inv1 = test_inventories.inv_on_grid_serie2_bis
    inv2 = test_inventories.inv_on_grid_serie2

    pytest.raises(ValueError, add_inventories, inv1, inv2)


def test_add_different_grid_with_option():
    """Different grids can be merged if 'remove_grid=True' is set."""

    inv1 = test_inventories.inv_on_grid_serie2_bis
    inv2 = test_inventories.inv_on_grid_serie2

    inv_added = add_inventories(inv1, inv2, remove_grid=True)

    # No more shared grid, everything went to gdfs
    assert inv_added.gdf is None

    tot_added = _clean_total(inv_added.total_emissions)
    tot1 = _clean_total(inv1.total_emissions, like=tot_added)
    tot2 = _clean_total(inv2.total_emissions, like=tot_added)

    assert tot_added.equals(tot1 + tot2)


def test_gdf_to_gdfs():
    """Test the conversion of the main gdf to per category gdfs."""

    inv = test_inventories.inv
    converted = gdf_to_gdfs(inv)

    assert converted.gdf is None
    assert converted.grid is None

    tot_before = inv.total_emissions
    tot_after = converted.total_emissions

    pd.testing.assert_frame_equal(tot_before, tot_after, check_like=True)


def test_gdf_to_gdfs_keeps_existing_gdfs():
    """Test that existing gdfs are kept when converting the main gdf."""

    inv = test_inventories.inv_with_pnt_sources
    converted = gdf_to_gdfs(inv)

    assert converted.gdf is None
    for cat in inv.gdfs:
        assert cat in converted.gdfs

    pd.testing.assert_frame_equal(
        inv.total_emissions, converted.total_emissions, check_like=True
    )


def test_gdf_to_gdfs_fails_on_profiles():
    """Test that gdf_to_gdfs fails if the inventory has profiles defined over cells."""

    inv = test_inventories.inv.copy()
    inv.set_profiles(
        temporal_profiles.three_composite_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )

    pytest.raises(NotImplementedError, gdf_to_gdfs, inv)


def test_add_different_grid_with_profiles():

    inv1 = test_inventories.inv_on_grid_serie2.copy()
    inv2 = test_inventories.inv_on_grid_serie2_bis.copy()

    inv1.set_profiles(
        temporal_profiles.three_composite_profiles,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )
    inv2.set_profiles(
        temporal_profiles.get_random_profiles(
            temporal_profiles.indexes_inv_catsub_missing.max().values + 1,
            profile_types=[HourOfYearProfile, WeeklyProfile],
        ),
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )

    with pytest.raises(NotImplementedError):
        # Currently the case
        add_inventories(inv1, inv2, remove_grid=True)


def test_profiles():
    """Test the addition of two inventories with profiles."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    inv1.set_profiles(
        temporal_profiles.three_composite_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )
    inv2.set_profiles(
        temporal_profiles.get_random_profiles(
            temporal_profiles.indexes_inv_catsub_missing.max().values + 1,
            profile_types=[HourOfYearProfile, WeeklyProfile],
        ),
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )

    summed_inv = add_inventories(inv1, inv2)

    total_summed = summed_inv.total_emissions

    pd.testing.assert_frame_equal(
        total_summed,
        inv1.total_emissions.add(inv2.total_emissions, fill_value=0),
        check_like=True,  # Ignore index ordering
    )


def test_profiles_values():
    """Test the addition of two inventories with profiles and values."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    inv1.set_profiles(
        temporal_profiles.three_composite_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )
    inv2.set_profiles(
        temporal_profiles.get_random_profiles(
            temporal_profiles.indexes_inv_catsub_missing.max().values + 1,
            profile_types=[HourOfYearProfile, WeeklyProfile],
        ),
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )

    summed_inv = add_inventories(inv1, inv2)

    total_summed = summed_inv.total_emissions

    pd.testing.assert_frame_equal(
        total_summed,
        inv1.total_emissions.add(inv2.total_emissions, fill_value=0),
        check_like=True,  # Ignore index ordering
    )
