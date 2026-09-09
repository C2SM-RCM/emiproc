"""Test the addition of two inventories."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from emiproc.inventories.utils import add_inventories, gdf_to_gdfs, scale_inventory
from emiproc.profiles.operators import add_profiles
from emiproc.profiles.temporal.profiles import (
    DailyProfile,
    HourOfYearProfile,
    MounthsProfile,
    WeeklyProfile,
)
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
        temporal_profiles.three_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )
    inv2.set_profiles(
        temporal_profiles.three_profiles,
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
    """Test the addition of two inventories with matching temporal profiles."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()
    weekly_1 = np.array([7, 6, 5, 4, 3, 2, 1], dtype=float)
    monthly_1 = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    weekly_2 = np.array([1, 3, 5, 7, 6, 4, 2], dtype=float)
    monthly_2 = np.array([1, 4, 2, 5, 3, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    inv2_profiles = [
        [
            WeeklyProfile(ratios=weekly_1 / weekly_1.sum()),
            MounthsProfile(ratios=monthly_1 / monthly_1.sum()),
        ],
        [
            WeeklyProfile(ratios=weekly_2 / weekly_2.sum()),
            MounthsProfile(ratios=monthly_2 / monthly_2.sum()),
        ],
    ]

    inv1.set_profiles(
        temporal_profiles.three_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )
    inv2.set_profiles(
        inv2_profiles,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )

    expected_profiles, expected_indexes = add_profiles(inv1, inv2)
    summed_inv = add_inventories(inv1, inv2)

    total_summed = summed_inv.total_emissions

    pd.testing.assert_frame_equal(
        total_summed,
        inv1.total_emissions.add(inv2.total_emissions, fill_value=0),
        check_like=True,  # Ignore index ordering
    )
    np.testing.assert_allclose(
        summed_inv.t_profiles_groups.ratios, expected_profiles.ratios
    )
    assert summed_inv.t_profiles_indexes.equals(expected_indexes)


def test_profiles_types_must_match():
    """Test that adding different temporal profile types raises an error."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()

    inv1.set_profiles(
        temporal_profiles.three_profiles,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )
    inv2.set_profiles(
        [
            [
                WeeklyProfile(
                    ratios=np.full(WeeklyProfile.size, 1 / WeeklyProfile.size)
                ),
                HourOfYearProfile(
                    ratios=np.full(
                        HourOfYearProfile.size,
                        1 / HourOfYearProfile.size,
                    )
                ),
            ]
        ],
        indexes=temporal_profiles.indexes_inv_catsub_missing * 0,
    )

    with pytest.raises(
        ValueError,
        match="Please interpolate the temporal profiles to a common temporal resolution",
    ):
        add_inventories(inv1, inv2)


def test_profiles_values_independent_of_type_order():
    """Test that temporal profile addition is independent of type ordering."""

    inv1 = test_inventories.inv.copy()
    inv2 = test_inventories.inv.copy()
    inv2_reversed = test_inventories.inv.copy()

    inv2.gdf[("adf", "CH4")] *= 3
    inv2_reversed.gdf[("adf", "CH4")] *= 3

    inv1_profiles = [
        [
            DailyProfile(ratios=np.full(DailyProfile.size, 1 / DailyProfile.size)),
            WeeklyProfile(ratios=np.array([1, 2, 3, 4, 5, 6, 7], dtype=float) / 28),
        ],
        [
            DailyProfile(
                ratios=np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 2, dtype=float
                )
                / 156
            ),
            WeeklyProfile(ratios=np.array([7, 6, 5, 4, 3, 2, 1], dtype=float) / 28),
        ],
    ]
    inv2_profiles = [
        [
            DailyProfile(
                ratios=np.array(
                    [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 2, dtype=float
                )
                / 156
            ),
            WeeklyProfile(ratios=np.array([7, 5, 3, 1, 2, 4, 6], dtype=float) / 28),
        ],
        [
            DailyProfile(ratios=np.arange(1, DailyProfile.size + 1, dtype=float) / 300),
            WeeklyProfile(ratios=np.array([2, 4, 6, 7, 5, 3, 1], dtype=float) / 28),
        ],
    ]
    inv2_profiles_reversed = [[profile[1], profile[0]] for profile in inv2_profiles]

    inv1.set_profiles(
        inv1_profiles,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )
    inv2.set_profiles(
        inv2_profiles,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )
    inv2_reversed.set_profiles(
        inv2_profiles_reversed,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )

    summed_inv = add_inventories(inv1, inv2)
    summed_reversed_inv = add_inventories(inv1, inv2_reversed)
    normalized_profiles = summed_inv.t_profiles_groups.broadcast(
        list(summed_inv.t_profiles_groups._profiles.keys())
    )
    normalized_reversed_profiles = summed_reversed_inv.t_profiles_groups.broadcast(
        list(summed_inv.t_profiles_groups._profiles.keys())
    )

    pd.testing.assert_frame_equal(
        summed_inv.total_emissions,
        summed_reversed_inv.total_emissions,
        check_like=True,
    )
    assert summed_inv.t_profiles_indexes.equals(summed_reversed_inv.t_profiles_indexes)
    np.testing.assert_allclose(
        normalized_profiles.ratios, normalized_reversed_profiles.ratios
    )
