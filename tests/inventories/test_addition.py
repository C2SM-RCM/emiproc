"""Test the addition of two inventories."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from emiproc.inventories.utils import add_inventories, gdf_to_gdfs, scale_inventory
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


def _normalized(values: list[float] | np.ndarray) -> np.ndarray:
    """Return normalized ratios."""
    ratios = np.asarray(values, dtype=float)
    return ratios / ratios.sum()


def _profiles_group(
    *profiles,
) -> list[DailyProfile | WeeklyProfile | MounthsProfile]:
    """Create a temporal profiles group from ordered profile definitions."""
    return [
        profile_type(ratios=_normalized(ratios)) for profile_type, ratios in profiles
    ]


def _composite_ratios(
    profiles: list[DailyProfile | WeeklyProfile | MounthsProfile],
    types_order: list[type] | None = None,
) -> np.ndarray:
    """Concatenate the ratios of a temporal profiles group."""
    profiles_by_type = {
        type(profile): profile.ratios.reshape(-1) for profile in profiles
    }
    if types_order is None:
        types_order = [type(profile) for profile in profiles]
    return np.concatenate(
        [profiles_by_type[profile_type] for profile_type in types_order]
    )


def _get_output_ratios(inv, category: str, substance: str) -> np.ndarray:
    """Return the output ratios associated with a category/substance."""
    indexes = inv.t_profiles_indexes.sel(category=category, substance=substance).values
    return inv.t_profiles_groups.ratios[indexes]


def _weighted_expected(
    ratios_1: np.ndarray,
    ratios_2: np.ndarray,
    weights_1: float | np.ndarray,
    weights_2: float | np.ndarray,
) -> np.ndarray:
    """Compute the expected weighted average of two profiles."""
    weights_1 = np.asarray(weights_1, dtype=float)
    weights_2 = np.asarray(weights_2, dtype=float)
    total_weights = weights_1 + weights_2

    if weights_1.ndim == 0:
        return (ratios_1 * weights_1 + ratios_2 * weights_2) / total_weights

    return (
        ratios_1 * np.expand_dims(weights_1, axis=-1)
        + ratios_2 * np.expand_dims(weights_2, axis=-1)
    ) / np.expand_dims(total_weights, axis=-1)


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
    inv2_profiles = [
        _profiles_group(
            (WeeklyProfile, [7, 6, 5, 4, 3, 2, 1]),
            (MounthsProfile, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
        ),
        _profiles_group(
            (WeeklyProfile, [1, 3, 5, 7, 6, 4, 2]),
            (MounthsProfile, [1, 4, 2, 5, 3, 6, 7, 8, 9, 10, 11, 12]),
        ),
    ]

    inv1.set_profiles(
        temporal_profiles.three_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )
    inv2.set_profiles(
        inv2_profiles,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )

    summed_inv = add_inventories(inv1, inv2)

    total_summed = summed_inv.total_emissions
    types_order = list(summed_inv.t_profiles_groups._profiles.keys())
    expected_ratios = _weighted_expected(
        _composite_ratios(temporal_profiles.three_profiles[1], types_order),
        _composite_ratios(inv2_profiles[0], types_order),
        inv1.gdf[("adf", "CH4")].to_numpy(),
        inv2.gdf[("adf", "CH4")].to_numpy(),
    )

    pd.testing.assert_frame_equal(
        total_summed,
        inv1.total_emissions.add(inv2.total_emissions, fill_value=0),
        check_like=True,  # Ignore index ordering
    )
    np.testing.assert_allclose(
        _get_output_ratios(summed_inv, "adf", "CH4"), expected_ratios
    )


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
    inv2_reversed = test_inventories.inv.copy()

    inv2_reversed.gdf[("adf", "CH4")] *= 3

    inv1_profiles = [
        _profiles_group(
            (DailyProfile, np.ones(DailyProfile.size)),
            (WeeklyProfile, [1, 2, 3, 4, 5, 6, 7]),
        ),
        _profiles_group(
            (DailyProfile, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 2),
            (WeeklyProfile, [7, 6, 5, 4, 3, 2, 1]),
        ),
    ]
    inv2_profiles_reversed = [
        _profiles_group(
            (WeeklyProfile, [7, 5, 3, 1, 2, 4, 6]),
            (DailyProfile, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 2),
        ),
        _profiles_group(
            (WeeklyProfile, [2, 4, 6, 7, 5, 3, 1]),
            (DailyProfile, np.arange(1, DailyProfile.size + 1)),
        ),
    ]

    inv1.set_profiles(
        inv1_profiles,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )
    inv2_reversed.set_profiles(
        inv2_profiles_reversed,
        indexes=temporal_profiles.indexes_inv_catsub_missing,
    )

    summed_reversed_inv = add_inventories(inv1, inv2_reversed)
    types_order = list(summed_reversed_inv.t_profiles_groups._profiles.keys())
    expected_ch4 = _weighted_expected(
        _composite_ratios(inv1_profiles[0], types_order),
        _composite_ratios(inv2_profiles_reversed[0], types_order),
        inv1.gdf[("adf", "CH4")].sum(),
        inv2_reversed.gdf[("adf", "CH4")].sum(),
    )
    expected_co2 = _weighted_expected(
        _composite_ratios(inv1_profiles[1], types_order),
        _composite_ratios(inv2_profiles_reversed[1], types_order),
        inv1.gdf[("adf", "CO2")].sum(),
        inv2_reversed.gdf[("adf", "CO2")].sum(),
    )
    np.testing.assert_allclose(
        _get_output_ratios(summed_reversed_inv, "adf", "CH4"), expected_ch4
    )
    np.testing.assert_allclose(
        _get_output_ratios(summed_reversed_inv, "adf", "CO2"), expected_co2
    )
