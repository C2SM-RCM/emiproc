"""Test the addition of two inventories."""

from __future__ import annotations

import pytest

from emiproc.inventories.utils import add_inventories, scale_inventory
from emiproc.profiles.temporal.profiles import HourOfYearProfile, WeeklyProfile
from emiproc.tests_utils import temporal_profiles, test_inventories


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
    tot_added = inv_added.total_emissions.fillna(0)
    # Also expand the dataframes to have the same columns and rows
    tot1 = inv1.total_emissions.reindex_like(tot_added).fillna(0)
    tot2 = inv2.total_emissions.reindex_like(tot_added).fillna(0)

    assert tot_added.equals(tot1 + tot2)


def test_cannot_add_different_grid():
    """Test that we can only add inventories on the same grid."""

    inv1 = test_inventories.inv_on_grid_serie2_bis
    inv2 = test_inventories.inv_on_grid_serie2

    pytest.raises(ValueError, add_inventories, inv1, inv2)


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
