from emiproc.tests_utils import temporal_profiles, test_inventories
from emiproc.inventories.utils import group_categories


def test_group_profiles_with_time_profiles():

    inv = test_inventories.inv.copy()

    inv.set_profiles(
        temporal_profiles.three_composite_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )

    groupped = group_categories(inv, {"all": inv.categories})

    assert groupped.categories == ["all"]

    assert groupped.t_profiles_indexes["category"].values == ["all"]
