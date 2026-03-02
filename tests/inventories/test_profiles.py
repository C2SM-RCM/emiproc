from emiproc.tests_utils import temporal_profiles, test_inventories



def test_set_profiles():

    inv = test_inventories.inv.copy()

    inv.set_profiles(
        temporal_profiles.three_composite_profiles,
        indexes=temporal_profiles.indexes_inv_catsubcell,
    )

    assert inv.t_profiles_indexes is not None 


def test_set_profiles_with_type():

    inv = test_inventories.inv_with_pnt_sources.copy()

    inv.set_profiles(
        temporal_profiles.three_composite_profiles,
        indexes=temporal_profiles.indexes_with_gridded_shapped,
    )
    assert inv.t_profiles_indexes is not None