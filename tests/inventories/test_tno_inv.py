"""Test file for tno inventory"""

from pathlib import Path

import pytest

import emiproc
from emiproc.inventories.tno import TNO_Inventory
from emiproc.inventories.utils import group_categories
from emiproc.profiles.vertical_profiles import check_valid_vertical_profile
from emiproc.utilities import get_country_mask

# TODO change the pth of that if you want to test it
tno_path = emiproc.FILES_DIR / "TNO_6x6_GHGco_v4_0/TNO_GHGco_v4_0_year2018.nc"


tno_template = emiproc.FILES_DIR / "test/tno/tno_test_minimal.nc"


def test_loading_template():
    """Test loading the template"""
    inv_tno = TNO_Inventory(tno_template)


# make this test only if the tno inventory is available
# otherwise skip it
@pytest.mark.slow
def test_loading_and_grouping():
    # Test vertical profiles on the TNO inventory
    if not tno_path.exists():
        raise FileNotFoundError(
            f"File {tno_path} not found, please add it to {tno_path}"
        )

    inv_tno = TNO_Inventory(tno_path)
    # Check the vertical profiles
    check_valid_vertical_profile(inv_tno.v_profiles)

    groupped_tno = group_categories(inv_tno, {"all": inv_tno.categories})
    check_valid_vertical_profile(groupped_tno.v_profiles)
    # test that we have the same number of point sources in both inventories
    # the number of point source is the number of rows of each of the gdfs

    assert sum([len(gdf) for gdf in inv_tno.gdfs.values()]) == len(
        groupped_tno.gdfs["all"]
    )


def test_country_ratios():
    inv_tno = TNO_Inventory(tno_template)
    # test getting the country mask of the test grid of tno
    ratios = get_country_mask(inv_tno.grid, return_fractions=True)


def test_with_t_profiles():
    inv_tno = TNO_Inventory(
        tno_template, profiles_dir=emiproc.FILES_DIR / "test/tno/profiles_cat"
    )
    assert "category" in inv_tno.t_profiles_indexes.dims
    assert "substance" not in inv_tno.t_profiles_indexes.dims


def test_with_t_profiles_catsub():
    inv_tno = TNO_Inventory(
        tno_template, profiles_dir=emiproc.FILES_DIR / "test/tno/profiles_catsub"
    )
    assert "category" in inv_tno.t_profiles_indexes.dims
    assert "substance" in inv_tno.t_profiles_indexes.dims


def test_with_t_profiles_misssing():
    inv_tno = TNO_Inventory(
        tno_template, profiles_dir=emiproc.FILES_DIR / "test/tno/profiles_missing_cat"
    )
    assert "category" in inv_tno.t_profiles_indexes.dims
    assert "substance" not in inv_tno.t_profiles_indexes.dims


def test_with_t_profiles_catsub_misssing():
    inv_tno = TNO_Inventory(
        tno_template,
        profiles_dir=emiproc.FILES_DIR / "test/tno/profiles_catsub_missing",
    )
    assert "category" in inv_tno.t_profiles_indexes.dims
    assert "substance" in inv_tno.t_profiles_indexes.dims
    missing_cat = "F2"
    other_cats = ["A", "B", "F1"]
    assert all(inv_tno.t_profiles_indexes.sel(category=missing_cat).values == -1)
    assert all(
        [cat in inv_tno.t_profiles_indexes["category"].values for cat in other_cats]
    )


if __name__ == "__main__":
    pytest.main([__file__])
