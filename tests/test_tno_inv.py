"""Test file for tno inventory"""
# %% Imports
from emiproc.inventories.tno import TNO_Inventory
from emiproc.inventories.utils import group_categories
from emiproc.profiles.vertical_profiles import check_valid_vertical_profile
from pathlib import Path
import pytest

# TODO change the pth of that if you want to test it
tno_path = Path(
    r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0\TNO_GHGco_v4_0_year2018.nc"
)


# make this test only if the tno inventory is available
# otherwise skip it
@pytest.mark.slow
@pytest.mark.skipif(not tno_path.exists(), reason="TNO inventory not found")
def test_loading_and_grouping():
    # %% Test vertical profiles on the TNO inventory
    if not tno_path.exists():
        raise ValueError("The path to the TNO inventory is not correct")

    inv_tno = TNO_Inventory(tno_path)
    # Check the vertical profiles
    check_valid_vertical_profile(inv_tno.v_profiles)

    inv_tno
    # %%
    groupped_tno = group_categories(inv_tno, {"all": inv_tno.categories})
    check_valid_vertical_profile(groupped_tno.v_profiles)
    # test that we have the same number of point sources in both inventories
    # the number of point source is the number of rows of each of the gdfs

    assert sum([len(gdf) for gdf in inv_tno.gdfs.values()]) == len(
        groupped_tno.gdfs["all"]
    )
    # %%
