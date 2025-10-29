
import pytest
from emiproc.inventories import Inventory
from emiproc.inventories.utils import clip_box
from emiproc.profiles.temporal.profiles import HourOfYearProfile, WeeklyProfile
from emiproc.tests_utils import temporal_profiles, test_inventories

@pytest.mark.parametrize(
    "inv",
    [
        test_inventories.inv_on_grid_serie2,
        test_inventories.inv_with_pnt_sources,
        test_inventories.inv,
    ],
)
def test_clip_box(inv):
    """Test the clipping of an inventory with a bounding box."""


    clipped_inv = clip_box(inv, minx=0, miny=0, maxx=10, maxy=10)
    assert isinstance(clipped_inv, Inventory)

def test_clip_box_bad_box():
    """Test the clipping of an inventory with a bounding box."""

    inv = test_inventories.inv_with_pnt_sources
    with pytest.raises(ValueError, match="Invalid bounding box coordinates"):
        clipped_inv = clip_box(inv, minx=20, miny=0, maxx=10, maxy=10)

