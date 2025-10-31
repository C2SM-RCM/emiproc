import pytest
from emiproc.grids import RegularGrid
from emiproc.inventories import Inventory
from emiproc.inventories.utils import clip_box
from emiproc.profiles.temporal.profiles import HourOfYearProfile, WeeklyProfile
from emiproc.tests_utils import temporal_profiles, test_inventories, african_case


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


@pytest.mark.parametrize(
    "coords",
    [
        # Directly on a grid cell edge
        dict(minx=-16, miny=15, maxx=-14, maxy=25),
        dict(minx=-25, miny=10, maxx=-5, maxy=30),
        dict(minx=-16.0001, miny=4.9999, maxx=-8.08172, maxy=27.08172),
    ],
)
def test_clip_regular_grid(coords):
    """Test the clipping of an inventory with a bounding box on a regular grid.

    xmin=-20.5,
    xmax=-9.25,
    ymin=4.65,
    ymax=20.9,
    """

    inv = african_case.african_inv_regular_grid
    clipped_inv = clip_box(inv, **coords)

    assert isinstance(clipped_inv, Inventory)
    grid_in: RegularGrid = inv.grid
    grid_out: RegularGrid = clipped_inv.grid

    assert grid_out.nx <= grid_in.nx
    assert grid_out.ny <= grid_in.ny

    print(grid_out.lon_bounds)
    print(grid_out.lat_bounds)

    assert grid_out.lat_bounds[0] in grid_in.lat_bounds
    assert grid_out.lat_bounds[-1] in grid_in.lat_bounds
    assert grid_out.lon_bounds[0] in grid_in.lon_bounds
    assert grid_out.lon_bounds[-1] in grid_in.lon_bounds


def test_clip_box_bad_box():
    """Test the clipping of an inventory with a bounding box."""

    inv = test_inventories.inv_with_pnt_sources
    with pytest.raises(ValueError, match="Invalid bounding box coordinates"):
        clipped_inv = clip_box(inv, minx=20, miny=0, maxx=10, maxy=10)
