import numpy as np

from emiproc.utilities import get_timezone_mask
from emiproc.grids import RegularGrid


def test_create_simple_mask():
    grid = RegularGrid(
        xmin=47.5,
        xmax=58.5,
        ymin=7.5,
        ymax=12.5,
        nx=13,
        ny=10,
    )
    arr = get_timezone_mask(grid)
    # Check that the array is a numpy array of strings
    assert isinstance(arr, np.ndarray)
    # We don't know the lengths of the strings
    assert arr.dtype.kind == "U"

    # Check the sized is the same as the grid
    assert arr.shape == grid.shape
