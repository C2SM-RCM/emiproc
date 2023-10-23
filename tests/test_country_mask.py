import numpy as np

from emiproc.utilities import get_country_mask
from emiproc.grids import RegularGrid


def test_create_simple_mask():
    arr = get_country_mask(
        RegularGrid(
            xmin=47.5,
            xmax=58.5,
            ymin=7.5,
            ymax=12.5,
            nx=10,
            ny=10,
        ),
        resolution="110m",
    )
    # check that there are some countries in there
    # Not just -1 values
    assert len(np.unique(arr)) > 1
