import pytest
from emiproc.plots import plot_inventory
import matplotlib
from emiproc.tests_utils.test_inventories import (
    inv,
    inv_on_grid_serie2,
    inv_with_pnt_sources,
)


@pytest.mark.parametrize("inventory", [inv, inv_on_grid_serie2, inv_with_pnt_sources])
def test_plot_inventory(inventory):
    # Prevent plots from showing during tests
    matplotlib.use("Agg")

    plot_inventory(inventory)
