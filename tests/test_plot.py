from warnings import catch_warnings, simplefilter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from emiproc.plots import plot_inventory
from emiproc.profiles.temporal.profiles import MounthsProfile
from emiproc.tests_utils.african_case import african_inv_regular_grid
from emiproc.tests_utils.test_inventories import (
    inv,
    inv_on_grid_serie2,
    inv_with_pnt_sources,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def plot_test_context():
    with catch_warnings():
        simplefilter("ignore", category=UserWarning)
        yield
    plt.close("all")


@pytest.fixture(
    params=[
        pytest.param(inv, id="inv"),
        pytest.param(inv_on_grid_serie2, id="inv_on_grid_serie2"),
        pytest.param(inv_with_pnt_sources, id="inv_with_pnt_sources"),
        pytest.param(african_inv_regular_grid, id="african_inv_regular_grid"),
    ]
)
def inventory(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param({}, id="default"),
        pytest.param({"bare_plot": True}, id="bare_plot"),
        pytest.param({"total_only": True}, id="total_only"),
        pytest.param({"add_country_borders": True}, id="add_country_borders"),
    ]
)
def plot_inventory_kwargs(request):
    return request.param


def test_plot_inventory(inventory, plot_inventory_kwargs):

    run_plot = lambda: plot_inventory(inventory, **plot_inventory_kwargs)

    add_country_borders = plot_inventory_kwargs.get("add_country_borders", False)
    if add_country_borders and inventory.grid.crs is None:
        with pytest.raises(
            ValueError, match="Grid has no CRS, cannot add country borders"
        ):
            run_plot()
    else:
        run_plot()


def test_plot_inventory_with_monthly_temporal_profile(monkeypatch):
    monthly_inv = inv.copy()
    monthly_inv.year = 2020

    profile_indexes = xr.DataArray(
        data=np.zeros(
            (len(monthly_inv.categories), len(monthly_inv.substances)), dtype=int
        ),
        dims=["category", "substance"],
        coords={
            "category": monthly_inv.categories,
            "substance": monthly_inv.substances,
        },
    )
    monthly_inv.set_profiles([[MounthsProfile()]], indexes=profile_indexes)

    step_x_dtypes = []
    original_step = matplotlib.axes.Axes.step

    def record_step(self, x, y, *args, **kwargs):
        step_x_dtypes.append(np.asarray(x).dtype)
        return original_step(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "step", record_step)
    plot_inventory(monthly_inv, total_only=True)
    assert step_x_dtypes
    assert all(dtype.kind == "M" for dtype in step_x_dtypes)
