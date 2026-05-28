from warnings import catch_warnings, simplefilter

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pytest
from shapely import LineString
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from emiproc.inventories import Inventory
from emiproc.inventories.utils import crop_with_shape
from emiproc.plots import plot_inventory
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


def test_plot_inventory_after_crop_with_empty_line_gdf_raises():
    shapes = {
        "cat": LineString([(0, 0), (1, 1)]),
        "cat_outside": LineString([(2, 2), (3, 3)]),
        "cat_cross": LineString([(-2, -2), (3, 3)]),
    }

    inv = Inventory.from_gdf(
        gdfs={
            cat: gpd.GeoDataFrame({"CO2": [1.0]}, geometry=[shape])
            for cat, shape in shapes.items()
        },
    )

    inv_cropped = crop_with_shape(
        inv,
        shape=Polygon(((-1, -1), (-1, 1), (1, 1), (1, -1))),
        modify_grid=True,
    )
    inv_cropped_grid_kept = crop_with_shape(
        inv,
        shape=Polygon(((-1, -1), (-1, 1), (1, 1), (1, -1))),
        modify_grid=False,
    )

    for cat in shapes.keys():
        assert len(inv.gdfs[cat]) == 1
        assert len(inv_cropped_grid_kept.gdfs[cat]) == 1
        assert inv_cropped_grid_kept.gdfs[cat].is_valid.all()
        assert len(inv_cropped.gdfs[cat]) == (1 if cat != "cat_outside" else 0)
        assert inv_cropped.gdfs[cat].is_valid.all()

    total_emissions = inv_cropped.total_emissions
    assert total_emissions.loc["CO2", "cat_outside"] == 0.0
    assert total_emissions.loc["CO2", "cat"] == 1.0
    assert total_emissions.loc["CO2", "cat_cross"] == 2.0 / 5.0
