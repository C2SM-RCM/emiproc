import warnings

import pytest

from emiproc.exports.geopackage import export_to_geopackage
from emiproc.inventories import Inventory
from emiproc.tests_utils.african_case import african_inv
from emiproc.tests_utils.test_inventories import (
    inv_on_grid_serie2,
    inv_with_pnt_sources,
)


@pytest.fixture(
    params=[african_inv, inv_with_pnt_sources, inv_on_grid_serie2],
    ids=["african_inv", "inv_with_pnt_sources", "inv_on_grid_serie2"],
)
def inv(request):
    return request.param


def test_export_to_geopackage(tmp_path, inv: Inventory):
    import geopandas as gpd

    output_path = tmp_path / f"test_inventory_export_{inv}.gpkg"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        export_to_geopackage(inv, output_path)

        # Test we can read it later
        gdf = gpd.read_file(output_path)
