from shapely.geometry import Polygon
from emiproc.regrid import get_weights_mapping

from emiproc.tests_utils.test_inventories import inv, inv_with_pnt_sources
from emiproc.tests_utils import WEIGHTS_DIR

polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
def test_get_weights_mapping():
    get_weights_mapping(
        WEIGHTS_DIR/ 'test_get_weights_mapping', inv.gdf, polys
        )

def test_no_weights():
    get_weights_mapping(
        None, inv.gdf, polys
        )