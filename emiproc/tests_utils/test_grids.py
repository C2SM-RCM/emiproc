import geopandas as gpd

from shapely.geometry import Point, Polygon

from emiproc.grids import RegularGrid, GeoPandasGrid

# Regular grid for testing
# Note that this grid is big enough to include the inventories from the
# test_utils/test_inventories.py module
regular_grid = RegularGrid(
    xmin=-1, xmax=5, ymin=-2, ymax=3, nx=10, ny=15, name="Test Regular Grid"
)

gpd_grid = GeoPandasGrid(
    gpd.GeoDataFrame(
        geometry=[
            Polygon(((0.5, 0.5), (0.5, 1.5), (1.5, 1.5))),
            Polygon(((0.5, 0.5), (1.5, 0.5), (1.5, 1.5))),
            Polygon(((2.5, 0.5), (1.5, 1.5), (1.5, 0.5))),
            Polygon(((2.5, 0.5), (2.5, 1.5), (1.5, 1.5))),
        ]
    )
)
