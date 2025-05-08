import geopandas as gpd

from shapely.geometry import Point, Polygon

from emiproc.grids import HexGrid, RegularGrid, GeoPandasGrid

# Regular grid for testing
# Note that this grid is big enough to include the inventories from the
# test_utils/test_inventories.py module
regular_grid = RegularGrid(
    xmin=-1, xmax=5, ymin=-2, ymax=3, nx=10, ny=15, name="Test Regular Grid"
)

hex_grid = HexGrid(xmin=-1, xmax=5, ymin=-2, ymax=3, nx=10, ny=15, name="Test Hex Grid")

basic_serie = gpd.GeoSeries(
    [
        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
        Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
        Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
        Polygon(((2, 1), (2, 2), (3, 2), (3, 1))),
    ]
)

# Another basic serie, on which it is intersting to remap the first serie
basic_serie_2 = gpd.GeoSeries(
    [
        Polygon(((0.75, 0.5), (0.75, 1.5), (1.75, 1.5), (1.75, 0.5))),
        Polygon(((1.75, 0.5), (1.75, 1.5), (2.75, 1.5), (2.75, 0.5))),
    ]
)
basic_grid = GeoPandasGrid(gpd.GeoDataFrame(geometry=basic_serie))
basic_grid_2 = GeoPandasGrid(gpd.GeoDataFrame(geometry=basic_serie_2))

basic_serie_of_size_2 = gpd.GeoSeries(
    [
        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
    ]
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

# A regular grid over some african countries
# It makes sure some grid cells are in one country only
# some countries are too small for country mask (Gambia (GMB))
# Some cells are ocean
regular_grid_africa = RegularGrid(
    # 19°24'03.0"N 20°30'34.3"W
    # 4°39'55.8"N 9°14'11.6"W
    xmin=-20.5,
    xmax=-9.25,
    ymin=4.65,
    ymax=20.9,
    nx=10,
    ny=10,
)
