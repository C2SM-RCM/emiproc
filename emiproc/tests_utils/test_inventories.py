"""Some inventories that can be used for test purposes."""

import geopandas as gpd
from shapely.geometry import Point, Polygon
from emiproc.inventories import Inventory


serie = gpd.GeoSeries(
    [
        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
        Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
        Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
        Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
        Polygon(((2, 1), (2, 2), (3, 2), (3, 1))),
    ]
)


inv = Inventory.from_gdf(
    gpd.GeoDataFrame(
        {
            ("adf", "CH4"): [i + 3 for i in range(len(serie))],
            ("adf", "CO2"): [i for i in range(len(serie))],
            ("liku", "CO2"): [i for i in range(len(serie))],
            ("test", "NH3"): [i + 1 for i in range(len(serie))],
        },
        geometry=serie,
    )
)

# An inventory with point sources
inv_with_pnt_sources = inv.copy()
inv_with_pnt_sources.gdfs["blek"] = gpd.GeoDataFrame(
    {
        "CO2": [1, 2, 3],
    },
    geometry=[Point(0.75, 0.75), Point(0.25, 0.25), Point(1.2, 1)],
)
inv_with_pnt_sources.gdfs["liku"] = gpd.GeoDataFrame(
    {
        "CO2": [1, 2],
    },
    geometry=[Point(0.65, 0.75), Point(1.1, 0.8)],
)
inv_with_pnt_sources.gdfs["other"] = gpd.GeoDataFrame(
    {
        "AITS": [1, 2],
    },
    geometry=[Point(0.65, 0.75), Point(1.1, 0.8)],
)