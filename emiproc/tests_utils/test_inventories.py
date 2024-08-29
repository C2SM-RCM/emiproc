"""Some inventories that can be used for test purposes."""

import geopandas as gpd
from shapely.geometry import Point, Polygon

from emiproc.inventories import Inventory

from emiproc.tests_utils.test_grids import (
    basic_serie,
    basic_serie_2,
    basic_serie_of_size_2,
)


serie = basic_serie

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
        "CO2": [1.0, 2.0, 3.0],
    },
    geometry=[Point(0.75, 0.75), Point(0.25, 0.25), Point(1.2, 1)],
)
inv_with_pnt_sources.gdfs["liku"] = gpd.GeoDataFrame(
    {
        "CO2": [1.0, 2.0],
    },
    geometry=[Point(0.65, 0.75), Point(1.1, 0.8)],
)
inv_with_pnt_sources.gdfs["other"] = gpd.GeoDataFrame(
    {
        "AITS": [1.0, 2.0],
    },
    geometry=[Point(0.65, 0.75), Point(1.1, 0.8)],
)

inv_only_one_gdfs = Inventory.from_gdf(
    gdfs={
        "adf": gpd.GeoDataFrame(
            {
                "CO2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            geometry=[
                # corner point
                Point(0.75, 0.75),
                # Outside point
                Point(0.5, 0.4),
                # Inside point
                Point(1.2, 1),
                # 1/8 inside polygon
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                # 1/4 inside polygon
                Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
                # outside polygon
                Polygon(((3, 3), (3, 4), (4, 4), (4, 3))),
                # fully inside polygon
                Polygon(((1, 0.5), (1.5, 0.5), (1.5, 1), (1, 1))),
            ],
        )
    }
)

inv_with_gdfs_bad_indexes = Inventory.from_gdf(
    gdfs={
        "adf": gpd.GeoDataFrame(
            {"CO2": [1.0, 2.0, 3.0]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            index=[0, 1, 100000000],
        )
    }
)

inv_on_grid_serie2 = Inventory.from_gdf(
    gpd.GeoDataFrame(
        {
            ("adf", "CH4"): [i + 3 for i in range(len(basic_serie_2))],
            ("adf", "CO2"): [i for i in range(len(basic_serie_2))],
            ("liku", "CO2"): [i for i in range(len(basic_serie_2))],
            ("test", "NH3"): [i + 1 for i in range(len(basic_serie_2))],
        },
        geometry=basic_serie_2,
    )
)

inv_on_grid_serie2_bis = Inventory.from_gdf(
    gpd.GeoDataFrame(
        {
            ("adf", "CH4"): [i + 3 for i in range(len(basic_serie_of_size_2))],
            ("adf", "CO2"): [i for i in range(len(basic_serie_of_size_2))],
            ("liku", "CO2"): [i for i in range(len(basic_serie_of_size_2))],
            ("test", "NH3"): [i + 1 for i in range(len(basic_serie_of_size_2))],
        },
        geometry=basic_serie_of_size_2,
    )
)
