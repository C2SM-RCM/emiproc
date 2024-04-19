import numpy as np
import geopandas as gpd
import xarray as xr

from shapely.geometry import Polygon, Point

from emiproc.inventories import Inventory

from emiproc.profiles.vertical_profiles import (
    VerticalProfile,
    VerticalProfiles,
)


def get_random_profiles(
    num: int, heights: list[int] = [14, 50, 89]
) -> VerticalProfiles:
    """Get random profiles for testing."""
    ratios = np.random.rand(num, len(heights))
    return VerticalProfiles(ratios / ratios.sum(axis=1)[:, None], height=heights)


# Create test profiles
list_of_three = [
    VerticalProfile(np.array([0, 0.3, 0.7, 0.0]), np.array([15, 30, 60, 100])),
    VerticalProfile(
        np.array([0.1, 0.3, 0.5, 0.0, 0.1]), np.array([10, 30, 40, 65, 150])
    ),
    VerticalProfile(np.array([1]), np.array([20])),
]
VerticalProfiles_instance = VerticalProfiles(
    np.array(
        [
            [0.0, 0.3, 0.7, 0.0],
            [0.1, 0.2, 0.7, 0.0],
            [0.0, 0.3, 0.2, 0.5],
        ]
    ),
    np.array([15, 30, 60, 100]),
)

# Create a test geodataframe
gdf = gpd.GeoDataFrame(
    {
        ("test_cat", "CO2"): [i for i in range(4)],
        ("test_cat", "CH4"): [i + 3 for i in range(4)],
        # ("test_cat", "NH3"): [2 * i for i in range(4)],
        ("test_cat2", "CO2"): [i + 1 for i in range(4)],
        ("test_cat2", "CH4"): [i + 1 for i in range(4)],
        # ("test_cat2", "NH3"): [i + 1 for i in range(4)],
    },
    geometry=gpd.GeoSeries(
        [
            Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
            Polygon(((0, 1), (0, 2), (1, 2), (1, 1))),
            Polygon(((1, 0), (1, 1), (2, 1), (2, 0))),
            Polygon(((1, 1), (1, 2), (2, 2), (2, 1))),
        ]
    ),
)

# Corresponding profiles integer array
# -1 is when the profile is not defined
corresponding_vertical_profiles = xr.DataArray(
    [
        [[0, -1, 0, 2], [0, 2, 0, 0], [0, 0, 0, 0]],
        [[0, -1, -1, 1], [0, 0, -1, 0], [0, 1, 0, 0]],
    ],
    dims=("category", "substance", "cell"),
    coords={
        "category": ["test_cat", "test_cat2"],
        "substance": (substances := ["CH4", "CO2", "NH3"]),
        "cell": (cells := [0, 1, 2, 3]),
    },
)
corresponding_2d_profiles = xr.DataArray(
    [
        [2, 1, 0],
        [-1, 1, 0],
    ],
    dims=("category", "substance"),
    coords={
        "category": ["test_cat", "test_cat2"],
        "substance": (substances := ["CH4", "CO2", "NH3"]),
    },
)
single_dim_profile_indexes = xr.DataArray(
    [0, 1, -1],
    dims="category",
    coords={
        "category": ["test_cat", "test_cat2", "test_cat3"],
    },
)
single_dim_weights = xr.DataArray(
    [0.5, 0.2, 0.0],
    dims="category",
    coords={
        "category": ["test_cat", "test_cat2", "test_cat3"],
    },
)


inv = Inventory.from_gdf(
    gdf,
    # Add some point sources
    gdfs={
        "test_cat": gpd.GeoDataFrame(
            {
                "CO2": [1, 2, 3],
                "__v_profile__": [-1, 2, 1],
            },
            geometry=[Point(0.75, 0.75), Point(0.25, 0.25), Point(1.2, 1)],
        ),
        "test_cat2": gpd.GeoDataFrame(
            {
                "CO2": [1.2, 2.7, 8],
                "CH4": [4, 2, 8],
                "__v_profile__": [1, 2, -1],
            },
            geometry=[Point(0.65, 0.75), Point(1.1, 0.8), Point(1.2, 1)],
        ),
        "test_cat3": gpd.GeoDataFrame(
            {"CO2": [1, 2]},
            geometry=[Point(0.65, 0.75), Point(1.1, 0.8)],
        ),
    },
)
inv_groups_dict = {"new_cat": ["test_cat"], "new_cat2": ["test_cat2", "test_cat3"]}
inv_groups_subs_dict = {"co2": ["CO2"], "others": ["CH4", "NH3"]}

inv.v_profiles = VerticalProfiles_instance
inv.v_profiles_indexes = corresponding_vertical_profiles
