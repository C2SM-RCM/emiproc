import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.tests_utils.test_grids import regular_grid_africa
from emiproc.inventories import Inventory
from emiproc.utilities import get_country_mask
from emiproc.tests_utils.temporal_profiles import (
    get_random_profiles,
    three_composite_profiles,
)

gdf = gpd.GeoDataFrame(
    data={
        ("test", "CO2"): np.random.rand(len(regular_grid_africa.gdf)),
        ("test", "NH3"): np.random.rand(len(regular_grid_africa.gdf)),
        ("test2", "CO2"): np.random.rand(len(regular_grid_africa.gdf)),
    },
    geometry=regular_grid_africa.gdf.geometry,
)


african_inv = Inventory.from_gdf(gdf)

# Now we make an invenotry but only the land cells have emissons (the ones that have a country in them)
da_african_ratios = get_country_mask(regular_grid_africa, return_fractions=True)
mask_cell_no_country = (da_african_ratios.sum(dim="country") == 0).to_numpy()

gdf_land_only = gdf.copy(deep=True)
# Set 0 emissions in the ocean
gdf_land_only.loc[
    mask_cell_no_country, [col for col in gdf.columns if gdf[col].dtype == float]
] = 0.0
african_inv_emissions_only_land = Inventory.from_gdf(gdf_land_only)

# For the african test set
african_countries_test_set = ["SEN", "MLI", "MRT", "GIN", "GNB", "LBR", "SLE", "GMB"]
indexes_african_2d = xr.DataArray(
    data=np.arange(len(african_countries_test_set) * 3).reshape(
        (len(african_countries_test_set), 3)
    ),
    dims=["country", "category"],
    coords={
        "country": african_countries_test_set,
        "category": ["liku", "blek", "test"],
    },
)
indexes_african_simple = xr.DataArray(
    data=np.arange(len(african_countries_test_set)),
    dims=["country"],
    coords={"country": african_countries_test_set},
)


african_inv_with_tprofiles = african_inv.copy()
african_inv_with_tprofiles_2d = african_inv.copy()

african_inv_with_tprofiles.set_profiles(
    get_random_profiles(8), indexes=indexes_african_simple
)
african_inv_with_tprofiles_2d.set_profiles(
    get_random_profiles(24), indexes=indexes_african_2d
)
