import geopandas as gpd
import numpy as np
import pandas as pd

from emiproc.tests_utils.test_grids import regular_grid_africa
from emiproc.inventories import Inventory
from emiproc.utilities import get_country_mask

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
