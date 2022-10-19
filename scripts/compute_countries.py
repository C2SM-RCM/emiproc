#%%
from time import time
from warnings import warn
from shapely.geometry import Polygon, MultiPolygon
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import numpy as np
import pandas as pd

from emiproc.grids import ICONGrid
import emiproc
from emiproc.utilities import ProgressIndicator, grid_polygon_intersects, compute_country_mask
from emiproc.country_code import country_codes
from emiproc.grids import WGS84


#%%
shpfilename = shpreader.natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)
# %%
countries = list(shpreader.Reader(shpfilename).records())

country_bounds = [country.bounds for country in countries]

#


# %%
# TODO 
grid = ICONGrid(r"C:\Users\coli\Documents\ZH-CH-emission\icon_europe_DOM01.nc")

#%%
mask = compute_country_mask(grid, '10m', 1)

#%%
gdf = grid.gdf.assign(**{'country_mask': mask.reshape(-1)})
gdf.iloc[::37].explore('country_mask')
# %%