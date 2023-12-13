"""Little script to show how to compute country masks for a grid."""
# %%
from emiproc.utilities import get_natural_earth


from emiproc.grids import ICONGrid
from emiproc.utilities import (
    get_country_mask,
)
from emiproc.country_code import country_codes
from emiproc.grids import WGS84


# %%
countries_gdf = get_natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)


# %%
grid = ICONGrid("/path/to/icon_grid.nc")

# %%
# Maks is an array of shape (nlat, nlon) of dtype "U<3" with the country code on each grid cell.
mask = get_country_mask(grid, "10m")

# %%
gdf = grid.gdf.assign(**{"country_mask": mask.reshape(-1)})
gdf.iloc[::37].explore("country_mask")
# %%
