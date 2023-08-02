"""This should test that a shape containing a hole is properly working.

One test should be on a test data and one on a country like south africa.
"""
#%%

from shapely.geometry import Polygon



# New version without cartopy 
from emiproc.utilities import get_natural_earth

countries = get_natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)

countries.set_index('SOVEREIGNT', inplace=True)


sa_geometry = countries.loc['South Africa'].geometry

# The country (no islands), this one has holes
main_geom = sa_geometry.geoms[0]
#%%
square_around = Polygon.from_bounds(*main_geom.bounds)
# %% play a bit with the shape to try to understand it
main_geom.intersection(square_around.difference(main_geom))



