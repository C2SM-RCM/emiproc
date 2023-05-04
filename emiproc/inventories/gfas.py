from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
import re
import geopandas as gpd
import pyogrio

from emiproc.grids import WGS84, RegularGrid, LatLonNcGrid
from emiproc.inventories import Inventory
from emiproc.utilities import SEC_PER_YR

class GFAS_Inventory(Inventory):
    """The GFAS inventory.
    
    `CAMS global biomass burning emissions based on fire radiative power  <https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-fire-emissions-gfas>`_
    
    GFAS has only grid cell sources.
    """
    grid: RegularGrid

    def __init__(self, nc_file: PathLike):
        """Create a GFAS.
        The GFAS directory contains gridded data of forest fires (co2 fluxes)
                
        """
        super().__init__()
        ds = xr.open_dataset(nc_file)
        categories = ['co2fire', 'mami']

        substances_mapping = {
            "co2fire": "CO2",
        }

        # Take the mean of the fire emissions over the time period submitted
        if 'time' in ds.keys():
            ds = ds.mean('time', keep_attrs=True)
        
        self.grid = RegularGrid(ds.longitude[0].values, 
                                ds.longitude[-1].values, 
                                ds.latitude[0].values, # Sorted from high to low
                                ds.latitude[-1].values,
                                len(ds.longitude),
                                len(ds.latitude))
                
        polys = self.grid.cells_as_polylist

        # Index in the polygon list (from the gdf) (index start at 1)
        poly_ind = np.arange(self.grid.nx * self.grid.ny)

        self.gdfs = {}
        mapping = {}
        for cat_idx, cat_name in enumerate(categories):
            for sub_in_nc, sub_emiproc in substances_mapping.items():
                tuple_idx = (cat_name, sub_emiproc)
                if tuple_idx not in mapping:
                    mapping[tuple_idx] = np.zeros(len(polys))

                # Add all the area sources corresponding to that category
                np.add.at(mapping[tuple_idx], poly_ind, ds[sub_in_nc].data.T.flatten())

        self.gdf = gpd.GeoDataFrame(
            mapping,
            geometry=polys,
            crs=WGS84,
        )

        self.cell_areas = self.grid.cell_areas

        # -- Convert to kg/yr
        self.gdf[list(mapping)] *= SEC_PER_YR * self.cell_areas[:, np.newaxis]  