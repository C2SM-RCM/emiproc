from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
import re
import geopandas as gpd

from emiproc.grids import WGS84, EDGARGrid
from emiproc.inventories import Inventory


class EDGAR_Inventory(Inventory):
    """The EDGAR inventory.
    
    `Emissions Database for Global Atmospheric Research <https://edgar.jrc.ec.europa.eu/>`_
    
    EDGAR has only grid cell sources.
    """
    grid: EDGARGrid

    def __init__(self, nc_file_pattern: PathLike) -> None:
        """Create a EDGAR_Inventory.
        
        :arg nc_file_pattern: Pattern of files, e.g "EDGAR/SF6/PRU/v7.0_FT2021_SF6_*_PRU.0.1x0.1.nc"
        
        """
        super().__init__()

        nc_file_pattern = Path(nc_file_pattern)

        self.name = nc_file_pattern.stem

        list_filepaths = list(nc_file_pattern.parent.glob(nc_file_pattern.name))
        list_ds = [xr.open_dataset(f) for f in list_filepaths]

        substance = list_filepaths[0].parent.parent.stem

        substances_mapping = {}
        categories = [re.search(r'{}_(.+?)_(.+?)\.'.format(substance), filepath.name)[2] for filepath in list_filepaths] 

        for i, ds in enumerate(list_ds):
            old_varname = list(ds.data_vars.keys())[0]
            new_varname = (substance+ '_' + categories[i]).upper()
            list_ds[i] = ds.rename_vars({old_varname: new_varname})
            substances_mapping[new_varname] = new_varname.upper()

        ds = xr.merge(list_ds)

        self.grid = EDGARGrid(list_filepaths[0])

        polys = self.grid.cells_as_polylist()

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

        self.cell_areas = self.grid.cell_areas()
