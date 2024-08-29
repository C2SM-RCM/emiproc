import logging
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
import re
import geopandas as gpd
import pyogrio

from emiproc.grids import WGS84, EDGARGrid, RegularGrid
from emiproc.inventories import Inventory
from emiproc.utilities import SEC_PER_YR


class EDGARv8(Inventory):
    """The EDGAR inventory.

    `Emissions Database for Global Atmospheric Research <https://edgar.jrc.ec.europa.eu/>`_

    EDGAR has only grid cell sources.
    """

    grid: EDGARGrid

    def __init__(
        self,
        nc_file_pattern: PathLike,
        year: int | None = None,
    ) -> None:
        """Create a EDGAR_Inventory.

        :arg nc_file_pattern: Pattern of files, e.g "path/to/edgarfiles/v8.0_FT2021_SF6_*_PRU.0.1x0.1.nc"

        """
        super().__init__()

        nc_file_pattern = Path(nc_file_pattern)
        logger = logging.getLogger(__name__)

        list_filepaths = list(nc_file_pattern.parent.glob(nc_file_pattern.name))

        columns = {}

        file_for_grid = None

        for filepath in list_filepaths:
            ds = xr.open_dataset(filepath)
            if "fluxes" not in ds:
                logger.warning(
                    f"File {filepath} does not contain 'fluxes' variable. Skipping."
                )
                continue
            da = ds["fluxes"]
            substance = da.attrs["substance"]
            category = da.attrs["long_name"]
            file_year = int(da.attrs["year"])
            if year is None:
                year = file_year
            if file_year != year:
                logger.warning(
                    f"File {filepath} has year {file_year}, expected {year}. Skipping."
                )
                continue
            units = da.attrs["units"]
            assert units == "kg m-2 s-1", f"Units are {units}, expected kg m-2 s-1"

            columns[(category, substance)] = da.data.T.reshape(-1)

            if file_for_grid is None:
                file_for_grid = ds

        self.grid = RegularGrid.from_centers(
            ds["lon"].data, ds["lat"].data, name="EDGARv8_grid"
        )
        cell_areas = self.grid.cell_areas

        # Swtich longitude from 0/360 to -180/180
        self.gdf = gpd.GeoDataFrame(
            data={
                # Convert to kg/yr
                col: value * SEC_PER_YR * cell_areas
                for col, value in columns.items()
            },
            geometry=self.grid.gdf.geometry,
            crs=WGS84,
        )

        # Empty shaped sources
        self.gdfs = {}


class EDGAR_Inventory(Inventory):
    """The EDGAR inventory.

    `Emissions Database for Global Atmospheric Research <https://edgar.jrc.ec.europa.eu/>`_

    EDGAR has only grid cell sources.
    """

    grid: EDGARGrid

    def __init__(
        self, nc_file_pattern: PathLike, grid_shapefile: PathLike | None = None
    ) -> None:
        """Create a EDGAR_Inventory.

        The EDGAR directory that contains the original datasets must be structured as ./substance/categories/dataset,
        e.g., ./SF6/NFE/v7.0_FT2021_SF6_*_NFE.0.1x0.1.nc and ./SF6/PRU/v7.0_FT2021_SF6_*_PRU.0.1x0.1.nc

        :arg nc_file_pattern: Pattern of files, e.g "EDGAR/SF6/PRU/v7.0_FT2021_SF6_*_PRU.0.1x0.1.nc"

        """
        super().__init__()

        nc_file_pattern = Path(nc_file_pattern)

        self.name = nc_file_pattern.stem

        list_filepaths = list(nc_file_pattern.parent.glob(nc_file_pattern.name))
        list_ds = [xr.open_dataset(f) for f in list_filepaths]

        substance = list_filepaths[0].parent.parent.stem

        substances_mapping = {}
        categories = [
            re.search(r"{}_(.+?)_(.+?)\.".format(substance), filepath.name)[2]
            for filepath in list_filepaths
        ]

        for i, ds in enumerate(list_ds):
            old_varname = list(ds.data_vars.keys())[0]
            new_varname = substance.upper()
            list_ds[i] = ds.rename_vars({old_varname: new_varname})
            substances_mapping[new_varname] = new_varname.upper()

        ds = xr.merge(list_ds)

        # Swtich longitude from 0/360 to -180/180
        attributes = ds.lon.attrs
        ds = ds.assign(lon=(ds.lon + 180) % 360 - 180).sortby("lon")
        ds.lon.attrs = attributes

        self.grid = EDGARGrid(list_filepaths[0])

        if grid_shapefile is None:
            polys = self.grid.cells_as_polylist
        else:
            gdf_geometry = gpd.read_file(grid_shapefile, engine="pyogrio")
            gdf_geometry = gdf_geometry.to_crs(WGS84)
            polys = gdf_geometry.geometry.values

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
