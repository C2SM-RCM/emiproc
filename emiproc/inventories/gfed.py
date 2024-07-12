from os import PathLike
from pathlib import Path
from emiproc.grids import RegularGrid, Grid
from datetime import date, datetime
from emiproc.profiles.temporal_profiles import (
    CompositeTemporalProfiles,
    DayOfYearProfile,
)

import numpy as np
import xarray as xr
import geopandas as gpd

from emiproc.inventories import Inventory


class GFED_Grid(RegularGrid):
    """Grid for the GFED inventory.


    The hdf5 file contains all the information about the grid coordinates.
    It is a regular grid.


    Not also that the grid cell area is provided in the file,
    but we do not use it here.
    Also the values are not exactly the same (around 10% errors).
    Geopandas calculated areas are a bit larger it seems.

    .. note::

        The profiles implemented are only the day of year profile.
        If you want hour of day, you will have to implement it.

    """

    def __init__(self, gfed_filepath: PathLike):

        gfed_filepath = Path(gfed_filepath)
        ds = xr.open_dataset(gfed_filepath)

        # Get the lon lat coordinates
        # This assumes (but checks) that the grid is regular
        unique_lons = np.unique(ds["lon"].values, axis=0)
        unique_lats = np.unique(ds["lat"].values.T, axis=0)
        assert len(unique_lons) == 1
        assert len(unique_lats) == 1

        self.lon_range = unique_lons[0]
        self.lat_range = unique_lats[0]

        # Get the grid cell size (here also ensure that the grid is regular)
        unique_dx = np.unique(np.diff(self.lon_range))
        unique_dy = np.unique(np.diff(self.lat_range))

        assert len(unique_dx) == 1
        assert len(unique_dy) == 1

        self.dx = abs(unique_dx[0])
        self.dy = abs(unique_dy[0])

        self.nx = len(self.lon_range)
        self.ny = len(self.lat_range)

        # Bypass the RegularGrid __init__ method because we already have the grid coordinates
        Grid.__init__(self, gfed_filepath.stem)


class GFED4_Inventory(Inventory):
    """Global Fire Emissions Database.

    Global inventory based on satellite data, burned areas, fuel consumption.

    https://www.globalfiredata.org/

    This uses the area inside the GFED file to calculate the total emissions, but
    we found out that the area is not exactly the same as the one calculated by geopandas.
    See :py:class:`GFED_Grid` for the grid information.

    The data set contains two variables:
        * C: Carbon emissions
        * DM: Dry matter emissions

    .. note:: This inventory applies only for GFED4 .
        GFED5 has changed the format and is not supported by this class.
    """

    def __init__(self, gfed_filepath: PathLike, year: int):

        super().__init__()

        self.gfed_filepath = Path(gfed_filepath)
        gfed_file = self.gfed_filepath
        self.grid = GFED_Grid(gfed_filepath)

        self.year = year

        # Units of C var: g C / m^2 / month
        # Units of DM var: kg DM / m^2 / month

        dss = []
        for mounth in range(1, 13):
            ds = xr.open_dataset(self.gfed_filepath, group=f"/emissions/{mounth:02}")
            # Rename the phony_dims to lat and lon
            phony_dims = [dim for dim in ds.dims if dim.startswith("phony_dim")]
            dss.append(
                ds.rename({phony_dims[0]: "lat", phony_dims[1]: "lon"}).expand_dims(
                    mounth=[mounth]
                )
            )
        ds = xr.concat(dss, dim="mounth")
        ds_yearly = ds.sum(dim="mounth")
        # Convert the C units to kg
        ds_yearly["C"] *= 1e-3

        # Get the grid cell areas
        grid_areas = xr.open_dataset(self.gfed_filepath, group="/ancill/")[
            "grid_cell_area"
        ]
        phony_dims = [dim for dim in grid_areas.dims if dim.startswith("phony_dim")]
        grid_areas = grid_areas.rename({phony_dims[0]: "lat", phony_dims[1]: "lon"})

        # Scale with the grid cell area to get kg / year / cell
        ds_yearly_per_cell = ds_yearly * grid_areas

        columns = {}

        for substance in ["C", "DM"]:
            columns[("GFED", substance)] = ds_yearly_per_cell[
                substance
            ].values.T.flatten()

        self.gdf = gpd.GeoDataFrame(columns, geometry=self.grid.gdf.geometry)
        self.gdfs = {}

        # Now we make the profiles

        dss_daily = []
        year = 2018
        for mounth in range(1, 13):
            ds_daily = xr.open_dataset(
                gfed_file, group=f"/emissions/{mounth:02}/daily_fraction"
            )
            # Rename the phony_dims to lat and lon
            phony_dims = [dim for dim in ds_daily.dims if dim.startswith("phony_dim")]
            # Merge the days into a new dimension
            da_daily = xr.concat(
                [
                    ds_daily[f"day_{i:d}"].expand_dims(
                        day=[datetime(year=year, month=mounth, day=i)]
                    )
                    for i in range(1, 32)
                    if f"day_{i:d}" in ds_daily
                ],
                dim="day",
            )
            dss_daily.append(
                da_daily.rename({phony_dims[0]: "lat", phony_dims[1]: "lon"})
                # Scale with the mounth
                * ds.sel(mounth=mounth)
            )
        ds_day_of_year = xr.concat(dss_daily, dim="day")
        ds_day_of_year = ds_day_of_year / ds_day_of_year.sum(dim="day")

        # Stack to have on the cell dimension
        ds_day_of_year_stacked = ds_day_of_year.stack(cell=("lon", "lat"))
        ds_day_of_year_stacked = ds_day_of_year_stacked.assign_coords(
            cell=range(ds_day_of_year_stacked.sizes["cell"])
        )

        profile_indexes = xr.DataArray(
            -1,
            coords={"cell": ds_day_of_year_stacked["cell"], "substance": ["C", "DM"]},
            dims=["cell", "substance"],
        )

        ds_stacked_cleaned_C = ds_day_of_year_stacked["C"].dropna("cell")
        profiles_C = ds_stacked_cleaned_C.values.T
        ds_stacked_cleaned_DM = ds_day_of_year_stacked["DM"].dropna("cell")
        profiles_DM = ds_stacked_cleaned_DM.values.T
        profile_indexes.loc[ds_stacked_cleaned_C["cell"], "C"] = range(
            len(ds_stacked_cleaned_C["cell"])
        )
        profile_indexes.loc[ds_stacked_cleaned_DM["cell"], "DM"] = np.array(
            range(len(ds_stacked_cleaned_DM["cell"]))
        ) + len(ds_stacked_cleaned_C["cell"])

        profiles = CompositeTemporalProfiles.from_ratios(
            np.concatenate([profiles_C, profiles_DM]), [DayOfYearProfile]
        )

        self.set_profiles(profiles, profile_indexes)
