from os import PathLike

import geopandas as gpd
import pandas as pd
import xarray as xr

from emiproc.grids import BoundingBox, RegularGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.profiles import (
    DayOfYearProfile,
    get_leap_year_or_normal,
)
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.utilities import SEC_PER_YR


class GFAS_Inventory(Inventory):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "GFAS_Inventory has been updated and renamed to GFAS. Please use GFAS instead."
        )


class GFAS(Inventory):
    """The GFAS inventory.

    Contains gridded data of forest fires from the Copernicus Atmosphere Monitoring
    Service (CAMS).

    You can access the data at
    `CAMS global biomass burning emissions based on fire radiative power
    <https://ads.atmosphere.copernicus.eu/datasets/cams-global-fire-emissions-gfas?tab=overview>`_

    """

    grid: RegularGrid

    def __init__(
        self,
        nc_file: PathLike,
        variables: list[str] = [],
        bbox: BoundingBox | None = None,
    ):
        """Create a GFAS inventory.

        The GFAS nc file contains gridded data of forest fires.
        By default, all variables in the nc file are included. If you want to include
        only a subset of the variables, you can specify them in the variables argument.

        :param nc_file: The path to the netCDF file.
            Make sure the file contains one year of data
        :param variables: A list of variables to include in the inventory.
        :param bbox: A bounding box to subset the data.
            `[minx, miny, maxx, maxy]`

        """
        super().__init__()
        ds = xr.open_dataset(nc_file)

        self.year = pd.Timestamp(ds["valid_time"].values[0]).year

        # Check that the file contains one year of data
        expected_profile = get_leap_year_or_normal(DayOfYearProfile, year=self.year)
        if expected_profile.size != ds["valid_time"].size:
            raise ValueError(
                f"Expected {expected_profile.size} timesteps for year {self.year}, "
                f"but got {ds['valid_time'].size} timesteps in the file."
                " Make sure the file contains one (and only one) full year of data."
            )

        # Apply optional bounding box to subset the data
        if bbox:
            lon_mask = (ds["longitude"] >= bbox[0]) & (ds["longitude"] <= bbox[2])
            lat_mask = (ds["latitude"] >= bbox[1]) & (ds["latitude"] <= bbox[3])
            ds = ds.sel(longitude=lon_mask, latitude=lat_mask)

        if not variables:
            variables = list(ds.data_vars.keys())
        var_2_emiproc = {var: var.replace("fire", "").upper() for var in variables}

        profiles = {
            sub: ds[key].expand_dims(substance=[sub])
            for key, sub in var_2_emiproc.items()
        }

        self.grid = RegularGrid.from_centers(
            x_centers=ds["longitude"].values,
            y_centers=ds["latitude"].values,
            name="GFAS",
            rounding=2,
            # Custom projection, to correct for the lon range of the data
            crs="+proj=longlat +datum=WGS84 +no_defs +type=crs +lon_wrap=180",
        )

        self.gdfs = {}

        # Reshape the data to the regular grid
        da_profiles: xr.DataArray = (
            xr.concat(list(profiles.values()), dim="substance")
            .stack(cell=("longitude", "latitude"))
            .drop_vars(["longitude", "latitude"])
            .rename({"valid_time": "ratio"})
        )

        def process_substance(sub):
            return (
                da_profiles.sel(substance=sub).mean("ratio")
                # Convert from kg m-2 s-1 to kg/yr
                * SEC_PER_YR
                * self.cell_areas.reshape(-1)
            )

        self.gdf = gpd.GeoDataFrame(
            {("gfas", sub): process_substance(sub) for sub in var_2_emiproc.values()},
            geometry=self.grid.gdf.geometry,
        )

        ratios, indices = ratios_dataarray_to_profiles(da_profiles)

        profiles = CompositeTemporalProfiles.from_ratios(
            ratios,
            [get_leap_year_or_normal(DayOfYearProfile, year=self.year)],
            rescale=True,
        )

        self.set_profiles(profiles, indices)
