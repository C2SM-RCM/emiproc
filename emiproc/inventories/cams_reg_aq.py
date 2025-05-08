from os import PathLike

import geopandas as gpd
import xarray as xr
import re
from pathlib import Path

from emiproc.grids import WGS84, RegularGrid
from emiproc.inventories import Inventory

UNIT_CONVERSION_FACTOR = 1e9  # Tg -> kg


class CAMS_REG_AQ(Inventory):
    """The CAMS regional air quality inventory.

    Contains gridded data of air pollutants (NOx, CO, CH4, VOC, NH3, SO2, PM2.5, PM10)
    from the Copernicus Atmosphere Monitoring Service (CAMS).

    You can access the data at
    `CAMS-REG-ANT v6.1-Ref2
    <https://permalink.aeris-data.fr/CAMS-REG-ANT>`_

    """

    grid: RegularGrid

    def __init__(
        self,
        nc_dir: PathLike,
        year: int = 2022,
        substances_mapping: dict[str, str] = {
            "nox": "NOx",
            "co": "CO",
            "ch4": "CH4",
            "nmvoc": "VOC",
            "sox": "SO2",
            "nh3": "NH3",
            "pm2_5": "PM25",
            "pm10": "PM10",
        },
        categories_mapping: dict[str, str] = {
            "A_PublicPower": "A",
            "B_Industry": "B",
            "C_OtherStationaryComb": "C",
            "D_Fugitives": "D",
            "E_Solvents": "E",
            "F_RoadTransport": "F",
            "G_Shipping": "G",
            "H_Aviation": "H",
            "I_OffRoad": "I",
            "J_Waste": "J",
            "K_AgriLivestock": "K",
            "L_AgriOther": "L",
        },
    ):
        """Create a CAMS_REG_ANT-inventory.

        :arg nc_dir: The directory containing the NetCDF emission datasets. One file
            per air pollutant.
        :arg year: Year of the inventory.
        :arg substances_mapping: How to map the names of air pollutants from the
            names of the NetCDF files to names for emiproc.
        :arg categories_mapping: How to map the names of the emission categories from
            the NetCDF files to names for emiproc.
        """

        super().__init__()

        filename_pattern = rf"CAMS-REG-ANT_EUR_0\.05x0\.1_anthro_(?P<substance>\w+)_v6\.1-Ref2_yearly_{year}\.nc"

        nc_dir = Path(nc_dir)
        if not nc_dir.is_dir():
            raise FileNotFoundError(f"Profiles directory {nc_dir} is not a directory.")
        nc_files = [
            f
            for f in nc_dir.iterdir()
            if f.is_file() and f.suffix == ".nc" and re.match(filename_pattern, f.name)
        ]
        self.logger.debug(f"{nc_files=}")

        if not nc_files:
            raise FileNotFoundError(
                f"No .nc files found matching the pattern '{filename_pattern}' in {nc_dir}"
            )

        # Read in emission data
        inv_data = {}

        substances_available = []

        for nc_file in nc_files:
            ds = xr.open_dataset(nc_file)

            match = re.match(filename_pattern, nc_file.name)
            sub_cams = match.group("substance")
            sub_name = substances_mapping.get(sub_cams, None)
            if sub_name is None:
                raise ValueError(f"No substance mapping fround for {sub_cams}")
            substances_available.append(sub_name)

            file_vars = ds.data_vars.keys()
            for var, cat in categories_mapping.items():
                if var in file_vars:
                    col_index = (cat, sub_name)
                    if ds[var].attrs["units"] != "Tg":
                        raise ValueError(
                            f"Units are {ds[var].attrs['units']}, expected Tg"
                        )
                    inv_data[col_index] = ds[var].expand_dims(cat_sub=[col_index])
                else:
                    raise ValueError(f"Category {var} not found in the file {nc_file}.")
            # Extract grid information
            if not hasattr(self, "grid"):
                self.grid = RegularGrid.from_centers(
                    x_centers=ds["lon"].values,
                    y_centers=ds["lat"].values,
                    name="CAMS_REG_AQ",
                    rounding=2,
                )

        # List of pairs (emis cat, sub)
        cat_sub_pairs = [
            (cat, sub)
            for cat in categories_mapping.values()
            for sub in substances_available
        ]

        # Reshape data to regular grid
        da_inventory: xr.DataArray = (
            xr.concat(list(inv_data.values()), dim="cat_sub")
            .stack(cell=("lon", "lat"))
            .drop_vars(["lon", "lat", "time"])
        )

        def process_cat_sub(cs):
            return (
                da_inventory.sel(cat_sub=cs).values.flatten() * UNIT_CONVERSION_FACTOR
            )

        self.gdf = gpd.GeoDataFrame(
            {cs: process_cat_sub(cs) for cs in cat_sub_pairs},
            geometry=self.grid.gdf.geometry,
        )

        self.gdfs = {}
