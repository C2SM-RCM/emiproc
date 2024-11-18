from os import PathLike

import geopandas as gpd
import xarray as xr
import re
from pathlib import Path

from emiproc.grids import WGS84, RegularGrid
from emiproc.inventories import Inventory
from emiproc.profiles.temporal_profiles import read_temporal_profiles
from emiproc.profiles.vertical_profiles import read_vertical_profiles

UNIT_CONVERSION_FACTOR = 1e9  # Tg -> kg

class CAMS_REG_AQ(Inventory):

    grid: RegularGrid

    def __init__(
        self,
        nc_dir: PathLike,
        profiles_dir: PathLike = None,
        year: int = 2022,
        substances_mapping: dict[str, str] = {
            "nox": "NOx",
            "co": "CO",
            "ch4": "CH4",
            "nmvoc": "VOC",
            "sox": "SO2",
            "nh3": "NH3",
            "pm2_5": "PM25",
            "pm10":"PM10"
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
            "L_AgriOther": "L"
        },
        substances_mapping_profiles: dict[str, str] = {
            "nox": "NOx",
            "co": "CO",
            "ch4": "CH4",
            "nmvoc": "VOC",
            "so2": "SO2",
            "nh3": "NH3",
            "pm2_5": "PM25",
            "pm10":"PM10"
        },
    ):
        
        super().__init__()

        filename_pattern = fr"CAMS-REG-ANT_EUR_0\.05x0\.1_anthro_(?P<substance>\w+)_v6\.1-Ref2_yearly_{year}\.nc"

        nc_dir = Path(nc_dir)
        if not nc_dir.is_dir():
                raise FileNotFoundError(
                    f"Profiles directory {nc_dir} is not a directory."
                )
        nc_files = [f for f in nc_dir.iterdir() if f.is_file()]

        if profiles_dir is None:
            profiles_dir = Path(nc_dir)
        else:
            profiles_dir = Path(profiles_dir)
            if not profiles_dir.is_dir():
                raise FileNotFoundError(
                    f"Profiles directory {profiles_dir} is not a directory."
                )
        
        # Read the vertical and temporal profile files
        v_profiles, v_profiles_indexes = read_vertical_profiles(profiles_dir)

        t_profiles, t_profiles_indexes = read_temporal_profiles(
            profiles_dir,
            profile_csv_kwargs={
                "encoding": "latin",
            },
        )
        # Rename substances in profiles according to dictionary
        if "substance" in t_profiles_indexes.dims:
            t_profiles_indexes = t_profiles_indexes.assign_coords(
                substance=[
                    substances_mapping_profiles[name] for name in t_profiles_indexes['substance'].values
                ]
            )
        if "substance" in v_profiles_indexes.dims:
            v_profiles_indexes = v_profiles_indexes.assign_coords(
                substance=[
                    substances_mapping_profiles[name] for name in v_profiles_indexes['substance'].values
                ]
            )

        # Read in emission data
        inv_data = {}

        for nc_file in nc_files:
            if nc_file.suffix == '.nc' and re.match(filename_pattern, nc_file.name):
        
                ds = xr.open_dataset(nc_file)

                match = re.match(filename_pattern, nc_file.name)
                sub_cams = match.group('substance')
                sub_name = substances_mapping.get(sub_cams, None)
                if sub_name is None: 
                    raise ValueError(f"No substance mapping fround for {sub_cams}")
            
                file_vars = ds.data_vars.keys()

                for var, cat in categories_mapping.items():
                    if var in file_vars:
                        col_index = (cat, sub_name)
                        inv_data[col_index] = ds[var].expand_dims(cat_sub=[col_index])
                    else:
                        raise ValueError(f"Category {var} not found in the file {nc_file}.")

                # Extract grid information
                if not hasattr(self, 'grid'):
                    self.grid = RegularGrid.from_centers(
                        x_centers=ds["lon"].values,
                        y_centers=ds["lat"].values,
                        name="CAMS_REG_AQ",
                        rounding=2,
                    )
            else:
                print(f"Skipping file: {nc_file} (does not match the expected pattern or not a .nc file)")

        # List of pairs (emis cat, sub)
        cat_sub_pairs = [(cat, sub) for cat in categories_mapping.values() for sub in substances_mapping.values()]
        
        # Reshape data to regular grid
        da_inventory: xr.DataArray = (
            xr.concat(list(inv_data.values()), dim="cat_sub")
            .stack(cell=("lon", "lat"))
            .drop_vars(["lon", "lat", "time"])
        )

        def process_cat_sub(cs):
            return (
                da_inventory.sel(cat_sub=cs).values.flatten() 
                * UNIT_CONVERSION_FACTOR 
            )

        self.gdf = gpd.GeoDataFrame(
            {cs: process_cat_sub(cs) for cs in cat_sub_pairs},
            geometry=self.grid.gdf.geometry
            )
        
        self.gdfs = {} 
        
        self.set_profiles(t_profiles, t_profiles_indexes)
        self.set_profiles(v_profiles, v_profiles_indexes)
