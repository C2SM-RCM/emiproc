from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from emiproc.grids import WGS84, TNOGrid
from emiproc.inventories import Inventory, Substance


from emiproc.profiles.vertical_profiles import (
    VerticalProfiles,
    check_valid_vertical_profile,
)


def read_vertical_profiles(file: PathLike) -> tuple[VerticalProfiles, list[str]]:
    """Read tno vertical profiles.

    Vertical profiles only depend on the category.

    :return: A tuple containing the vertical profiles and a list of the
        categories that matches each profiles.
    """

    # These are the names in the sectors column of tno
    # seems that these should be changed with new versions
    sectors_column = "TNO GNFR sectors Sept 2018"
    alternative_name_for_sectors_column = "GNFR_Category"

    df_vertical = pd.read_csv(
        file,
        header=17,
        sep="\t",
    )
    boundarys = df_vertical.columns[3:]
    starts = []
    ends = []
    for boundary_str in boundarys:
        a, b = boundary_str.split("-")
        starts.append(int(a))
        ends.append(int(b.replace("m", "")))
    tops = np.array(ends)
    bots = np.array(starts)
    
    # Store the profiles in the object and check the validity
    profiles = VerticalProfiles(df_vertical[boundarys].to_numpy(), tops)
    check_valid_vertical_profile(profiles)

    # Categories are the sectors
    categories = df_vertical[sectors_column].to_list()
    
    return profiles, categories


class TNO_Inventory(Inventory):
    """The TNO inventory.

    TNO has grid cell sources and point sources.
    This handles both.

    https://topas.tno.nl/emissions/
    """

    grid: TNOGrid

    def __init__(
        self,
        nc_file: PathLike,
        substances: list[Substance] = ["CO2", "CO", "NOx", "CH4", "VOC"],
    ) -> None:
        """Create a TNO_Inventory.

        :arg nc_file: The TNO NetCDF dataset.
        """
        super().__init__()

        nc_file = Path(nc_file)
        if not nc_file.is_file():
            raise FileNotFoundError(
                f"TNO Inventory file {nc_file} is not a file."
            )

        self.name = nc_file.stem

        ds = xr.load_dataset(nc_file, engine="netcdf4")

        self.grid = TNOGrid(nc_file)

        mask_area_sources = ds["source_type_index"] == 1
        mask_point_sources = ds["source_type_index"] == 2

        # Maps substances inside the nc file to the emiproc names
        substances_mapping = {
            "co2_ff": "CO2",
            "co2_bf": "CO2",
            "co_ff": "CO",
            "co_bf": "CO",
            "nox": "NOx",
            "ch4": "CH4",
            "nmvoc": "VOC",
        }

        # Select only the requested substances
        substances_mapping = {
            k: v for k, v in substances_mapping.items() if v in substances
        }

        polys = self.grid.cells_as_polylist

        # I assume it is (no info in nc file)
        crs = WGS84

        # Index in the polygon list (from the gdf) (index start at 1 )
        poly_ind = (
            (ds["longitude_index"] - 1) * self.grid.ny + (ds["latitude_index"] - 1)
        ).to_numpy()
        mapping = {}
        self.gdfs = {}
        for cat_idx, cat_name in enumerate(ds["emis_cat_code"].to_numpy()):
            cat_name = cat_name.decode("utf-8")
            # Indexes start at 1
            mask_this_category = ds["emission_category_index"] == cat_idx + 1

            # Extract point sources information
            mask_points_this_cat = mask_this_category & mask_point_sources
            point_sources_gdf = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries.from_xy(
                    ds["longitude_source"][mask_points_this_cat],
                    ds["latitude_source"][mask_points_this_cat],
                ),
                crs=crs,
            )
            # Extract the emissions for the points sources of this cat
            ds_point_sources = ds[substances_mapping.keys()].sel(
                {"source": mask_points_this_cat}
            )

            for sub_in_nc, sub_emiproc in substances_mapping.items():
                tuple_idx = (cat_name, sub_emiproc)
                if tuple_idx not in mapping:
                    mapping[tuple_idx] = np.zeros(len(polys))
                # Add all the area sources corresponding to that category
                mask = mask_this_category & mask_area_sources
                np.add.at(
                    mapping[tuple_idx], poly_ind[mask], ds[sub_in_nc][mask].to_numpy()
                )

                # Add the point sources
                point_sources_values = ds_point_sources[sub_in_nc]
                if sub_emiproc in point_sources_gdf:
                    point_sources_gdf[sub_emiproc] += point_sources_values.to_numpy()
                else:
                    point_sources_gdf[sub_emiproc] = point_sources_values.to_numpy()

            if len(point_sources_gdf):
                # Add only the non emtpy
                self.gdfs[cat_name] = point_sources_gdf

        self.gdf = gpd.GeoDataFrame(
            mapping,
            geometry=polys,
            crs=crs,
        )

        self.cell_areas = ds["area"].T.to_numpy().reshape(-1)

        # Vertical Profiles are read from a file
        self.v_profiles, profiles_categories = read_vertical_profiles(nc_file.with_name("TNO_height-distribution_GNFR.csv"))
        
        # Set the matching of the profiles
        self.v_profiles_indexes = xr.DataArray(
            np.arange(len(profiles_categories), dtype=int),
            dims=("category"),
            coords={"category": profiles_categories}
        )

        # Set the vertical profiles to the points sources
        for cat_name, gdf in self.gdfs.items():
            gdf["_v_profile"] = profiles_categories.index(cat_name)
        

if __name__ == "__main__":
    # %%
    v_prof, categories = read_vertical_profiles(
        r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0\TNO_height-distribution_GNFR.csv"
    )
    v_prof

    #%% Read the time profiles
    def read_tno_time_profile_csv(file: PathLike):
        df = pd.read_csv(file, header=6, sep=";", encoding="latin")

        return df

    df_hod = read_tno_time_profile_csv(
        r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0\timeprofiles-hour-in-day_GNFR.CSV"
    )
    df_dow = read_tno_time_profile_csv(
        r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0\timeprofiles-day-in-week_GNFR.CSV"
    )
    df_moy = read_tno_time_profile_csv(
        r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0\timeprofiles-month-in-year_GNFR.csv"
    )

    sectors_columns = [
        (
            df[sectors_column]
            if sectors_column in df.columns
            else df[alternative_name_for_sectors_column]
        ).to_numpy()
        for df in [df_vertical, df_hod, df_dow, df_moy]
    ]

    # Check matching of the categories
    if not np.all(sectors_columns == sectors_columns[0]):
        raise ValueError("Some categories are not the same in the profiles files.")

    profiles = xr.Dataset(
        {
            "vertical": (("category", "level"), df_vertical[boundarys].to_numpy()),
        },
        {
            "level": list(range(len(boundarys))),
            "layer_top": (
                "level",
                tops,
                {"long_name": "top of layer above ground"} | layer_attrs,
            ),
            "layer_bot": (
                "level",
                bots,
                {"long_name": "bottom of layer above ground"} | layer_attrs,
            ),
            "categories": df_vertical[sectors_column].to_list(),
        },
    )
    profiles
    # %%
