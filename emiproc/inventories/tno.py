from os import PathLike
from pathlib import Path
from typing import Any


import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from emiproc.grids import WGS84, TNOGrid
from emiproc.inventories import Inventory, Substance
from emiproc.profiles.operators import group_profiles_indexes
from emiproc.profiles.temporal_profiles import (
    AnyTimeProfile,
    DailyProfile,
    MounthsProfile,
    TemporalProfile,
    WeeklyProfile,
    from_csv,
)
from emiproc.profiles.utils import remove_objects_of_type_from_list, type_in_list


from emiproc.profiles.temporal_profiles import read_temporal_profiles
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
        substances_mapping: dict[str, str] = {
            "co2_ff": "CO2",
            "co2_bf": "CO2",
            "co_ff": "CO",
            "co_bf": "CO",
            "nox": "NOx",
            "ch4": "CH4",
            "nmvoc": "VOC",
        },
    ) -> None:
        """Create a TNO_Inventory.

        :arg nc_file: The TNO NetCDF dataset.
        :arg substances: A list of substances to load in the inventory.
        :arg substances_mapping: How to mapp the names from the nc files,
            to names for empiproc.
        """
        super().__init__()

        nc_file = Path(nc_file)
        if not nc_file.is_file():
            raise FileNotFoundError(f"TNO Inventory file {nc_file} is not a file.")

        self.name = nc_file.stem

        ds = xr.load_dataset(nc_file, engine="netcdf4")

        self.grid = TNOGrid(nc_file)

        mask_area_sources = ds["source_type_index"] == 1
        mask_point_sources = ds["source_type_index"] == 2

        categories = ds["emis_cat_code"].to_numpy()

        weights = xr.DataArray(
            data=np.zeros((len(categories), len(substances_mapping))),
            coords={
                "category": [c.decode() for c in categories],
                "substance": list(substances_mapping.keys()),
            },
            dims=["category", "substance"],
        )

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

        # Vertical Profiles are read from a file
        self.v_profiles, profiles_categories = read_vertical_profiles(
            nc_file.with_name("TNO_height-distribution_GNFR.csv")
        )

        # Set the matching of the profiles
        self.v_profiles_indexes = xr.DataArray(
            np.arange(len(profiles_categories), dtype=int),
            dims=("category"),
            coords={"category": profiles_categories},
        )

        # Set the Temporal profiles
        # Time profiles can vary on category and also on substance
        t_profiles, t_profiles_indexes = read_temporal_profiles(
            nc_file.parent,
            profile_csv_kwargs={
                "cat_colname": "GNFR_Category",
                "read_csv_kwargs": {"sep": ";", "header": 6, "encoding": "latin"},
            },
        )
        # Check that the substances in the profiles match the ones in the
        # nc file
        for sub in t_profiles_indexes["substance"].data:
            if sub not in substances_mapping.keys():
                self.logger.error(
                    f"Substance {sub} in temporal profiles is not in the nc file. Please check the names in the profile files."
                )

        for cat_idx, cat_name in enumerate(categories):
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
                emissions = ds[sub_in_nc][mask].to_numpy()
                np.add.at(mapping[tuple_idx], poly_ind[mask], emissions)
                weights.loc[dict(category=cat_name, substance=sub_in_nc)] += np.sum(
                    emissions
                )

                # Add the point sources
                point_sources_values = ds_point_sources[sub_in_nc].to_numpy()
                if sub_emiproc in point_sources_gdf:
                    point_sources_gdf[sub_emiproc] += point_sources_values
                else:
                    point_sources_gdf[sub_emiproc] = point_sources_values
                weights.loc[dict(category=cat_name, substance=sub_in_nc)] += np.sum(
                    point_sources_values
                )

            if len(point_sources_gdf):
                # Add only the non emtpy
                self.gdfs[cat_name] = point_sources_gdf

        self.gdf = gpd.GeoDataFrame(
            mapping,
            geometry=polys,
            crs=crs,
        )

        self.cell_areas = ds["area"].T.to_numpy().reshape(-1)

        # We are going to apply the mapping of substances to the

        groupp_mapping = {}
        for sub_from, sub_to in substances_mapping.items():
            if sub_to not in groupp_mapping:
                groupp_mapping[sub_to] = []
            groupp_mapping[sub_to].append(sub_from)

        self.t_profiles_groups, self.t_profiles_indexes = group_profiles_indexes(
            t_profiles,
            t_profiles_indexes,
            weights,
            groupp_mapping,
            groupping_dimension='substance',
        )


if __name__ == "__main__":
    # %%

    tno_dir = r"C:\Users\coli\Documents\emiproc\files\TNO_6x6_GHGco_v4_0"

    profiles, indexes = read_temporal_profiles(
        tno_dir,
        profile_csv_kwargs={
            "cat_colname": "GNFR_Category",
            "read_csv_kwargs": {"sep": ";", "header": 6, "encoding": "latin"},
        },
    )
    profiles
# %%
