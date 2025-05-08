"""File containing the TNO inventory functions."""

import logging
from os import PathLike
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.creation import polygons

from emiproc.grids import WGS84, GeoPandasGrid, TNOGrid
from emiproc.inventories import Inventory
from emiproc.profiles import naming
from emiproc.profiles.operators import group_profiles_indexes
from emiproc.profiles.temporal.profiles import (
    DayOfYearProfile,
    get_leap_year_or_normal,
)
from emiproc.profiles.temporal.io import read_temporal_profiles

from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.profiles.vertical_profiles import read_vertical_profiles

logger = logging.getLogger(__name__)


class TNO_Inventory(Inventory):
    """The TNO inventory.

    TNO has grid cell sources and point sources.
    This handles both.

    https://topas.tno.nl/emissions/


    All the information of the inventory is stored in a netcdf file.

    Each substance is a separate variable in the netcdf file.
    This class will read the `long_name` attribute of each variable to
    determine the substance variables. The `long_name` attribute should
    start with `emission of` .
    You can then merge the substances from the file to a new set of
    substances using the `substances_mapping` argument.
    A default mapping which should work for general cases is provided.

    In the profile files, if you specifiy the substances, you will have to
    use the names created by the mapping, not the names in the nc file.


    :param tno_ds: The xarray dataset with the TNO emission data.
    """

    grid: TNOGrid
    tno_ds: xr.Dataset

    def __init__(
        self,
        nc_file: PathLike,
        substances_mapping: dict[str, str] = {
            "co2_ff": "CO2",
            "co2_bf": "CO2",
            "co_ff": "CO",
            "co_bf": "CO",
            "nox": "NOx",
            "ch4": "CH4",
            "nmvoc": "VOC",
        },
        profiles_dir: PathLike = None,
        vertical_profiles_dir: PathLike = None,
        temporal_profiles_dir: PathLike = None,
        # I assume it is (no info in nc file)
        crs: str = WGS84,
    ) -> None:
        """Create a TNO_Inventory.

        :arg nc_file: Path to the TNO NetCDF dataset.
        :arg substances_mapping: How to map the names from the nc files
            to names for emiproc. See in :py:class:`TNO_Inventory` for more
            information.
            Specifiying the mapping is important if the names in the nc file
            are not the same as the ones you want to use in emiproc.
        :arg profiles_dir: The directory where the profiles are stored.
            If None the same directory as the nc_file is used.
        :arg vertical_profiles_dir: The directory where the vertical profiles
            are stored. If None profiles_dir is used.
        :arg temporal_profiles_dir: The directory where the temporal profiles
            are stored. If None profiles_dir is used.
        """
        super().__init__()

        nc_file = Path(nc_file)
        if not nc_file.is_file():
            raise FileNotFoundError(f"TNO Inventory file {nc_file} is not a file.")

        if profiles_dir is None:
            profiles_dir = nc_file.parent
        else:
            profiles_dir = Path(profiles_dir)
            if not profiles_dir.is_dir():
                raise FileNotFoundError(
                    f"Profiles directory {profiles_dir} is not a directory."
                )

        if vertical_profiles_dir is None:
            vertical_profiles_dir = profiles_dir
        else:
            vertical_profiles_dir = Path(vertical_profiles_dir)

        if temporal_profiles_dir is None:
            temporal_profiles_dir = profiles_dir
        else:
            temporal_profiles_dir = Path(temporal_profiles_dir)
            if not temporal_profiles_dir.is_dir():
                raise FileNotFoundError(
                    f"Temporal profiles directory {temporal_profiles_dir} is not a"
                    " directory."
                )

        # Read the vertical profiles files
        v_profiles, v_profiles_indexes = read_vertical_profiles(vertical_profiles_dir)

        # Set the Temporal profiles
        # Time profiles can vary on category and also on substance
        t_profiles, t_profiles_indexes = read_temporal_profiles(
            temporal_profiles_dir,
            profile_csv_kwargs={
                "encoding": "latin",
            },
        )
        logging.debug(f"Temporal profiles indexes: {t_profiles_indexes}")

        self.name = nc_file.stem

        ds = xr.load_dataset(nc_file, engine="netcdf4")
        self.tno_ds = ds

        self.grid = TNOGrid(nc_file)

        # Read the source types codes
        source_types = ds["source_type_code"].to_numpy()
        for i, source_type in enumerate(source_types):
            # Indexes start at 1 🤦‍♀️
            mask = ds["source_type_index"] == i + 1
            if source_type == b"a":
                mask_area_sources = mask
            elif source_type == b"p":
                mask_point_sources = mask
            else:
                raise NotImplementedError(f"Unknown `source_type_code` {source_type}.")
        # Check that we got all the sources
        masks = [mask_area_sources, mask_point_sources]
        if not all(np.logical_xor.reduce(masks)):
            raise ValueError(
                "The masks overlap or not all point sources were assigned. This is"
                " probably a problem with how the source types are defined or assigned."
            )
        categories = ds["emis_cat_code"].to_numpy()
        # Decode from bytes
        categories = [c.decode("utf-8") for c in categories]

        # Read the variables in the file
        file_substances = [
            var
            for var, xr_var in ds.variables.items()
            if xr_var.attrs.get("long_name", "").startswith("emission of")
        ]
        if substances_mapping:
            # Check that all the file substances are in the mapping
            missing_substances = set(file_substances) - set(substances_mapping.keys())
            if missing_substances:
                self.logger.warning(
                    f"Substances {missing_substances} in the nc file are not in the"
                    f" mapping: {substances_mapping}.\n"
                    "They will be ignored."
                )
                file_substances = [
                    s for s in file_substances if s not in missing_substances
                ]
            # Check if all the mapping substances are in the file
            missing_substances = set(substances_mapping.keys()) - set(file_substances)
            if missing_substances:
                raise ValueError(
                    f"Substances {missing_substances} in the mapping are not in the"
                    f" nc file: {file_substances}.\n"
                    "Please check the names in the mapping."
                )
        else:
            substances_mapping = {s: s for s in file_substances}

        weights = xr.DataArray(
            data=np.zeros((len(categories), len(substances_mapping))),
            coords={
                "category": categories,
                "substance": list(substances_mapping.keys()),
            },
            dims=["category", "substance"],
        )

        polys = self.grid.cells_as_polylist

        # Index in the polygon list (from the gdf) (index start at 1 )
        poly_ind = (
            (ds["longitude_index"] - 1) * self.grid.ny + (ds["latitude_index"] - 1)
        ).to_numpy()
        mapping = {}
        self.gdfs = {}

        for cat_idx, cat_name in enumerate(categories):
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

        for profile_type, profiles_indexes in {
            "vertical": v_profiles_indexes,
            "temporal": t_profiles_indexes,
        }.items():
            if profiles_indexes is None:
                continue
            if "substance" in profiles_indexes.dims:
                # Check that the substances in the profiles match the ones in the
                # nc file
                for sub in profiles_indexes["substance"].data:
                    if sub not in substances_mapping.keys():
                        self.logger.warning(
                            f"Substance {sub} in {profile_type} profiles is not in the"
                            f" nc file ({file_substances}). Please check the names in"
                            " the profile files."
                        )
            if "category" in profiles_indexes.dims:
                # Check that the categories in the profiles match the ones in the
                # nc file
                for cat in profiles_indexes["category"].data:
                    if cat not in categories:
                        self.logger.warning(
                            f"Category {cat} in {profile_type} profiles is not in the"
                            f" nc file {categories}. Please check the names in the"
                            " profile files."
                        )

        # We are going to apply the mapping of substances to the profiles

        groupp_mapping = {}
        for sub_from, sub_to in substances_mapping.items():
            if sub_to not in groupp_mapping:
                groupp_mapping[sub_to] = []
            groupp_mapping[sub_to].append(sub_from)
        if t_profiles is not None:
            if "substance" in t_profiles_indexes.dims:
                (
                    t_profiles,
                    t_profiles_indexes,
                ) = group_profiles_indexes(
                    t_profiles,
                    t_profiles_indexes,
                    weights,
                    groupp_mapping,
                    groupping_dimension="substance",
                )
            self.set_profiles(t_profiles, t_profiles_indexes)
        if v_profiles is not None:
            if "substance" in v_profiles_indexes.dims:
                v_profiles, v_profiles_indexes = group_profiles_indexes(
                    v_profiles,
                    v_profiles_indexes,
                    weights,
                    groupp_mapping,
                    groupping_dimension="substance",
                )
            self.set_profiles(v_profiles, v_profiles_indexes)


def read_tno_gridded_profiles(
    file_path: Path, year: int
) -> tuple[CompositeTemporalProfiles, xr.DataArray, GeoPandasGrid]:
    """Read the TNO gridded profiles.

    :arg file_path: The path to the file.
    :arg year: The year for which to read the profiles.

    :returns: A tuple with the profiles, the indexes and the grid.

        * Profiles: The profiles from the file
        * Indexes: The indexes of the profiles
        * Grid: The grid on which the profiles are defined. Usually not the same
            as the inventory grid.

    """

    df = pd.read_csv(file_path, sep=",")
    df = df.loc[df["year"] == year]

    # We need to create a stacked dataset to extract the profiles and indices
    ds_profiles = df.to_xarray()
    da_profiles = (
        ds_profiles.set_coords(["latitude", "longitude", "POLL", "day", "GNFR"])
        .drop_vars(["year"])["Factor"]
        .rename({"POLL": "substance", "GNFR": "category", "day": "day_of_year"})
    )

    coords_multi = ["category", "substance", "latitude", "longitude", "day_of_year"]
    multi_index = pd.MultiIndex.from_arrays(
        [da_profiles[var].values for var in coords_multi], names=coords_multi
    )

    da_profiles = da_profiles.assign_coords(index=multi_index)
    da_profiles = da_profiles.drop_duplicates("index").unstack()
    da_profiles_stacked_cell = da_profiles.stack(cell=["latitude", "longitude"])

    # Get a grid based on the cell dimension
    def get_delta_over_2(da, dim):
        values = da[dim].values
        diffs = np.diff(values)
        # Check that the diffs are the same
        if not np.allclose(diffs, diffs[0]):
            raise ValueError(f"Differences in {dim} are not the same.")
        return diffs[0] / 2.0

    dx = get_delta_over_2(da_profiles, "longitude")
    dy = get_delta_over_2(da_profiles, "latitude")

    x_coords = da_profiles_stacked_cell.longitude.values
    y_coords = da_profiles_stacked_cell.latitude.values

    coords = np.array(
        [
            [x, y]
            for x, y in zip(
                [x_coords - dx, x_coords - dx, x_coords + dx, x_coords + dx],
                [y_coords - dy, y_coords + dy, y_coords + dy, y_coords - dy],
            )
        ]
    )

    coords = np.rollaxis(coords, -1, 0)
    gdf = gpd.GeoDataFrame(geometry=polygons(coords), crs="EPSG:4326")

    grid = GeoPandasGrid(gdf, name="TNO_gridded_profiles")

    da_scaling_factors = da_profiles_stacked_cell.drop_vars(
        ["latitude", "longitude"]
    ).rename(day_of_year="ratio")
    sfs, indexes = ratios_dataarray_to_profiles(da_scaling_factors)

    profiles = CompositeTemporalProfiles.from_ratios(
        ratios=sfs / sfs.sum(axis=1).reshape(-1, 1),
        types=[get_leap_year_or_normal(DayOfYearProfile, year=year)],
    )

    return profiles, indexes, grid
