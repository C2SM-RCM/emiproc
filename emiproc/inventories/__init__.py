"""Inventories of emissions.

Contains the classes and functions to work with inventories of emissions.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from typing import NewType, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.grids import GeoPandasGrid, Grid
from emiproc.profiles import naming
from emiproc.profiles.temporal.profiles import (
    AnyTimeProfile,
    TemporalProfile,
)
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.utils import check_valid_indexes
from emiproc.profiles.vertical_profiles import (
    VerticalProfile,
    VerticalProfiles,
    resample_vertical_profiles,
)

# Represent a substance that is emitted and can be present in a dataset.
Substance = NewType("Substance", str)
# Represent a category of emissions.
Category = NewType("Category", str)
# A colum from the gdf
CatSub = tuple[Category, Substance]

TemporalProfiles = Union[list[list[TemporalProfile]], CompositeTemporalProfiles]


@dataclass
class EmissionInfo:
    """Information about an emission category.

    This additional information is used for some models.
    It concerns only the :py:attr:`Inventory.gdfs` features.

    :param height: The height of the emission source (over the ground). [m]
    :param height_over_buildings: If True, the height is taken over buildings.
    :param width: The width of the emission source. [m]
    :param vertical_extension:
        The vertical extension (thickness) of the emission source. [m]
        This implies that the emission starts at height
        and ends at height + vertical_extension.
    :param temperature: The temperature of the emission source. [K]
    :param speed: The speed of the emission of the substances. [m/s]
    :param comment: A comment about the emission source.

    """

    # Height
    height: float = 0.0
    # weight the height is taken over buildings
    height_over_buildings: bool = True

    width: float = 0.5
    vertical_extension: float = 3.0
    temperature: float = 353.0
    speed: float = 5.0
    comment: str = ""


class Inventory:
    """Parent class for inventories.

    :param name: The name of the inventory. This is going to be used
        for adding metadata to the output files, and also for the reggridding
        weights files.
    :param year: The year of the inventory. (optional)
    :param grid: The grid on which the inventory is. Can be none if the invenotry
        is not defined on a grid. (only shapped emissions)
    :param substances: The :py:class:`Substance` present in this inventory.
    :param categories: List of the categories present in the inventory.

    :param emission_infos: Information about the emissions.
        Concerns only the :py:attr:`gdfs` features.
        This is optional, but mandatory for some models (ex. Gramm-Gral).

    :param gdf: The GeoPandas DataFrame that represent the whole inventory.
        The geometry column contains geometric objects for all the grid cells.
        The other columns should contain the emission value for the substances
        and the categories.

    :param gdfs: Some inventories are given on more than one grid.
        For example, :py:class:`MapLuftZurich` is given on a grid
        where every category has different shape file.
        In this case gdf must be set to None and gdfs will be
        a dictionnary mapping only the categories desired.

    :param v_profiles: A vertical profiles object.
    :param v_profiles_indexes: A :py:class:`xarray.DataArray` storing the information
        of which vertical profile belongs to which cell/category/substance.
        This allow to map each single emission value from the gdf to a specific
        profile.
        See :ref:`vertical_profiles` for more information.

    :param t_profiles_groups: A list  of temporal profiles groups.
        One temporal pattern can be defined by more than one temporal profile.
        (ex you can combine hour of day and day of week).
        The main list contains the different groups of temporal profiles.
        Each group is a list of :py:class:`TemporalProfile`.
    :param t_profiles_indexes: Same as :py:attr:`v_profiles_indexes`.
        For the temporal profiles, the indexes point to one of the groups.


    :param history: Stores all the operations that happened to this inventory.


    """

    name: str
    year: int | None = None

    grid: Grid | None
    substances: list[Substance]
    categories: list[Category]
    emission_infos: dict[Category, EmissionInfo]

    gdf: gpd.GeoDataFrame | None
    gdfs: dict[str, gpd.GeoDataFrame]
    geometry: gpd.GeoSeries

    v_profiles: VerticalProfiles | None = None
    v_profiles_indexes: xr.DataArray | None = None

    t_profiles_groups: TemporalProfiles | None = None
    t_profiles_indexes: xr.DataArray | None = None

    logger: logging.Logger
    history: list[str]

    def __init__(self) -> None:
        class_name = type(self).__name__
        if not hasattr(self, "name"):
            self.name = class_name
        self.history = [f"{self} created as type:'{class_name}'"]
        self.logger = logging.getLogger(f"emiproc.Inventory.{self.name}")

    def __repr__(self) -> str:
        return f"Inventory({self.name})"

    @property
    def emission_infos(self) -> dict[Category, EmissionInfo]:
        if hasattr(self, "_emission_infos"):
            return self._emission_infos
        else:
            raise ValueError(f"'emission_infos' were not set for {self}")

    @emission_infos.setter
    def emission_infos(self, emission_infos: dict[Category, EmissionInfo]):
        # Check that the cat is in the emission_infos
        missing_cats = set(self.categories) - set(emission_infos.keys())
        if missing_cats:
            raise ValueError(
                f"{missing_cats} are not in the emission_infos. "
                "Please add it to the emission_infos dict."
            )
        self._emission_infos = emission_infos

    @property
    def geometry(self) -> gpd.GeoSeries:
        return self.gdf.geometry

    @property
    def cell_areas(self) -> np.ndarray:
        """Area of the cells in m2 .

        These match the geometry from the gdf.
        """
        if not hasattr(self, "_cell_area"):
            # Compute with the grid
            if self.grid is None:
                raise ValueError(
                    "No grid set, cannot compute cell areas. "
                    f"implement or assign 'cell_areas' in {self.name}"
                )
            self._cell_area = np.array(self.grid.cell_areas)

        return self._cell_area

    @cell_areas.setter
    def cell_areas(self, cell_areas):
        if len(cell_areas) != len(self.gdf):
            raise ValueError(
                f"size does not match, got {len(cell_areas) }, expected {len(self.gdf)}"
            )

        self._cell_area = cell_areas

    @property
    def crs(self) -> int | None:
        if self.gdf is not None:
            return self.gdf.crs
        else:
            return self.gdfs[list(self.gdfs.keys())[0]].crs

    @property
    def categories(self) -> list[str]:
        return list(
            set(
                [
                    cat
                    for cat, _ in (self.gdf.columns if self.gdf is not None else [])
                    if not isinstance(self.gdf[(cat, _)].dtype, gpd.array.GeometryDtype)
                ]
            )
            | set(self.gdfs.keys())
        )

    @property
    def substances(self) -> list[Substance]:
        # Unique substances in the inventories
        subs = list(
            set(
                [
                    sub
                    for _, sub in (self.gdf.columns if self.gdf is not None else [])
                    if not isinstance(self.gdf[(_, sub)].dtype, gpd.array.GeometryDtype)
                ]
            )
            | set(sum([gdf.keys().to_list() for gdf in self.gdfs.values()], []))
        )
        if "geometry" in subs:
            subs.remove("geometry")
        return subs

    @property
    def total_emissions(self) -> pd.DataFrame:
        """Calculate the total emissions, returning a DataFrame.

        Simple accessor to the function
        :py:func:`~emiproc.inventories.utils.get_total_emissions`.
        """
        from emiproc.inventories.utils import get_total_emissions

        return pd.DataFrame(get_total_emissions(self)).T

    def copy(self, no_gdfs: bool = False, profiles: bool = True) -> Inventory:
        """Copy the inventory.

        :arg no_gdfs: Whether the gdfs should not be copied (main gdf and the gdfs).
        :arg profiles: Whether the profiles should be copied.
        """
        inv = Inventory()
        inv.__class__ = self.__class__
        inv.history = deepcopy(self.history)
        inv.year = self.year
        if hasattr(self, "grid"):
            inv.grid = self.grid

        if profiles and self.v_profiles is not None:
            inv.v_profiles = self.v_profiles.copy()
            inv.v_profiles_indexes = self.v_profiles_indexes.copy()
        if profiles and self.t_profiles_groups is not None:
            inv.t_profiles_groups = self.t_profiles_groups.copy()
            inv.t_profiles_indexes = self.t_profiles_indexes.copy()

        if no_gdfs or self.gdf is None:
            inv.gdf = None
        else:
            inv.gdf = self.gdf.copy(deep=True)

        if self.gdfs and not no_gdfs:
            inv.gdfs = {key: gdf.copy(deep=True) for key, gdf in self.gdfs.items()}
        else:
            inv.gdfs = {}

        # Copy the internal property
        if hasattr(self, "_emission_infos"):
            inv._emission_infos = deepcopy(self._emission_infos)

        inv.history.append(f"Copied from {type(self).__name__} to {inv}.")
        return inv

    @classmethod
    def from_gdf(
        cls,
        gdf: gpd.GeoDataFrame | None = None,
        name: str = "custom_from_gdf",
        gdfs: dict[str, gpd.GeoDataFrame] = {},
    ) -> Inventory:
        """The gdf must be a two level gdf with (category, substance)."""
        inv = Inventory()
        inv.name = name
        inv.gdf = gdf
        inv.gdfs = gdfs
        inv.grid = None if gdf is None else GeoPandasGrid(gdf)

        return inv

    @property
    def _gdf_columns(self) -> list[tuple[str, Substance]]:
        """All the columsn but not the geometric columns."""
        if self.gdf is None:
            return []
        return [
            col
            for col in self.gdf.columns
            if not isinstance(self.gdf[col].dtype, gpd.array.GeometryDtype)
        ]

    def to_crs(self, *args, **kwargs):
        """Same as geopandas.to_crs() but for inventories.

        Perform the conversion in place.
        """
        if self.gdf is not None:
            self.gdf.to_crs(*args, **kwargs, inplace=True)
        for gdf in self.gdfs.values():
            gdf.to_crs(*args, **kwargs, inplace=True)

    def set_crs(self, *args, **kwargs):
        """Same as geopandas.set_crs() but for inventories.

        Perform the conversion in place.
        """
        if self.gdf is not None:
            self.gdf.set_crs(*args, **kwargs, inplace=True)
        for gdf in self.gdfs.values():
            gdf.set_crs(*args, **kwargs, inplace=True)

    def add_gdf(self, category: Category, gdf: gpd.GeoDataFrame):
        """Add a gdf contaning emission sources to the inventory.

        This will add the category to the inventory and add the data to the
        inventory.gdfs dictionary.

        :arg category: The category to add.
        :arg gdf: The geodataframe containing the data for the category.
        """

        if category not in self.gdfs:
            self.gdfs[category] = gdf
            return

        # if the category is already present, we need to merge the data
        # this is a bit tricky, because we need to make sure that the
        # columns are the same

        # Set the values on missing columns to 0
        missing_columns = set(self.gdfs[category].columns) - set(gdf.columns)
        for col in missing_columns:
            gdf[col] = 0
        # Also set the substance to 0 in the gdfs if not there
        missing_columns = set(gdf.columns) - set(self.gdfs[category].columns)
        for col in missing_columns:
            self.gdfs[category][col] = 0

        # Now we can append
        self.gdfs[category] = pd.concat((self.gdfs[category], gdf), ignore_index=True)

    def set_profile(
        self,
        profile: VerticalProfile | list[TemporalProfile],
        category: str | None = None,
        substance: str | None = None,
    ) -> None:
        """Set a vertical or temporal profile to a specific category and/or substance.

        This happens in place.
        The profile is appened to the existing profiles.

        If only one of 'category' or 'substance' is specified, the profile is assigned
        to all the categories or substances on the non specified dimension.


        :arg profile: The profile to set.
            If Vertical profile, a vertical profile.
            If Temporal profile, a list of temporal profiles.
        :arg category: The category to set the profile to.
        :arg substance: The substance to set the profile to.
        """

        if isinstance(profile, VerticalProfile):
            indexes_array = self.v_profiles_indexes
            if self.v_profiles is None:
                # Set the profile for the first time
                self.v_profiles = VerticalProfiles(
                    ratios=np.array([profile.ratios]),
                    height=profile.height,
                )
            else:
                self.v_profiles = resample_vertical_profiles(
                    self.v_profiles, profile, specified_levels=self.v_profiles.height
                )
            profiles = self.v_profiles
        elif isinstance(profile, list):
            # Temporal profiles
            indexes_array = self.t_profiles_indexes
            if self.t_profiles_groups is None:
                self.t_profiles_groups = []
            self.t_profiles_groups.append(profile)

            profiles = self.t_profiles_groups
        else:
            raise ValueError(f"Unknown profile type {type(profile)}")

        if indexes_array is None:
            # Create it if it does not exist, axis is substance and category
            indexes_array = xr.DataArray(
                np.full(
                    (len(self.categories), len(self.substances)),
                    fill_value=-1,
                    dtype=int,
                ),
                dims=("category", "substance"),
                coords={"category": self.categories, "substance": self.substances},
            )

        # Add to the index the profile
        sel_dict = {}
        if category is not None:
            if "category" not in indexes_array.dims:
                # Exapnd the array over the categories of the inventory
                indexes_array = indexes_array.expand_dims({"category": self.categories})
            if category not in indexes_array.coords["category"].values:
                if category not in self.categories:
                    raise ValueError(
                        f"Category {category} is not in the inventory. "
                        "Please add it before setting the profile."
                    )
                # Otherwise add it to the coords and set the values to -1 (unassigned)
                # emtpy da with the new category being the only one in the
                new_cat_array = xr.DataArray(
                    -1,
                    dims=indexes_array.dims,
                    coords={
                        "category": [category],
                        **{
                            coord: indexes_array.coords[coord]
                            for coord in indexes_array.dims
                            if coord != "category"
                        },
                    },
                )
                indexes_array = xr.concat(
                    [indexes_array, new_cat_array], dim="category"
                )

            sel_dict["category"] = category
        if substance is not None:
            if "substance" not in indexes_array.dims:
                # Exapnd the array over the substnaces of the inventory
                indexes_array = indexes_array.expand_dims(
                    {"substance": self.substances}
                )
            sel_dict["substance"] = substance

        if sel_dict:
            # Set the index for the new profile
            profile_idx = len(profiles) - 1
            indexes_array.loc[sel_dict] = profile_idx
        else:
            self.logger.warning(
                "No category or substance specified for the profile."
                "The profile was not set."
            )
            return

        if isinstance(profile, VerticalProfile):
            self.v_profiles_indexes = indexes_array
        elif isinstance(profile, list):
            self.t_profiles_indexes = indexes_array

        self.history.append(f"Set profile {profile_idx} to {category}, {substance}.")

    def set_profiles(
        self,
        profiles: (
            VerticalProfiles | CompositeTemporalProfiles | list[list[AnyTimeProfile]]
        ),
        indexes: xr.DataArray,
    ):
        """Replace the profiles of the invenotry with the new profiles given.

        This checks that the indexes are correct and that the coords are matching.
        If they are not, gives a warning.

        If the profiles given are not valid (ex. not the same number of profiles
        as categories), raises an error.
        """

        indexes = indexes.copy()

        if isinstance(profiles, list):
            profiles = CompositeTemporalProfiles(profiles)

        # Check that the indexes are correct
        check_valid_indexes(indexes, profiles)

        # Check the coord if they match correctly what is given in the inventory
        for coord in naming.type_of_dim.keys():
            if coord not in indexes.dims:
                continue
            if coord == "cell":
                values_in_inv = list(range(len(self.grid)))
            elif coord == "category":
                values_in_inv = self.categories
            elif coord == "substance":
                values_in_inv = self.substances
            elif coord in ["country", "day_type", "time"]:
                # No check needed to be performed
                continue
            else:
                raise ValueError(f"Unknown coord {coord}")
            values_in_indexes = indexes.coords[coord].values
            values_not_in_inv = set(values_in_indexes) - set(values_in_inv)
            values_not_in_indexes = set(values_in_inv) - set(values_in_indexes)
            if values_not_in_inv:
                self.logger.warning(
                    f"{coord} {values_not_in_inv} from profiles are not in"
                    " the inventory."
                )
                indexes = indexes.drop_sel({coord: list(values_not_in_inv)})
            if values_not_in_indexes:
                self.logger.warning(
                    f"{coord} {values_not_in_indexes} from inventory are not in"
                    " the profiles."
                )

        # Assign to the inventory
        if isinstance(profiles, VerticalProfiles):
            self.v_profiles = profiles
            self.v_profiles_indexes = indexes
        elif isinstance(profiles, CompositeTemporalProfiles):
            self.t_profiles_groups = profiles
            self.t_profiles_indexes = indexes
        else:
            raise ValueError(f"Unknown profile type {type(profiles)}")


class EmiprocNetCDF(Inventory):
    """An output from emiproc.

    Useful if you need to process again an inventory.
    """

    def __init__(self, file: PathLike) -> None:
        super().__init__()


if __name__ == "__main__":
    test_inv = Inventory()
