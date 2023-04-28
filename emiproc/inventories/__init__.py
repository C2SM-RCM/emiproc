"""Inventories of emissions."""
from __future__ import annotations
import logging
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import NewType
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import xarray as xr

from emiproc.grids import LV95, Grid, SwissGrid
from emiproc.profiles.temporal_profiles import TemporalProfile
from emiproc.profiles.utils import get_desired_profile_index
from emiproc.regrid import get_weights_mapping, weights_remap
from emiproc.profiles.vertical_profiles import VerticalProfile, VerticalProfiles

from emiproc.grids import Grid
from emiproc.regrid import get_weights_mapping, weights_remap

# Represent a substance that is emitted and can be present in a dataset.
Substance = NewType("Substance", str)
# Represent a category of emissions.
Category = NewType("Category", str)
# A colum from the gdf
CatSub = tuple[Category, Substance]


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
    """Base class for inventories.

    :attr name: The name of the inventory. This is going to be used
        for adding metadata to the output files, and also for the reggridding
        weights files.
    :attr grid: The grid on which the inventory is.
    :attr substances: The :py:class:`Substance` present in this inventory.
    :attr categories: List of the categories present in the inventory.

    :attr emission_infos: Information about the emissions.
        Concerns only the :attr:`Inventory.gdfs` features.
        This is optional, but mandoatory for some models (ex. Gramm-Gral).

    :attr gdf: The GeoPandas DataFrame that represent the whole inventory.
        The geometry column contains all the grid cells.
        The other columns should contain the emission value for the substances
        and the categories.

    :attr gdfs: Some inventories are given on more than one grid.
        For example, :py:class:`MapLuftZurich` is given on a grid
        where every category has different shape file.
        In this case gdf must be set to None and gdfs will be
        a dictionnary mapping only the categories desired.

    :attr v_profiles: A vertical profiles object.
    :attr v_profiles_indexes: A :py:class:`xarray.DataArray` storing the information
        of which vertical profile belongs to which cell/category/substance.
        This allow to map each single emission value from the gdf to a specific
        profile.
        See :ref:`vertical_profiles` for more information.

    :attr t_profiles_groups: A list  of temporal profiles groups.
        One temporal pattern can be defined by more than one temporal profile.
        (ex you can combine hour of day and day of week).
        The main list contains the different groups of temporal profiles.
        Each group is a list of :py:class:`TemporalProfile`.
    :attr t_profiles_indexes: Same as :py:attr:`v_profiles_indexes`.
        For the temporal profiles, the indexes point to one of the groups.


    :attr history: Stores all the operations that happened to this inventory.

    .. note::
        If your data contains point sources, the data on them must be stored in
        the gdfs, as :attr:`gdf` is only valid for the inventory grid.
        A gdf should contain only point sources.

    """

    name: str

    grid: Grid
    substances: list[Substance]
    categories: list[Category]
    emission_infos: dict[Category, EmissionInfo]

    gdf: gpd.GeoDataFrame | None
    gdfs: dict[str, gpd.GeoDataFrame]
    geometry: gpd.GeoSeries

    v_profiles: VerticalProfiles | None = None
    v_profiles_indexes: xr.DataArray | None = None

    t_profiles_groups: list[list[TemporalProfile]] | None = None
    t_profiles_indexes: xr.DataArray | None = None

    logger: logging.Logger
    history: list[str]

    _groupping: dict[str, list[str]] | None = None

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
        if hasattr(self, "_cell_area"):
            return self._cell_area
        raise NotImplementedError(f"implement or assign 'cell_areas' in {self.name}")

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

    def copy(self, no_gdfs: bool = False, no_v_profiles: bool = False) -> Inventory:
        """Copy the inventory."""
        inv = Inventory()
        inv.__class__ = self.__class__
        inv.history = deepcopy(self.history)
        if hasattr(self, "grid"):
            inv.grid = self.grid

        
        if not no_v_profiles and self.v_profiles is not None:
            inv.v_profiles = self.v_profiles.copy()
            inv.v_profiles_indexes = self.v_profiles_indexes.copy()

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

    def get_emissions(
        self, category: str, substance: str, ignore_point_sources: bool = False
    ):
        """Get the emissions of the requested category and substance.

        In case you have point sources the will be assigned their correct grid cells.

        :arg ignore_point_sources: Whether points sources should not be counted.
        .. note::
            Internally emiproc stores categories and substances as a tuple
            in the header of the gdf: (category, substance),
            or uses the gdfs dictonary for {category: df} where the
            df has substances in the header.
            If you combined the two, a category not in the df should
            be present in the gdfs.
            If you have an optimized way of doing this, you can reimplement
            this function in your :py:class:`Inventory` .
        """
        tuple_name = (category, substance)
        if tuple_name in self.gdf:
            return self.gdf[tuple_name]
        if category in self.gdfs.keys():
            gdf = self.gdfs[category]
            # check if it is point sources
            if len(gdf) == 0 or isinstance(gdf.geometry.iloc[0], Point):
                if ignore_point_sources:
                    return np.zeros(len(gdf))
                else:
                    return weights_remap(
                        get_weights_mapping(
                            Path(".emiproc")
                            / f"Point_source_{type(self).__name__}_{category}",
                            gdf.geometry,
                            self.gdf.geometry,
                            loop_over_inv_objects=True,
                        ),
                        gdf[substance],
                        len(self.gdf),
                    )
            else:
                return gdf[substance]
        raise IndexError(f"Nothing found for {category}, {substance}")

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
        self.gdfs[category] = self.gdfs[category].append(gdf, ignore_index=True)


class EmiprocNetCDF(Inventory):
    """An output from emiproc.

    Useful if you need to process again an inventory.
    """

    def __init__(self, file: PathLike) -> None:
        super().__init__()


if __name__ == "__main__":
    test_inv = Inventory()
