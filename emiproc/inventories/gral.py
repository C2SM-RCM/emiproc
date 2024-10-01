"""Read an the input for a gramgral simulation as an inventory."""
from __future__ import annotations
from enum import Enum, IntEnum
import json
import logging
from os import PathLike
from pathlib import Path
from typing import Any
from emiproc.grids import LV95, WGS84
from emiproc.inventories import Category, Inventory, Substance
from emiproc.exports.gral import EmissionWriter
from emiproc.utilities import HOUR_PER_YR
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon


class PointsCols(IntEnum):
    """Columns of the point sources file."""

    X = 0
    Y = 1
    Z = 2
    EMISSION = 3  # kg/h
    EXIT_VELOCITY = 7
    STACK_DIAMETER = 8
    EXIT_TEMPERATURE = 9
    SOURCE_GROUP = 10


class LinesCols(IntEnum):
    """Columns of the line sources file."""

    NAME = 0
    SECTION = 1
    SOURCE_GROUP = 2
    X_START = 3
    Y_START = 4
    Z_START = 5
    X_END = 6
    Y_END = 7
    Z_END = 8
    WIDTH = 9
    EXTENSION = 10
    EMISSION = 13  # kg/h/km

class CadastreCols(IntEnum):
    """Columns of the cadastre file."""

    X = 0
    Y = 1
    Z = 2
    X_EXTENSION = 3
    Y_EXTENSION = 4
    Z_EXTENSION = 5
    EMISSION = 6  # kg/h
    SOURCE_GROUP = 10

class GralInventory(Inventory):
    """Gral inventory.

    Load the inputs of a gral simulation.
    
    .. warning:: There are some things not implemented yet:
        - The portals file
        - Emission Infos (exit velocity, heights, ...)
        - Shared utilites between thefiles (many code is duplicated accros funcitons)

    """

    def __init__(
        self,
        emissions_dir: PathLike,
        source_group_mapping: dict[int, tuple[Substance, Category]] | None = None,
        crs: str | int = WGS84,
    ) -> None:

        emissions_dir = Path(emissions_dir)
        if not emissions_dir.is_dir():
            raise ValueError(f"{emissions_dir=} is not a directory")
        self.path = emissions_dir

        self.file_points = self.path / "point.dat"
        self.file_lines = self.path / "line.dat"
        self.file_cadastre = self.path / "cadastre.dat"
        self.file_portals = self.path / "portals.dat"

        group_mapping_file = self.path / "source_groups.json"

        super().__init__()

        # Read the source group mapping
        if source_group_mapping is None:
            if not group_mapping_file.is_file():
                raise FileNotFoundError(
                    f"{group_mapping_file=} not found. Please provide"
                    " 'source_group_mapping' of source groups to substances and"
                    " categories. Or generate the file."
                )
            with open(group_mapping_file) as f:
                source_group_mapping = json.load(f)
                self.logger.debug(f"From {group_mapping_file}: {source_group_mapping=}")
                # Convert the keys to int
                source_group_mapping = {
                    int(k): v for k, v in source_group_mapping.items()
                }
        else:
            if group_mapping_file.is_file():
                self.logger.warning(
                    f"{group_mapping_file=} exist, but 'source_group_mapping' was"
                    " provided, so the file will be ignored."
                )

        self.source_group_mapping = source_group_mapping
        self.gdf = None
        self.gdfs = {}

        # assign the crs
        self._requested_crs = crs

        # Now we can read the files
        self._read_points()
        self._read_lines()
        self._read_cadastre()
        self._read_portals()

    def _get_sub_cat(self, source_group: int) -> tuple[Substance, Category]:
        """Get the substance and category for a source group."""
        if source_group not in self.source_group_mapping:
            raise ValueError(
                f"{source_group=} not found in {self.source_group_mapping=}"
            )
        sub_cat = self.source_group_mapping[source_group]

        if isinstance(sub_cat, str):
            substance = "unknown"
            category = sub_cat
            sub_cat = (substance, category)

        return sub_cat

    def _add_gdf(
        self, gdf: gpd.GeoDataFrame, substance: Substance, category: Category
    ) -> None:
        """Add a GeoDataFrame to the inventory."""
        if category not in self.gdfs:
            self.gdfs[category] = gdf
        else:
            # Add to the existing GeoDataFrame
            # Set the values on missing columns to 0
            missing_columns = set(self.gdfs[category].columns) - set(gdf.columns)
            for col in missing_columns:
                gdf[col] = 0
            # Also set the substance to 0 in the gdfs if not there
            if substance not in self.gdfs[category].columns:
                self.gdfs[category][substance] = 0
            # Now we can append
            self.gdfs[category] = pd.concat([self.gdfs[category], gdf], ignore_index=True)

    def _read_points(self) -> None:
        """Read the point sources."""
        if not self.file_points.is_file():
            self.logger.warning(f"{self.file_points=} not found.")
            return

        df = pd.read_csv(self.file_points, sep=",", header=1)
        # find all the source categories
        self.logger.debug(f"{df.columns=}")
        source_groups = df[df.columns[PointsCols.SOURCE_GROUP]].unique()

        for source_group in source_groups:
            substance, category = self._get_sub_cat(source_group)
            mask_source_group = df[df.columns[PointsCols.SOURCE_GROUP]] == source_group
            # Create the GeoDataFrame
            emission_col = df.columns[PointsCols.EMISSION]
            # Select only the emission column renamed to the substance
            emissions_values = df[mask_source_group].rename(
                columns={emission_col: substance}
            )[substance].to_numpy().astype(float)
            # Convert the units from kg/h to kg/y
            emissions_values *= HOUR_PER_YR
            gdf = gpd.GeoDataFrame(
                {substance: emissions_values},
                geometry=gpd.points_from_xy(
                    df.loc[mask_source_group, df.columns[PointsCols.X]],
                    df.loc[mask_source_group, df.columns[PointsCols.Y]],
                ),
                crs=self._requested_crs,
            )
            self._add_gdf(gdf, substance, category)
            # TODO: add the source information as well

    def _read_lines(self) -> None:
        """Read the line sources."""
        if not self.file_lines.is_file():
            self.logger.warning(f"{self.file_lines=} not found.")
            return

        df = pd.read_csv(self.file_lines, sep=",", header=4)
        # find all the source categories
        self.logger.debug(f"{df.columns=}")
        source_groups = df[df.columns[LinesCols.SOURCE_GROUP]].unique()

        for source_group in source_groups:
            substance, category = self._get_sub_cat(source_group)
            self.logger.debug(f"{source_group=}, {substance=}, {category=}")
            mask_source_group = df[df.columns[LinesCols.SOURCE_GROUP]] == source_group
            # Create the GeoDataFrame
            emission_col = df.columns[LinesCols.EMISSION]
            # Create geseries with the start points and end points
            gs_start = gpd.points_from_xy(
                df.loc[mask_source_group, df.columns[LinesCols.X_START]],
                df.loc[mask_source_group, df.columns[LinesCols.Y_START]],
            )
            gs_end = gpd.points_from_xy(
                df.loc[mask_source_group, df.columns[LinesCols.X_END]],
                df.loc[mask_source_group, df.columns[LinesCols.Y_END]],
            )
            # Create the linestrings
            gs_lines = gpd.GeoSeries(
                [LineString([start, end]) for start, end in zip(gs_start, gs_end)],
                crs=self._requested_crs,
            )
            self.logger.debug(f"{gs_lines=}")

            # Select only the emission column renamed to the substance
            emission_values = df[mask_source_group].rename(
                columns={emission_col: substance}
            )[substance].to_numpy()
            # Convert the units from kg/h/km to kg/y(/shape)
            line_lenghts = gs_lines.length * 1e-3
            emission_values *= HOUR_PER_YR * line_lenghts

            self.logger.debug(f"{emission_values=}")
            # Create the GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {substance: emission_values},
                geometry=gs_lines,
                crs=self._requested_crs,
            )
            self.logger.debug(f"{gdf=}")
            self._add_gdf(gdf, substance, category)

    def _read_cadastre(self) -> None:
        """Read the cadastre sources."""
        if not self.file_cadastre.is_file():
            self.logger.debug(f"{self.file_cadastre=} not found.")
            return

        df = pd.read_csv(self.file_cadastre, sep=",", header=0, index_col=False)
        self.logger.debug(f"{df}")
        # find all the source categories
        self.logger.debug(f"{df[df.columns[CadastreCols.SOURCE_GROUP]]=}")
        source_groups = df[df.columns[CadastreCols.SOURCE_GROUP]].unique()
        self.logger.debug(f"{source_groups=}")
        for source_group in source_groups:
            substance, category = self._get_sub_cat(source_group)
            self.logger.debug(f"{source_group=}, {substance=}, {category=}")
            mask_source_group = df[df.columns[CadastreCols.SOURCE_GROUP]] == source_group
            # Create the GeoDataFrame
            emission_col = df.columns[CadastreCols.EMISSION]
            # Create geseries with the start points and end points
            df_group = df.loc[mask_source_group]

            gs_sqaures = gpd.GeoSeries(
                [Polygon(((x, y), (x + x_ext, y), (x + x_ext, y + y_ext), (x, y + y_ext), (x, y))) 
                for x, y, x_ext, y_ext in zip(
                    df_group[df.columns[CadastreCols.X]],
                    df_group[df.columns[CadastreCols.Y]],
                    df_group[df.columns[CadastreCols.X_EXTENSION]],
                    df_group[df.columns[CadastreCols.Y_EXTENSION]],
                )],
                crs=self._requested_crs,
            )
            self.logger.debug(f"{gs_sqaures=}")

            # Select only the emission column renamed to the substance
            emission_values = df[mask_source_group].rename(
                columns={emission_col: substance}
            )[substance].to_numpy()
            # Convert the units from kg/h(/shape?) to kg/y(/shape)
            emission_values *= HOUR_PER_YR 

            self.logger.debug(f"{emission_values=}")
            # Create the GeoDataFrame
            gdf = gpd.GeoDataFrame(
                {substance: emission_values},
                geometry=gs_sqaures,
                crs=self._requested_crs,
            )
            self.logger.debug(f"{gdf=}")
            self._add_gdf(gdf, substance, category)



    def _read_portals(self) -> None:
        """Read the portals sources."""
        if not self.file_portals.is_file():
            self.logger.debug(f"{self.file_portals=} not found.")
            return
        else:
            self.logger.error(
                f"{self.file_portals=} found. But reading portals is not implemented"
                " yet."
            )




