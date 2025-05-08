"""Emission generation for the GRAL model: https://gral.tugraz.at/ .

This module contains functions to prepare emissions for GRAL.

Gral only has one polluant available, but can contain more than one category.

To counter that issue, we make a source group for each substance/category.


.. warning:: For PM, it seems that there are some additional parameters:
    share_pm25,share_pm10,diamter_pm30[μm],density[kg/m3],dry_dep_vel_pm25[m/s],
    dry_dep_vel_pm10[m/s],dry_dep_vel_pm30[m/s],mode

    We don't use them for now.

.. warning:: The current version of emiproc does not support lines and portals.


4 types of emissions are supported with their respective files:

- point sources: point.dat
- line sources: line.dat
- area sources: cadastre.dat
- tunnel sources: portals.dat

Each file is optional.


The following methods are applied:

- points: simply write the coordinates and the emission rate
- lines: write the coordinates of the line and the emission rate
    The Multi-lines have to be split into single lines, which increases the
    size of the problem.
- areas: polygons have to be rasterized into squares before writing them.
- tunnels: not implemented 



"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from emiproc.inventories import EmissionInfo, Inventory

if TYPE_CHECKING:
    # pygg module for gram gral processing
    from pygg.grids import GralGrid


class EmissionWriter:
    def __init__(
        self,
        path: os.PathLike,
        inventory: Inventory,
        grid: GralGrid,
        # Egde size of the rasterized polygons (crs units)
        polygon_raster_size: float = 1.0,
    ) -> None:

        # Path where to write the emissons
        self.path = Path(path)

        self.inventory = inventory
        self.grid = grid
        self.polygon_raster_size = polygon_raster_size

        # Maps the (cat/sub) to source groups
        self.source_groups = {
            (sub, cat): i * len(self.inventory.categories) + j
            for i, sub in enumerate(self.inventory.substances)
            for j, cat in enumerate(self.inventory.categories)
        }

        self._make_files()

    def _make_files(self):
        """Create empty emissions files with the headers.

        Called durint init of the class.
        """
        if not self.path.is_dir():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        # Make the filepaths
        self.file_points = self.path / "point.dat"
        self.file_lines = self.path / "line.dat"
        self.file_cadastre = self.path / "cadastre.dat"
        self.file_portals = self.path / "portals.dat"

        with open(self.file_points, "w") as f:
            # 2 lines ignored
            header = datetime.datetime.now().strftime("Generated: %d/%m/%Y %H:%M\n")
            f.write(header)
        self.points_dfs = []
        with open(self.file_lines, "w") as f:
            # 5 lines ignored
            header = (
                datetime.datetime.now().strftime("Generated: %d/%m/%Y %H:%M\n")
                + 3 * "Generated: \n"
                + "Name,Section,source_group,x1,y1,z1,x2,y2,z2,width,vert. ext.,-,-,"
                "emission_rate[kg/h/km],-,-,-,-\n"
            )
            f.write(header)
        with open(self.file_cadastre, "w") as f:
            # 1 line ignored
            header = "x,y,z,dx,dy,dz,emission_rate[kg/h],-,-,-,source_group\n"
            f.write(header)
        with open(self.file_portals, "w") as f:
            # 2 lines ignored
            header = (
                datetime.datetime.now().strftime("Generated: %d/%m/%Y %H:%M\n")
                + "x1,y1,x2,y2,z0,z1,emission_rate[kg/h],-,-,-,source_group\n"
            )
            f.write(header)

        # File to save the source groups values
        self.file_source_groups = self.path / "source_groups.json"
        with open(self.file_source_groups, "w") as f:
            # reverse the dict (items become keys and vice versa)
            reversed_source_groups = {v: k for k, v in self.source_groups.items()}
            json.dump(reversed_source_groups, f, indent=2)

    def write_gdfs(self):
        """Write the gdfs emissions to the files."""
        for cat, gdf in self.inventory.gdfs.items():
            info = self.inventory.emission_infos[cat]
            for sub in self.inventory.substances:
                source_group = self.source_groups[(sub, cat)]
                if sub not in gdf.columns:
                    continue

                mask_polygons = gdf.geom_type.isin(["Polygon", "MultiPolygon"])
                if any(mask_polygons):
                    gdf_polygons = gdf.loc[mask_polygons]
                    self._write_polygons(
                        gdf_polygons.geometry, gdf_polygons[sub], info, source_group
                    )

                mask_points = gdf.geom_type == "Point"
                if any(mask_points):
                    gdf_points = gdf.loc[mask_points]
                    self._add_points(
                        gdf_points.geometry, gdf_points[sub], info, source_group
                    )

                mask_lines = gdf.geom_type.isin(["LineString"])
                if any(mask_lines):
                    gdf_lines = gdf.loc[mask_lines]
                    self._write_lines(
                        gdf_lines.geometry, gdf_lines[sub], info, source_group
                    )

                mask_multilines = gdf.geom_type.isin(["MultiLineString"])
                if any(mask_multilines):
                    gdf_multilines = gdf.loc[mask_multilines]
                    # Split all the multilines into lines
                    for shape, shape_emission in zip(
                        gdf_multilines.geometry, gdf_multilines[sub]
                    ):
                        lenghts = np.array([line.length for line in shape.geoms])
                        proprtions = lenghts / shape.length
                        for line, prop in zip(shape.geoms, proprtions):
                            self._write_line(
                                line, shape_emission * prop, info, source_group
                            )
                mask_missing = ~(
                    mask_multilines | mask_lines | mask_points | mask_polygons
                )
                if any(mask_missing):
                    raise NotImplementedError(
                        f"Shapes of type: '{gdf.loc[mask_missing].geom_type.unique()}'"
                        " are not implemented."
                    )

        # Write all the points as a singl batch
        pd.concat(self.points_dfs).to_csv(
            self.file_points,
            mode="a",
            index=False,
        )

    def _add_points(
        self,
        shapes: gpd.GeoSeries,
        emissions: pd.Series,
        info: EmissionInfo,
        source_group: int,
    ):
        z = info.height
        if info.height_over_buildings:
            z += self.grid.building_heights[self.grid.get_index(shapes.x, shapes.y)]
        self.points_dfs.append(
            pd.DataFrame(
                {
                    "x": shapes.x,
                    "y": shapes.y,
                    "z": z,
                    "emission[kg/h]": emissions,
                    # Fill the unused columsn with zeros as GRAL cannot handle empty columns
                    "unused_0": 0,
                    "unused_1": 0,
                    "unused_2": 0,
                    "exit_velocity[m/s]": info.speed,
                    "diameter[m]": info.width,
                    "temperature[K]": info.temperature,
                    "source_group": source_group,
                }
            )
        )

    def _write_lines(
        self,
        shapes: gpd.GeoSeries,
        emissions: pd.Series,
        info: EmissionInfo,
        source_group: int,
    ):
        for shape, emission in zip(shapes, emissions):
            self._write_line(shape, emission, info, source_group)

    def _write_line(
        self,
        shape: LineString,
        emission: float,
        info: EmissionInfo,
        source_group: int,
    ):
        """Write a line source to the file."""
        # split the line string in single lines
        previous_coord = shape.coords[0]

        lines = [LineString([previous_coord, coord]) for coord in shape.coords[1:]]

        # split the emission between the lines based on the length
        line_lengths = np.array([line.length for line in lines])
        line_emission_ratios = line_lengths / line_lengths.sum()
        line_emissions = line_emission_ratios * emission

        for i, (line, line_emission) in enumerate(zip(lines, line_emissions)):
            self._write_staight_line(
                Point(line.coords[0]),
                Point(line.coords[1]),
                line_emission,
                info,
                source_group,
                section=i,
            )

    def _write_staight_line(
        self,
        start: Point,
        end: Point,
        emission: float,
        info: EmissionInfo,
        source_group: int,
        section: int,  # Section number of the line
    ):
        """Write a straight line source to the file."""
        z_start = info.height
        z_end = info.height
        if info.height_over_buildings:
            z_start += self.grid.building_heights[self.grid.get_index(start.x, start.y)]
            z_end += self.grid.building_heights[self.grid.get_index(end.x, end.y)]

        with open(self.file_lines, "a") as f:
            # Write the line
            f.write(
                f"unnamed,{section},{source_group},{start.x},{start.y},{z_start},"
                f"{end.x},{end.y},{z_end},{info.width},-{info.vertical_extension},0,0,"
                f"{emission},0,0,0,0\n"
            )

    def _write_polygons(
        self,
        shapes: Iterable[Polygon],
        emissions: Iterable[float],
        info: EmissionInfo,
        source_group: int,
    ):
        """Write a polygon source to the file."""

        # Rasterize the polygon on a grid
        shapes_serie = gpd.GeoSeries(shapes)
        # get polygon bounds
        minx, miny, maxx, maxy = shapes_serie.total_bounds
        # Create a grid for the rasterization
        x = np.arange(minx, maxx, self.polygon_raster_size)
        y = np.arange(miny, maxy, self.polygon_raster_size)

        # Get the emission per cell
        average_cells_proportion = (self.polygon_raster_size**2) / shapes_serie.area
        cell_emissions = np.array(emissions) * average_cells_proportion

        # WARNING: this might be not exactly mass convserving
        rasterized_emissions = rasterize(
            shapes=zip(shapes, cell_emissions),
            out_shape=(len(x), len(y)),
            transform=from_bounds(minx, miny, maxx, maxy, len(x), len(y)),
            all_touched=False,
            merge_alg=MergeAlg.add,
        )[
            ::-1, :
        ]  # flip the y axis

        # Get the coordinates of the rasterized polygon
        indices = np.array(np.where(rasterized_emissions)).T

        # Write the polygon
        with open(self.file_cadastre, "a") as f:
            for i_x, i_y in indices:
                f.write(
                    f"{x[i_x]},{y[i_y]},{info.height},"
                    f"{self.polygon_raster_size},{self.polygon_raster_size},{info.vertical_extension},"
                    f"{rasterized_emissions[i_x, i_y]},0,0,0,{source_group},\n"
                )


def export_to_gral(
    inventory: Inventory,
    grid: GralGrid,
    path: os.PathLike,
    polygon_raster_size: float = 1.0,
) -> None:
    """Export an inventory to GRAL.

    .. note:: This requires the external python package `pygg` to be installed.

    :param inventory: Inventory to export.
    :param path: Path where to write the emissions.
    :param grid: Grid to use.
    """

    writer = EmissionWriter(Path(path), inventory, grid, polygon_raster_size)

    writer.write_gdfs()


if __name__ == "__main__":
    # Small test inventory for saving to gral format
    import logging

    import geopandas as gpd
    import numpy as np
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPolygon,
        Point,
        Polygon,
    )

    from emiproc.exports.gral import export_to_gral
    from emiproc.inventories import EmissionInfo, Inventory
    from emiproc.tests_utils import TEST_OUTPUTS_DIR
    from emiproc.tests_utils.test_inventories import inv_with_pnt_sources

    gral_test_inv = inv_with_pnt_sources.copy()
    # Addintg some shapes that should be supported
    gral_test_inv.gdfs["adf"] = gpd.GeoDataFrame(
        {
            "CO2": [1, 2, 1, 3, 1],
            "CH4": [3, 1, 1, 1, 1],
        },
        geometry=[
            LineString([(0.75, 0.75), (1.25, 1.25), (2.25, 1.25)]),
            Polygon([(0.25, 0.25), (0.75, 0.75), (0.25, 0.75)]),
            MultiPolygon(
                [
                    # Square from 2 to 4
                    Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
                    # Sqare from 5 to 6
                    Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
                    # A triangle from 8 to 12
                    Polygon([(8, 8), (12, 8), (12, 12)]),
                ]
            ),
            MultiLineString(
                [
                    LineString([(0.25, 0.25), (0.75, 0.75), (3.25, 1.25)]),
                    LineString([(0.75, 0.75), (0.5, 0.25)]),
                ]
            ),
            Point(0.5, 0.5),
        ],
    )

    from pygg.grids import GralGrid

    inv = gral_test_inv

    grid = GralGrid(
        dz0=1.5,
        stretch=0.5,
        nx=20,
        ny=20,
        nslice=20,
        sourcegroups="",
        xmin=0,
        xmax=10,
        ymin=0,
        ymax=4,
        latitude=0,
        crs=inv.crs,
    )
    grid.building_heights = np.zeros(size=(grid.ny, grid.nx))

    inv.emission_infos = {
        "adf": EmissionInfo(),
        "blek": EmissionInfo(),
        "liku": EmissionInfo(),
        "other": EmissionInfo(),
        "test": EmissionInfo(),
    }

    export_to_gral(inv, grid, TEST_OUTPUTS_DIR, 1)
