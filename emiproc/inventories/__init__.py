"""Inventories of emissions."""

from enum import Enum, auto
from os import PathLike
from pathlib import Path
from emiproc.grids import LV95, Grid, SwissGrid
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np


class Substance(Enum):
    """Represent a substance that is emitted and can be present in a dataset."""

    def _generate_next_value_(name, start, count, last_values):
        # Auto assign name from the enum variable
        return name

    CO2 = auto()
    CO = auto()
    SO2 = auto()
    NO2 = auto()
    NOx = auto()
    N2O = auto()
    NH3 = auto()
    CH4 = auto()
    VOC = auto()


class Inventory:
    """Base class for inventories.

    :attr name: The name of the inventory. This is going to be used
        for adding metadata to the output files, and also for the reggridding
        weights files.
    :attr grid: The grid on which the inventory is.
    :attr substances: The :py:class:`Substance` present in this inventory.
    :attr categories: List of the categories present in the inventory.
    :attr gdf: The GeoPandas DataFrame that represent the whole inventory.
        The geometry column contains all the grid cells.
        The other columns should contain the emission value for the substances
        and the categories.
    """

    name: str
    grid: Grid
    substances: list[Substance]
    categories: list[str]
    gdf: gpd.GeoDataFrame


class SwissRasters(Inventory):
    """An inventory of Switzerland based on swiss rasters."""

    edge_size: int = 100
    grid: SwissGrid

    def __init__(
        self,
        rasters_dir: PathLike,
        rasters_str_dir: PathLike,
        df_eipwp: gpd.GeoDataFrame,
        df_emission: pd.DataFrame,
    ) -> None:
        """Create a swiss raster inventory.

        :arg rasters_dir: The folder where the rasters are found.
        :arg rasters_str_dir: The folder where the rasters pro substance are found.
        :arg df_eipwp: The geodataframe contaning the point sources.
            Must have a column for each substance  in [kg/y]
            and have the gemoetry column
            containing the point sources.
        :arg df_emission: A dataframe, where columns are the substances
            and rows are the rasters name.
            The data is the total emission for the category in [kg/y].
        """
        super().__init__()

        rasters_dir = Path(rasters_dir)
        normal_rasters = [r for r in rasters_dir.rglob("*.asc")]
        # Rasters with emission
        rasters_str_dir = Path(rasters_str_dir)
        str_rasters = [
            r for r in rasters_str_dir.rglob("*.asc") if "_tun" not in r.stem
        ]

        all_rasters = normal_rasters + str_rasters

        raster_categories = [r.stem for r in normal_rasters] + [
            r.stem[:-2] for r in str_rasters
        ]
        # All categories
        self.categories = raster_categories + ["eipwp"]

        self.df_eipwp = df_eipwp
        self.df_emission = df_emission

        # Grid on which the inventory is created
        self.grid = SwissGrid(
            "ch_emissions",
            3600,
            2400,
            xmin=2480000,
            ymin=1060000,
            dx=self.edge_size,
            dy=self.edge_size,
            crs=LV95,
        )

    # Make this a property as creating it is expensive
    @property
    def gdf(self) -> gpd.GeoDataFrame:
        if not hasattr(self, "_gdf"):
            xs = np.arange(
                self.grid.xmin,
                self.grid.xmin + self.grid.nx * self.grid.dx,
                step=self.grid.dx,
            )
            ys = np.arange(
                self.grid.ymin,
                self.grid.ymin + self.grid.ny * self.grid.dy,
                step=self.grid.dy,
            )
            self._gdf = gpd.GeoDataFrame(
                # This vector is same as raster data reshaped using reshape(-1)
                geometry=[
                    Polygon(
                        (
                            (x, y),
                            (x, y + self.grid.dy),
                            (x + self.grid.dx, y + self.grid.dy),
                            (x + self.grid.dx, y),
                        )
                    )
                    for y in reversed(ys)
                    for x in xs
                ],
                crs=LV95,
            )

        return self._gdf


class MapLuftZurich(Inventory):

    mapluft_gdb: Path

    def __init__(self, mapluft_gdb: PathLike) -> None:
        self.mapluft_gdb = Path(mapluft_gdb)

        super().__init__()


class EmiprocNetCDF(Inventory):
    """An output from emiproc.

    Useful if you need to process again an inventory.
    """

    def __init__(self, file: PathLike) -> None:
        super().__init__()
