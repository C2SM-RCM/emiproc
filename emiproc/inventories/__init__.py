"""Inventories of emissions."""
from __future__ import annotations
from enum import Enum, auto
from os import PathLike
from pathlib import Path
from emiproc.grids import LV95, Grid, SwissGrid
from emiproc.regrid import get_weights_mapping, weights_remap
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import rasterio


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
    :attr gdfs: Some inventories are given on more than one grid.
        For example, :py:class:`MapLuftZurich` is given on a grid
        where every category has different shape file.
        In this case gdf must be set to None and gdfs will be
        a dictionnary mapping only the categories desired.
    :attr history: Stores all the operations that happened to this inventory.

    .. note::
        If your data contains point sources, the data on them must be stored in
        the gdfs, as :attr:`gdf` is only valid for the inventory grid.
        A gdf should contain only point sources.

    """

    name: str
    grid: Grid
    substances: list[Substance]
    categories: list[str]
    gdf: gpd.GeoDataFrame | None
    gdfs: dict[str, gpd.GeoDataFrame] | None
    geometry: gpd.GeoSeries

    history: list[str]

    def __init__(self) -> None:
        self.history = [f"Created as {type(self).__name__}"]

    @property
    def geometry(self) -> gpd.GeoSeries:
        return self.gdf.geometry

    @property
    def categories(self) -> list[str]:
        return list(
            set(
                [
                    cat
                    for cat, _ in self.gdf.columns
                    if not isinstance(self.gdf[(cat, _)].dtype, gpd.array.GeometryDtype)
                ]
            )
            | set(self.gdfs.keys())
        )

    def copy(self, no_gdfs: bool = False) -> Inventory:
        """Copy the inventory."""
        inv = Inventory()
        inv.__class__ = self.__class__
        inv.history = self.history.copy()
        if hasattr(self, "grid"):
            inv.grid = self.grid

        if not no_gdfs:
            if self.gdf is not None:
                inv.gdf = self.gdf.copy(deep=True)
            else:
                inv.gdf = None
            if self.gdfs is not None:
                inv.gdfs = {key: gdf.copy(deep=True) for key, gdf in self.gdfs.items()}
            else:
                inv.gdfs = None

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
        gdf: gpd.GeoDataFrame,
        name: str = "custom_from_gdf",
        gdfs: dict[str, gpd.GeoDataFrame] = {},
    ) -> Inventory:
        """The gdf must be a two level gdf with (category, substance)."""
        inv = Inventory()
        inv.name = name
        inv.gdf = gdf
        inv.gdfs = gdfs

        return inv


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

        self.all_raster_files = normal_rasters + str_rasters

        self.raster_categories = [r.stem for r in normal_rasters] + [
            r.stem[:-2] for r in str_rasters
        ]
        # All categories
        self.categories = self.raster_categories + ["eipwp"]

        self.df_eipwp = df_eipwp
        self.df_emission = df_emission

        self.substances = df_emission.columns.to_list()

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

            # Loading all the raster categories
            for raster_file, category in zip(
                self.all_raster_files, self.raster_categories
            ):
                self._gdf[category] = self.load_raster(raster_file).reshape(-1)
            # Finally add the point sources
            self.gdfs = {}
            self.gdfs["eipwp"] = self.df_eipwp

        return self._gdf

    @gdf.setter
    def gdf(self, gdf=gpd.GeoDataFrame):
        self._gdf = gdf

    def load_raster(self, raster_file: Path) -> np.ndarray:
        # Load and save as npy fast reading format
        if raster_file.with_suffix(".npy").exists():
            inventory_field = np.load(raster_file.with_suffix(".npy"))
        else:
            print(f"Parsing {raster_file}")
            src = rasterio.open(raster_file).read(1)
            np.save(raster_file.with_suffix(".npy"), src)
            inventory_field = src
        return inventory_field

    def get_emissions(self, category: str, substance: Substance):

        if category in self.gdf:
            # Scale the category with the emission factor
            return self.gdf[category] * self.df_emission.loc[category, substance]
        if category in self.gdfs.keys():
            # Point sources
            return super().get_emissions(category, substance)
        raise IndexError(f"Nothing found in zh inventory for {category}, {substance}")

    def copy(self, no_gdfs: bool = False) -> Inventory:
        inv = super().copy(no_gdfs)
        inv.df_eipwp = self.df_eipwp
        inv.df_emission = self.df_emission
        return inv


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


if __name__ == "__main__":
    test_inv = Inventory()
