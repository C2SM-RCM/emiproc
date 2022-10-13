from __future__ import annotations
from os import PathLike
from pathlib import Path
from emiproc.grids import LV95, Grid, SwissGrid
from emiproc.inventories import Inventory
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from emiproc.inventories.utils import load_category
import numpy as np
import rasterio


class SwissRasters(Inventory):
    """An inventory of Switzerland based on swiss rasters."""

    edge_size: int = 100
    grid: SwissGrid
    df_eipwp: gpd.GeoDataFrame
    df_emission: pd.DataFrame

    def __init__(
        self,
        data_path: Path,
        rasters_dir: PathLike,
        rasters_str_dir: PathLike,
        requires_grid: bool = True,
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
        :arg requires_grid: Whether the grid should be created as well.
            Creating the shapes for the swiss grid is quite expensive process.
            Most of the weights for remapping can be cached so if you
            have them generated already, set that to false.
        """
        super().__init__()

        # Load the file with the point sources
        df_eipwp = load_category(
            data_path / "ekat_ch_basisraster.gdb" / "ekat_ch_basisraster.gdb",
            "eipwp" + "_2015",
        )
        cols_eipwp = {
            "CO2_15": "CO2",
            "CH4_15": "CH4",
            "N2O_15": "N2O",
            "NOx_15": "NOx",
            "CO_15": "CO",
            "NMVOC_15": "VOC",
            "SO2_15": "SO2",
            "NH3_15": "NH3",
        }

        df_eipwp = df_eipwp.rename(columns=cols_eipwp)
        for col in cols_eipwp.values():
            # t/y -> kg/y
            df_eipwp[col] *= 1000
            df_eipwp.loc[pd.isna(df_eipwp[col]), col] = 0.
        df_eipwp["F-Gase"] = 0.0

        # Load the excel sheet with the total emissions
        df_emissions = pd.read_excel(
            data_path / "Emissionen-2015-je-Emittentengruppe.xlsx",
            header=2,
            index_col="Basisraster",
            usecols=[5, 6, 7, 8, 10, 11, 12, 13, 14, 16],
        )
        df_emissions = df_emissions.rename(columns={"CO2 foss/geog": "CO2"})
        df_emissions = df_emissions.loc[~pd.isna(df_emissions.index)]

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

        self.df_eipwp = df_eipwp[
            [
                "CO2",
                "CH4",
                "N2O",
                "NOx",
                "CO",
                "VOC",
                "SO2",
                "NH3",
                "F-Gase",
                "geometry",
            ]
        ]
        self.df_emission = df_emissions

        self._substances = df_emissions.columns.to_list()

        self.requires_grid = requires_grid

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
                crs=LV95,
                geometry=(
                    [
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
                    ]
                    if self.requires_grid
                    else np.full(self.grid.nx * self.grid.ny, np.nan)
                ),
            )
            mapping = {}
            # Loading all the raster categories
            for raster_file, category in zip(
                self.all_raster_files, self.raster_categories
            ):
                _raster_array = self.load_raster(raster_file).reshape(-1)
                if "_" in category:
                    cat, sub = category.split("_")
                    sub_name = sub.upper()
                    if sub_name == "NOX":
                        sub_name = "NOx"
                    # t/ y -> kg/y
                    # These rasters don't need the mutliplication by factor
                    mapping[(cat, sub_name)] = _raster_array * 1000
                else:
                    for sub in self._substances:
                        # emissions are in t/ y in the file -> kg/y
                        factor = self.df_emission.loc[category, sub] * 1000
                        if factor:
                            mapping[(category, sub)] = _raster_array * factor

            self._gdf = gpd.GeoDataFrame(
                # This vector is same as raster data reshaped using reshape(-1)
                data=mapping,
                crs=LV95,
                geometry=[
                    Polygon(
                        (
                            (x, y),
                            (x, y + self.grid.dy),
                            (x + self.grid.dx, y + self.grid.dy),
                            (x + self.grid.dx, y),
                        )
                    )
                    if self.requires_grid
                    else None
                    for y in reversed(ys)
                    for x in xs
                ],
            )
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

    def get_emissions(self, category: str, substance: str):

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
