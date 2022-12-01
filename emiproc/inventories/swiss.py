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
        year: int = 2015,
    ) -> None:
        """Create a swiss raster inventory.

        :arg rasters_dir: The folder where the rasters are found.
        :arg rasters_str_dir: The folder where the rasters pro substance are found.
        :arg df_eipwp: The geodataframe contaning the point sources.
            Must have a column for each substance  in [kg/y]
            and have the geometry column
            containing the point sources.
        :arg df_emission: A dataframe, where columns are the substances
            and rows are the rasters name.
            The data is the total emission for the category in [kg/y].
        :arg requires_grid: Whether the grid should be created as well.
            Creating the shapes for the swiss grid is quite expensive process.
            Most of the weights for remapping can be cached so if you
            have them generated already, set that to false.
        :arg year: The year of the inventory that should be used.
            Currently accepted 2015 or 2020.
            The raster files are the same for both years. Only the scaling
            of the full raster pro substance changes.
            The original rasters were made for year 2015.
        """
        super().__init__()

        valid_years = [2015, 2020]
        if year not in valid_years:
            raise ValueError(f"year must be one of {valid_years}.")

        total_emission_file = (
            data_path / "Emissionen-2015-2020-je-Emittentengruppe_2022-11-02.xlsx"
        )

        # Load the file with the point sources
        header_of_year = {
            2015: 5,
            2020: 33,
        }
        df_eipwp = pd.read_excel(
            total_emission_file,
            sheet_name='Punktquellen',
            nrows=20,
            header=header_of_year[year],
        )
        # The following way of loading, using the gdb
        # was replaced by reading from the spreadsheet
        # also the units now cahnged in the excel spreadsheet
        ##df_eipwp = load_category(
        ##    data_path / "ekat_ch_basisraster.gdb" / "ekat_ch_basisraster.gdb",
        ##    "eipwp" + "_2015",
        ##)
        ##cols_eipwp = {
        ##    "CO2_15": "CO2",
        ##    "CH4_15": "CH4",
        ##    "N2O_15": "N2O",
        ##    "NOx_15": "NOx",
        ##    "CO_15": "CO",
        ##    "NMVOC_15": "VOC",
        ##    "SO2_15": "SO2",
        ##    "NH3_15": "NH3",
        ##}
        cols_eipwp = {
            "CO2": "CO2",
            "CH4": "CH4",
            "N2O": "N2O",
            "NOx": "NOx",
            "CO": "CO",
            "NMVOC": "VOC",
            "SO2": "SO2",
            "NH3": "NH3",
        }
        
        df_eipwp = df_eipwp.rename(columns=cols_eipwp)
        for col in cols_eipwp.values():
            serie = df_eipwp[col].copy()
            serie.loc[serie=="k.A."] = 0.0
            serie.loc[pd.isna(serie)] = 0.0
            serie = serie.astype(float)
            # kt/y -> kg/y  # This was changed since the reading from the gdb
            serie *= 1e6
            df_eipwp[col] = serie
        df_eipwp["F-gases"] = 0.0

        # Point location hardcoded as not in the excel sheet
        # They were taken from the points data given in the gdb
        points = [
            Point(2634640.000, 1127354.000),
            Point(2563500.000, 1122500.000),
            Point(2712500.000, 1137900.000),
            Point(2495114.000, 1118010.000),
            Point(2533117.000, 1172768.000),
            Point(2563500.000, 1122500.000),
            Point(2569384.000, 1209968.000),
            Point(2649655.000, 1212590.000),
            Point(2497000.000, 1118130.000),
            Point(2751520.000, 1188050.000),
            Point(2644825.000, 1215313.000),
            Point(2608999.000, 1219558.000),
            Point(2640650.000, 1266464.000),
            Point(2634150.000, 1127975.000),
            Point(2729835.000, 1278930.000),
            Point(2708128.000, 1268256.000),
            Point(2587754.000, 1209929.000),
            Point(2609500.000, 1224900.000),
            Point(2624370.000, 1128968.000),
            Point(2662982.000, 1213561.000),
        ]
        df_eipwp = gpd.GeoDataFrame(df_eipwp, geometry=points, crs=LV95)

        columns_of_year = {
            # Hard coded columns that should be read in the excel sheet of total emissions
            2015: [5, 6, 7, 8, 10, 11, 12, 13, 14, 27],
            2020: [16, 17, 18, 19, 21, 22, 23, 24, 25, 27],
        }

        # Load the excel sheet with the total emissions
        df_emissions = pd.read_excel(
            total_emission_file,
            header=7,
            index_col="Basisraster",
            usecols=columns_of_year[year],
        )
        # Handle the duplicate column names of the file, by removing the .1 str that is added by pandas
        df_emissions.columns = [
            col[:-2] if col[-2:] == ".1" else col for col in df_emissions.columns
        ]
        df_emissions = df_emissions.rename(
            columns={
                "CO2 foss/geog": "CO2",
                "NMVOC": "VOC",
                "F-Gase": "F-gases",
            }
        )
        # Remove empty lines at the end of the document
        df_emissions = df_emissions.loc[~pd.isna(df_emissions.index)]

        rasters_dir = Path(rasters_dir)
        normal_rasters = [r for r in rasters_dir.rglob("*.asc")]
        # Rasters with emission
        rasters_str_dir = Path(rasters_str_dir)
        str_rasters = [
            r
            for r in rasters_str_dir.rglob("*.asc")
            # Don't include the tunnel specific rasters as already included in the files
            if "_tun" not in r.stem
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
                "F-gases",
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

        mapping = {}

        # Loading all the raster categories
        for raster_file, category in zip(self.all_raster_files, self.raster_categories):
            _raster_array = self.load_raster(raster_file).reshape(-1)
            if "_" in category:
                # this is for the rasters of the traffic category
                # each substance has a different raster
                cat, sub = category.split("_")
                sub_name = sub.upper()
                if sub_name == "NOX":
                    sub_name = "NOx"
                if sub_name == "NMVOC":
                    sub_name = "VOC"
                # t/ y -> kg/y
                factor = self.df_emission.loc[category, sub_name] * 1000
                # Normalize the array to ensure the factor will be the sum
                # Note: this is to ensure consistency if the data provider
                # change the df_emission values in the future but not the rasters
                _normalized_raster_array = _raster_array / _raster_array.sum()
                mapping[(cat, sub_name)] = _normalized_raster_array * factor
            else:
                for sub in self._substances:
                    # emissions are in t/ y in the file -> kg/y
                    factor = self.df_emission.loc[category, sub] * 1000
                    if factor > 0:
                        mapping[(category, sub)] = _raster_array * factor

        self.gdf = gpd.GeoDataFrame(
            mapping,
            crs=LV95,
            # This vector is same as raster data reshaped using reshape(-1)
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
        # Finally add the point sources
        self.gdfs = {}
        self.gdfs["eipwp"] = self.df_eipwp

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

if __name__ == "__main__":
    swiss_data_path =  Path(r"C:\Users\coli\Documents\ZH-CH-emission\Data\CHEmissionen")

    inv_ch = SwissRasters(
        data_path=swiss_data_path,
        rasters_dir=swiss_data_path / "ekat_gridascii",
        rasters_str_dir=swiss_data_path / "ekat_str_gridascii",
        requires_grid=False,
        year=2020,
    )