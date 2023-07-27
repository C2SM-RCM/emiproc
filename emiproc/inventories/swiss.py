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

        # Emission data file
        total_emission_file = (
            data_path / "Emissions_CH_test.xlsx"
        )

        # Load excel sheet with the total emissions (excluding point sources)
        df_emissions = pd.read_excel(total_emission_file)

        # Dictionary to rename chemical species according to emiproc conventions
        dict_spec = {
            "NOX": "NOx",
            "NMVOC": "VOC",
            "PM2.5": "PM25",
            "F-Gase": "F-gases"
        }

        # Rename chemical specis according to emiproc conventions
        df_emissions['Chemical Species'] = df_emissions['Chemical Species'].replace(dict_spec) 

        # Add indexing column consisting of both grid and species' name
        df_emissions['Grid_Spec'] = df_emissions['Grids'] + '_' + df_emissions['Chemical Species']
        df_emissions = df_emissions.set_index('Grid_Spec')

        # Check if selected year is in dataset 
        if year not in df_emissions.columns:
            raise ValueError("Selected year not in dataset.")
        
        # ---------------------------------------------------------------------
        # -- Emissions from point sources
        # ---------------------------------------------------------------------

        # Initialize dataframe with columns=chemical species and rows=point sources
        df_eipwp = pd.DataFrame()

        # Load data
        df_eipwp_ori = pd.read_excel(
            total_emission_file,
            sheet_name='Point Sources',
        )
        df_loc_ps = pd.read_excel(
            total_emission_file,
            sheet_name='Location Point Sources'
        )

        # Check if selected year is in dataset 
        if year not in df_eipwp_ori.columns:
            raise ValueError("Selected year not in dataset.")

        # Set indexing column 
        df_eipwp_ori = df_eipwp_ori.set_index('Chemical Species')

        # Location of point sources 
        easting = df_loc_ps['Easting'].tolist()
        northing = df_loc_ps['Northing'].tolist()
        points = [Point(x,y) for x,y in zip(easting,northing)]

        # Species with point source emissions 
        # Rmk: *set() removes duplicates
        spec_emis_ps =  [*set(df_eipwp_ori.index.tolist())]

        # Set "k.A." and NaN-values to zero
        #df_eipwp_ori[year] = df_eipwp_ori[year].replace({"k.A.":0})
        df_eipwp_ori[year] = df_eipwp_ori[year].fillna(0)

        # Extract emisssions for each chemical species
        for sub in spec_emis_ps:
            # Transform units [kt/y] -> [kg/y]
            factors = df_eipwp_ori[year].loc[sub] * 1e6
            df_eipwp[sub] = factors.tolist()
        
        # Rename columns according to emiproc convention
        df_eipwp = df_eipwp.rename(columns=dict_spec)

        # Store as GeoDataFrame
        df_eipwp = gpd.GeoDataFrame(df_eipwp, geometry=points, crs=LV95)
        self.df_eipwp = df_eipwp

        #self.df_eipwp = df_eipwp[
        #    [
        #        "CO2",
        #        "CH4",
        #        "N2O",
        #        "NOx",
        #        "CO",
        #        "VOC",
        #        "SO2",
        #        "NH3",
        #        "F-gases",
        #        "geometry",
        #    ]
        #]

        # ---------------------------------------------------------------------
        # -- Grids
        # ---------------------------------------------------------------------

        rasters_dir = Path(rasters_dir)

        # List with Raster categories for which we have emissions
        raster_sub = df_emissions.index.tolist()
        rasters = []
        for t in raster_sub:
            cat, sub = t.split("_")
            subname = sub.lower()
            if cat == 'evstr':
                # Grid for non-methane VOCs is named "evstr_nmvoc"
                if subname == 'voc':
                    subname = 'nmvoc'
                rasters.append(cat + '_'+ subname)
            else:
                rasters.append(cat)
        # Remove duplicates
        rasters = [*set(rasters)]

        # Grids that depend on chemical species (road transport)
        rasters_str_dir = Path(rasters_str_dir)
        str_rasters = [
            r
            for r in rasters_str_dir.rglob("*.asc")
            # Don't include the tunnel specific grids as they are already included in the grids for road transport
            if "_tun" not in r.stem
        ]

        # Grids that do not depend on chemical species    
        normal_rasters = [r 
                          for r in rasters_dir.rglob("*.asc") 
                          ]

        self.all_raster_files = normal_rasters + str_rasters

        self.raster_categories = [r.stem for r in normal_rasters] + [
            r.stem for r in str_rasters
        ]

        # ---------------------------------------------------------------------
        # -- Emissions without point sources
        # ---------------------------------------------------------------------

        self.df_emission = df_emissions
        self.requires_grid = requires_grid

        # Fill NaN values with zeros
        self.df_emission[year] = self.df_emission[year].fillna(0)

        # List with chemical species 
        self._substances =  [*set(self.df_emission['Chemical Species'].tolist())]

        # Grid on which the inventory is created
        self.grid = SwissGrid(
            "ch_emissions",
            nx=3600,
            ny=2400,
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

        # Loading Raster categories and assigning respective emissions
        for raster_file, category in zip(self.all_raster_files, self.raster_categories):
            _raster_array = self.load_raster(raster_file).reshape(-1)
            if "_" in category:
                cat, sub = category.split("_")
                sub_name = sub.upper()
                if sub_name in dict_spec.keys():
                    sub_name = dict_spec[sub_name]
                idx = cat + '_' + sub_name
                # Transform units [t/y] -> [kg/y]
                factor = self.df_emission[year].loc[idx] * 1000
                # Normalize the array to ensure the factor will be the sum
                # Note: this is to ensure consistency if the data provider
                # change the df_emission values in the future but not the rasters
                _normalized_raster_array = _raster_array / _raster_array.sum()
                mapping[(cat, sub_name)] = _normalized_raster_array * factor
            else:
                for sub in self._substances:
                    idx = category + '_' + sub
                    # transform units [t/y] -> [kg/y]
                    factor = self.df_emission[year].loc[idx] * 1000
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

        # Add point sources
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
    swiss_data_path =  Path(r"/users/ckeller/emission_data/CH_Emissions")

    inv_ch = SwissRasters(
        data_path=swiss_data_path,
        rasters_dir=swiss_data_path / "ekat_gridascii_test",
        rasters_str_dir=swiss_data_path / "ekat_str_gridascii_test",
        requires_grid=True,
        year=2015,
    )