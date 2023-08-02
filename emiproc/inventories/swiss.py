from __future__ import annotations
from os import PathLike
from pathlib import Path
from emiproc.grids import LV95, SwissGrid
from emiproc.inventories import Inventory
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
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

        :arg data_path: Folder containing the data.
            A file called `Emissions_CH.xlsx` must be present in this folder.
            It contains total emissionsfor each category/substance
            for different years.
        :arg rasters_dir: The folder where the rasters are found.
        :arg rasters_str_dir: The folder where the rasters pro substance are found.
        :arg requires_grid: Whether the grid should be created as well.
            Creating the shapes for the swiss grid is quite expensive process.
            Most of the weights for remapping can be cached so if you
            have them generated already, set that to false.
        :arg year: The year of the inventory that should be used.
            This should be present in the `Emissions_CH.xlsx` file.
            The raster files are the same for all years. Only the scaling
            of the full raster pro substance changes.
        """
        super().__init__()

        # Emission data file
        total_emission_file = data_path / "Emissions_CH.xlsx"

        # Load excel sheet with the total emissions (excluding point sources)
        df_emissions = pd.read_excel(total_emission_file)

        # Dictionary to rename chemical species according to emiproc conventions
        dict_spec = {"NOX": "NOx", "NMVOC": "VOC", "PM2.5": "PM25", "F-Gase": "F-gases"}

        # Rename chemical specis according to emiproc conventions
        df_emissions["Chemical Species"] = df_emissions["Chemical Species"].replace(
            dict_spec
        )

        # Add indexing column consisting of both grid and species' name
        df_emissions["Grid_Spec"] = (
            df_emissions["Grids"] + "_" + df_emissions["Chemical Species"]
        )
        df_emissions = df_emissions.set_index("Grid_Spec")

        # Check if selected year is in dataset
        if year not in df_emissions.columns:
            raise ValueError("Selected year not in dataset.")

        # ---------------------------------------------------------------------
        # -- Emissions from point sources
        # ---------------------------------------------------------------------

        # Load data
        df_eipwp_ori = pd.read_excel(
            total_emission_file,
            sheet_name="Point Sources",
        )
        df_loc_ps = pd.read_excel(
            total_emission_file, sheet_name="Location Point Sources"
        )

        # Check if selected year is in dataset
        if year not in df_eipwp_ori.columns:
            raise ValueError("Selected year not in dataset.")

        # Add indexing column consisting of the company name and the species' name
        df_eipwp_ori["Comp_Spec"] = (
            df_eipwp_ori["Company"] + "_" + df_eipwp_ori["Chemical Species"]
        )
        df_eipwp_ori = df_eipwp_ori.set_index("Comp_Spec")

        # Set company name as index for location of point sources
        df_loc_ps = df_loc_ps.set_index("Company")

        # Companies with point source emissions
        # Rmk: *set() removes duplicates
        comp_ps = [*set(df_eipwp_ori["Company"].tolist())]

        # Species with point source emissions
        spec_ps = [*set(df_eipwp_ori["Chemical Species"].tolist())]

        # Set NaN-values to zero
        df_eipwp_ori[year] = df_eipwp_ori[year].fillna(0)

        # Initialize dataframe with columns=chemical species
        df_eipwp = pd.DataFrame(columns=spec_ps)

        # Extract location of point sources together with its emissions for each chemical species
        points = []
        for comp in comp_ps:
            lst_row = []
            # Location of point source
            x = df_loc_ps["Easting"].loc[comp]
            y = df_loc_ps["Northing"].loc[comp]
            points.append(Point(x, y))
            for sub in spec_ps:
                idx = comp + "_" + sub
                # Transform units [kt/y] -> [kg/y]
                factor = df_eipwp_ori[year].loc[idx] * 1e6
                lst_row.append(factor)
            df_eipwp.loc[len(df_eipwp)] = lst_row

        # Rename columns according to emiproc convention
        df_eipwp = df_eipwp.rename(columns=dict_spec)

        # Store as GeoDataFrame
        df_eipwp = gpd.GeoDataFrame(df_eipwp, geometry=points, crs=LV95)
        self.df_eipwp = df_eipwp

        # ---------------------------------------------------------------------
        # -- Grids
        # ---------------------------------------------------------------------

        rasters_dir = Path(rasters_dir)

        # Grids that depend on chemical species (road transport)
        rasters_str_dir = Path(rasters_str_dir)
        str_rasters = [
            r
            for r in rasters_str_dir.rglob("*.asc")
            # Don't include the tunnel specific grids as they are already included in the grids for road transport
            if "_tun" not in r.stem
        ]

        # Grids that do not depend on chemical species
        normal_rasters = [r for r in rasters_dir.rglob("*.asc")]

        self.all_raster_files = normal_rasters + str_rasters

        self.raster_categories = [r.stem for r in normal_rasters] + [
            r.stem for r in str_rasters
        ]

        # List with Raster categories for which we have emissions
        raster_sub = df_emissions.index.tolist()
        rasters_w_emis = []
        for t in raster_sub:
            cat, sub = t.split("_")
            subname = sub.lower()
            if cat == "evstr":
                # Grid for non-methane VOCs is named "evstr_nmvoc"
                if subname == "voc":
                    subname = "nmvoc"
                rasters_w_emis.append(cat + "_" + subname)
            else:
                rasters_w_emis.append(cat)
        # Remove duplicates
        rasters_w_emis = [*set(rasters_w_emis)]

        # Compare Raster categories of input emission file with Raster categories of grids
        # Raise error if the two don't agree
        if not sorted(self.raster_categories) == sorted(rasters_w_emis):
            missing_raster_files = [
                r for r in rasters_w_emis if r not in self.raster_categories
            ]
            missing_emissions_values = [
                r for r in self.raster_categories if r not in rasters_w_emis
            ]
            raise ValueError(
                "Raster categories of emission file don't match:"
                f"\nMissing raster files: {missing_raster_files}"
                f"\nMissing emissions values: {missing_emissions_values}"
            )

        # ---------------------------------------------------------------------
        # -- Emissions without point sources
        # ---------------------------------------------------------------------

        self.df_emission = df_emissions
        self.requires_grid = requires_grid

        # Fill NaN values with zeros
        self.df_emission[year] = self.df_emission[year].fillna(0)

        # List with chemical species
        self._substances = [*set(self.df_emission["Chemical Species"].tolist())]

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
                idx = cat + "_" + sub_name
                # Transform units [t/y] -> [kg/y]
                factor = self.df_emission[year].loc[idx] * 1000
                # Normalize the array to ensure the factor will be the sum
                # Note: this is to ensure consistency if the data provider
                # change the df_emission values in the future but not the rasters
                _normalized_raster_array = _raster_array / _raster_array.sum()
                mapping[(cat, sub_name)] = _normalized_raster_array * factor
            else:
                for sub in self._substances:
                    idx = category + "_" + sub
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
            self.logger.info(f"Parsing {raster_file}")
            src = rasterio.open(raster_file).read(1)
            np.save(raster_file.with_suffix(".npy"), src)
            inventory_field = src
        return inventory_field


if __name__ == "__main__":
    swiss_data_path = Path(r"/users/ckeller/emission_data/CH_Emissions")

    inv_ch = SwissRasters(
        data_path=swiss_data_path,
        rasters_dir=swiss_data_path / "ekat_gridascii",
        rasters_str_dir=swiss_data_path / "ekat_str_gridascii",
        requires_grid=True,
        year=2015,
    )
