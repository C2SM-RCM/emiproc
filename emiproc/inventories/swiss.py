from __future__ import annotations
from enum import Enum, auto
from os import PathLike
from pathlib import Path
from emiproc.grids import LV95, SwissGrid
from emiproc.inventories import Category, Inventory, Substance
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.creation import polygons
import numpy as np
import rasterio


class PointSourceCorrection(Enum):
    """Enum for the point source correction method."""

    KEEP_RASTER_ONLY = auto()
    KEEP_POINT_SOURCE_ONLY_SCALED_TO_RASTER_TOTAL = auto()
    REMOVE_POINT_SOURCE_FROM_RASTER_TOTAL = auto()
    IS_ONLY_POINT_SOURCE = auto()


default_point_source_correction = {
    "eipro": PointSourceCorrection.REMOVE_POINT_SOURCE_FROM_RASTER_TOTAL,
    "eipzm": PointSourceCorrection.KEEP_POINT_SOURCE_ONLY_SCALED_TO_RASTER_TOTAL,
    "eipkv": PointSourceCorrection.KEEP_POINT_SOURCE_ONLY_SCALED_TO_RASTER_TOTAL,
    "eikla": PointSourceCorrection.KEEP_RASTER_ONLY,
    "eidep": PointSourceCorrection.KEEP_RASTER_ONLY,
    "eiprd": PointSourceCorrection.REMOVE_POINT_SOURCE_FROM_RASTER_TOTAL,
}


class SwissRasters(Inventory):
    """An inventory of Switzerland based on swiss rasters."""

    edge_size: int = 100
    grid: SwissGrid
    df_eipwp: gpd.GeoDataFrame
    df_emissions: pd.DataFrame
    emission: pd.Series

    def __init__(
        self,
        filepath_csv_totals: Path,
        filepath_point_sources: Path,
        rasters_dir: PathLike,
        rasters_str_dir: PathLike,
        requires_grid: bool = True,
        year: int = 2015,
        point_source_correction: dict[
            Category, PointSourceCorrection
        ] = default_point_source_correction,
    ) -> None:
        """Create a swiss raster inventory.

        :arg filepath_csv_totals: Csv file containing total emissions
            for each category/substance for different years.
        :arg filepath_point_sources: Excel file containing point sources.
            See in :py:func:`emiproc.inventories.swiss.read_prtr` for more details.
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

        self.year = year

        filepath_csv_totals = Path(filepath_csv_totals)

        # Emission data file
        if filepath_csv_totals.is_file():
            total_emission_file = filepath_csv_totals
        else:
            raise FileNotFoundError(
                f"Data path {filepath_csv_totals} is not an existing file or a folder."
            )

        # Load excel sheet with the total emissions (excluding point sources)
        df_emissions = pd.read_csv(total_emission_file, comment="#")

        # Add indexing column consisting of both grid and species' name
        df_emissions["cat_sub"] = (
            df_emissions["category"] + "_" + df_emissions["substance"]
        )
        df_emissions = df_emissions.set_index("cat_sub")
        self.df_emissions = df_emissions

        year_str = str(year)

        # Check if selected year is in dataset
        if year_str not in df_emissions.columns:
            raise ValueError(
                f"Selected {year=} not in dataset with columns={df_emissions.columns}."
            )

        emissions = df_emissions[year_str].copy()

        # List with substance
        substances = df_emissions["substance"].unique()

        # ---------------------------------------------------------------------
        # -- Emissions from point sources
        # ---------------------------------------------------------------------

        # Load data
        gdfs = read_prtr(filepath_point_sources, year, substances=substances)

        # Remove the total values in the rasters
        for cat, gdf in gdfs.items():

            if "CO2" in gdf.columns:
                # We need to split CO2 in CO2 and CO2_biog
                catsub_co2 = cat + "_CO2"
                catsub_biog = cat + "_CO2_biog"
                assert (
                    "CO2_biog" not in gdf.columns
                ), "Unexpected CO2_biog in the point sources"

                if (
                    point_source_correction[cat]
                    == PointSourceCorrection.IS_ONLY_POINT_SOURCE
                ):
                    biog_fracton = 0.0
                else:
                    # Get the biogenic fraction in the total emissions
                    # and apply it to the pointsources
                    biog_fracton = emissions.loc[catsub_biog] / (
                        emissions.loc[catsub_co2] + emissions.loc[catsub_biog]
                    )
                # Split the CO2 emissions in two
                base_col = gdf["CO2"].copy()
                gdf["CO2"] = base_col * (1.0 - biog_fracton)
                gdf["CO2_biog"] = base_col * biog_fracton

            totals = gdf.drop(columns=["geometry"]).sum(axis="rows")
            for sub in totals.index:
                catsub = cat + "_" + sub

                if cat not in point_source_correction:
                    raise ValueError(
                        f"Category {cat} with point source emissions not in point_source_correction dictionary."
                    )

                correction = point_source_correction[cat]

                if correction == PointSourceCorrection.KEEP_RASTER_ONLY:
                    # Remove the emissions of point sources
                    gdf[sub] = 0.0
                elif correction == PointSourceCorrection.IS_ONLY_POINT_SOURCE:
                    # Make sure the raster is empty
                    if catsub in emissions and emissions.loc[catsub] != 0:
                        raise ValueError(
                            f"Raster {catsub} is not empty for {correction}."
                        )
                    emissions.loc[catsub] = 0.0
                elif (
                    correction
                    == PointSourceCorrection.KEEP_POINT_SOURCE_ONLY_SCALED_TO_RASTER_TOTAL
                ):
                    # Scale the point sources to match
                    total = emissions.loc[catsub]
                    gdf[sub] = gdf[sub] / totals[sub] * total
                    # Remove the emissions of the raster
                    emissions.loc[catsub] = 0.0
                elif (
                    correction
                    == PointSourceCorrection.REMOVE_POINT_SOURCE_FROM_RASTER_TOTAL
                ):
                    if emissions.loc[catsub] < totals[sub]:
                        self.logger.warning(
                            f"Total emissions for {cat=} and {sub=} are negative."
                            f" point sources: {totals[sub]}, inventory total: {emissions.loc[catsub]}"
                            " Only the point sources will be used."
                        )
                        # If we have more emissions in the point sources than in the inventory
                        # We remove the rasters emissions
                        emissions.loc[catsub] = 0.0
                    else:
                        emissions.loc[catsub] -= totals[sub]
                else:
                    raise ValueError(f"Unknown correction {correction}")

        # ---------------------------------------------------------------------
        # -- Grids
        # ---------------------------------------------------------------------

        rasters_dir = Path(rasters_dir)

        # Grids that depend on substance (road transport)
        rasters_str_dir = Path(rasters_str_dir)
        str_rasters = [
            r
            for r in rasters_str_dir.glob("*.asc")
            # Don't include the tunnel specific grids as they are already included in the grids for road transport
            if "_tun" not in r.stem
        ]

        # Grids that do not depend on substance
        normal_rasters = [r for r in rasters_dir.glob("*.asc")]

        self.all_raster_files = normal_rasters + str_rasters

        self.raster_categories = [r.stem for r in normal_rasters] + [
            r.stem for r in str_rasters
        ]

        # List with Raster categories for which we have emissions
        raster_sub = emissions.index.tolist()
        rasters_w_emis = []

        evstr_subname_to_subname = {}
        for t in raster_sub:
            split = t.split("_")
            assert len(split) > 1
            cat = split[0]
            sub = "_".join(split[1:])
            if "evstr" in cat:
                # Grid for non-methane VOCs is named "evstr_nmvoc"
                subname = sub.lower()
                if subname == "voc":
                    subname = "nmvoc"
                rasters_w_emis.append(cat + "_" + subname)
                evstr_subname_to_subname[subname] = sub
            elif (
                cat in point_source_correction
                and point_source_correction[cat]
                == PointSourceCorrection.IS_ONLY_POINT_SOURCE
            ):
                # If the category is only point source, we don't need the raster
                pass
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

        self.emission = emissions
        self.requires_grid = requires_grid

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
                split = category.split("_")
                cat = split[0]
                sub = "_".join(split[1:])
                sub_name = evstr_subname_to_subname[sub]
                idx = cat + "_" + sub_name
                total_emissions = emissions.loc[idx]
                # Normalize the array to ensure the factor will be the sum
                # Note: this is to ensure consistency if the data provider
                # change the df_emission values in the future but not the rasters
                _normalized_raster_array = _raster_array / _raster_array.sum()
                mapping[(cat, sub_name)] = _normalized_raster_array * total_emissions
            else:
                for sub in substances:
                    idx = category + "_" + sub
                    total_emissions = emissions.loc[idx]
                    if total_emissions > 0:
                        mapping[(category, sub)] = _raster_array * total_emissions

        if self.requires_grid:
            x_coords, y_coords = np.meshgrid(xs, ys[::-1])
            # Reshape to 1D
            x_coords = x_coords.flatten()
            y_coords = y_coords.flatten()
            dx = self.grid.dx
            dy = self.grid.dy
            coords = np.array(
                [
                    [x, y]
                    for x, y in zip(
                        [x_coords, x_coords, x_coords + dx, x_coords + dx],
                        [y_coords, y_coords + dy, y_coords + dy, y_coords],
                    )
                ]
            )
            coords = np.rollaxis(coords, -1, 0)
        self.gdf = gpd.GeoDataFrame(
            mapping,
            crs=LV95,
            # This vector is same as raster data reshaped using reshape(-1)
            geometry=(
                polygons(coords)
                if self.requires_grid
                else np.full(self.grid.nx * self.grid.ny, np.nan)
            ),
        )

        # Add point sources
        self.gdfs = gdfs

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


polluant_matching = {
    #'PCDD +PCDF (Dioxine + Furane) (als Teq)',
    #'Gesamtphosphor',
    #'Ethylenoxid',
    #'Gesamtstickstoff ',
    "Schwefeloxide (SOx/SO2)": "SO2",
    #'Halone',
    #'Arsen und Verbindungen (als As)',
    #'Dichlormethan (DCM)',
    #'Nonylphenolethoxylate (NP/NPEs) und verwandte Stoffe',
    "flüchtige organische Verbindungen ohne Methan (NMVOC)": "VOC",
    "Kohlenmonoxid (CO)": "CO",
    "Stickstoffoxide (NOx/NO2)": "NOx",
    #'Xylole (als BTEX) (a)',
    #'gesamter organischer Kohlenstoff (TOC) (als Gesamt-C oder CSB/3)',
    #'Toluol (als BTEX) (a)',
    "Kohlendioxid (CO2)": "CO2",
    #'Benzol (als BTEX) (a)',
    "Fluoride (als Gesamt-F)": "F-Gases",
    "Ammoniak (NH3)": "NH3",
    #'Chloride (als Gesamt-Cl)',
    #'Quecksilber und Verbindungen (als Hg)', 'Cyanide (als Gesamt-CN)',
    #'Chrom und Verbindungen (als Cr)',
    # 'Kupfer und Verbindungen (als Cu)',
    # 'Nickel und Verbindungen (als Ni)',
    # 'Blei und Verbindungen (als Blei)',
    # 'Cadmium und Verbindungen (als Cd)',
    # 'Zink und Verbindungen (als Zn)',
    "Feinstaub (PM10)": "PM10",
    # 'Fluor und anorganische Verbindungen (als HF)',
    # 'Chlor und anorganische Verbindungen (als HCl)',
    "Methan (CH4)": "CH4",
    # 'Halogenierte organische Verbindungen (als AOX)',
    # 'Tributylzinn und Verbindungen', 'Ethylbenzol (als BTEX) (a)',
    # 'Phenole (als Gesamt-C)',
    "Distickstoffoxid (N2O)": "N2O",
    # 'Trichlorethen',
    "Schwefelhexafluorid (SF6)": "SF6",
    # 'teilfluorierte Kohlenwasserstoffe (HFKW)',
    # 'Polychlorierte Biphenyle (PCB)', 'Diuron',
    # 'Tetrachlorethen (PER)', 'Naphthalin',
    # 'Polyzyklische aromatische Kohlenwasserstoffe (PAK) (b)',
    # '1,2-Dichlorethan (EDC)', 'Cyanwasserstoff (HCN) '
}


activities_to_categories = {
    # 1 - Energiesektor
    "1.a": "eipro",  # Mineralöl- und Gasraffinerien
    "1.b": "eipro",  # Vergasungs- und Verflüssigungsanlagen
    "1.c": "eipro",  # Wärmekraftwerke und andere Verbrennungsanlagen
    # 2 - Herstellung und Verarbeitung von Metallen
    "2.b": "eipro",
    "2.c.1": "eipro",
    "2.c.2": "eipro",
    "2.e.1": "eipro",
    "2.e.2": "eipro",
    "2.f": "eipro",
    # 3 - Mineralverarbeitende Industrie zement: eipzm=point source zement
    "3.c.1": "eipzm",  # c - Anlagen zur Herstellung von 1 - Zementklinkern in Drehrohröfen
    # 3 - Mineralverarbeitende Industrie others
    "3.e": "eipro",  # Glas
    "3.f": "eipro",  # Schmelzen mineralischer Stoffe
    "3.g": "eipro",  #  Herstellung von keramischen Erzeugnissen
    # 4 - Chemische Industrie
    "4.a.1": "eipro",
    "4.a.10": "eipro",
    "4.a.11": "eipro",
    "4.a.2": "eipro",
    "4.a.5": "eipro",
    "4.a.8": "eipro",
    "4.b.5": "eipro",
    "4.d": "eipro",
    "4.e": "eipro",
    "4.f": "eipro",
    # 5 - Abfall- und Abwasserbewirtschaftung
    # Punktquellen KVA (Kehrichtverbrennungsanlagen ) == Waste incinerators
    "5.a": "eipkv",  #  Anlagen zur Verbrennung, Pyrolyse, Verwertung, chemischen Behandlung, oder Deponierung von Sonderabfällen
    "5.b": "eipkv",  # Kehrichtverbrennungsanlagen für Siedlungsmüll
    "5.d": "eidep",  # Deponien
    "5.f": "eikla",  # Kommunale Abwasserbehandlungsanlagen
    "5.g": "eikla",  # Eigenständig betriebene Industrie¬abwasserbehandlungsanlagen
    # 6 - Be- und Verarbeitung von Papier und Holz
    "6.b": "eipro",
    # 8 - Tierische und pflanzliche Produkte aus dem Lebensmittel- und Getränkesektor
    "8.b.2": "eipro",
    "8.c": "eipro",
    # 9 - Sonstige Tätigkeiten
    "9.c": "eipro",
    "9.d": "eipro",
}


def read_prtr(
    prtr_file: PathLike, year: int, substances: list[Substance] | None = None
) -> dict[Category, gpd.GeoDataFrame]:
    """Read the PRTR file and return the gdfs.

    If you want to change the substances or the categories, you can do it
    by changing the dictionaries in the `emiproc.inventories.swiss` module.

    :arg prtr_file: Path to the PRTR file.
        This file must be downloaded from the Swiss PRTR website
        https://www.bafu.admin.ch/bafu/de/home/themen/chemikalien/zustand/schadstoffregister-swissprtr.html
        then go in the 'Datenpublikation' section and download the Excel file
    :arg year: The year of the data to use.
    :arg substances: List of substances to use.
        If None, all substances will be used.

    :returns: A dictionary with the categories as keys and the GeoDataFrames as values.
        Can be used to create the Inventory object.
    """

    df_prtr = pd.read_excel(prtr_file, skiprows=[0, 1, 3])

    substance_matching = {
        key: value
        for key, value in polluant_matching.items()
        if substances is None or value in substances
    }
    # Check that all substances are in the dictionary
    if substances is not None:
        for sub in substances:
            if sub not in substance_matching.values() and sub not in [
                # Compounds which are not in the point sources data
                "CO2_biog",
                "PM25",
            ]:
                raise ValueError(
                    f"Unkown substance `{sub}` not in the "
                    "`emiproc.inventories.swiss.polluant_matching` dictionary."
                )

    columns_of_interest = [
        "Year",
        "North coordinate (CH1903+)",
        "East coordinate (CH1903+)",
        "Facility",
        "Value",
        "Unit",
        "Pollutant_name",
        "Installation_main activity",
    ]

    mask_point_source = df_prtr["Source type"] == "Punktquelle"
    if year not in df_prtr["Year"].unique():
        raise ValueError(f"Year {year} not in the data.")
    mask_year = df_prtr["Year"] == year
    mask_has_value = df_prtr["Value"].notnull()
    mask_known_pollutants = df_prtr["Pollutant_name"].isin(substance_matching.keys())
    df_cleaned = df_prtr.loc[
        mask_point_source & mask_year & mask_has_value & mask_known_pollutants,
        columns_of_interest,
    ]

    # Correct the units to have kg year
    factors = {
        "t/a": 1e3,
        "kg/a": 1.0,
    }
    mask_corrected = pd.Series(0, index=df_cleaned.index, dtype=bool)
    for unit, factor in factors.items():
        mask = df_cleaned["Unit"] == unit
        mask_corrected = mask_corrected | mask
        df_cleaned.loc[mask, "Value"] *= factor
    if not mask_corrected.all():
        raise ValueError(
            f"Units not corrected for {df_cleaned.loc[~mask_corrected, 'Unit'].unique()}."
            " Fix the `emiproc.inventories.swiss.polluant_matching` dictionary."
        )
    # Drop the unit column
    df_cleaned = df_cleaned.drop(columns="Unit")
    # Rename the substances
    df_cleaned["Substance"] = df_cleaned["Pollutant_name"].replace(substance_matching)
    # Check the categories
    mask_matching = df_cleaned["Installation_main activity"].isin(
        activities_to_categories.keys()
    )
    if not mask_matching.all():
        raise ValueError(
            f"Missing categories for {df_cleaned.loc[~mask_matching, 'Installation_main activity'].unique()}"
            " Fix the `emiproc.inventories.swiss.activities_to_categories` dictionary."
        )
    df_cleaned["Category"] = df_cleaned["Installation_main activity"].replace(
        activities_to_categories
    )
    df_cleaned["x"] = df_cleaned["East coordinate (CH1903+)"]
    df_cleaned["y"] = df_cleaned["North coordinate (CH1903+)"]

    cols_for_emiproc = ["Category", "Substance", "Value", "x", "y"]
    df_emiproc = df_cleaned[cols_for_emiproc].copy()
    df_emiproc

    gdfs = {}
    for category in df_emiproc["Category"].unique():
        cat_df = df_emiproc.loc[df_emiproc["Category"] == category].copy()
        for substances in cat_df["Substance"].unique():
            mask_substance = cat_df["Substance"] == substances
            df_substance = cat_df.loc[mask_substance].copy()
            cat_df[substances] = df_substance["Value"]
        df_groupped = (
            cat_df.drop(columns=["Value", "Substance", "Category"])
            .groupby(["x", "y"])
            .sum()
            .reset_index()
        )
        gdfs[category] = gpd.GeoDataFrame(
            df_groupped.drop(columns=["x", "y"]),
            geometry=gpd.points_from_xy(df_groupped["x"], df_groupped["y"]),
            crs="LV95",
        )

    return gdfs
