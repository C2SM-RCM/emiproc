"""DuckDB-based inventory."""

from __future__ import annotations
import logging
from pathlib import Path
from os import PathLike
import re

import geopandas as gpd
from shapely import from_wkb

try:
    import duckdb
except ImportError:
    duckdb = None

from emiproc.inventories import Inventory, Substance, Category

logger = logging.getLogger(__name__)


def _parse_duckdb_table_name(
    table_name: str,
    con: duckdb.DuckDBPyConnection,
    year: int,
    geometry_column: str = "geom",
    year_column: str = "jahr",
) -> gpd.GeoDataFrame:
    """Parse the duck db table to a GeoDataFrame."""
    columns = con.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    columns = [col[1] for col in columns]  # col[1] is the column name
    if geometry_column not in columns:
        raise ValueError(
            f"Geometry column '{geometry_column}' not found in table '{table_name}'."
        )

    # Get the year columns
    query_year = f"SELECT DISTINCT {year_column} FROM {table_name};"
    years = con.execute(query_year).fetchall()
    years = [y[0] for y in years]
    if year not in years:
        raise ValueError(
            f"Year '{year}' not found in table '{table_name}'. Available years: {years }"
        )

    query = f"""
    INSTALL spatial;
    LOAD spatial;
    SELECT ST_AsWKB({geometry_column}) FROM {table_name} WHERE {year_column} = {year};
    """
    # Get the geometry column and other columns
    column = con.execute(query).fetchall()
    geometry = [from_wkb(g[0]) for g in column]

    emission_columns = [col for col in columns if col.startswith("emission_")]

    query_emissions = f"SELECT {', '.join(emission_columns)} FROM {table_name} WHERE {year_column} = {year};"
    df_emissions = con.execute(query_emissions).fetchdf()

    # Get the columns that match the pattern
    df_emissions = df_emissions.rename(columns=lambda x: re.sub(r"^emission_", "", x))
    df_emissions = df_emissions.fillna(0.0)
    gdf = gpd.GeoDataFrame(df_emissions, geometry=geometry, crs="LV95")

    return gdf


class DuckDBInventory(Inventory):
    """Inventory backed by a DuckDB-file.

    Organization of emissions in duckDB database.

    * tables for each category
    * columns for each substance (should follow a pattern eg emission_{substance})
    * geometry column for shapes
    * year column if multiple years are present


    :param duckdb_filepath: Path to the duckDB file.
    :param year: Year of the emissions to load.
    :param skip_suffixes: List of suffixes to skip when loading tables.

    """

    duckdb_filepath: Path

    def __init__(
        self,
        duckdb_filepath: PathLike,
        year: int,
        skip_suffixes: list[str] = ["_ef", "_p"],
    ) -> None:

        if duckdb is None:
            raise ImportError(
                "duckdb package is required to use DuckDBInventory. "
                "Please install it via 'pip install duckdb'."
            )

        duckdb_filepath = Path(duckdb_filepath)

        if not duckdb_filepath.is_file():
            raise FileNotFoundError(
                f"{duckdb_filepath!s} does not exist or is not reachable."
            )

        self.name = f"DuckDBInventory({duckdb_filepath.stem})"

        super().__init__()
        # This inventory currently does not implement reading logic.
        # We keep the API surface compatible with other shaped inventories
        # by setting `gdf` to None and preparing `gdfs` as an empty dict.
        self.gdf = None
        self.grid = None

        # Get all tables in the DuckDB file (not loaded, just names)
        with duckdb.connect(duckdb_filepath, read_only=True) as con:
            tables = con.execute("SHOW TABLES;").fetchall()
            available_tables = [t[0] for t in tables]

            self.gdfs = {}
            for table in available_tables:
                if any(table.endswith(suffix) for suffix in skip_suffixes):
                    self.logger.debug(f"Skipping table {table} due to suffix filter.")
                    continue
                try:
                    gdf = _parse_duckdb_table_name(table, con, year=year)
                    self.gdfs[table] = gdf
                except Exception as e:
                    self.logger.warning(f"Skipping table {table}: {e}")

        self.logger.info(
            f"DuckDBInventory initialized from {duckdb_filepath!s}. "
            f"Available tables: {available_tables}"
        )

        if len(self.gdfs) == 0:
            raise ValueError(
                f"No valid tables loaded from {duckdb_filepath!s} for year {year}."
            )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("emiproc").setLevel(logging.DEBUG)

    logger.setLevel(logging.DEBUG)

    # Example usage
    duckdb_file = "/home/coli/Data/zurich/Emissionskataster/emikat.db"
    inv = DuckDBInventory(duckdb_file, year=2024)

    inv.logger.info(f"Inventory initialized: {inv.name}")

    inv.logger.info(
        f"Loaded categories: {inv.categories} and substances: {inv.substances}"
    )

    inv.logger.info(f"{inv.total_emissions['klärschlammverwertung']=}")
