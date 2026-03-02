import logging
from pathlib import Path

import geopandas as gpd

from emiproc.inventories import Inventory


def export_to_geopackage(
    inv: Inventory,
    filepath: Path,
    name_gridded: str = "gridded_emissions",
) -> None:
    """Export an inventory to a GeoPackage file.

    The gridded emissions will be saved in a layer with :param:`name_gridded` .

    For shaped emissions, each category will be saved in a separate layer,
    with the name being the category name.

    :param inv:
        The inventory to export.
    :param filepath:
        The path to the output GeoPackage file.
    :param name_gridded:
        The name of the layer for gridded emissions.
    """

    logger = logging.getLogger(__name__)

    gdfs = inv.gdfs
    if gdfs is None:
        gdfs = {}

    if filepath.is_file():
        logger.info(f"Removing existing file {filepath!s}")
        filepath.unlink()

    for cat, gdf in gdfs.items():

        gdf.to_file(filepath, layer=cat, driver="GPKG")

    gdf = getattr(inv, "gdf", None)
    if gdf is not None:
        gdf: gpd.GeoDataFrame
        # Check that gridded is not a category already exported
        if name_gridded in gdfs:
            raise ValueError(
                f"Category name '{name_gridded}' already exists in inventory categories."
            )
        # Rename tuple columns to avoid conflicts
        gdf = gdf.copy()
        gdf.columns = gdf.columns.map(lambda col: "_".join(col))
        gdf.to_file(filepath, layer=name_gridded, driver="GPKG")
