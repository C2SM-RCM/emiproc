from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
import geopandas as gpd

from emiproc.grids import WGS84, TNOGrid
from emiproc.inventories import Inventory, Substance


class TNO_Inventory(Inventory):
    """The TNO inventory.

    TNO has grid cell sources and point sources.
    This handles both.

    https://topas.tno.nl/emissions/
    """

    grid: TNOGrid

    def __init__(
        self,
        nc_file: PathLike,
        substances: list[Substance] = ["CO2", "CO", "NOx", "CH4", "VOC"],
    ) -> None:
        """Create a TNO_Inventory.

        :arg nc_file: The TNO NetCDF dataset.
        """
        super().__init__()

        nc_file = Path(nc_file)

        self.name = nc_file.stem

        ds = xr.load_dataset(nc_file)

        self.grid = TNOGrid(nc_file)

        mask_area_sources = ds["source_type_index"] == 1
        mask_point_sources = ds["source_type_index"] == 2

        # Maps substances inside the nc file to the emiproc names
        substances_mapping = {
            "co2_ff": "CO2",
            "co2_bf": "CO2",
            "co_ff": "CO",
            "co_bf": "CO",
            "nox": "NOx",
            "ch4": "CH4",
            "nmvoc": "VOC",
        }

        # Select only the requested substances
        substances_mapping = {
            k: v for k, v in substances_mapping.items() if v in substances
        }

        polys = self.grid.cells_as_polylist

        # I assume it is (no info in nc file)
        crs = WGS84

        # Index in the polygon list (from the gdf) (index start at 1 )
        poly_ind = (
            (ds["longitude_index"] - 1) * self.grid.ny + (ds["latitude_index"] - 1)
        ).to_numpy()
        mapping = {}
        self.gdfs = {}
        for cat_idx, cat_name in enumerate(ds["emis_cat_code"].to_numpy()):
            cat_name = cat_name.decode("utf-8")
            # Indexes start at 1
            mask_this_category = ds["emission_category_index"] == cat_idx + 1

            # Extract point sources information
            mask_points_this_cat = mask_this_category & mask_point_sources
            point_sources_gdf = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries.from_xy(
                    ds["longitude_source"][mask_points_this_cat],
                    ds["latitude_source"][mask_points_this_cat],
                ),
                crs=crs,
            )
            # Extract the emissions for the points sources of this cat
            ds_point_sources = ds[substances_mapping.keys()].sel(
                {"source": mask_points_this_cat}
            )

            for sub_in_nc, sub_emiproc in substances_mapping.items():
                tuple_idx = (cat_name, sub_emiproc)
                if tuple_idx not in mapping:
                    mapping[tuple_idx] = np.zeros(len(polys))
                # Add all the area sources corresponding to that category
                mask = mask_this_category & mask_area_sources
                np.add.at(
                    mapping[tuple_idx], poly_ind[mask], ds[sub_in_nc][mask].to_numpy()
                )

                # Add the point sources
                point_sources_values = ds_point_sources[sub_in_nc]
                if sub_emiproc in point_sources_gdf:
                    point_sources_gdf[sub_emiproc] += point_sources_values.to_numpy()
                else:
                    point_sources_gdf[sub_emiproc] = point_sources_values.to_numpy()

            if len(point_sources_gdf):
                # Add only the non emtpy
                self.gdfs[cat_name] = point_sources_gdf

        self.gdf = gpd.GeoDataFrame(
            mapping,
            geometry=polys,
            crs=crs,
        )

        self.cell_areas = ds["area"].T.to_numpy().reshape(-1)
