"""Conservative CAMS-GLOB-ANT export to a projected MicroHH grid."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import pyproj
import xarray as xr

from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.grids import RegularGrid, WGS84
from emiproc.inventories.cams_glob_ant import CAMSGlobAnt, CAMS_FLUX_UNIT
from emiproc.regrid import get_weights_mapping, weights_remap


def _inventory_bbox_for_grid(grid: RegularGrid) -> tuple[float, float, float, float]:
    """Return the target grid's longitude/latitude bounding box."""
    target = grid.gdf.to_crs(WGS84)
    west, south, east, north = target.total_bounds
    return float(west), float(south), float(east), float(north)


def export_cams_glob_ant_to_microhh(
    inventory: CAMSGlobAnt,
    grid: RegularGrid,
    output_file: PathLike,
    weights_file: PathLike | None = None,
    add_total: bool = True,
) -> Path:
    """Conservatively remap monthly CAMS fluxes to a MicroHH NetCDF file.

    The output emission variable has dimensions ``(time, category, y, x)`` and
    units ``kg m-2 s-1``. Source emissions are first converted to ``kg s-1``,
    distributed using polygon-overlap weights, and divided by target-cell area.
    """
    target_crs = pyproj.CRS.from_user_input(grid.crs)
    if target_crs.is_geographic:
        raise ValueError("The target MicroHH grid must use a projected CRS.")

    source_shapes = inventory.grid.gdf.geometry.to_crs(target_crs).reset_index(
        drop=True
    )
    target_shapes = grid.gdf.geometry.to_crs(target_crs).reset_index(drop=True)
    source_areas = source_shapes.area.to_numpy()
    target_areas = target_shapes.area.to_numpy()

    mapping = get_weights_mapping(
        Path(weights_file) if weights_file is not None else None,
        source_shapes,
        target_shapes,
        loop_over_inv_objects=False,
    )

    remapped = np.empty(
        (
            inventory.monthly_flux.sizes["time"],
            inventory.monthly_flux.sizes["category"],
            grid.ny,
            grid.nx,
        ),
        dtype=np.float64,
    )
    for time_index in range(inventory.monthly_flux.sizes["time"]):
        for category_index in range(inventory.monthly_flux.sizes["category"]):
            source_flux = inventory.monthly_flux.isel(
                time=time_index, category=category_index
            ).values
            target_rate = weights_remap(
                mapping, source_flux * source_areas, len(target_shapes)
            )
            remapped[time_index, category_index] = (
                target_rate / target_areas
            ).reshape(grid.shape).T

    x = grid.lon_range - grid.xmin
    y = grid.lat_range - grid.ymin
    projected_x, projected_y = np.meshgrid(grid.lon_range, grid.lat_range)
    to_lonlat = pyproj.Transformer.from_crs(target_crs, WGS84, always_xy=True)
    longitude, latitude = to_lonlat.transform(projected_x, projected_y)

    categories = inventory.monthly_flux["category"].values.astype(str)
    emission_name = f"emi_{inventory.substance_name}"
    additional_attributes: dict[str, str | float | int] = {
        "inventory_year": str(inventory.year),
        "target_grid": grid.name,
        "emiproc_history": str(inventory.history),
    }
    if hasattr(inventory, "corso_subtraction_summary"):
        additional_attributes.update(
            {
                f"corso_{key}": value
                for key, value in inventory.corso_subtraction_summary.items()
            }
        )
    ds = xr.Dataset(
        data_vars={
            emission_name: (
                ("time", "category", "y", "x"),
                remapped,
                {
                    "long_name": (
                        f"monthly mean {inventory.substance_name} emission flux"
                    ),
                    "units": CAMS_FLUX_UNIT,
                    "substance": inventory.substance_name,
                    "grid_mapping": "crs",
                },
            ),
            "cell_area": (
                ("y", "x"),
                target_areas.reshape(grid.shape).T,
                {"long_name": "horizontal cell area", "units": "m2"},
            ),
            "longitude": (
                ("y", "x"),
                longitude,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "latitude": (
                ("y", "x"),
                latitude,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "time_bnds": (
                ("time", "bounds"),
                inventory.time_bounds,
                {"long_name": "monthly averaging interval"},
            ),
            "crs": ((), 0, target_crs.to_cf()),
        },
        coords={
            "time": (
                "time",
                inventory.monthly_flux["time"].values,
                {"standard_name": "time", "bounds": "time_bnds"},
            ),
            "category": categories,
            "x": (
                "x",
                x,
                {
                    "long_name": "MicroHH local scalar-cell x coordinate",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                "y",
                y,
                {
                    "long_name": "MicroHH local scalar-cell y coordinate",
                    "units": "m",
                    "axis": "Y",
                },
            ),
            "easting": (
                "x",
                grid.lon_range,
                {"standard_name": "projection_x_coordinate", "units": "m"},
            ),
            "northing": (
                "y",
                grid.lat_range,
                {"standard_name": "projection_y_coordinate", "units": "m"},
            ),
            "bounds": [0, 1],
        },
        attrs=nc_cf_attributes(
            title="CAMS-GLOB-ANT emissions on a MicroHH grid",
            source="CAMS-GLOB-ANT",
            history=f"Conservatively remapped by emiproc from {inventory.name}.",
            additional_attributes=additional_attributes,
        ),
    )
    ds["category"].attrs = {
        "long_name": "CAMS-GLOB-ANT emission sector code",
        "descriptions": "; ".join(
            f"{cat}: {inventory.category_descriptions[cat]}" for cat in categories
        ),
    }
    if add_total:
        ds[f"{emission_name}_total"] = ds[emission_name].sum("category")
        ds[f"{emission_name}_total"].attrs = {
            "long_name": (
                f"monthly mean total {inventory.substance_name} emission flux"
            ),
            "units": CAMS_FLUX_UNIT,
            "substance": inventory.substance_name,
            "grid_mapping": "crs",
        }

    output_path = Path(output_file).with_suffix(".nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_encoding = {
        "units": f"hours since {inventory.year}-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }
    ds["time"].encoding.update(time_encoding)
    ds["time_bnds"].encoding.update(time_encoding)
    ds.to_netcdf(output_path)
    return output_path


def load_cams_glob_ant(
    nc_file: PathLike,
    grid: RegularGrid,
    output_file: PathLike,
    variables: list[str] | None = None,
    substance: str | None = None,
    weights_file: PathLike | None = None,
    corso_dataset: xr.Dataset | None = None,
    corso_substance: str | None = None,
    corso_sector_mapping: Mapping[str, str] | None = None,
    corso_negative_policy: Literal["clip", "raise", "allow"] = "clip",
) -> Path:
    """Crop, import, remap, and export one CAMS-GLOB-ANT file.

    When ``corso_dataset`` is provided, its raw annual point-source emissions
    are subtracted from the corresponding CAMS cells before conservative
    remapping. See :class:`emiproc.inventories.cams_glob_ant.CAMSGlobAnt` for
    species mapping, sector mapping, and negative-residual behavior.
    """
    inventory = CAMSGlobAnt(
        nc_file,
        variables=variables,
        bbox=_inventory_bbox_for_grid(grid),
        substance=substance,
        corso_dataset=corso_dataset,
        corso_substance=corso_substance,
        corso_sector_mapping=corso_sector_mapping,
        corso_negative_policy=corso_negative_policy,
    )
    return export_cams_glob_ant_to_microhh(
        inventory, grid, output_file, weights_file=weights_file
    )


# Preserve the original public name while also supporting the shorter loader
# name used by current callers.
import_and_export_cams_glob_ant = load_cams_glob_ant


__all__ = [
    "export_cams_glob_ant_to_microhh",
    "import_and_export_cams_glob_ant",
    "load_cams_glob_ant",
]
