"""Conservative CAMS-GLOB-ANT export to a projected MicroHH grid."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import pyproj
import xarray as xr

from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.grids import RegularGrid, WGS84, WGS84_NSIDC
from emiproc.inventories.cams_global import (
    CAMSGlobal,
    CAMS_FLUX_UNIT,
    resolve_cams_substances,
    select_cams_global_variable,
)
from emiproc.regrid import get_weights_mapping, weights_remap

CAMSFile = str | PathLike
CAMSFiles = CAMSFile | Sequence[CAMSFile]


def _inventory_bbox_for_grid(grid: RegularGrid) -> tuple[float, float, float, float]:
    """Return the target grid's longitude/latitude bounding box."""
    target = grid.gdf.to_crs(WGS84)
    west, south, east, north = target.total_bounds
    return float(west), float(south), float(east), float(north)


def _summation_variable(nc_file: CAMSFile) -> str:
    """Select the one flux variable used when summing multiple files."""
    return select_cams_global_variable(nc_file)


def _validate_summation_compatibility(
    reference: CAMSGlobal,
    other: CAMSGlobal,
    other_file: Path,
) -> None:
    """Ensure two selected CAMS inventories describe the same time interval."""
    errors = []
    if reference.year != other.year:
        errors.append(f"year {other.year} != {reference.year}")
    if not np.array_equal(
        reference.monthly_flux["time"].values,
        other.monthly_flux["time"].values,
    ):
        errors.append("time coordinates differ")
    if errors:
        raise ValueError(
            f"Cannot sum CAMS inventory {other_file}: " + "; ".join(errors) + "."
        )


def _grids_match(reference: CAMSGlobal, other: CAMSGlobal) -> bool:
    """Return whether two source grids have identical cells and ordering."""
    return (
        reference.grid.crs == other.grid.crs
        and reference.grid.shape == other.grid.shape
        and np.allclose(reference.grid.lon_bounds, other.grid.lon_bounds)
        and np.allclose(reference.grid.lat_bounds, other.grid.lat_bounds)
    )


def _remap_flux_to_inventory_grid(
    source: CAMSGlobal,
    target: CAMSGlobal,
) -> np.ndarray:
    """Conservatively remap one selected CAMS field to another CAMS grid."""
    source_shapes = source.grid.gdf.geometry.to_crs(WGS84_NSIDC).reset_index(
        drop=True
    )
    target_shapes = target.grid.gdf.geometry.to_crs(WGS84_NSIDC).reset_index(
        drop=True
    )
    source_areas = source_shapes.area.to_numpy()
    target_areas = target_shapes.area.to_numpy()
    mapping = get_weights_mapping(
        None,
        source_shapes,
        target_shapes,
        loop_over_inv_objects=False,
    )
    remapped = np.empty(
        (source.monthly_flux.sizes["time"], len(target.grid)),
        dtype=np.float64,
    )
    for time_index in range(source.monthly_flux.sizes["time"]):
        source_flux = source.monthly_flux.isel(
            time=time_index, category=0
        ).values
        target_rate = weights_remap(
            mapping,
            source_flux * source_areas,
            len(target.grid),
        )
        remapped[time_index] = target_rate / target_areas
    return remapped


def _sum_cams_global_files(
    nc_files: Sequence[CAMSFile],
    bbox: tuple[float, float, float, float],
    substance: str | None,
    corso_dataset: xr.Dataset | None,
    corso_substance: str | None,
    corso_sector_mapping: Mapping[str, str] | None,
    corso_negative_policy: Literal["clip", "raise", "allow"],
    date: datetime | None,
) -> CAMSGlobal:
    """Load compatible CAMS files, sum their fluxes, then subtract CORSO."""
    paths = [Path(path) for path in nc_files]
    if not paths:
        raise ValueError("At least one CAMS inventory file must be provided.")

    inventories = [
        CAMSGlobal(
            path,
            variables=[_summation_variable(path)],
            bbox=bbox,
            substance=substance,
            date=date,
        )
        for path in paths
    ]
    temporal_reference = inventories[0]
    for path, inventory in zip(paths[1:], inventories[1:]):
        _validate_summation_compatibility(temporal_reference, inventory, path)

    aggregate_substance = resolve_cams_substances(
        [inventory.source_substance_name for inventory in inventories],
        override=substance,
    )
    reference = min(
        inventories,
        key=lambda inventory: float(np.median(inventory.grid.cell_areas)),
    )
    summed_values = np.zeros(
        (reference.monthly_flux.sizes["time"], len(reference.grid)),
        dtype=np.float64,
    )
    remapped_files = 0
    for inventory in inventories:
        if _grids_match(reference, inventory):
            summed_values += inventory.monthly_flux.isel(category=0).values
        else:
            summed_values += _remap_flux_to_inventory_grid(inventory, reference)
            remapped_files += 1

    reference.monthly_flux = (
        reference.monthly_flux.isel(category=[0])
        .assign_coords(category=["sum"])
        .astype(np.float64)
    )
    reference.monthly_flux.data = summed_values[:, None, :]
    reference.substance_name = aggregate_substance
    reference.monthly_flux.attrs.update(
        {
            "long_name": (
                f"summed {reference.substance_name} emission flux for selected "
                "time intervals"
            ),
            "summed_input_files": len(paths),
        }
    )
    reference.category_descriptions = {
        "sum": f"Sum of {len(paths)} CAMS global inventory files"
    }
    reference.name = f"CAMS_GLOBAL_SUM_{len(paths)}_FILES"
    reference.source_files = [str(path) for path in paths]
    reference.source_file_metadata = [
        (
            f"{path}: variable={inventory.selected_variables[0]}, "
            f"source_substance={inventory.source_substance_name}"
        )
        for path, inventory in zip(paths, inventories)
    ]
    reference.history.append(
        f"Summed {len(paths)} CAMS global inventory files before CORSO subtraction"
    )
    if remapped_files:
        reference.history.append(
            f"Conservatively remapped {remapped_files} input files to the "
            "finest CAMS source grid before summation"
        )

    if corso_dataset is not None:
        reference._subtract_corso_emissions(
            corso_dataset,
            corso_substance=corso_substance,
            sector_mapping=corso_sector_mapping,
            negative_policy=corso_negative_policy,
        )
    reference._rebuild_derived_data()
    return reference


def export_cams_global_to_microhh(
    inventory: CAMSGlobal,
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
    temporal_label = (
        "monthly mean"
        if inventory.monthly_flux.sizes["time"] == 12
        else "selected-time"
    )
    emission_name = f"emi_{inventory.substance_name}"
    additional_attributes: dict[str, str | float | int] = {
        "inventory_year": str(inventory.year),
        "target_grid": grid.name,
        "emiproc_history": str(inventory.history),
    }
    if hasattr(inventory, "source_files"):
        additional_attributes["cams_global_input_file_count"] = len(
            inventory.source_files
        )
        additional_attributes["cams_global_input_files"] = " | ".join(
            inventory.source_files
        )
    if hasattr(inventory, "source_file_metadata"):
        additional_attributes["cams_global_input_details"] = " | ".join(
            inventory.source_file_metadata
        )
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
                        f"{temporal_label} {inventory.substance_name} emission flux"
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
                f"{temporal_label} total {inventory.substance_name} emission flux"
            ),
            "units": CAMS_FLUX_UNIT,
            "substance": inventory.substance_name,
            "grid_mapping": "crs",
        }

    output_path = Path(output_file).with_suffix(".nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_encoding = {
        "units": f"seconds since {inventory.year}-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }
    ds["time"].encoding.update(time_encoding)
    ds["time_bnds"].encoding.update(time_encoding)
    ds.to_netcdf(output_path)
    return output_path


def load_cams_global(
    nc_file: CAMSFiles,
    grid: RegularGrid,
    output_file: PathLike,
    variables: list[str] | None = None,
    substance: str | None = None,
    weights_file: PathLike | None = None,
    corso_dataset: xr.Dataset | None = None,
    corso_substance: str | None = None,
    corso_sector_mapping: Mapping[str, str] | None = None,
    corso_negative_policy: Literal["clip", "raise", "allow"] = "clip",
    date: datetime | None = None,
) -> Path:
    """Crop, import, remap, and export CAMS-GLOB-ANT files.

    A scalar path retains the standard :class:`CAMSGlobal` variable-selection
    behavior. For a sequence of paths, ``sum`` is selected from each file when
    present; otherwise the file must contain exactly one data variable. Those
    single-variable fluxes are summed after compatibility validation. The
    ``variables`` argument is consequently not accepted with multiple-file
    input.

    ``date`` selects the matching month from monthly inventories and the exact
    day from daily shipping or oceanic inventories. It must be provided for
    inputs that do not contain 12 monthly time steps.

    When ``corso_dataset`` is provided, its raw annual point-source emissions
    are subtracted from the corresponding CAMS cells after all CAMS files have
    been summed and before conservative remapping. See
    :class:`emiproc.inventories.cams_global.CAMSGlobal` for species mapping,
    sector mapping, and negative-residual behavior.
    """
    bbox = _inventory_bbox_for_grid(grid)
    if isinstance(nc_file, Sequence) and not isinstance(nc_file, (str, bytes)):
        if variables is not None:
            raise ValueError(
                "`variables` cannot be provided with a sequence of CAMS files; "
                "each file uses 'sum' or its sole data variable."
            )
        inventory = _sum_cams_global_files(
            nc_file,
            bbox=bbox,
            substance=substance,
            corso_dataset=corso_dataset,
            corso_substance=corso_substance,
            corso_sector_mapping=corso_sector_mapping,
            corso_negative_policy=corso_negative_policy,
            date=date,
        )
    else:
        selected_variables = variables
        if date is not None and selected_variables is None:
            selected_variables = [_summation_variable(nc_file)]
        inventory = CAMSGlobal(
            nc_file,
            variables=selected_variables,
            bbox=bbox,
            substance=substance,
            corso_dataset=corso_dataset,
            corso_substance=corso_substance,
            corso_sector_mapping=corso_sector_mapping,
            corso_negative_policy=corso_negative_policy,
            date=date,
        )
    return export_cams_global_to_microhh(
        inventory, grid, output_file, weights_file=weights_file
    )


# Public convenience name plus compatibility aliases for the former inventory
# name.
import_and_export_cams_global = load_cams_global
export_cams_glob_ant_to_microhh = export_cams_global_to_microhh
load_cams_glob_ant = load_cams_global
import_and_export_cams_glob_ant = load_cams_global


__all__ = [
    "export_cams_global_to_microhh",
    "import_and_export_cams_global",
    "load_cams_global",
    "export_cams_glob_ant_to_microhh",
    "import_and_export_cams_glob_ant",
    "load_cams_glob_ant",
]
