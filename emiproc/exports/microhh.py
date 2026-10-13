"""Export an emiproc inventory to the horizontal grid of MicroHH."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from emiproc import deprecated
from emiproc.exports.netcdf import nc_cf_attributes
from emiproc.exports.utils import (
    FINGERPRINT_ATTRIBUTE,
    file_fingerprint,
    get_temporally_scaled_array,
    is_up_to_date,
    make_fingerprint,
    write_netcdf,
)
from emiproc.grids import RegularGrid, WGS84
from emiproc.inventories import Category, Inventory, Substance
from emiproc.inventories.cams_global import (
    CAMSFiles,
    CAMS_FLUX_UNIT,
    read_cams_global,
)
from emiproc.inventories.utils import (
    NegativePolicy,
    get_total_emissions,
    merge_inventories_without_double_counting,
)
from emiproc.regrid import remap_inventory
from emiproc.utils.translators import inv_to_xarray
from emiproc.utils.constants import SEC_PER_YR

logger = logging.getLogger(__name__)

#: Unit of the emissions written in the MicroHH files.
MICROHH_FLUX_UNIT = CAMS_FLUX_UNIT


def _inventory_bbox_for_grid(grid: RegularGrid) -> tuple[float, float, float, float]:
    """Return the target grid\'s longitude/latitude bounding box."""
    target = grid.gdf.to_crs(WGS84)
    west, south, east, north = target.total_bounds
    return float(west), float(south), float(east), float(north)


def _default_time_range(inv: Inventory, freq: str) -> pd.DatetimeIndex:
    """Build a one year time range for the inventory."""
    if inv.year is None:
        raise ValueError(
            "The inventory has no year, so the time range cannot be inferred."
            " Give `time_range=` explicitly."
        )
    return pd.date_range(
        start=f"{inv.year}-01-01",
        end=f"{inv.year + 1}-01-01",
        freq=freq,
        inclusive="left",
    )


def _constant_time_range(inv: Inventory) -> pd.DatetimeIndex:
    """Return the single time step used for time independent emissions."""
    if inv.year is None:
        raise ValueError(
            "The inventory has no year and no temporal profiles, so the time"
            " range cannot be inferred. Give `time_range=` explicitly."
        )
    return pd.DatetimeIndex([pd.Timestamp(f"{inv.year}-01-01")])


def _time_bounds(time_range: pd.DatetimeIndex, year: int | None) -> np.ndarray:
    """Return the ``(n_times, 2)`` averaging intervals of the time steps."""
    starts = time_range.to_numpy()
    if len(starts) > 1:
        last = starts[-1] + (starts[-1] - starts[-2])
    elif year is not None:
        last = np.datetime64(f"{year + 1}-01-01", "ns")
    else:
        last = starts[-1] + np.timedelta64(1, "D")
    ends = np.concatenate([starts[1:], [last]])
    return np.column_stack([starts, ends])


def _grid_fingerprint(grid: RegularGrid) -> dict:
    """Everything of the grid that changes the written file."""
    return {
        "name": grid.name,
        "crs": str(grid.crs),
        "nx": int(grid.nx),
        "ny": int(grid.ny),
        "dx": float(grid.dx),
        "dy": float(grid.dy),
        "xmin": float(grid.xmin),
        "ymin": float(grid.ymin),
    }


def _inventory_fingerprint(inv: Inventory) -> dict:
    """Identify the emissions of an inventory without writing them out."""
    totals = get_total_emissions(inv)
    return {
        "name": inv.name,
        "year": inv.year,
        "categories": sorted(map(str, inv.categories)),
        "substances": sorted(map(str, inv.substances)),
        "totals_kg_y": totals,
        "has_t_profiles": getattr(inv, "t_profiles_groups", None) is not None,
        "has_v_profiles": getattr(inv, "v_profiles", None) is not None,
    }


def _time_range_fingerprint(time_range) -> object:
    """A time range given as an index, a year, or not given at all."""
    if time_range is None or isinstance(time_range, (int, np.integer)):
        return time_range
    return [pd.Timestamp(t).isoformat() for t in time_range]


def export_to_microhh(
    inv: Inventory,
    grid: RegularGrid,
    output_file: PathLike,
    time_range: pd.DatetimeIndex | int | None = None,
    freq: str = "MS",
    substances: list[Substance] | None = None,
    categories: list[Category] | None = None,
    weights_file: PathLike | None = None,
    add_total: bool = True,
    category_descriptions: Mapping[str, str] | None = None,
    overwrite: bool = False,
    fingerprint: str | None = None,
) -> Path:
    """Export an inventory to a MicroHH emission NetCDF file.

    The inventory is remapped conservatively onto ``grid`` when it is not
    already on it, and its temporal profiles are used to resolve the emissions
    on the time axis. One ``emi_<substance>`` variable is written per
    substance, with dimensions ``(time, category, y, x)`` and units
    ``kg m-2 s-1``.

    Any :py:class:`~emiproc.inventories.Inventory` can be exported, so this
    works for CAMS-GLOB-ANT, TNO, EDGAR, a point-source database such as
    :py:class:`~emiproc.inventories.corso.CorsoInventory`, or a combination
    of them built with
    :py:func:`~emiproc.inventories.utils.merge_inventories_without_double_counting`.

    :arg inv: The inventory to export. It must have temporal profiles.
    :arg grid: The MicroHH grid, see
        :py:class:`~emiproc.grids.MicroHHGrid`. It must use a projected CRS.
    :arg output_file: Path of the NetCDF file to write.
    :arg time_range: The time steps to write. An integer is read as a year.
        By default the year of the inventory is used with ``freq``.
    :arg freq: Frequency of the default time range, monthly by default.
        See :py:func:`pandas.date_range`.
    :arg substances: Substances to export, all of them by default.
    :arg categories: Categories to export, all of them by default.
    :arg weights_file: Optional file storing the remapping weights.
        See :py:func:`~emiproc.regrid.remap_inventory`.
    :arg add_total: Whether to add an ``emi_<substance>_total`` variable
        holding the sum over the categories.
    :arg category_descriptions: Optional descriptions of the categories, added
        to the metadata of the ``category`` coordinate.
    :arg overwrite: Whether the output must be rebuilt even when it is already
        up to date. When False (the default), an existing file built from the
        same inputs is kept and simply returned, and one built from different
        inputs is replaced. See :py:func:`~emiproc.exports.utils.make_fingerprint`.
    :arg fingerprint: Optional fingerprint of the inputs, stored in the output
        and used for the check above. By default it is computed from ``inv``,
        ``grid`` and the arguments of this function. Pass it when the caller
        knows about inputs that this function cannot see, such as the files an
        inventory was read from.

    :return: The path of the file written, or of the up to date file found.

    .. note::
        Emissions are converted from the emiproc unit ``kg/y`` to
        ``kg m-2 s-1`` using the area of the target cells, so the export is
        mass conservative up to the cells that fall outside of ``grid``.
    """
    target_crs = pyproj.CRS.from_user_input(grid.crs)
    if target_crs.is_geographic:
        raise ValueError("The target MicroHH grid must use a projected CRS.")

    output_path = Path(output_file).with_suffix(".nc")
    if fingerprint is None:
        fingerprint = make_fingerprint(
            {
                "exporter": "emiproc.exports.microhh.export_to_microhh",
                "inventory": _inventory_fingerprint(inv),
                "grid": _grid_fingerprint(grid),
                "time_range": _time_range_fingerprint(time_range),
                "freq": freq,
                "substances": substances,
                "categories": categories,
                "add_total": add_total,
                "category_descriptions": category_descriptions,
            }
        )
    if not overwrite and is_up_to_date(output_path, fingerprint):
        logger.info("%s is already up to date, keeping it.", output_path)
        return output_path

    # Remapping copies the inventory, and `Inventory.copy` only keeps the
    # standard attributes, so read the reader specific metadata beforehand.
    source_metadata: dict[str, str | float | int] = {}
    if hasattr(inv, "source_files"):
        source_metadata["input_file_count"] = len(inv.source_files)
        source_metadata["input_files"] = " | ".join(inv.source_files)
    if hasattr(inv, "source_file_metadata"):
        source_metadata["input_details"] = " | ".join(inv.source_file_metadata)
    if hasattr(inv, "subtraction_summary"):
        summary = inv.subtraction_summary
        source_metadata.update(
            {
                "double_counting_negative_policy": str(summary["negative_policy"]),
                "double_counting_clipped_values": int(summary["clipped_values"]),
                "double_counting_removed_kg_y": str(summary["removed_kg_y"]),
                "double_counting_requested_kg_y": str(summary["requested_kg_y"]),
            }
        )
    source_descriptions = dict(getattr(inv, "category_descriptions", {}) or {})
    source_time_bounds = getattr(inv, "time_bounds", None)

    if getattr(inv, "grid", None) is not grid:
        logger.debug("Remapping %s on the MicroHH grid.", inv)
        inv = remap_inventory(inv, grid, weights_file=weights_file)

    has_profiles = (
        getattr(inv, "t_profiles_groups", None) is not None
        and getattr(inv, "t_profiles_indexes", None) is not None
    )

    if time_range is None:
        time_range = (
            _default_time_range(inv, freq)
            if has_profiles
            else _constant_time_range(inv)
        )
    elif isinstance(time_range, (int, np.integer)):
        time_range = pd.date_range(
            start=f"{int(time_range)}-01-01",
            end=f"{int(time_range) + 1}-01-01",
            freq=freq,
            inclusive="left",
        )
    elif not isinstance(time_range, pd.DatetimeIndex):
        raise TypeError(
            f"`time_range` must be a pd.DatetimeIndex or an int, got"
            f" {type(time_range)}."
        )

    # kg/y per cell, scaled along the time axis by the temporal profiles.
    if has_profiles:
        da = get_temporally_scaled_array(inv, time_range, sum_over_cells=False)
    else:
        logger.warning(
            "%s has no temporal profiles, the emissions are exported as"
            " constant over the %d time step(s).",
            inv,
            len(time_range),
        )
        da = inv_to_xarray(inv).expand_dims(time=time_range)
    # `Inventory.categories` and `.substances` are built from sets, so their
    # order varies between runs. Sort them to keep the output reproducible.
    da = da.sel(
        substance=(
            sorted(da["substance"].values, key=str)
            if substances is None
            else list(substances)
        ),
        category=(
            sorted(da["category"].values, key=str)
            if categories is None
            else list(categories)
        ),
    )

    target_areas = np.asarray(grid.cell_areas, dtype=float)
    # kg/y/cell -> kg/s/cell -> kg m-2 s-1
    da = da / SEC_PER_YR / xr.DataArray(target_areas, dims="cell")

    # Unstack the flat cell dimension onto the 2D grid.
    nx, ny = grid.shape
    da = da.assign_coords(
        x=("cell", np.repeat(np.arange(nx), ny)),
        y=("cell", np.tile(np.arange(ny), nx)),
    )
    da = da.assign_coords(
        xr.Coordinates.from_pandas_multiindex(
            pd.MultiIndex.from_arrays([da.x.values, da.y.values], names=["x", "y"]),
            "cell",
        )
    )
    da = da.unstack("cell").transpose("time", "category", "substance", "y", "x")

    exported_categories = [str(category) for category in da["category"].values]
    x = grid.lon_range - grid.xmin
    y = grid.lat_range - grid.ymin
    projected_x, projected_y = np.meshgrid(grid.lon_range, grid.lat_range)
    to_lonlat = pyproj.Transformer.from_crs(target_crs, WGS84, always_xy=True)
    longitude, latitude = to_lonlat.transform(projected_x, projected_y)

    data_vars = {}
    for substance in da["substance"].values:
        values = da.sel(substance=substance).values
        name = f"emi_{substance}"
        data_vars[name] = (
            ("time", "category", "y", "x"),
            values,
            {
                "long_name": f"{substance} emission flux",
                "units": MICROHH_FLUX_UNIT,
                "substance": str(substance),
                "grid_mapping": "crs",
            },
        )
        if add_total:
            data_vars[f"{name}_total"] = (
                ("time", "y", "x"),
                values.sum(axis=1),
                {
                    "long_name": f"total {substance} emission flux",
                    "units": MICROHH_FLUX_UNIT,
                    "substance": str(substance),
                    "grid_mapping": "crs",
                },
            )

    data_vars.update(
        {
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
                (
                    np.asarray(source_time_bounds)
                    if source_time_bounds is not None
                    and len(source_time_bounds) == len(time_range)
                    else _time_bounds(time_range, inv.year)
                ),
                {"long_name": "emission averaging interval"},
            ),
            "crs": ((), 0, target_crs.to_cf()),
        }
    )

    additional_attributes: dict[str, str | float | int] = {
        "target_grid": grid.name,
        "emiproc_history": str(inv.history),
        FINGERPRINT_ATTRIBUTE: fingerprint,
    }
    if inv.year is not None:
        additional_attributes["inventory_year"] = str(inv.year)
    additional_attributes.update(source_metadata)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": (
                "time",
                time_range.to_numpy(),
                {"standard_name": "time", "bounds": "time_bnds"},
            ),
            "category": exported_categories,
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
            title="emiproc emissions on a MicroHH grid",
            source=inv.name,
            history=f"Remapped and temporally resolved by emiproc from {inv}.",
            additional_attributes=additional_attributes,
        ),
    )
    descriptions = dict(category_descriptions or source_descriptions)
    ds["category"].attrs = {"long_name": "emission category"}
    if descriptions:
        ds["category"].attrs["descriptions"] = "; ".join(
            f"{category}: {descriptions[category]}"
            for category in exported_categories
            if category in descriptions
        )

    reference_year = inv.year if inv.year is not None else time_range[0].year
    time_encoding = {
        "units": f"seconds since {reference_year}-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }
    ds["time"].encoding.update(time_encoding)
    ds["time_bnds"].encoding.update(time_encoding)
    return write_netcdf(ds, output_path)


@deprecated("Use `export_to_microhh` instead, it works for any inventory.")
def export_cams_global_to_microhh(
    inventory: Inventory,
    grid: RegularGrid,
    output_file: PathLike,
    weights_file: PathLike | None = None,
    add_total: bool = True,
) -> Path:
    """Export a CAMS-GLOB-ANT inventory to a MicroHH file.

    .. deprecated::
        Use :py:func:`export_to_microhh`, which accepts any inventory.
    """
    return export_to_microhh(
        inventory,
        grid,
        output_file,
        weights_file=weights_file,
        add_total=add_total,
    )


def import_and_export_cams_global(
    nc_file: CAMSFiles,
    grid: RegularGrid,
    output_file: PathLike,
    variables: list[str] | None = None,
    substance: str | None = None,
    weights_file: PathLike | None = None,
    point_sources: Inventory | None = None,
    category_mapping: Mapping[str, str] | None = None,
    substance_mapping: Mapping[str, str] | None = None,
    negative_policy: NegativePolicy = "clip",
    date: datetime | None = None,
    keep_point_sources: bool = False,
    overwrite: bool = False,
) -> Path:
    """Read CAMS-GLOB-ANT files and export them on a MicroHH grid.

    This is a convenience wrapper over the standard emiproc pipeline:
    :py:func:`~emiproc.inventories.cams_global.read_cams_global` to read,
    :py:func:`~emiproc.inventories.utils.merge_inventories_without_double_counting`
    to remove the emissions already described by ``point_sources``, and
    :py:func:`export_to_microhh` to write. Call these three yourself when you
    need to apply other operators in between.

    :arg nc_file: One path, or several paths, to CAMS-GLOB-ANT files.
        See :py:func:`~emiproc.inventories.cams_global.read_cams_global`.
    :arg grid: The MicroHH grid to export to.
    :arg output_file: Path of the NetCDF file to write.
    :arg variables: Sector variables to import, only for a single file.
    :arg substance: Optional species name, normally inferred from the metadata.
    :arg weights_file: Optional file storing the remapping weights.
    :arg point_sources: Optional inventory of sources that CAMS already
        includes, ex. :py:class:`~emiproc.inventories.corso.CorsoInventory`.
        Its emissions are removed from the CAMS cells so that they are not
        counted twice.
    :arg category_mapping: Maps the categories of ``point_sources`` to the CAMS
        categories, ex.
        :py:data:`~emiproc.inventories.cams_global.DEFAULT_CORSO_SECTOR_MAPPING`.
    :arg substance_mapping: Maps the substances of ``point_sources`` to the
        CAMS substances.
    :arg negative_policy: What to do when the point sources exceed the CAMS
        emissions of a cell. See
        :py:func:`~emiproc.inventories.utils.subtract_inventories`.
    :arg date: Selects the matching month or day of the CAMS input.
    :arg keep_point_sources: Whether the point sources are added back to the
        inventory after having been removed from the CAMS cells. When False
        (the default) only the CAMS emissions that the point sources do not
        already describe are exported.
    :arg overwrite: Whether the output must be rebuilt even when it is already
        up to date. When False (the default), an existing file built from the
        same CAMS files, grid, point sources and options is kept and returned
        without reading the CAMS files again, and one built from anything
        different is replaced. The CAMS files are identified by their path,
        size and modification time.

    :return: The path of the file written, or of the up to date file found.

    .. note::
        When several CAMS files are summed, the result holds the single
        category ``"sum"``. The categories of ``point_sources`` are then all
        mapped onto it, as there is nothing finer to subtract them from, and
        ``category_mapping`` is not needed.
    """
    output_path = Path(output_file).with_suffix(".nc")
    paths = (
        [Path(nc_file)]
        if isinstance(nc_file, (str, PathLike))
        else [Path(path) for path in nc_file]
    )
    fingerprint = make_fingerprint(
        {
            "exporter": "emiproc.exports.microhh.import_and_export_cams_global",
            "cams_files": [file_fingerprint(path) for path in paths],
            "summed": not isinstance(nc_file, (str, PathLike)),
            "grid": _grid_fingerprint(grid),
            "variables": variables,
            "substance": substance,
            "date": date,
            "point_sources": (
                None if point_sources is None else _inventory_fingerprint(point_sources)
            ),
            "category_mapping": category_mapping,
            "substance_mapping": substance_mapping,
            "negative_policy": negative_policy,
            "keep_point_sources": keep_point_sources,
        }
    )
    if not overwrite and is_up_to_date(output_path, fingerprint):
        logger.info("%s is already up to date, keeping it.", output_path)
        return output_path
    inv = read_cams_global(
        nc_file,
        variables=variables,
        bbox=_inventory_bbox_for_grid(grid),
        substance=substance,
        date=date,
    )

    if point_sources is not None:
        if list(inv.categories) == ["sum"]:
            # Summed CAMS files hold one single category, so there is nothing
            # finer to subtract the point sources from. This is what the CAMS
            # specific subtraction used to do before it became an operator.
            summed_mapping = {category: "sum" for category in point_sources.categories}
            if category_mapping is not None and set(
                category_mapping.values()
            ).isdisjoint(inv.categories):
                logger.info(
                    "The CAMS files were summed into the single category"
                    " 'sum', so `category_mapping` is ignored and every"
                    " category of %s is subtracted from it.",
                    point_sources,
                )
            category_mapping = summed_mapping

        # The two remappings (point sources onto the CAMS grid, then CAMS onto
        # the MicroHH grid) must not share a weights file: `get_weights_mapping`
        # caches by file name without checking that the shapes still match.
        point_weights_file = (
            None
            if weights_file is None
            else Path(weights_file)
            .with_suffix(".npz")
            .with_stem(Path(weights_file).stem + "_point_sources")
        )
        if keep_point_sources:
            inv = merge_inventories_without_double_counting(
                inv,
                point_sources,
                category_mapping=category_mapping,
                substance_mapping=substance_mapping,
                negative_policy=negative_policy,
                weights_file=point_weights_file,
            )
        else:
            from emiproc.inventories.utils import subtract_inventories

            inv = subtract_inventories(
                inv,
                point_sources,
                category_mapping=category_mapping,
                substance_mapping=substance_mapping,
                negative_policy=negative_policy,
                weights_file=point_weights_file,
            )

    return export_to_microhh(
        inv,
        grid,
        output_path,
        # A `date` selects one single time step out of the CAMS input.
        time_range=None if date is None else pd.DatetimeIndex([date]),
        weights_file=weights_file,
        overwrite=True,
        fingerprint=fingerprint,
    )


#: Deprecated alias of :py:func:`import_and_export_cams_global`.
load_cams_global = import_and_export_cams_global


#: Deprecated aliases kept from the former CAMS-GLOB-ANT specific module.
export_cams_glob_ant_to_microhh = export_cams_global_to_microhh
load_cams_glob_ant = import_and_export_cams_global
import_and_export_cams_glob_ant = import_and_export_cams_global


__all__ = [
    "MICROHH_FLUX_UNIT",
    "export_to_microhh",
    "export_cams_global_to_microhh",
    "import_and_export_cams_global",
    "load_cams_global",
    "export_cams_glob_ant_to_microhh",
    "import_and_export_cams_glob_ant",
    "load_cams_glob_ant",
]
