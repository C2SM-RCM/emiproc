"""Import CAMS-GLOB-ANT monthly emissions as an emiproc inventory.

CAMS-GLOB-ANT contains monthly mean emission fluxes on a regular 0.1 degree
latitude/longitude grid. This module keeps those monthly fluxes unchanged;
model-specific export functions live in :mod:`emiproc.exports`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Literal
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.grids import (
    BoundingBox,
    RectilinearGrid,
    RegularGrid,
    WGS84,
    WGS84_NSIDC,
)
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import MounthsProfile
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.utilities import SEC_PER_YR

CAMS_FLUX_UNIT = "kg m-2 s-1"

#: One CAMS-GLOB-ANT file.
CAMSFile = str | PathLike
#: One or several CAMS-GLOB-ANT files.
CAMSFiles = CAMSFile | Sequence[CAMSFile]

DEFAULT_CORSO_SECTOR_MAPPING = {
    "power": "ene",
    "cement": "ind",
    "iron&steel": "ind",
    "syntheticfuel": "ref",
}

DEFAULT_CORSO_SUBSTANCE_MAPPING = {
    "CH4": "ch4",
    "CO": "co",
    "CO2": "co2",
    "CO2EXSC": "co2",
    "CO2SC": "co2",
    "NOX": "nox",
    "SO2": "sox",
    "SOX": "sox",
}

CorsoNegativePolicy = Literal["clip", "raise", "allow"]

SUBSTANCE_ALIASES = {
    "CARBONDIOXIDE": "CO2",
    "CO2": "CO2",
    "CO2EXSC": "CO2",
    "CO2SC": "CO2",
    "CARBONMONOXIDE": "CO",
    "CO": "CO",
    "NMVOC": "VOCs",
    "NMVOCS": "VOCs",
    "NONMETHANEVOC": "VOCs",
    "NONMETHANEVOCS": "VOCs",
    "OTHERVOC": "VOCs",
    "OTHERVOCS": "VOCs",
    "VOC": "VOCs",
    "VOCS": "VOCs",
    "VOCSALL": "VOCs",
    "VOLATILEORGANICCOMPOUNDS": "VOCs",
    "VOLATILEORGANICCOMPOUNDSALL": "VOCs",
}

VOC_COMPONENT_KEYS = {
    "ISOPRENE",
    "MONOTERPENE",
    "MONOTERPENES",
    *(key for key, canonical in SUBSTANCE_ALIASES.items() if canonical == "VOCs"),
}


def _canonical_substance(substance: str) -> str:
    """Return a common name for equivalent CAMS substance labels."""
    normalized = re.sub(r"[^A-Z0-9]", "", substance.upper())
    return SUBSTANCE_ALIASES.get(normalized, substance.strip())


def resolve_cams_substances(
    substances: list[str],
    override: str | None = None,
) -> str:
    """Resolve source labels to one order-independent aggregate substance."""
    if override is not None:
        return _canonical_substance(override)
    canonical = [_canonical_substance(value) for value in substances]
    if len(set(canonical)) == 1:
        return canonical[0]
    keys = {re.sub(r"[^A-Z0-9]", "", value.upper()) for value in substances}
    if keys.issubset(VOC_COMPONENT_KEYS):
        return "VOCs"
    raise ValueError(
        "Cannot sum CAMS inventories with different substances: "
        + ", ".join(repr(value) for value in substances)
        + ". Pass `substance=` only when these files intentionally form one "
        "aggregate substance."
    )


def _infer_year(ds: xr.Dataset, path: Path) -> int:
    """Infer the inventory year without decoding CAMS' ``months since`` axis."""
    units = str(ds["time"].attrs.get("units", ""))
    match = re.search(r"(?:months|days) since (\d{4})-", units)
    if match:
        return int(match.group(1))

    match = re.search(r"_(\d{4})\.nc$", path.name)
    if match:
        return int(match.group(1))

    raise ValueError(
        "Could not infer the CAMS-GLOB-ANT year from the time units or filename."
    )


def _infer_substance(ds: xr.Dataset, variables: list[str], path: Path) -> str:
    """Infer the chemical species name from CAMS variable metadata."""
    candidates = ["sum", *variables]
    for variable in candidates:
        if variable not in ds:
            continue
        long_name = str(ds[variable].attrs.get("long_name", ""))
        match = re.search(r"Emissions of (.+?) for ", long_name)
        if match:
            return match.group(1).strip()
        match = re.match(r"(.+?) emissions(?:\s|$)", long_name, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        standard_name = str(ds[variable].attrs.get("standard_name", ""))
        match = re.search(r"mass_content_of_(.+?)_due_to_emission", standard_name)
        if match:
            return match.group(1).replace("_", " ").strip()

        bare_long_name = re.sub(
            r"\s*\([^)]*\)\s*$", "", long_name, flags=re.IGNORECASE
        ).strip()
        if bare_long_name:
            return bare_long_name

    match = re.match(
        r"CAMS-GLOB-(?:ANT|BIO|OCE|SHIP)_v[^_]+_(.+)_\d{4}\.nc$",
        path.name,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).replace("-", " ")

    raise ValueError(
        "Could not infer the substance. Pass it explicitly with `substance=`."
    )


def _time_index_for_date(
    ds: xr.Dataset,
    path: Path,
    requested_date: datetime,
) -> int:
    """Return the monthly or daily time index containing ``requested_date``."""
    if not isinstance(requested_date, datetime):
        raise TypeError("`date` must be a datetime.datetime instance.")

    units = str(ds["time"].attrs.get("units", ""))
    match = re.fullmatch(
        r"(months|days) since (\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?",
        units.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Cannot select a date from {path}: unsupported time units {units!r}."
        )

    origin_text = match.group(2)
    if match.group(3):
        origin_text += f" {match.group(3)}"
    origin = datetime.fromisoformat(origin_text)
    requested_day = datetime(
        requested_date.year,
        requested_date.month,
        requested_date.day,
    )
    if match.group(1).lower() == "months":
        offset = (requested_day.year - origin.year) * 12 + (
            requested_day.month - origin.month
        )
    else:
        offset = (requested_day - origin).days

    values = np.asarray(ds["time"].values, dtype=float)
    matches = np.flatnonzero(np.isclose(values, offset, rtol=0, atol=1e-6))
    if len(matches) != 1:
        raise ValueError(f"Date {requested_date:%Y-%m-%d} is not available in {path}.")
    return int(matches[0])


def _validate_flux_variable(da: xr.DataArray, variable: str) -> None:
    required_dims = {"time", "lat", "lon"}
    if not required_dims.issubset(da.dims):
        raise ValueError(
            f"CAMS variable {variable!r} must contain dimensions {required_dims}, "
            f"found {da.dims}."
        )
    unit = str(da.attrs.get("units", ""))
    normalized = unit.replace(" ", "")
    if normalized not in {"kgm-2s-1", "kg/m2/s"}:
        raise ValueError(
            f"CAMS variable {variable!r} has unsupported units {unit!r}; "
            f"expected {CAMS_FLUX_UNIT!r}."
        )


def select_cams_global_variable(nc_file: str | PathLike) -> str:
    """Select ``sum`` or the sole emission-flux field from a CAMS file."""
    path = Path(nc_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, decode_times=False) as dataset:
        candidates = []
        for name, variable in dataset.data_vars.items():
            if not {"time", "lat", "lon"}.issubset(variable.dims):
                continue
            unit = str(variable.attrs.get("units", "")).replace(" ", "")
            if unit in {"kgm-2s-1", "kg/m2/s"}:
                candidates.append(name)

    if "sum" in candidates:
        return "sum"
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot select one emission variable from {path}: expected a 'sum' "
        "field or exactly one time/latitude/longitude field with supported "
        f"flux units, found {len(candidates)} candidates ({', '.join(candidates)})."
    )


class CAMSGlobal(Inventory):
    """A cropped CAMS-GLOB-ANT inventory for one substance and year.

    Parameters
    ----------
    nc_file:
        CAMS-GLOB-ANT NetCDF file containing one chemical species.
    variables:
        Sector variables to import. By default, all variables carrying a
        ``sector_name`` attribute are selected and the aggregate ``sum``
        variable is excluded to prevent double counting.
    bbox:
        Optional longitude/latitude bounding box ``(west, south, east, north)``.
        Every source cell intersecting the box is retained.
    substance:
        Optional species name. It is normally inferred from CAMS metadata.
    date:
        Optional :class:`datetime.datetime` selecting the applicable monthly
        value or exact daily value. Required for inputs that do not contain 12
        monthly time steps.

    Notes
    -----
    The emissions are stored in the standard emiproc
    :py:attr:`~emiproc.inventories.Inventory.gdf` in ``kg year-1`` per cell,
    and the monthly cycle is stored as regular emiproc temporal profiles. Every
    emiproc operator and exporter therefore works on this inventory, including
    :py:func:`~emiproc.exports.microhh.export_to_microhh`.

    :py:attr:`monthly_flux` keeps the fluxes exactly as they were read from the
    CAMS files, in ``kg m-2 s-1`` with dimensions ``(time, category, cell)``.
    It is provenance of the input file only: emiproc operators modify ``gdf``
    and the profiles, not ``monthly_flux``, so do not use it after the
    inventory has been transformed.

    To combine this inventory with an explicit point-source database without
    counting the same emissions twice, use
    :py:func:`~emiproc.inventories.utils.merge_inventories_without_double_counting`
    together with :py:data:`DEFAULT_CORSO_SECTOR_MAPPING` and
    :py:data:`DEFAULT_CORSO_SUBSTANCE_MAPPING`.
    """

    grid: RegularGrid | RectilinearGrid
    monthly_flux: xr.DataArray
    category_descriptions: dict[str, str]
    time_bounds: np.ndarray

    def __init__(
        self,
        nc_file: PathLike,
        variables: list[str] | None = None,
        bbox: BoundingBox | None = None,
        substance: str | None = None,
        date: datetime | None = None,
    ) -> None:
        path = Path(nc_file)
        if not path.is_file():
            raise FileNotFoundError(path)

        self.name = f"CAMS_GLOBAL_{path.stem}"
        super().__init__()

        with xr.open_dataset(path, decode_times=False) as source_ds:
            if not {"time", "lat", "lon"}.issubset(source_ds.variables):
                raise ValueError(
                    "CAMS dataset must define time, lat, and lon variables."
                )

            self.year = _infer_year(source_ds, path)
            if date is None and source_ds.sizes["time"] != 12:
                raise ValueError(
                    "CAMS input must contain 12 monthly time steps when `date` "
                    f"is not provided, found {source_ds.sizes['time']}. Pass a "
                    "datetime with `date=` for daily inventories."
                )
            time_index = (
                _time_index_for_date(source_ds, path, date)
                if date is not None
                else None
            )

            if variables is None:
                selected = [
                    name
                    for name, da in source_ds.data_vars.items()
                    if name != "sum" and "sector_name" in da.attrs
                ]
                if not selected:
                    selected = [select_cams_global_variable(path)]
            else:
                selected = list(variables)
            if not selected:
                raise ValueError("No CAMS-GLOB-ANT sector variables were selected.")
            if "sum" in selected and len(selected) > 1:
                raise ValueError(
                    "The aggregate 'sum' variable cannot be selected together with "
                    "component sectors because that would double-count emissions."
                )
            missing = sorted(set(selected) - set(source_ds.data_vars))
            if missing:
                raise ValueError(f"Variables not found in CAMS dataset: {missing}")

            for variable in selected:
                _validate_flux_variable(source_ds[variable], variable)

            self.source_substance_name = substance or _infer_substance(
                source_ds, selected, path
            )
            self.substance_name = _canonical_substance(self.source_substance_name)
            self.selected_variables = list(selected)

            ds = source_ds[selected]
            if time_index is not None:
                ds = ds.isel(time=[time_index])
            if bbox is not None:
                west, south, east, north = bbox
                if west > east:
                    raise ValueError(
                        "Dateline-crossing bounding boxes are not supported."
                    )
                lon_step = float(np.abs(np.diff(source_ds["lon"].values)).max())
                lat_step = float(np.abs(np.diff(source_ds["lat"].values)).max())
                ds = ds.sel(
                    lon=slice(west - lon_step, east + lon_step),
                    lat=slice(south - lat_step, north + lat_step),
                )
            ds = ds.load()

            if ds.sizes["lon"] == 0 or ds.sizes["lat"] == 0:
                raise ValueError(
                    "The requested bounding box does not intersect the data."
                )

            try:
                self.grid = RegularGrid.from_centers(
                    x_centers=ds["lon"].values,
                    y_centers=ds["lat"].values,
                    name="CAMS-GLOBAL",
                    crs=WGS84,
                    # Nominally regular coordinates are commonly float32.
                    rounding=4,
                )
            except ValueError:
                self.grid = RectilinearGrid(
                    x_centers=ds["lon"].values,
                    y_centers=ds["lat"].values,
                    name="CAMS-GLOBAL",
                    crs=WGS84,
                )
            self.category_descriptions = {
                variable: str(ds[variable].attrs.get("sector_name", variable))
                for variable in selected
            }

            if date is None:
                interval_starts = pd.date_range(
                    f"{self.year}-01-01", periods=13, freq="MS"
                )
                time_coordinates = (
                    interval_starts[:-1]
                    + (interval_starts[1:].values - interval_starts[:-1].values) / 2
                )
                self.time_bounds = np.column_stack(
                    [interval_starts[:-1].values, interval_starts[1:].values]
                )
            else:
                selected_day = datetime(date.year, date.month, date.day)
                time_coordinates = pd.DatetimeIndex([date])
                self.time_bounds = np.asarray(
                    [[selected_day, selected_day + timedelta(days=1)]],
                    dtype="datetime64[ns]",
                )

            monthly = xr.concat(
                [ds[var].expand_dims(category=[var]) for var in selected],
                dim="category",
            ).assign_coords(time=time_coordinates)
            monthly.attrs = {
                "long_name": f"monthly mean {self.substance_name} emission flux",
                "units": CAMS_FLUX_UNIT,
            }
            self.monthly_flux = (
                monthly.transpose("time", "category", "lon", "lat")
                .stack(cell=("lon", "lat"))
                .reset_index("cell")
                .drop_vars(["lon", "lat"])
                .assign_coords(cell=np.arange(len(self.grid)))
            )

        self._rebuild_derived_data()

    def _rebuild_derived_data(self) -> None:
        """Rebuild standard inventory data from authoritative monthly fluxes."""
        source_areas = np.asarray(self.grid.cell_areas)
        annual_mean = self.monthly_flux.astype(np.float64).mean("time")
        categories = self.monthly_flux["category"].values.astype(str)
        self.gdf = gpd.GeoDataFrame(
            {
                (category, self.substance_name): (
                    annual_mean.sel(category=category).values
                    * SEC_PER_YR
                    * source_areas
                )
                for category in categories
            },
            geometry=self.grid.gdf.geometry,
            crs=self.grid.crs,
        )
        self.gdfs = {}

        if self.monthly_flux.sizes["time"] != 12:
            self.t_profiles_groups = None
            self.t_profiles_indexes = None
            return

        # CAMS files are float32; compute the ratios in double precision so
        # that the profiles reproduce the input fluxes to float64 accuracy.
        ratios_da = (
            self.monthly_flux.astype(np.float64)
            .rename({"time": "ratio"})
            .expand_dims(substance=[self.substance_name])
        )
        ratios_da = ratios_da / ratios_da.sum("ratio")
        ratios, indexes = ratios_dataarray_to_profiles(ratios_da.fillna(0.0))
        self.set_profiles(
            CompositeTemporalProfiles.from_ratios(
                ratios, types=[MounthsProfile], rescale=True
            ),
            indexes,
        )


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
    from emiproc.regrid import get_weights_mapping, weights_remap

    source_shapes = source.grid.gdf.geometry.to_crs(WGS84_NSIDC).reset_index(drop=True)
    target_shapes = target.grid.gdf.geometry.to_crs(WGS84_NSIDC).reset_index(drop=True)
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
        source_flux = source.monthly_flux.isel(time=time_index, category=0).values
        target_rate = weights_remap(
            mapping,
            source_flux * source_areas,
            len(target.grid),
        )
        remapped[time_index] = target_rate / target_areas
    return remapped


def read_cams_global(
    nc_files: CAMSFiles,
    variables: list[str] | None = None,
    bbox: BoundingBox | None = None,
    substance: str | None = None,
    date: datetime | None = None,
) -> CAMSGlobal:
    """Read one or several CAMS-GLOB-ANT files as a single inventory.

    A single path (not in a sequence) behaves exactly like
    :py:class:`CAMSGlobal`, keeping the standard variable selection.
    A sequence always takes the summation path, even with one element.

    For several paths, the ``sum`` variable is selected from each file when it
    is present, and otherwise the file must contain exactly one data variable.
    Those single-variable fluxes are summed after checking that the files
    describe the same time interval, which makes ``variables`` meaningless and
    therefore rejected. Files that are not on the same grid are conservatively
    remapped onto the finest of the input grids before being summed.

    :arg nc_files: One path, or a sequence of paths, to CAMS-GLOB-ANT files.
    :arg variables: Sector variables to import, only for a single file.
    :arg bbox: Optional ``(west, south, east, north)`` bounding box in WGS84.
    :arg substance: Optional species name, normally inferred from the metadata.
    :arg date: Selects the matching month of a monthly inventory, or the exact
        day of a daily one. It must be given for inputs that do not contain 12
        monthly time steps.

    :return: The inventory holding the summed emissions.
    """
    if isinstance(nc_files, (str, PathLike)):
        return CAMSGlobal(
            nc_files,
            variables=variables,
            bbox=bbox,
            substance=substance,
            date=date,
        )

    paths = [Path(path) for path in nc_files]
    if not paths:
        raise ValueError("At least one CAMS inventory file must be provided.")
    if variables is not None:
        raise ValueError(
            "`variables` cannot be provided when the files are given as a"
            " sequence, because the summation variable is selected from each"
            " file. Pass a single path to choose the variables yourself."
        )

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
    reference.history.append(f"Summed {len(paths)} CAMS global inventory files")
    if remapped_files:
        reference.history.append(
            f"Conservatively remapped {remapped_files} input files to the "
            "finest CAMS source grid before summation"
        )

    reference._rebuild_derived_data()
    return reference


CAMS_GLOBAL = CAMSGlobal

# Compatibility aliases for the former inventory name.
CAMSGlobAnt = CAMSGlobal
CAMS_GLOB_ANT = CAMSGlobal


__all__ = [
    "CAMSFile",
    "CAMSFiles",
    "CAMSGlobal",
    "CAMS_GLOBAL",
    "CAMSGlobAnt",
    "CAMS_GLOB_ANT",
    "CAMS_FLUX_UNIT",
    "DEFAULT_CORSO_SECTOR_MAPPING",
    "DEFAULT_CORSO_SUBSTANCE_MAPPING",
    "SUBSTANCE_ALIASES",
    "VOC_COMPONENT_KEYS",
    "read_cams_global",
    "resolve_cams_substances",
    "select_cams_global_variable",
]
