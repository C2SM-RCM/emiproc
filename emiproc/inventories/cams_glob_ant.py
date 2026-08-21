"""Import CAMS-GLOB-ANT monthly emissions as an emiproc inventory.

CAMS-GLOB-ANT contains monthly mean emission fluxes on a regular 0.1 degree
latitude/longitude grid. This module keeps those monthly fluxes unchanged;
model-specific export functions live in :mod:`emiproc.exports`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal
from warnings import warn

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.grids import BoundingBox, RegularGrid, WGS84
from emiproc.inventories import Inventory
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import MounthsProfile
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.utilities import SEC_PER_YR

CAMS_FLUX_UNIT = "kg m-2 s-1"

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


def _infer_year(ds: xr.Dataset, path: Path) -> int:
    """Infer the inventory year without decoding CAMS' ``months since`` axis."""
    units = str(ds["time"].attrs.get("units", ""))
    match = re.search(r"months since (\d{4})-", units)
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

    match = re.match(r"CAMS-GLOB-ANT_[^_]+_(.+)_\d{4}\.nc$", path.name)
    if match:
        return match.group(1)

    raise ValueError(
        "Could not infer the substance. Pass it explicitly with `substance=`."
    )


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


class CAMSGlobAnt(Inventory):
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
    corso_dataset:
        Optional raw dataset returned by
        :meth:`emiproc.inventories.corso.CorsoEmissions.get_dataset`. Matching
        annual CORSO point-source emissions are removed from their CAMS cells.
    corso_substance:
        Optional CORSO species prefix or ``*_kty`` variable name. By default it
        is inferred from the CAMS substance.
    corso_sector_mapping:
        Optional additions or overrides to the default mapping from CORSO
        sectors to CAMS-GLOB-ANT categories.
    corso_negative_policy:
        Behavior when CORSO exceeds a monthly CAMS value: ``"clip"`` sets the
        residual to zero, ``"raise"`` rejects the import, and ``"allow"``
        retains a negative residual.

    Notes
    -----
    ``monthly_flux`` is the authoritative representation and has dimensions
    ``(time, category, cell)`` in ``kg m-2 s-1``. The standard emiproc ``gdf``
    stores the equal-month annual mean in ``kg year-1 cell-1`` for compatibility
    with the :class:`~emiproc.inventories.Inventory` interface.
    """

    grid: RegularGrid
    monthly_flux: xr.DataArray
    category_descriptions: dict[str, str]
    time_bounds: np.ndarray
    corso_subtraction_summary: dict[str, float | int | str]

    def __init__(
        self,
        nc_file: PathLike,
        variables: list[str] | None = None,
        bbox: BoundingBox | None = None,
        substance: str | None = None,
        corso_dataset: xr.Dataset | None = None,
        corso_substance: str | None = None,
        corso_sector_mapping: Mapping[str, str] | None = None,
        corso_negative_policy: CorsoNegativePolicy = "clip",
    ) -> None:
        path = Path(nc_file)
        if not path.is_file():
            raise FileNotFoundError(path)

        self.name = f"CAMS_GLOB_ANT_{path.stem}"
        super().__init__()

        with xr.open_dataset(path, decode_times=False) as source_ds:
            if not {"time", "lat", "lon"}.issubset(source_ds.variables):
                raise ValueError(
                    "CAMS dataset must define time, lat, and lon variables."
                )

            self.year = _infer_year(source_ds, path)
            if source_ds.sizes["time"] != 12:
                raise ValueError(
                    "CAMS-GLOB-ANT input must contain 12 monthly time steps, "
                    f"found {source_ds.sizes['time']}."
                )

            selected = (
                [
                    name
                    for name, da in source_ds.data_vars.items()
                    if name != "sum" and "sector_name" in da.attrs
                ]
                if variables is None
                else list(variables)
            )
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

            self.substance_name = substance or _infer_substance(
                source_ds, selected, path
            )

            ds = source_ds[selected]
            if bbox is not None:
                west, south, east, north = bbox
                if west > east:
                    raise ValueError(
                        "Dateline-crossing bounding boxes are not supported."
                    )
                lon_step = float(np.abs(np.diff(source_ds["lon"].values)).min())
                lat_step = float(np.abs(np.diff(source_ds["lat"].values)).min())
                ds = ds.sel(
                    lon=slice(west - lon_step, east + lon_step),
                    lat=slice(south - lat_step, north + lat_step),
                )
            ds = ds.load()

            if ds.sizes["lon"] == 0 or ds.sizes["lat"] == 0:
                raise ValueError(
                    "The requested bounding box does not intersect the data."
                )

            self.grid = RegularGrid.from_centers(
                x_centers=ds["lon"].values,
                y_centers=ds["lat"].values,
                name="CAMS-GLOB-ANT",
                crs=WGS84,
                # CAMS stores nominally regular coordinates as float32. At
                # mid-latitudes, a 0.1 degree step consequently varies by a
                # few 1e-6 degrees solely due to representation precision.
                rounding=4,
            )
            self.category_descriptions = {
                variable: str(ds[variable].attrs.get("sector_name", variable))
                for variable in selected
            }

            month_starts = pd.date_range(
                f"{self.year}-01-01", periods=13, freq="MS"
            )
            month_midpoints = month_starts[:-1] + (
                month_starts[1:].values - month_starts[:-1].values
            ) / 2
            self.time_bounds = np.column_stack(
                [month_starts[:-1].values, month_starts[1:].values]
            )

            monthly = xr.concat(
                [ds[var].expand_dims(category=[var]) for var in selected],
                dim="category",
            ).assign_coords(time=month_midpoints)
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

        if corso_dataset is not None:
            self._subtract_corso_emissions(
                corso_dataset,
                corso_substance=corso_substance,
                sector_mapping=corso_sector_mapping,
                negative_policy=corso_negative_policy,
            )

        source_areas = np.asarray(self.grid.cell_areas)
        annual_mean = self.monthly_flux.mean("time")
        self.gdf = gpd.GeoDataFrame(
            {
                (category, self.substance_name): (
                    annual_mean.sel(category=category).values
                    * SEC_PER_YR
                    * source_areas
                )
                for category in selected
            },
            geometry=self.grid.gdf.geometry,
            crs=self.grid.crs,
        )
        self.gdfs = {}

        ratios_da = self.monthly_flux.rename({"time": "ratio"}).expand_dims(
            substance=[self.substance_name]
        )
        ratios_da = ratios_da / ratios_da.sum("ratio")
        ratios, indexes = ratios_dataarray_to_profiles(ratios_da.fillna(0.0))
        self.set_profiles(
            CompositeTemporalProfiles.from_ratios(
                ratios, types=[MounthsProfile], rescale=True
            ),
            indexes,
        )

    def _subtract_corso_emissions(
        self,
        corso_dataset: xr.Dataset,
        corso_substance: str | None,
        sector_mapping: Mapping[str, str] | None,
        negative_policy: CorsoNegativePolicy,
    ) -> None:
        """Remove annual-mean CORSO point-source rates from CAMS cells."""
        if negative_policy not in {"clip", "raise", "allow"}:
            raise ValueError(
                "`corso_negative_policy` must be 'clip', 'raise', or 'allow'."
            )

        required = {"longitude", "latitude"}
        missing = sorted(required - set(corso_dataset.variables))
        if missing:
            raise ValueError(f"CORSO dataset is missing variables: {missing}")

        coordinate_dims = corso_dataset["longitude"].dims
        if (
            len(coordinate_dims) != 1
            or corso_dataset["latitude"].dims != coordinate_dims
        ):
            raise ValueError(
                "CORSO longitude and latitude must share one source dimension."
            )
        source_dim = coordinate_dims[0]

        if corso_substance is None:
            normalized_substance = re.sub(
                r"[^A-Z0-9]", "", self.substance_name.upper()
            )
            try:
                corso_substance = DEFAULT_CORSO_SUBSTANCE_MAPPING[
                    normalized_substance
                ]
            except KeyError as exc:
                raise ValueError(
                    f"No default CORSO species mapping for CAMS substance "
                    f"{self.substance_name!r}. Pass `corso_substance=` explicitly."
                ) from exc
        corso_substance = corso_substance.lower()
        emission_variable = (
            corso_substance
            if corso_substance.endswith("_kty")
            else f"{corso_substance}_kty"
        )
        if emission_variable not in corso_dataset:
            raise ValueError(
                f"CORSO dataset does not contain {emission_variable!r}. "
                "Expected raw emissions in kilotonnes per year."
            )
        if corso_dataset[emission_variable].dims != (source_dim,):
            raise ValueError(
                f"CORSO variable {emission_variable!r} must have only the "
                f"{source_dim!r} dimension."
            )

        categories = self.monthly_flux["category"].values.astype(str)
        subtract_from_sum = len(categories) == 1 and categories[0] == "sum"
        if not subtract_from_sum:
            if "sector" not in corso_dataset:
                raise ValueError(
                    "CORSO dataset must contain 'sector' when CAMS component "
                    "categories are selected."
                )
            if corso_dataset["sector"].dims != (source_dim,):
                raise ValueError(
                    f"CORSO variable 'sector' must have only the {source_dim!r} "
                    "dimension."
                )

        category_indexes = {category: i for i, category in enumerate(categories)}
        resolved_sector_mapping = DEFAULT_CORSO_SECTOR_MAPPING.copy()
        if sector_mapping is not None:
            resolved_sector_mapping.update(
                {
                    str(key).strip().lower(): str(value).strip()
                    for key, value in sector_mapping.items()
                }
            )

        longitudes = np.asarray(corso_dataset["longitude"].values, dtype=float)
        latitudes = np.asarray(corso_dataset["latitude"].values, dtype=float)
        emissions_kty = np.asarray(
            corso_dataset[emission_variable].values, dtype=float
        )
        if np.any(emissions_kty[np.isfinite(emissions_kty)] < 0):
            raise ValueError("CORSO emissions must be non-negative.")

        finite = (
            np.isfinite(longitudes)
            & np.isfinite(latitudes)
            & np.isfinite(emissions_kty)
        )
        inside = (
            finite
            & (longitudes >= self.grid.xmin)
            & (longitudes < self.grid.xmax)
            & (latitudes >= self.grid.ymin)
            & (latitudes < self.grid.ymax)
        )

        requested_rates = np.zeros(
            (len(categories), len(self.grid)), dtype=np.float64
        )
        mapped_sources = 0
        unmapped_sources = 0
        sectors = (
            np.asarray(corso_dataset["sector"].values).astype(str)
            if not subtract_from_sum
            else None
        )
        for source_index in np.flatnonzero(inside & (emissions_kty > 0)):
            if subtract_from_sum:
                category = "sum"
            else:
                sector = sectors[source_index].strip().lower()
                category = resolved_sector_mapping.get(sector)
                if category not in category_indexes:
                    unmapped_sources += 1
                    continue

            lon_index = int(
                np.searchsorted(
                    self.grid.lon_bounds, longitudes[source_index], side="right"
                )
                - 1
            )
            lat_index = int(
                np.searchsorted(
                    self.grid.lat_bounds, latitudes[source_index], side="right"
                )
                - 1
            )
            cell_index = lon_index * self.grid.ny + lat_index
            rate_kg_s = emissions_kty[source_index] * 1_000_000 / SEC_PER_YR
            requested_rates[category_indexes[category], cell_index] += rate_kg_s
            mapped_sources += 1

        cell_areas = np.asarray(self.grid.cell_areas, dtype=float)
        requested_flux = requested_rates / cell_areas[None, :]
        original_flux = np.asarray(self.monthly_flux.values, dtype=np.float64)
        residual_flux = original_flux - requested_flux[None, :, :]
        negative_mask = residual_flux < 0
        clipped_values = int(np.count_nonzero(negative_mask))
        if clipped_values and negative_policy == "raise":
            largest_deficit = float(-residual_flux[negative_mask].min())
            raise ValueError(
                f"CORSO subtraction would create {clipped_values} negative CAMS "
                f"values (largest deficit {largest_deficit:.6g} "
                f"{CAMS_FLUX_UNIT}). Use `corso_negative_policy='clip'` or "
                "'allow' to choose an explicit alternative."
            )
        if negative_policy == "clip":
            residual_flux = np.maximum(residual_flux, 0.0)
            if clipped_values:
                warn(
                    f"CORSO subtraction exceeded {clipped_values} monthly CAMS "
                    "cell/category values; those residuals were clipped to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.monthly_flux.data = residual_flux
        actual_removed_rates = (
            (original_flux - residual_flux) * cell_areas[None, None, :]
        ).sum(axis=(1, 2))
        requested_rate = float(requested_rates.sum())
        self.corso_subtraction_summary = {
            "emission_variable": emission_variable,
            "negative_policy": negative_policy,
            "mapped_sources": mapped_sources,
            "unmapped_sources": unmapped_sources,
            "outside_sources": int(np.count_nonzero(finite & ~inside)),
            "invalid_sources": int(np.count_nonzero(~finite)),
            "negative_values": clipped_values,
            "clipped_values": clipped_values if negative_policy == "clip" else 0,
            "requested_annual_mean_kg_s": requested_rate,
            "actual_monthly_mean_kg_s": float(actual_removed_rates.mean()),
        }
        self.monthly_flux.attrs.update(
            {
                "corso_subtracted": "true",
                "corso_emission_variable": emission_variable,
                "corso_negative_policy": negative_policy,
                "corso_requested_annual_mean_kg_s": requested_rate,
                "corso_actual_monthly_mean_kg_s": float(
                    actual_removed_rates.mean()
                ),
            }
        )
        self.history.append(
            f"Subtracted CORSO {emission_variable}: "
            f"{mapped_sources} sources mapped, {unmapped_sources} skipped, "
            f"{clipped_values if negative_policy == 'clip' else 0} values clipped"
        )


CAMS_GLOB_ANT = CAMSGlobAnt


__all__ = [
    "CAMSGlobAnt",
    "CAMS_GLOB_ANT",
]
