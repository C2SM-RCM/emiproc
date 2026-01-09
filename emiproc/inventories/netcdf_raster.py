from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from emiproc.grids import RegularGrid
from emiproc.inventories import Category, Inventory, Substance
from emiproc.profiles.temporal.composite import (
    AnyTimeProfile,
    CompositeTemporalProfiles,
)
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.utils.units import get_scaling_factor_to_emiproc


def get_year_from_attrs(attrs: dict) -> int | None:
    """Get the year from the attributes dictionary.

    :param attrs: Attributes dictionary.
    :return: Year if found, None otherwise.
    """
    if "year" in attrs:
        try:
            return int(attrs["year"])
        except (ValueError, TypeError):
            return None

    return None


def _array_to_series(da: xr.DataArray, time_name: str) -> pd.Series:
    """Convert a DataArray to a Pandas Series, averaging over time if necessary."""
    if time_name in da.dims:
        da = da.mean(dim=time_name)

    # Check that there is only one dimension left (cell)
    if len(da.dims) != 1 or da.dims[0] != "cell":
        raise ValueError(
            f"Expected DataArray with only 'cell' dimension, found {da.dims}."
            " Make sure you specified the name of the time dimension correctly."
        )

    return da.to_pandas()


def _read_variable_to_catsub_mapping(
    variable_to_catsub: dict[str, tuple[Category, Substance]] | None,
    ds: xr.Dataset,
    lon_name: str,
    lat_name: str,
) -> tuple[list[str], list[tuple[Category, Substance]]]:
    """Read the variable to (category, substance) mapping.

    If None, assume all variables in the dataset are to be read,
    and that they have category and substance coordinates.

    :param variable_to_catsub: Mapping from variable names to (category, substance) tuples.
    :return: Tuple of (variables, catsubs).
    """
    logger = logging.getLogger(__name__)

    if variable_to_catsub is not None:
        variables = list(variable_to_catsub.keys())
        catsubs = list(variable_to_catsub.values())
        return variables, catsubs

    variables = []
    catsubs = []

    for var in ds.data_vars:
        da = ds[var]
        if lon_name not in da.dims or lat_name not in da.dims:
            logger.debug(
                f"Variable '{var}' does not have "
                f"'{lon_name}' and '{lat_name}' dimensions and is skipped."
            )
            continue
        attrs = da.attrs
        substance = attrs.get("substance")
        if substance and (category := attrs.get("category")):
            variables.append(var)
            catsubs.append((category, substance))
        elif "category" in da.coords:
            if substance is None:
                substance = var
            for cat in da.coords["category"].values:
                variables.append(var)
                catsubs.append((cat, substance))
        else:
            logger.debug(
                f"Variable '{var}' does not have 'category' and 'substance' "
                f"attributes and is skipped."
            )

    if not variables:
        raise ValueError(
            "variable_to_catsub is None and could not be inferred from the dataset. "
            "Please provide a mapping from variable names to (category, substance) tuples."
            "or make sure you have variables following this convention:"
            f" ({lon_name=}, {lat_name=}, category and substance attributes)."
        )

    return variables, catsubs


class NetcdfRaster(Inventory):
    """Netcdf inventory.

    Useful for custom inventories defined on a regular grid.

    Can read inventories created by
    :py:func:`~emiproc.exports.rasters.export_raster_netcdf`.


    :param file: Path to the netcdf file.
    :param variable_to_catsub: Dictionary mapping variable names in the netcdf file
        to (category, substance) tuples.
    :param lat_name: Name of the latitude variable in the netcdf file.
    :param lon_name: Name of the longitude variable in the netcdf file.
    :param time_name: Name of the time variable in the netcdf file.
    :param unit: Unit of the variables in the netcdf file. If None, the unit will be read
        from the netcdf file.

    """

    def __init__(
        self,
        file: PathLike,
        variable_to_catsub: dict[str, tuple[Category, Substance]] | None = None,
        lat_name: str = "lat",
        lon_name: str = "lon",
        time_name: str = "time",
        unit: str | None = None,
        temporal_profile: type[AnyTimeProfile] | None = None,
        year: int | None = None,
    ) -> None:
        file = Path(file)
        self.name = f"NetcdfRaster_Inventory_{file.stem}"
        super().__init__()

        with xr.open_dataset(file) as ds:

            self.grid = RegularGrid.from_centers(
                x_centers=ds[lon_name].values,
                y_centers=ds[lat_name].values,
                name="NetcdfRaster_grid",
            )

            cell_areas = self.grid.cell_areas

            if time_name in ds.variables and len(ds[time_name]) == 1 and year is None:
                # Case when time dimension is a single variable
                year = pd.to_datetime(ds[time_name].values[0]).year
            elif time_name in ds.variables:
                # Time dimension, must read the temporal profile
                # Check the size of the time dimension
                if temporal_profile is None:
                    raise ValueError(
                        "Temporal profile must be provided for inventories "
                        "with multiple time steps."
                    )
                years_in_data = pd.to_datetime(ds[time_name].values).year
                if year is None:
                    # Check only one year is present
                    unique_years = pd.unique(years_in_data)
                    if len(unique_years) > 1:
                        raise ValueError(
                            "Multiple years found in the data. Please specify the year to select."
                        )
                    year = unique_years[0]
                else:
                    # Ensure the data is given for that year
                    self.logger.info(
                        f"Selecting data for year {year} "
                        f" available: {sum(years_in_data == year)} time steps."
                    )
                    ds = ds.sel({time_name: years_in_data == year})
                    if len(ds[time_name]) == 0:
                        raise ValueError(
                            f"No data found for year {year} in the inventory."
                        )
                if temporal_profile.size != len(ds[time_name]):
                    raise ValueError(
                        f"Temporal profile size {temporal_profile.size} does not "
                        f"match number of time steps {len(ds[time_name])}."
                    )

            elif "year" in ds.attrs and year is None:
                # No time variable, probably a constant inventory
                year = get_year_from_attrs(ds.attrs)
                if year is None:
                    self.logger.warning(
                        "Year attribute found in the dataset, but could not be converted to int. Setting year to None."
                    )
            else:
                pass

            self.year = year

            das = {}
            das_ratios = []

            variables, catsubs = _read_variable_to_catsub_mapping(
                variable_to_catsub, ds, lon_name, lat_name
            )

            for var, (category, substance) in zip(variables, catsubs):
                da = ds[var]
                if "category" in da.coords:
                    da = da.sel(category=category)
                da_stacked = (
                    da.stack(cell=(lon_name, lat_name))
                    # Reset the cell index
                    .reset_index("cell")
                    .drop_vars([lon_name, lat_name])
                    .assign_coords(cell=np.arange(len(self.grid.gdf), dtype=int))
                    # .expand_dims(dim=dict(catsub=[(category, substance)]))
                    .fillna(0.0)
                )
                if temporal_profile is not None:
                    # Calculate ratios for profiles
                    mask_zero = da_stacked.sum(dim=time_name) == 0
                    da_ratios = da_stacked.sel({"cell": ~mask_zero})
                    # Ensure high precision for the ratios, so that they sum to 1
                    da_ratios = da_ratios.astype("float64")
                    da_ratios = da_ratios / da_ratios.sum(dim=time_name)
                    da_ratios = da_ratios.expand_dims(
                        dim={"catsub": [(category, substance)]}
                    )
                    das_ratios.append(da_ratios)
                # Unit conversion
                this_unit = da.attrs.get("units") if unit is None else unit
                if this_unit is None:
                    raise ValueError(
                        f"Unit for variable '{var}' is not specified in the dataset and no unit was provided."
                    )
                scaling_factor, multiply_by_area = get_scaling_factor_to_emiproc(
                    this_unit, substance=substance
                )
                if multiply_by_area:
                    da_cell_areas = xr.DataArray(
                        cell_areas.values,
                        dims=["cell"],
                        coords={"cell": da_stacked.cell.values},
                    )
                    da_stacked = da_stacked * da_cell_areas
                da_stacked = da_stacked * scaling_factor
                das[(category, substance)] = da_stacked

        # Convert to pandas
        series = {catsub: _array_to_series(da, time_name) for catsub, da in das.items()}

        self.gdf = gpd.GeoDataFrame(series, geometry=self.grid.gdf.geometry)
        self.gdfs = {}

        if temporal_profile is None:
            return

        # Following is only for temporal profiles

        da_ratios: xr.DataArray = xr.concat(das_ratios, dim="catsub", join="outer")
        # Put catsub to category and substance coordinates
        da_ratios = da_ratios.assign_coords(
            xr.Coordinates.from_pandas_multiindex(
                pd.MultiIndex.from_tuples(
                    da_ratios["catsub"].values, names=["category", "substance"]
                ),
                dim="catsub",
            )
        )
        da_ratios = da_ratios.unstack("catsub")
        ratios, indices = ratios_dataarray_to_profiles(
            da_ratios.rename({time_name: "ratio"})
        )
        self.set_profiles(
            profiles=CompositeTemporalProfiles.from_ratios(
                ratios=ratios, types=[temporal_profile]
            ),
            indexes=indices,
        )
