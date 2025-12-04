from os import PathLike
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np

from emiproc.grids import RegularGrid
from emiproc.inventories import Category, Inventory, Substance
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.utilities import DAY_PER_YR, SEC_PER_DAY
from emiproc.utils.constants import get_molar_mass


def get_unit_scaling_factor_to_kg_per_year_per_cell(
    unit: str, substance: str | None = None
) -> tuple[float, bool]:
    """Get the scaling factor to convert from the given unit to kg/year/cell.

    Supported units:
    - "kg/m2/s"

    :param unit: Unit string.

    :return: Scaling factor. and a boolean indicating that we need to scale (multiply) with the cell area.
    """
    if unit == "kg/m2/s":
        # kg/m2/s * day/year * s/day * m2/cell = kg/year/cell
        return DAY_PER_YR * SEC_PER_DAY, True  # seconds to year
    elif unit in ["kg/y/cell", "kg y-1 cell-1", "kg/year/cell"]:
        return 1.0, False
    elif unit == "PgC/yr":
        # Carbon to CO2 conversion
        if substance != "CO2":
            raise ValueError("PgC/yr unit can only be used for CO2 substance.")
        return 1e12 * (44.01 / 12.01), False
    elif unit == "micromol/m2/s":
        molar_mass = get_molar_mass(substance)  # g/mol
        # micromol/m2/s * kg/g * g/mol * mol/micromol * s/year * m2/cell
        return 1e-3 * molar_mass * 1e-6 * SEC_PER_DAY * DAY_PER_YR, True
    else:
        raise NotImplementedError(f"Unit {unit} not supported.")


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
                        f" availabe: {sum(years_in_data == year)} time steps."
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
                year = year
            self.year = year

            das = {}
            das_ratios = []

            if variable_to_catsub is None:
                variable_to_catsub = {}
                for var in ds.data_vars:
                    attrs = ds[var].attrs
                    if (substance := attrs.get("substance")) and (
                        category := attrs.get("category")
                    ):
                        variable_to_catsub[var] = (category, substance)
                if not variable_to_catsub:
                    raise ValueError(
                        "variable_to_catsub is None and could not be inferred from the dataset. "
                        "Please provide a mapping from variable names to (category, substance) tuples."
                    )
            for var, (category, substance) in variable_to_catsub.items():
                da_stacked = (
                    ds[var]
                    .stack(cell=(lon_name, lat_name))
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
                this_unit = ds[var].units if unit is None else unit
                scaling_factor, multiply_by_area = (
                    get_unit_scaling_factor_to_kg_per_year_per_cell(
                        this_unit, substance=substance
                    )
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
