from os import PathLike
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

from emiproc.grids import RegularGrid
from emiproc.inventories import Category, Inventory, Substance
from emiproc.profiles.temporal.composite import CompositeTemporalProfiles
from emiproc.profiles.temporal.profiles import MounthsProfile
from emiproc.profiles.utils import ratios_dataarray_to_profiles
from emiproc.utilities import DAY_PER_YR, HOUR_PER_YR, SEC_PER_DAY


def get_unit_scaling_factor_to_kg_per_year_per_cell(unit: str) -> tuple[float, bool]:
    """Get the scaling factor to convert from the given unit to kg/year/cell.

    Supported units:
    - "kg/m2/s"

    :param unit: Unit string.
    :return: Scaling factor. and a boolean indicating that we need to scale (multiply) with the cell area.
    """
    unit = unit.lower()
    if unit == "kg/m2/s":
        # kg/m2/s * day/year * s/day * m2/cell = kg/year/cell
        return DAY_PER_YR * SEC_PER_DAY, True  # seconds to year
    elif unit in ["kg/y/cell", "kg y-1 cell-1", "kg/year/cell"]:
        return 1.0, False
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

def _array_to_series(da: xr.DataArray) -> pd.Series:
    """Convert a DataArray to a Pandas Series, averaging over time if necessary."""
    if 'time' in da.dims:
        da = da.mean(dim='time')
    
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

            if time_name in ds.variables:
                if len(ds[time_name]) != 1:
                    raise NotImplementedError(
                        f"Expected only one time step in the dataset, found {len(ds[time_name])}."
                        "If you want to implement time-varying inventories, please contact the developers."
                    )
                year = pd.to_datetime(ds[time_name].values[0]).year 
            elif "year" in ds.attrs:
                # No time variable, probably a constant inventory
                year = get_year_from_attrs(ds.attrs)
                if year is None:
                    self.logger.warning(
                        "Year attribute found in the dataset, but could not be converted to int."
                        "Setting year to None."
                    )
            else:
                year = None
            self.year = year

            das = {}

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
                    # .expand_dims(dim=dict(catsub=[(category, substance)]))
                    .fillna(0.0)
                )
                # Unit conversion
                this_unit = ds[var].units.lower() if unit is None else unit
                scaling_factor, multiply_by_area = (
                    get_unit_scaling_factor_to_kg_per_year_per_cell(this_unit)
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

        # For implementation of temporal profiles in the future:

        # da_ratios = da_stacked / da_stacked.sum(dim="time")
        # da_ratios = da_ratios.rename(time="ratio")
        # ratios, indices = ratios_dataarray_to_profiles(da_ratios)

        # Convert to pandas

        series = {
            catsub: _array_to_series(da)
            for catsub, da in das.items()
        }

        self.gdf = gpd.GeoDataFrame(series, geometry=self.grid.gdf.geometry)
        self.gdfs = {}

        # self.set_profiles(
        #    profiles=CompositeTemporalProfiles.from_ratios(
        #        ratios=ratios, types=[MounthsProfile]
        #    ),
        #    indexes=indices,
        # )
