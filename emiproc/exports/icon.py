
from datetime import datetime
from os import PathLike
from pathlib import Path
import xarray as xr
import numpy as np
from emiproc.grids import ICONGrid
from emiproc.inventories import Inventory
from emiproc.utilities import SEC_PER_YR, compute_country_mask



def export_icon_oem(
    inv: Inventory,
    icon_grid_file: PathLike,
    output_file: PathLike,
    group_dict: dict[str, list[str]] = {},
    country_resolution: str = "10m",
):
    """Export to a netcdf file for ICON OEM.

    The inventory should have already been remapped to the
    :py:class:`emiproc.grids.IconGrid` .

    Values will be convergted from kg/y to kg/m2/s .

    :arg group_dict: If you groupped some categories, you can optionally
        add the groupping in the metadata.
    :arg country_resolution: The resolution
        can be either '10m', '50m' or '110m'

    .. warning::

        Country codes are not yet implemented

    """
    icon_grid_file = Path(icon_grid_file)
    output_file = Path(output_file)
    # Load the output xarray

    ds_out: xr.Dataset = xr.load_dataset(icon_grid_file)

    for (categorie, sub) in inv._gdf_columns:
        name = f"{categorie}-{sub}"

        # Convert from kg/year to kg/m2/s
        emissions = (
            inv.gdf[(categorie, sub)].to_numpy() / ds_out["cell_area"] / SEC_PER_YR
        )

        attributes = {
            "units": "kg/m2/s",
            "standard_name": name,
            "long_name": f"Emission of {sub} from {categorie}",
            "created_by_emiproc": f"{datetime.now()}",
        }
        if group_dict:
            attributes["group_made_from"] = f"{group_dict[categorie]}"

        emission_with_metadata = emissions.assign_attrs(attributes)

        ds_out = ds_out.assign({name: emission_with_metadata})

    # Find the proper contry codes
    mask_file = (
        output_file.parent
        / f".emiproc_country_mask_{country_resolution}_{icon_grid_file.stem}"
    ).with_suffix(".npy")
    if mask_file.is_file():
        country_mask = np.load(mask_file)
    else:
        icon_grid = ICONGrid(icon_grid_file)
        country_mask = compute_country_mask(icon_grid, country_resolution, 1)
        np.save(mask_file, country_mask)

    # Add the country ids variable for oem
    ds_out = ds_out.assign(
        {
            "country_ids": (
                ("cell"),
                country_mask.reshape(-1),
                {
                    "standard_name": "country_ids",
                    "long_name": "EMEP_country_code",
                    "history": f"Added by emiproc",
                    "country_resolution": f"country_resolution",
                },
            )
        }
    )

    ds_out.to_netcdf(output_file)


def create_profiles(*args, **kwargs):
    """Create the profiles for the icon oem.
    Script to create netCDF-files containing temporal emission factors.
    The structure of the generated files is as follows:

    4 files :
    hourofday, dayofweek, monthofyear, hourofyear

    For 'hourofday.nc'
    dimensions:
            hourofday = 24 ;
            country = <Ncountry> ;
    variables:
            float varname1(hourofday, country) ;
                    varname1:units = '1' ;
                    varname1:long_name = 'diurnal scaling factor for 24 hours' ;
                    varname1:comment = 'first hour is 00h, last 23h local time' ;
            float varname2(hourofday, country) ;
                    varname2:units = '1' ;
                    varname2:long_name = 'diurnal scaling factor for 24 hours' ;
                    varname2:comment = 'first hour is 00h, last 23h local time' ;
    [...]
            short countryID(country) ;
                    country:long_name = 'EMEP country code' ;

    For 'dayofweek.nc':
    dimensions:
            dayofweek = 7 ;
            country = <Ncountry> ;
    variables:
            float varname1(dayofweek, country) ;
                    varname1:units = '1' ;
                    varname1:long_name = 'day-of-week scaling factor for 7 days' ;
                    varname1:comment = 'first day is Monday, last day is Sunday' ;
    [...]
            short country(country) ;
                    countryID:long_name = 'EMEP country code' ;

    For 'monthofyear.nc':
    dimensions:
            monthofyear = 12 ;
            country = <Ncountry> ;
    variables:
            float varname1(monthofyear, country) ;
                    varname1:units = '1' ;
                    varname1:long_name = 'monthly scaling factor for 12 months';
                    varname1:comment = 'first month is Jan, last month is Dec';
    [...]
            short countryID(country) ;
                    countryID:long_name = 'EMEP country code' ;

    For 'hourofyear.nc':
    The country dependency includes the shift for each country timezone.
    dimensions:
        hourofyear = 8784 ;
            country = <Ncountry> ;
    variables:
            float varname1(hourofyear, category) ;
                    varname1:units = '1' ;
                    varname1:long_name = 'hourly scaling factor' ;
            [...]
            short countryID(country) ;
                    countryID:long_name = 'EMEP country code' ;
        
    """

def create_netcdf(path, countries, metadata):
    """\
    Create a netcdf file containing the list of countries and the dimensions.

    Parameters
    ----------
    path: String
        Path to the output netcdf file
    countries: List(int)
        List of countries
    metadata : dict(str : str)
        Containing global file attributes. Used as argument to
        netCDF4.Dataset.setncatts.
    """
    for (profile, size) in zip(
        ["hourofday", "dayofweek", "monthofyear", "hourofyear"],
        [N_HOUR_DAY, N_DAY_WEEK, N_MONTH_YEAR, N_HOUR_YEAR],
    ):
        filename = os.path.join(path, profile + ".nc")

        with netCDF4.Dataset(filename, "w") as nc:

            # global attributes (add input data)
            nc.setncatts(metadata)

            # create dimensions
            nc.createDimension(profile, size=size)
            nc.createDimension("country", size=len(countries))

            nc_cid = nc.createVariable("country", "i2", ("country"))
            nc_cid[:] = np.array(countries, "i2")
            nc_cid.long_name = "EMEP country code"

def write_single_variable(path, profile, values, tracer, category,
                          varname_format):
    """Add a profile to the output netcdf

    Parameters
    ----------
    path: String
        Path to the output netcdf file
    profile: String
        Type of profile to output
        (within ["hourofday", "dayofweek", "monthofyear", "hourofyear"])
    values: list(float)
        The profile
    tracer: string
        Name of tracer
    category: String
        Name of the category
    """
    filename = os.path.join(path, profile + ".nc")
    if profile == "hourofday":
        descr = "diurnal scaling factor"
        comment = "first hour is 00h, last 23h local time"
    if profile == "dayofweek":
        descr = "day-of-week scaling factor"
        comment = "first day is Monday, last day is Sunday"
    if profile == "monthofyear":
        descr = "month-of-year scaling factor"
        comment = "first month is Jan, last month is Dec"
    if profile == "hourofyear":
        descr = "hour-of-year scaling factor"
        comment = "first hour is on Jan 1. 00h"

    with netCDF4.Dataset(filename, "a") as nc:

        varname = varname_format.format(tracer=tracer, category=category)

        nc_var = nc.createVariable(varname, "f4", (profile, "country"))
        nc_var.long_name = "%s for GNFR %s" % (descr, category)
        nc_var.units = "1"
        nc_var.comment = comment
        nc_var[:] = values


def main_complex(cfg):

    os.makedirs(cfg.output_path, exist_ok=True)

    # read all data
    countries, snaps, daily, weekly, annual = io.read_tracer_profiles(cfg.tracers,
                                                            cfg.hod_input_file,
                                                            cfg.dow_input_file,
                                                            cfg.moy_input_file)
    countries = [0] + countries
    n_countries = len(countries)

    create_netcdf(cfg.output_path, countries, cfg.nc_metadata)

    country_tz = get_country_tz(countries, cfg.country_tz_file, cfg.winter)

    for (tracer, snap) in itertools.product(cfg.tracers, snaps):

        # day of week and month of year
        dow = np.ones((7, n_countries))
        moy = np.ones((12, n_countries))
        hod = np.ones((24, n_countries))

        if not cfg.only_ones:
            for i, country in enumerate(countries):

                if country in country_tz:
                    hod[:, i] = permute_cycle_tz(
                        country_tz[country], daily[snap]
                    )

                try:
                    dow[:, i] = weekly[tracer][country, snap]
                    if cfg.mean:
                        dow[:5, i] = (
                            np.ones(5)
                            * weekly[tracer][country, snap][:5].mean()
                        )
                except KeyError:
                    pass

                try:
                    moy[:, i] = annual[tracer][country, snap]
                except KeyError:
                    pass

        write_single_variable(cfg.output_path, "hourofday", hod, tracer, snap,
                             cfg.varname_format)
        write_single_variable(cfg.output_path, "dayofweek", dow, tracer, snap,
                             cfg.varname_format)
        write_single_variable(cfg.output_path, "monthofyear", moy, tracer,
                              snap, cfg.varname_format)



def main_simple(cfg):
    """ The main script for producing profiles from the csv files from TNO.
    Takes an output path as a parameter"""

    os.makedirs(cfg.output_path, exist_ok=True)

    # Arbitrary list of countries including most of Europe.
    countries = np.arange(74)
    countries = np.delete(
        countries, [5, 26, 28, 29, 30, 31, 32, 33, 34, 35, 58, 64, 67, 70, 71]
    )
    n_countries = len(countries)

    country_tz = get_country_tz(countries, cfg.country_tz_file, cfg.winter)

    create_netcdf(cfg.output_path, countries, cfg.nc_metadata)

    cats, daily = read_temporal_profile(cfg.hod_input_file)
    cats, weekly = read_temporal_profile(cfg.dow_input_file)
    cats, monthly = read_temporal_profile(cfg.moy_input_file)


    for cat_ind, cat in enumerate(cats):

        # day of week and month of year
        hod = np.ones((N_HOUR_DAY, n_countries))
        dow = np.ones((N_DAY_WEEK, n_countries))
        moy = np.ones((N_MONTH_YEAR, n_countries))

        if not cfg.only_ones:
            for i, country in enumerate(countries):
                try:
                    hod[:, i] = permute_cycle_tz(
                        country_tz[country], daily[cat_ind, :]
                    )
                except KeyError:
                    pass

                try:
                    dow[:, i] = weekly[cat_ind, :]
                    if cfg.mean:
                        dow[:5, i] = np.ones(5) * weekly[cat_ind, :5].mean()
                except KeyError:
                    pass

                try:
                    moy[:, i] = monthly[cat_ind]
                except KeyError:
                    pass

        write_single_variable(cfg.output_path, "hourofday", hod, None, cat,
                              cfg.varname_format)
        write_single_variable(cfg.output_path, "dayofweek", dow, None, cat,
                              cfg.varname_format)
        write_single_variable(cfg.output_path, "monthofyear", moy, None, cat,
                              cfg.varname_format)


