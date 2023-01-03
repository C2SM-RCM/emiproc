#!/usr/bin/env python
# coding: utf-8
"""Legacy file from v1"""

"""
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

import itertools
import os
import time
import numpy as np
import netCDF4

from ..country_code import country_codes as cc
from . import io


# Constants
N_HOUR_DAY = 24
N_DAY_WEEK = 7
N_MONTH_YEAR = 12
N_HOUR_YEAR = 8784


def permute_cycle_tz(tz, cycle):
    """
    Permute a daily cycle by a given amount of hours.
    This is a way to convert from local time to UTC

    Parameters
    ----------
    tz: String
        String corresponding to the timezone
    cycle: List of length N_HOUR_DAY
        Hourly scaling factors to be permuted

    Returns
    -------
    Permuted cycle scaling factors
    """
    shift = int(tz)

    # half hours for the time zone are not considered
    try:
        answer = [cycle[shift - i] for i in range(len(cycle), 0, -1)]
    except IndexError:
        answer = [cycle[shift - i + 24] for i in range(len(cycle), 0, -1)]

    return answer


def load_country_tz(filename, winter=True):
    """
    Load the list of country timzones from a csv file

    Parameters
    ----------
    filename: String
        Path to the csv file containing the timezone information
    winter: bool (defaut to True)
        Wether the winter time or the summertime is considered

    Returns
    -------
    dict(str : int)
        A dictionary mapping the iso3 country codes to the corresponding
        timezone (hours offset from UTC)
    """
    ctz = dict()
    with open(filename) as f:
        for line in f:
            words = line.split(";")
            if len(words[0].strip()) == 3:
                ctz[words[0].strip()] = int(words[2])
                if not winter:
                    ctz[words[0].strip()] += int(words[3])
    return ctz


def validate_tz(filename, all_tz):
    """Checks that all countries in the gridded emission file have a time zone

    Parameters
    ----------
    filename: String
        Path to the gridded emissions
    all_tz:
        List of time zones
    """
    with netCDF4.Dataset(filename) as f:
        clist = set(f["country_ids"][:].flatten())
        print(clist)
        for c in clist:
            if all_tz[c] == "None":
                print(c, "is missing")




def read_temporal_profile(path):
    """Read the temporal profile from a csv file.

    Parameters
    ----------
    path: String
        Path to the profile as a csv file

    Returns
    -------
    list of categories, np.array(scaling factors)
    """
    started = False
    data = []
    cats = []
    with open(path, encoding="latin") as prof:
        for l in prof:
            splitted = l.split(";")
            if splitted[0] == "1":
                started = True
            if started:
                if splitted[1] == "F1":
                    cats.append("F")
                elif "F" in splitted[1]:
                    continue
                else:
                    cats.append(splitted[1])
                data.append([float(i) for i in splitted[3:]])

    return cats, np.array(data)


def get_country_tz(countries, country_tz_file, winter):
    """
    Get the time zone of every country

    Parameters
    ----------
    countries: list (int)
        List of emep country codes

    Returns
    -------
    Dictionary linking country names to time zone
    """

    tz = load_country_tz(country_tz_file, winter)
    all_tz = dict()
    for country in countries:
        if country == 0:  # Seas
            continue

        country_names = [name for name, code in cc.items() if (code == country)]
        # Try find the name of the country, with 3 characters
        for name in country_names:
            if len(name) == 3:
                try:
                    all_tz[country] = tz[name]
                    break
                except KeyError:
                    continue

    return all_tz



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


