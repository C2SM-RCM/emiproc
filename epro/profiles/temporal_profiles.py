#!/usr/bin/env python
# coding: utf-8

# 4 files :
# hourofday, dayofweek, monthofyear, hourofyear

# For "hourofday.nc"
# dimensions:
# 	hourofday = 24 ;
# 	country = <Ncountry> ;
# variables:
# 	float varname1(hourofday, country) ;
# 		varname1:units = "1" ;
# 		varname1:long_name = "diurnal scaling factor for 24 hours" ;
# 		varname1:comment = "first hour is 00h, last 23h local time" ;
# 	float varname2(hourofday, country) ;
# 		varname2:units = "1" ;
# 		varname2:long_name = "diurnal scaling factor for 24 hours" ;
# 		varname2:comment = "first hour is 00h, last 23h local time" ;
# [...]
# 	short countryID(country) ;
# 		country:long_name = "EMEP country code" ;

# For “dayofweek.nc":
# dimensions:
# 	dayofweek = 7 ;
# 	country = <Ncountry> ;
# variables:
# 	float varname1(dayofweek, country) ;
# 		varname1:units = "1" ;
# 		varname1:long_name = "day-of-week scaling factor for 7 days" ;
# 		varname1:comment = "first day is Monday, last day is Sunday" ;
# [...]
# 	short country(country) ;
# 		countryID:long_name = "EMEP country code" ;
# For “monthofyear.nc":
# dimensions:
# 	monthofyear = 12 ;
# 	country = <Ncountry> ;
# variables:
# 	float varname1(monthofyear, country) ;
# 		varname1:units = "1" ;
# 		varname1:long_name = "monthly scaling factor for 12 months";
# 		varname1:comment = "first month is Jan, last month is Dec";
# [...]
# 	short countryID(country) ;
# 		countryID:long_name = "EMEP country code" ;

# For “hourofyear.nc":
# The country dependency includes the shift for each country timezone.
# dimensions:
#       hourofyear = 8784 ;
# 	country = <Ncountry> ;
# variables:
# 	float varname1(hourofyear, category) ;
# 		varname1:units = "1" ;
# 		varname1:long_name = "hourly scaling factor" ;
# 	[...]
# 	short countryID(country) ;
# 		countryID:long_name = "EMEP country code" ;


import itertools
import os
import time
import numpy as np
import netCDF4
import pytz
from datetime import datetime
from .country_code import country_codes as cc


"""Weekly and annual profiles are availble for these tracers"""
# TRACERS = ['CO', 'CO2', 'NH3', 'NMVOC', 'NO', 'NO2', 'NOx', 'P10', 'P25', 'SOx']
TRACERS = ["CO2", "CO", "CH4"]
only_ones = False

"""Set output type (normal, CH0, outCH0)"""
output_type = "normal"
# output_type = 'CH0'
# output_type = 'outCH0'


def permute_cycle_tz(tz, cycle):
    # tz in the shape "+0400"
    shift = int(int(tz[1:3]) * ((tz[0] == "+") - 0.5) * 2)

    # for now I don't consider half hours for the time zone
    try:
        answer = [cycle[shift - i] for i in range(len(cycle), 0, -1)]
    except IndexError:
        answer = [cycle[shift - i + 24] for i in range(len(cycle), 0, -1)]

    return answer

def load_country_tz_TNO(filename):
    winter = False
    ctz = dict()
    with open(filename) as f:
        for line in f:
            words = line.split(';')
            if len(words[0].strip())==3:
                ctz[words[0].strip()] = int(words[2])
                if not winter:
                    ctz[words[0].strip()] += int(words[3])
    return ctz

def validate_tz(filename,all_tz):
    """Checks that all countries in the emission file have a corresponding time zone"""
    with netCDF4.Dataset(filename) as f:
        clist = set(f['country_ids'][:].flatten())
        print(clist)
        for c in clist:
            if all_tz[c]=='None':
                print(c,'is missing')
        
def tz_int2str(time):
    string = ''
    if time>=0:
        string = '+'
    else:
        string = '-'
    if abs(time)>10:
        string+=str(abs(time))+'00'
    else:
        string+='0'+str(abs(time))+'00'
    
    return string

def get_country_tz_TNO(countries):
    tz_TNO = load_country_tz_TNO("CHE_input/country_tz.csv")
    all_tz = dict()
    for country in countries:
        if country==0: #Seas
            continue #all_tz[country] = tz_int2str(0)
        country_names = [name  for name,code in cc.items() if (code==country)]
        country_name=""
        # Try find the name of the country, with 3 characters
        for name in country_names:
            if len(name)==3:
                country_name = name
                try:
                    all_tz[country] = tz_int2str(tz_TNO[name])
                    break
                except KeyError:
                    continue
                    #all_tz[country] = "None"
        # if country_name=="":
        #     all_tz[country] = "None"
    return all_tz

def get_country_tz(countries):
    country_exception = dict(
        FGD="+0100",
        FFR="+0100",
        RUO="+0400",
        RUP="+0300",
        RUA="+0200",
        RUR="+0300",  # Moscau
        RU="+0300",  # Moscau
        YUG="+0100",
        CS="+0100",
        NOA="+0100",
        PT="+0000",  # Portugal mainland, not azores
        KZ="+0500",  # West of Kazakhstan
    )

    all_tz = dict()
    for country in countries:
        try:
            country_names = [
                name for name, code in cc.items() if (code == country)
            ]
            country_name = ""
            # Try find the name of the country, with 2 character
            for name in country_names:
                if len(name) == 2:
                    country_name = name
            # If it's not there, maybe it's one of the exceptions
            if country_name == "":
                for name in country_names:
                    try:
                        print("exception : ", name, country_exception[name])
                        country_name = name
                        continue
                    except KeyError:
                        pass
            # If I still didn't find it, then nevermind
            if country_name == "":
                raise KeyError
        except KeyError:
            print(country_names, "not a proper country name in pytz")
            continue
        # For countries with several time zones, they are listed in the exception and one tz is assigned.
        if country_name in country_exception:
            all_tz[country]= country_exception[country_name]
            print(country,country_name,country_exception[country_name])
        else:                
            zones = pytz.country_timezones[country_name]

            fmt = "%z"
            zero = pytz.utc.localize(datetime(2015, 1, 1, 0, 0, 0))
            for zone in zones:
                loc_tz = pytz.timezone(zone)
                loc_zero = zero.astimezone(loc_tz)
                hour = loc_zero.strftime("%z")
                all_tz[country]= hour
                print(country,country_name,hour)

    return all_tz


def read_daily_profiles(path):
    filename = os.path.join(path, "HourlyFac.dat")

    snaps = []
    data = {}

    with open(filename) as profile_file:
        for line in profile_file:
            values = line.split()
            snap = values[0].strip()
            data[snap] = np.array(values[1:], "f4")

    return snaps, data


def read_temporal_profile_simple(path):    
    started = False
    data = []
    snaps = []
    with open(path, encoding="latin") as prof:
        for l in prof:
            splitted = l.split(";")
            if splitted[0] == "1":
                started = True
            # if splitted[0]=="11":
            #     started=False
            if started:
                if splitted[1] == "F1":
                    snaps.append("F")
                elif "F" in splitted[1]:
                    continue
                else:
                    snaps.append(splitted[1])
                data.append([float(i) for i in splitted[3:]])

    return snaps, np.array(data)


def read_temporal_profiles(tracer, kind, path="timeprofiles"):
    """\
    Read temporal profiles for given `tracer` for
    'weekly' or 'annual' profiles.
    """
    data = {}
    countries = []
    snaps = []

    if kind == "weekly":
        filename = os.path.join(path, "DailyFac_%s.dat" % tracer)
    elif kind == "annual":
        filename = os.path.join(path, "MonthlyFac_%s.dat" % tracer)
    else:
        raise ValueError("kind has to be 'weekly' or 'annual' not '%s'" % kind)

    with open(filename, "r") as profile_file:
        for line in profile_file:
            values = line.split()
            country, snap = int(values[0]), str(values[1])

            countries.append(country)
            snaps.append(snap)

            data[country, snap] = np.array(values[2:], "f4")

    return list(set(countries)), list(set(snaps)), data


def read_all_data(tracers, path="timeprofiles"):

    daily_profiles = {}
    weekly_profiles = {}
    annual_profiles = {}

    countries = []

    # daily profiles
    snaps, daily_profiles = read_daily_profiles(path)

    for tracer in tracers:

        # weekly
        c, s, d = read_temporal_profiles(tracer, "weekly", path)
        weekly_profiles[tracer] = d
        countries += c
        snaps += s

        # weekly
        c, s, d = read_temporal_profiles(tracer, "annual", path)
        annual_profiles[tracer] = d
        countries += c
        snaps += s

    return (
        sorted(set(countries)),
        sorted(set(snaps)),
        daily_profiles,
        weekly_profiles,
        annual_profiles,
    )


def create_netcdf(path, countries):
    profile_names = ["hourofday", "dayofweek", "monthofyear", "hourofyear"]
    var_names = ["hourofday", "dayofweek", "monthofyear", "hourofyear"]
    if output_type == "normal":
        pass
    elif output_type == "CH0":
        profile_names[0] = "hourofday_CH0"
    elif output_type == "outCH0":
        profile_names[0] = "hourofday_outCH0"
    else:
        raise ValueError(
            "Variable 'output_type' has to be 'normal' or 'CH0' "
            "or 'outCH0', not '%s'" % output_type
        )

    for (profile, var, size) in zip(
        profile_names, var_names, [24, 7, 12, 8784]
    ):
        filename = os.path.join(path, profile + ".nc")

        with netCDF4.Dataset(filename, "w") as nc:

            # global attributes (add input data)
            nc.setncatts(
                {
                    "DESCRIPTION": "Temporal profiles for emissions",
                    "DATAORIGIN": "TNO time profiles",
                    "CREATOR": "Jean-Matthieu Haussaire",
                    "EMAIL": "jean-matthieu.haussaire@empa.ch",
                    "AFFILIATION": "Empa Duebendorf, Switzerland",
                    "DATE CREATED": time.ctime(time.time()),
                }
            )

            # create dimensions
            nc.createDimension(var, size=size)
            nc.createDimension("country", size=len(countries))

            nc_cid = nc.createVariable("country", "i2", ("country"))
            nc_cid[:] = np.array(countries, "i2")
            nc_cid.long_name = "EMEP country code"


def write_single_variable(path, profile, values, tracer, snap):
    filename = os.path.join(path, profile + ".nc")
    if "hourofday" in profile:
        descr = "diurnal scaling factor"
        varname = "hourofday"
    if profile == "dayofweek":
        descr = "day-of-week scaling factor"
        varname = "dayofweek"
    if profile == "monthofyear":
        descr = "month-of-year scaling factor"
        varname = "monthofyear"
    if profile == "hourofyear":
        descr = "hour-of-year scaling factor"
        varname = "hourofyear"

    with netCDF4.Dataset(filename, "a") as nc:
        # create variables and attributes
        # if profile == "hourofday" or profile == "hourofyear":
        #     nc_var = nc.createVariable(tracer+"_"+snap, 'f4', (profile))
        # else:
        nc_var = nc.createVariable(
            tracer + "_" + snap, "f4", (varname, "country")
        )
        if complex_profile:
            nc_var.long_name = "%s for species %s and SNAP %s" % (
                descr,
                tracer,
                snap,
            )
        else:
            nc_var.long_name = "%s for GNFR %s" % (descr, snap)
        nc_var.units = "1"
        nc_var[:] = values


def main_complex(path):

    mean = True

    # read all data
    countries, snaps, daily, weekly, annual = read_all_data(TRACERS)
    countries = [0] + countries
    n_countries = len(countries)

    create_netcdf(path, countries)

    # day of week and month of year
    dow = np.ones((7, n_countries))
    moy = np.ones((12, n_countries))
    hod = np.ones((24, n_countries))

    country_tz = get_country_tz(countries)

    for (tracer, snap) in itertools.product(TRACERS, snaps):
        if not only_ones:
            for i, country in enumerate(countries):
                try:
                    hod[:, i] = permute_cycle_tz(
                        country_tz[country], daily[snap]
                    )
                except KeyError:
                    pass

                try:
                    dow[:, i] = weekly[tracer][country, snap]
                    if mean:
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
        write_single_variable(path, "hourofday", hod, tracer, snap)
        write_single_variable(path, "dayofweek", dow, tracer, snap)
        write_single_variable(path, "monthofyear", moy, tracer, snap)

    # TODO: hour of year
    hours_in_year = np.arange(0, 8784)


def main_simple(path):
    mean = False

    # snaps = ["A","B","C","D","E","F","G","H","I","J"]

    countries = np.arange(74)
    # remove all countries not known, and not worth an exception
    countries = np.delete(
        countries, [5, 26, 28, 29, 30, 31, 32, 33, 34, 35, 58, 64, 67, 70, 71]
    )
    n_countries = len(countries)

    #country_tz = get_country_tz(countries)
    country_tz = get_country_tz_TNO(countries)
    #validate_tz('../testdata/CHE_TNO_v1_1_2018_12/CHE_TNO_offline/emis_2015_Europe.nc',country_tz)
    print(country_tz)

    snaps, daily = read_temporal_profile_simple(
        "CHE_input/timeprofiles-hour-in-day_GNFR.csv"
    )
    snaps, weekly = read_temporal_profile_simple(
        "CHE_input/timeprofiles-day-in-week_GNFR.csv"
    )
    snaps, monthly = read_temporal_profile_simple(
        "CHE_input/timeprofiles-month-in-year_GNFR.csv"
    )

    print(snaps)

    # day of week and month of year
    dow = np.ones((7, n_countries))
    moy = np.ones((12, n_countries))
    hod = np.ones((24, n_countries))

    for snap_ind, snap in enumerate(snaps):
        for i, country in enumerate(countries):
            try:
                hod[:, i] = permute_cycle_tz(
                    country_tz[country], daily[snap_ind, :]
                )
            except KeyError:
                pass

            try:
                dow[:, i] = weekly[snap_ind, :]
                if mean:
                    dow[:5, i] = np.ones(5) * weekly[snap_ind, :5].mean()
            except KeyError:
                pass

            try:
                moy[:, i] = monthly[snap_ind]
            except KeyError:
                pass

        # Use hourofday profile to mask CH if needed
        if output_type == "normal":
            write_single_variable(path, "hourofday", hod, "GNFR", snap)
        elif output_type == "CH0":
            hod[:, 23] = 0.0
            write_single_variable(path, "hourofday_CH0", hod, "GNFR", snap)
        elif output_type == "outCH0":
            for i, country in enumerate(countries):
                if i != 23:
                    hod[:, i] = 0.0
            write_single_variable(path, "hourofday_outCH0", hod, "GNFR", snap)
        else:
            raise ValueError(
                "Variable 'output_type' has to be 'normal' or 'CH0' "
                "or 'outCH0', not '%s'" % output_type
            )

        # Write other profiles
        write_single_variable(path, "dayofweek", dow, "GNFR", snap)
        write_single_variable(path, "monthofyear", moy, "GNFR", snap)


if __name__ == "__main__":
    complex_profile = False
    if complex_profile:
        main_complex("./output")
    else:
        main_simple("./CHE_output_todel")
