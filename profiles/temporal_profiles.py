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
from country_code import country_codes as cc



# weekly and annual profiles are availble for these tracers
#TRACERS = ['CO', 'CO2', 'NH3', 'NMVOC', 'NO', 'NO2', 'NOx', 'P10', 'P25', 'SOx']
TRACERS = ['CO2']
only_ones = False

def permute_cycle_tz(tz,cycle):
    # tz in the shape "+0400"
    shift = int(int(tz[1:3])*((tz[0]=="+")-0.5)*2)

    # for now I don't consider half hours for the time zone
    try:
        answer = [cycle[shift-i] for i in range(len(cycle),0,-1)]
    except IndexError:
        answer = [cycle[shift-i+24] for i in range(len(cycle),0,-1)]

    return answer

def get_country_tz(countries):
    country_exception = dict(
        FGD="+0100",
        FFR="+0100",
        RUO="+0400",
        RUP="+0300",
        RUA="+0200",
        RUR="+0300", # Moscau
        RU ="+0300", #Moscau
        YUG="+0100",
        CS ="+0100",
        NOA="+0100",
        PT ="+0000", #Portugal mainland, not azores
        KZ ="+0500", #West of Kazakhstan
    )

    all_tz = dict()
    for country in countries:
        try:
            country_names = [name  for name,code in cc.items() if (code==country)]
            country_name=""
            # Try find the name of the country, with 2 character
            for name in country_names:
                if len(name)==2:
                    country_name = name
            # If it's not there, maybe it's one of the exceptions
            if country_name=="":
                for name in country_names:
                    try:
                        print("exception : ",name,country_exception[name])
                        country_name = name
                        continue
                    except KeyError:
                        pass
            # If I still didn't find it, then nevermind
            if country_name=="":
                raise KeyError
        except KeyError:
            print(country_names,"not a proper country name in pytz")
            continue
        # For countries with several time zones, they are listed in the exception and one tz is assigned.
        if country_name in country_exception:
            all_tz[country]= country_exception[country_name]
        else:                
            zones = pytz.country_timezones[country_name]
            
            fmt = "%z"
            zero = pytz.utc.localize(datetime(2015, 1, 1, 0, 0, 0))                
            for zone in zones:
                loc_tz = pytz.timezone(zone)
                loc_zero = zero.astimezone(loc_tz)
                hour = loc_zero.strftime("%z")
                all_tz[country]= hour

    return all_tz


def read_daily_profiles(path):
    filename = os.path.join(path, 'HourlyFac.dat')

    snaps = []
    data = {}

    with open(filename) as profile_file:
        for line in profile_file:
            values = line.split()
            snap = values[0].strip()
            data[snap] = np.array(values[1:], 'f4')

    return snaps, data

def read_temporal_profile_simple(path):
    started = False
    data = []
    snaps = []
    with open(path,encoding="latin") as prof:
        for l in prof:
            splitted = l.split(";")
            if splitted[0]=="1":
                started=True
            if splitted[0]=="11":
                started=False
            if started:
                if splitted[1]=="F1":
                    snaps.append("F")
                elif "F" in splitted[1]:
                    continue
                else:
                    snaps.append(splitted[1])
                data.append([float(i) for i in splitted[3:]])

    return snaps,np.array(data)

def read_temporal_profiles(tracer, kind, path='timeprofiles'):
    """\
    Read temporal profiles for given `tracer` for
    'weekly' or 'annual' profiles.
    """
    data = {}
    countries = []
    snaps = []

    if kind == 'weekly':
        filename = os.path.join(path, 'DailyFac_%s.dat' % tracer)
    elif kind == 'annual':
        filename = os.path.join(path, 'MonthlyFac_%s.dat' % tracer)
    else:
        raise ValueError("kind has to be 'weekly' or 'annual' not '%s'" % kind)

    with open(filename, 'r') as profile_file:
        for line in profile_file:
            values = line.split()
            country, snap = int(values[0]), str(values[1])

            countries.append(country)
            snaps.append(snap)

            data[country, snap] = np.array(values[2:], 'f4')

    return list(set(countries)), list(set(snaps)), data



def read_all_data(tracers, path='timeprofiles'):

    daily_profiles = {}
    weekly_profiles = {}
    annual_profiles = {}

    countries = []

    # daily profiles
    snaps, daily_profiles = read_daily_profiles(path)

    for tracer in tracers:

        # weekly
        c,s,d = read_temporal_profiles(tracer, 'weekly', path)
        weekly_profiles[tracer] = d
        countries += c
        snaps += s

        # weekly
        c,s,d = read_temporal_profiles(tracer, 'annual', path)
        annual_profiles[tracer] = d
        countries += c
        snaps += s

    return sorted(set(countries)), sorted(set(snaps)), daily_profiles, weekly_profiles, annual_profiles


def create_netcdf(path,countries):
    for (profile,size) in zip(["hourofday", "dayofweek", "monthofyear", "hourofyear"],[24,7,12,8784]):
        filename = os.path.join(path,profile+".nc")
        
        with netCDF4.Dataset(filename, 'w') as nc:

            # global attributes (add input data)
            nc.setncatts({
                'DESCRIPTION':  'Temporal profiles for emissions',
                'DATAORIGIN':   'TNO time profiles',
                'CREATOR':      'Jean-Matthieu Haussaire',
                'EMAIL':        'jean-matthieu.haussaire@empa.ch',
                'AFFILIATION':  'Empa Duebendorf, Switzerland',
                'DATE CREATED': time.ctime(time.time()),
            })

            # create dimensions     
            nc.createDimension(profile, size=size)
            nc.createDimension('country', size=len(countries))

            nc_cid = nc.createVariable('country', 'i2', ('country'))
            nc_cid[:] = np.array(countries, 'i2')
            nc_cid.long_name = 'EMEP country code'

    
def write_single_variable(path, profile, values, tracer, snap):
    filename = os.path.join(path,profile+".nc")
    if profile =="hourofday":
        descr = 'diurnal scaling factor'
    if profile ==  "dayofweek":
        descr ='day-of-week scaling factor'
    if profile == "monthofyear":
        descr = 'month-of-year scaling factor'
    if profile == "hourofyear":
        descr = 'hour-of-year scaling factor'
            
        
    with netCDF4.Dataset(filename, 'a') as nc:
        # create variables and attributes
        # if profile == "hourofday" or profile == "hourofyear":
        #     nc_var = nc.createVariable(tracer+"_"+snap, 'f4', (profile))
        # else:
        nc_var = nc.createVariable(tracer+"_"+snap, 'f4', (profile, 'country'))
        if complex_profile:
            nc_var.long_name = "%s for species %s and SNAP %s" % (descr,tracer,snap)
        else:
            nc_var.long_name = "%s for GNFR %s" % (descr,snap)
        nc_var.units = '1'
        nc_var[:]= values





def main_complex(path):

    mean = True

    # read all data
    countries, snaps, daily, weekly, annual = read_all_data(TRACERS)
    countries = [0]+countries        
    n_countries = len(countries)

    create_netcdf(path,countries)

    # day of week and month of year
    dow = np.ones((7, n_countries))
    moy = np.ones((12, n_countries))
    hod = np.ones((24, n_countries))
    
    country_tz = get_country_tz(countries)
    
    
    for (tracer,snap) in itertools.product(TRACERS, snaps):
        if not only_ones:
            for i, country in enumerate(countries):            
                try:
                    hod[:,i] = permute_cycle_tz(country_tz[country],daily[snap])
                except KeyError:
                    pass

                try:
                    dow[:,i] = weekly[tracer][country, snap]
                    if mean:
                        dow[:5,i] = np.ones(5)*weekly[tracer][country, snap][:5].mean()
                except KeyError:
                    pass

                try:
                    moy[:,i] = annual[tracer][country, snap]
                except KeyError:
                    pass
        write_single_variable(path,"hourofday",hod,tracer,snap)                
        write_single_variable(path,"dayofweek",dow,tracer,snap)
        write_single_variable(path,"monthofyear",moy,tracer,snap) 

    # TODO: hour of year
    hours_in_year = np.arange(0, 8784)

def main_simple(path):
    mean = False

    #snaps = ["A","B","C","D","E","F","G","H","I","J"]

    countries = np.arange(74)
    # remove all countries not known, and not worth an exception
    countries = np.delete(countries,
                          [5,26,28,29,30,31,32,33,34,35,58,64,67,70,71])
    n_countries = len(countries)
    country_tz = get_country_tz(countries)


    create_netcdf(path,countries)

    snaps,daily = read_temporal_profile_simple("CHE_input/timeprofiles-hour-in-day_GNFR.csv")
    snaps,weekly = read_temporal_profile_simple("CHE_input/timeprofiles-day-in-week_GNFR.csv")
    snaps,monthly = read_temporal_profile_simple("CHE_input/timeprofiles-month-in-year_GNFR.csv")

    print(snaps)
    # day of week and month of year
    dow = np.ones((7, n_countries))
    moy = np.ones((12, n_countries))
    hod = np.ones((24, n_countries))

    for snap_ind,snap in enumerate(snaps):
        for i, country in enumerate(countries):            
            try:
                hod[:,i] = permute_cycle_tz(country_tz[country],daily[snap_ind,:])
            except KeyError:
                pass

            try:
                dow[:,i] = weekly[snap_ind,:]
                if mean:
                    dow[:5,i] = np.ones(5)*weekly[snap_ind,:5].mean()
            except KeyError:
                pass

            try:
                moy[:,i] = monthly[snap_ind]
            except KeyError:
                pass

        write_single_variable(path,"hourofday",hod,"GNFR",snap)
        write_single_variable(path,"dayofweek",dow,"GNFR",snap)
        write_single_variable(path,"monthofyear",moy,"GNFR",snap) 
    
    

if __name__ == "__main__":
    complex_profile=True
    if complex_profile:
        main_complex("./output")
    else:
        main_simple("./output")


