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


TRACERS = ['CO2']
only_ones = False
winter = False
mean = False

def permute_cycle_tz(tz,cycle):
    """
    Permute a daily cycle by a given amount of hours. 
    This is a way to convert from local time to UTC

    Parameters
    ----------
    tz: String in the shape "+0400"
        String corresponding to the timezone
    cycle: List of length 24
        Daily emissions to be permuted
    """
    
    shift = int(int(tz[1:3])*((tz[0]=="+")-0.5)*2)

    # half hours for the time zone are note considered
    try:
        answer = [cycle[shift-i] for i in range(len(cycle),0,-1)]
    except IndexError:
        answer = [cycle[shift-i+24] for i in range(len(cycle),0,-1)]

    return answer

def load_country_tz(filename,winter = True):
    """
    Load the list of country timzones from a csv file

    Parameters
    ----------
    filename: String
        Path to the csv file containing the timezone information
    winter: bool (defaut to True)
        Wether the winter time or the summertime is considered
    """
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
    """Checks that all countries in the gridded emission file have a corresponding time zone
    
    Parameters
    ----------
    filename: String
        Path to the gridded emissions
    all_tz: 
        List of time zones    
    """
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

def get_country_tz(countries):
    tz = load_country_tz("CHE_input/country_tz.csv",winter)
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
                    all_tz[country] = tz_int2str(tz[name])
                    break
                except KeyError:
                    continue
                    #all_tz[country] = "None"
        # if country_name=="":
        #     all_tz[country] = "None"
    return all_tz


def read_temporal_profile(path):    
    started = False
    data = []
    cats = []
    with open(path,encoding="latin") as prof:
        for l in prof:
            splitted = l.split(";")
            if splitted[0]=="1":
                started=True
            # if splitted[0]=="11":
            #     started=False
            if started:
                if splitted[1]=="F1":
                    cats.append("F")
                elif "F" in splitted[1]:
                    continue
                else:
                    cats.append(splitted[1])
                data.append([float(i) for i in splitted[3:]])

    return cats,np.array(data)

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

    
def write_single_variable(path, profile, values, tracer, cat):
    filename = os.path.join(path,profile+".nc")
    if profile =="hourofday":
        descr = 'diurnal scaling factor'
        comment = "first hour is 00h, last 23h local time"
    if profile ==  "dayofweek":
        descr ='day-of-week scaling factor'
        comment = "first day is Monday, last day is Sunday"
    if profile == "monthofyear":
        descr = 'month-of-year scaling factor'
        comment = "first month is Jan, last month is Dec"
    if profile == "hourofyear":
        descr = 'hour-of-year scaling factor'
        comment = "first hour is on Jan 1. 00h" 
        
    with netCDF4.Dataset(filename, 'a') as nc:
        nc_var = nc.createVariable(tracer+"_"+cat, 'f4', (profile, 'country'))
        nc_var.long_name = "%s for GNFR %s" % (descr,cat)
        nc_var.units = '1'
        nc_var.comment = comment
        nc_var[:]= values


def main(path):
    countries = np.arange(74)
    # remove all countries not known, and not worth an exception
    countries = np.delete(countries,
                          [5,26,28,29,30,31,32,33,34,35,58,64,67,70,71])
    n_countries = len(countries)

    country_tz = get_country_tz(countries)
    #validate_tz('../testdata/CHE_TNO_v1_1_2018_12/CHE_TNO_offline/emis_2015_Europe.nc',country_tz)
    print(country_tz)

    create_netcdf(path,countries)

    cats,daily = read_temporal_profile("CHE_input/timeprofiles-hour-in-day_GNFR.csv")
    cats,weekly = read_temporal_profile("CHE_input/timeprofiles-day-in-week_GNFR.csv")
    cats,monthly = read_temporal_profile("CHE_input/timeprofiles-month-in-year_GNFR.csv")

    print(cats)

    # day of week and month of year
    dow = np.ones((7, n_countries))
    moy = np.ones((12, n_countries))
    hod = np.ones((24, n_countries))

    for cat_ind, cat in enumerate(cats):
        for i, country in enumerate(countries):            
            try:
                hod[:,i] = permute_cycle_tz(country_tz[country],daily[cat_ind,:])
            except KeyError:
                pass

            try:
                dow[:,i] = weekly[cat_ind,:]
                if mean:
                    dow[:5,i] = np.ones(5)*weekly[cat_ind,:5].mean()
            except KeyError:
                pass

            try:
                moy[:,i] = monthly[cat_ind]
            except KeyError:
                pass

        write_single_variable(path,"hourofday",hod,"GNFR",cat)
        write_single_variable(path,"dayofweek",dow,"GNFR",cat)
        write_single_variable(path,"monthofyear",moy,"GNFR",cat) 
        
if __name__ == "__main__":
    main("./example_output")


