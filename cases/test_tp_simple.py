"""
Config for processing time profiles.
"""

import os
import time

# use different time profiles for different species
profile_depends_on_species = False

# list of species (only used if `profile_depend_on_species == True`)
tracers = [] #['CO', 'NH3', 'NMVOC', 'NOx', 'P10', 'P25', 'SOx']

# Sets all the profiles to a uniform 1
only_ones = False

# Produces profiles for winter time
winter = False

# Averages the first five days of the week in the profile.
mean = False

# Input files
input_path = os.path.join(os.path.dirname(__file__), '..', 'files')

# Path to the csv file containing the time zones of each country
country_tz_file = os.path.join(input_path, "time_profiles", "country_tz.csv")

# Path to the csv file containing the hour in day profile
hod_input_file = os.path.join(input_path, "time_profiles",
                              "timeprofiles-hour-in-day_GNFR.csv")

# Path to the csv file containing the day in week profile
dow_input_file = os.path.join(input_path, "time_profiles",
                             "timeprofiles-day-in-week_GNFR.csv")

# Path to the csv file containing the month in year profile
moy_input_file = os.path.join(input_path, "time_profiles",
                              "timeprofiles-month-in-year_GNFR.csv")

# Output path
output_path = os.path.join('outputs','profiles','profiles_simple')

# Name of output variable
varname_format = 'GNFR_{category}' # FIXME: tracer -> species

# Metadata added to netCDF output files
nc_metadata = {
    "DESCRIPTION": "Temporal profiles for emissions",
    "DATAORIGIN": "TNO time profiles",
    "CREATOR": "Gerrit Kuhlmann",
    "EMAIL": "gerrit.kuhlmann@empa.ch",
    "AFFILIATION": "Empa Duebendorf, Switzerland",
    "DATE CREATED": time.ctime(time.time()),
}
