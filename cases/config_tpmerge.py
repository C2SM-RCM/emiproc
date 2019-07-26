
# list of input inventories
inv_1 = 'TNO'
inv_2 = 'Carbocount'

# List of countries to set to 0 in inv_1
# All others countries will be set to 0 in inv_2
countries = ['CH']

# input and output paths for time profile files
profile_path_in = [
    './example_output/hourofday.nc',
    './example_output/dayofweek.nc',
    './example_output/monthofyear.nc'
]
profile_path_out = [
    './example_output/hourofday_merged.nc',
    './example_output/dayofweek_merged.nc',
    './example_output/monthofyear_merged.nc'
]

