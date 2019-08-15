import os 

# list of input inventories
inv1 = 'TNO'
inv2 = 'Carbocount'

# List of countries to set to 0 in inv_1
# All others countries will be set to 0 in inv_2
countries = ['CH']

# input and output paths for time profile files
profile_path_in = [
    os.path.join('outputs','profiles','profiles_simple','hourofday.nc'),
    os.path.join('outputs','profiles','profiles_simple','dayofweek.nc'),
    os.path.join('outputs','profiles','profiles_simple','monthofyear.nc'),
]
profile_path_out = [
    os.path.join('outputs','profiles','hourofday.nc'),
    os.path.join('outputs','profiles','dayofweek.nc'),
    os.path.join('outputs','profiles','monthofyear.nc'),
]

