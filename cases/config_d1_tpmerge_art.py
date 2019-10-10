import os

# list of input inventories
inv1 = 'EU'
inv2 = 'CH'

# List of countries to set to 0 in inv_1
# All others countries will be set to 0 in inv_2
countries = ['CH']

# input and output paths for time profile files
profile_path_in = [
    os.path.join('oae-art-example','profiles','raw','hourofday.nc'),
    os.path.join('oae-art-example','profiles','raw','dayofweek.nc'),
    os.path.join('oae-art-example','profiles','raw','monthofyear.nc'),
]
profile_path_out = [
    os.path.join('oae-art-example','profiles','hourofday.nc'),
    os.path.join('oae-art-example','profiles','dayofweek.nc'),
    os.path.join('oae-art-example','profiles','monthofyear.nc'),
]
