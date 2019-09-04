# Online emission processing

Preprocessing of the emissions for the online emission module of cosmo.
Produces gridded annual emissions as well as temporal and vertical profiles.

## Installation
To use the code, just download or clone the repository. The package can be installed with
```
    $ python setup.py install
```

The following requirements on third-party packages:

* Python (>= 3.6)
* cartopy
* netCDF4
* numpy
* shapely
* xarray

Please note emission inventories are not included in the repository, but have to
be obtained separately.

## Quickstart

1. Take one of the configuration files in the cases folder and adjust it to your case.

2. Generate the emission files:
```
    $ python -m epro grid --case config_{tno|ch}
```

3. Generate the profiles:
```
    $ python -m epro tp --case-file <filename>  # for temporal profiles
    $ python -m epro vp                         # for vertical profiles
```

## Gridded annual emissions

Emissions are read from the inventory and projected onto the COSMO grid.

The necessary information, such as grid characterstics and species, are supplied via
a config file. Since emission inventories can be structured quite differently, it may
also be necessary to adapt the main script. The provided examples are a good starting
point.

### Grids

In the file `grids.py` you can find definitions for classes handling common gridtypes
(COSMO, TNO, swiss). Use them in your configuration file to specify your grid.

If your grid can not be represented by an existing one, implement your own grid class
by inheriting from the `Grid` baseclass and implementing the required methods.

## License

This work is licensed under the Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or
send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
