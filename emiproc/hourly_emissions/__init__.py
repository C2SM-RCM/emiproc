import time
import datetime
import os

import numpy as np

from netCDF4 import Dataset
from multiprocessing import Pool


def daterange(start_date, end_date):
    """Yield a consequitve dates from the range [start_date, end_date).

    From https://stackoverflow.com/a/1060330.

    Parameters
    ----------
    start_date : datetime.date
    end_date : datetime.date

    Yields
    ------
    datetime.date
        Consecutive dates, starting from start_date and ending the day
        before end_date.
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + datetime.timedelta(n)


def country_id_mapping(country_codes, grid):
    """Map a country index to each gridpoint

    grid contains an EMEP country code at each gridpoint.
    country_codes is a 1D-array with all EMEP country codes.
    Assign to each gridpoint the index in country_codes
    of the EMEP-code given at that gridpoint by grid-mapping.

    This is useful as profiles belonging to countries are stored
    in the same order as the country_codes.

    Parameters
    ----------
    country_codes : np.array(dtype=int, shape=(n, ))
        Contains EMEP country codes
    grid : np.array(dtype=int)
        Each element points is a EMEP code which should be present in
        country_codes

    Returns
    -------
    np.array(dtype=int, shape=grid.shape)
        Contains at each gridpoint the position of the EMEP code (from grid)
        in country_codes
    """
    # There are 134 different EMEP country-codes, this fits in an uint8
    res = np.empty(shape=grid.shape, dtype=np.uint8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            location = np.where(country_codes[:] == grid[i, j])
            try:
                res[i, j] = location[0][0]
            except IndexError:
                raise IndexError("Country-ID not found")

    return res


def extract_to_grid(grid_to_index, country_vals):
    """Extract the values from each country_vals [1D] on gridpoints given
    by grid_to_index (2D)

    Parameters
    ----------
    grid_to_index : np.array(dtype=int)
        grid_to_index has at every gridpoint an integer which serves as an
        index into country_vals.
        grid_to_index.shape == (x, y)
        grid_to_index[i,j] < n_k forall i < x, j < y, k < len(country_vals)
    country_vals : np.array(dtype=float)
        country_vals.shape == (n,)

    Returns
    -------
    np.array(dtype=float)
        np.array, containing at (i,j) the value from the country_vals-array
        indicated by the grid_to_indexarray.

        >>> res = extract_to_grid(x, y)
        >>> y.dtype == res.dtype
        True
        >>> x.shape == res.shape
        True
        >>> res[i,j] == y[x[i,j]]
        True
    """
    res = np.empty(shape=grid_to_index.shape, dtype=np.float32)
    for i in range(grid_to_index.shape[0]):
        for j in range(grid_to_index.shape[1]):
            res[i, j] = country_vals[grid_to_index[i, j]]
    return res


class CountryToGridTranslator:
    """A wrapper for extract_to_grid that stores the grid_to_index matrix.
    The job of this class could be done more elegantly by a lambda, but that
    doesn't work with Pool (lambdas are not pickleable).

        >>> val_on_emigrid = lambda x: extract_to_grid(emigrid_to_index, x)

    behaves the same as

        >>> val_on_emigrid = CountryToGridTranslator(emigrid_to_index)
    """

    def __init__(self, grid_to_index):
        self.grid_to_index = grid_to_index

    def __call__(self, x):
        return extract_to_grid(self.grid_to_index, x)


def write_metadata(outfile, emi_file, ver_file, variables, model):
    """Write the metadata of the outfile.

    Determine rlat, rlon from emi_file, levels from ver_file.

    Create "time", "rlon", "rlat", "bnds", "level" dimensions.

    Create "rlon", "rlat" variables from emi_file.

    Create an emtpy variable with dimensions ("time", "level", "rlat", "rlon")
    for element of variables.

    Parameters
    ----------
    outfile : netCDF4.Dataset
        Opened with 'w'-option. Where the data is written to.
    emi_file : netCDF4.Dataset
        Containing dimensions "rlon", "rlat"
    ver_file : netCDF4.Dataset
        Containing the variable "level_bnds" and varaible & dimension "level"
    variables : list(str)
        List of variable-names to be created
    model : str
        'cosmo-ghg' or 'cosmo-art'
    """
    rlat = emi_file.dimensions["rlat"].size
    rlon = emi_file.dimensions["rlon"].size
    level = ver_file.dimensions["level"].size

    outfile.createDimension("time")
    outfile.createDimension("rlon", rlon)
    outfile.createDimension("rlat", rlat)
    outfile.createDimension("bnds", 2)
    outfile.createDimension("level", level)

    time = outfile.createVariable(
        varname="time",
        datatype="double",
        dimensions="time",
        )
    time.units = "seconds since 2015-01-01 00:00"
    time.calendar = "proleptic_gregorian"

    lvl = outfile.createVariable(
        varname="level",
        datatype="float",
        dimensions="level",
        )
    lvl.axis = "Z"
    lvl.units = "meter"
    lvl.standard_name = "height"


    lvl_bnd = outfile.createVariable(
        varname="level_bnds",
        datatype="float",
        dimensions=("bnds","level"),
    )    
    lvl_bnd.units = "meters"

    rpol = outfile.createVariable(
        varname="rotated_pole",
        datatype="c",
        dimensions=("time", "level", "rlat", "rlon"),
        )
    rpol.grid_north_pole_latitude = 43
    rpol.grid_north_pole_longitude = -170
    rpol.north_pole_grid_longitude = 0
    rpol.grid_mapping_name = "rotated_latitude_longitude" ;


    
    var_srcfile = [
        ("rlon", emi_file),
        ("rlat", emi_file),
    ]

    for varname, src_file in var_srcfile:
        outfile.createVariable(
            varname=varname,
            datatype=src_file[varname].datatype,
            dimensions=src_file[varname].dimensions,
        )
        outfile[varname].setncatts(src_file[varname].__dict__)
        outfile[varname].grid_mapping = "rotated_pole"

    outfile["time"][:] = 0
    outfile["level"][:] = ver_file["layer_mid"][:]
    outfile["level_bnds"][:] = np.array(
        [ver_file["layer_bot"][:], ver_file["layer_top"][:]]
    )
    outfile["rlat"][:] = emi_file["rlat"][:]
    outfile["rlon"][:] = emi_file["rlon"][:]

    for varname in variables:
        outfile.createVariable(
            varname=varname,
            datatype="float32",
            dimensions=("time", "level", "rlat", "rlon"),
        )
        if model == 'cosmo-ghg':
            outfile[varname].units = "kg m-2 s-1"
        elif model == 'cosmo-art':
            outfile[varname].units = "kg h-1 cell-1"
        else:
            raise ValueError('model has to be "cosmo-ghg" or "cosmo-art"')


def extract_matrices(infile, var_list, indices, transform=None):
    """Extract the array specified by indices for each variable in var_list
    from infile.

    If transform is not None, it is applied to the extracted array.

    Parameters
    ----------
    infile : str
        Path to netcdf file
    var_list : list(list(str))
    indices : slice()
    transform : function
        Takes as input the extracted array (infile[var][indices]).
        Default: None (== identity)

    Returns
    -------
    dict()
        Return a dictionary of varname : np.array() pairs.

        >>> mats = extract_matrices(if, ['myvar'], np.s_[0, :])
        >>> np.allclose(mats['myvar'], if['myvar'][0, :], rtol=0, atol=0)
        True
        >>> mats2 = extract_matrics(if, ['myvar'], np.s_[0, :], np.max)
        >>> mats2['myvar'] == np.max(if['myvar'][0, :])
        True
    """
    res = dict()

    with Dataset(infile) as data:
        if transform is None:
            for subvar_list in var_list:
                for var in subvar_list:
                    res[var] = np.array(data[var][indices])
        else:
            for subvar_list in var_list:
                for var in subvar_list:
                    res[var] = np.array(transform(data[var][indices]))

    return res


def process_day(date, path_template, lists, matrices, datasets, model):
    """Process one day of emissions, resulting in 24 hour-files.

    Loop over all hours of the day, create one file for each hour.

    For each file, the netcdf-file is created with the name specified by
    name_template. The metadata is written by write_metadata() using
    information from datasets (possible bottleneck as all processes read from
    the same netcdf files).

    Then, for each variable it's values at all points are computed using data
    from the matrices. The variables are specified in lists.

    Parameters
    ----------
    date : datetime.date
        Date for which the emissions are computed.
    path_template : str
        Path to output-file containg date-placeholders (such as
        '%Y%m%d%H' for YYYYMMDDHH).
    lists : dict
        A namedtuple containing the following key/value pairs:
        -   'variables' : list(str)
                List of all the variable names.
        -   'cats' : list(str)
                List of all categories (used as key for matrices.emi_mats).
        -   'tps' : list(str)
                List of all temporal profiles (used as key for
                matrices.hod_mats, matrices.dow_mats, matrices.moy_mats).
                Should have the same shape as cat_list.
        -   'vps' : list(str)
                List of all vertical profiles (used as key for
                matrices.vp_mats). Should have the same shape as cat_list.
        -  'contribution_list' : dict

    matrices : dict
        A dict containing the actual emission-data and the profiles applied
        to it:
        -   'emi_mats' : dict(str: np.array)
                Dict mapping categories to 2D np.arrays containig emission
                values on the grid. Shape of the arrays: (rlat, rlon).
        -   'hod_mats' : list(dict(str: np.array)
                List of 24 dicts mapping temporal profiles to 2D np.arrays
                containing hour-profile-values for each gridpoint. Shape of the
                arrays: (rlat, rlon).
        -   'dow_mats' : dict(str: np.array)
                Dict mapping temporal profiles to 2D np.arrays containing day-
                profile-values for each gridpoint. Shape of the arrays:
                (rlat, rlon).
        -   'moy_mats' : dict(str: np.array)
                Dict mapping temporal profiles to 2D np.arrays containing
                month-profile-values for each gridpoint. Shape of the arrays:
                (rlat, rlon).
        -   'ver_mats' : dict(str: np.array)
                Dict mapping vertical profiles to 1D np.arrays containg
                vertical profile-values for each level.
                Shape of the arrays: (7, ).
    datasets : dict
        A dict containing paths to netcdf-files with information about the
        metadata of the produced emissions-file, most importantly rlat, rlon,
        and levels. See the detailed usage of these datasets in write_metadata
        -   'emi_path' : str
                Dataset containing grid information, such as rlat, rlon.
        -   'ver_path' : str
                Dataset containing vertical information, such as levels.

    Returns
    -------
    datetime.date
        Returns the date of the processed day (the first argument)
    """
    print(date.strftime("Processing %x..."))

    with Dataset(datasets["emi_path"]) as emi, \
         Dataset(datasets["ver_path"]) as ver:
        rlat = emi.dimensions["rlat"].size
        rlon = emi.dimensions["rlon"].size
        levels = ver.dimensions["level"].size

        for hour in range(24):
            day_hour = datetime.datetime.combine(date, datetime.time(hour))
            of_path = day_hour.strftime(path_template)
            with Dataset(of_path, "w") as of:
                write_metadata(
                    outfile=of,
                    emi_file=emi,
                    ver_file=ver,
                    variables=lists["variables"],
                    model=model
                )

                for v, var in enumerate(lists["variables"]):
                    oae_vals = np.zeros((levels, rlat, rlon))
                    for cat, tp, vp in zip(
                        lists["cats"][v], lists["tps"][v], lists["vps"][v]
                    ):
                        emi_mat = matrices["emi_mats"][cat]
                        hod_mat = matrices["hod_mats"][hour][tp]
                        dow_mat = matrices["dow_mats"][tp]
                        moy_mat = matrices["moy_mats"][tp]
                        ver_mat = matrices["ver_mats"][vp]

                        # Add dimensions so numpy can broadcast
                        ver_mat = np.reshape(ver_mat, (levels, 1, 1))

                        # Compute emissions
                        value = emi_mat * hod_mat * dow_mat * moy_mat * ver_mat

                        # Apply speciation
                        if lists['contribution_list'] is not None:
                            value *= lists['contribution_list'][var].get_wildcard(cat)

                        oae_vals += value
                    # Careful, automatic reshaping!
                    of[var][0, :] = oae_vals

    return date


def process_day_one_arg(arg):
    """Simple wrapper permitting process_day to be called with one argument -
    a tuple of it's arguments"""
    return process_day(*arg)


def generate_arguments(
    start_date,
    end_date,
    emi_path,
    ver_path,
    hod_path,
    dow_path,
    moy_path,
    var_list,
    catlist,
    tplist,
    vplist,
    contribution_list,
    output_path,
    output_prefix,
    model
):
    """Prepare the arguments for process_day() (mainly extract the relevant data
    from netcdf-files) and yield them as tuples.

    Create the path_template for the output-files from output_path and
    output_prefix.

    Create the namedtuples to hold the argument-groups.

    Extract the data from the netcdf-files.

    For each day, pack the data into the tuples an yield them.

    Parameters
    ----------
    start_date : datetime.date
    end_date : datetime.date
        Emission files up to but not including this date are processed.
    emi_path : str
        Path to dataset containing the yearly emission average.
    ver_path : str
        Path to dataset containing vertical profiles per country.
    hod_path : str
        Dataset containing hour-profiles per country.
    dow_path : str
        Path to dataset containing day-profiles per country.
    moy_path : str
        Path to dataset containing month-profiles per country.
    var_list : list(str)
        List of variable names to be written into the output file.
        The i-th variable is composed of all the categories in the list
        at the i-th position in the catlist.
    catlist : list(list(str))
        List containing the emission category identifiers (keys for emi).
    tplist : list(list(str))
        List containing temporal profile identifiers (keys for hod, dow, moy).
    vplist : list(list(str))
        List containing vertical profile identifiers (keys for ver).
    output_path : str
        Path to directory where the output files are generated.
    output_prefix : str
        Prefix of the filename of the generated files.
        output_name="emis_" -> output_path/emis_YYYYMMDDHH.nc

    Yields
    ------
    tuple
        A tuple containing the arguments for process_day() for each day,
        starting at start_date and ending the day before end_date.
    """
    start = time.time()
    path_template = os.path.join(output_path, output_prefix + "%Y%m%d%H.nc")

    with Pool(16) as pool:  # have 32 parallel processes later
        # Create grid-country-mapping
        print("Creating gridpoint -> index mapping...")
        with Dataset(emi_path) as emi, Dataset(moy_path) as moy:
            mapping_result = pool.apply_async(
                country_id_mapping,
                args=(moy["country"][:], emi["country_ids"][:]),
            )

        # Time- and (grid-country-mapping)-independent data
        args_indep = [
            (emi_path, catlist, np.s_[:]),
            (ver_path, vplist, np.s_[:]),
        ]

        print("Extracting average emissions and vertical profiles...")
        res_indep = pool.starmap(extract_matrices, args_indep)
        emi_mats = res_indep[0]
        ver_mats = res_indep[1]
        print("... finished average emissions and vertical profiles")

        # get() blocks until apply_async is finished
        emigrid_to_index = mapping_result.get()
        val_on_emigrid = CountryToGridTranslator(emigrid_to_index)
        print("... finished gridpoint -> index mapping")

        # Time- and (grid-country-mapping)-dependent data
        assert start_date.month == end_date.month, (
            "Adjust the script to " "prepare data for multiple" " months."
        )
        month_id = start_date.month - 1
        args_dep = [(moy_path, tplist, np.s_[month_id, :], val_on_emigrid)]

        for day in range(7):  # always extract the whole week
            args_dep.append(
                tuple([dow_path, tplist, np.s_[day, :], val_on_emigrid])
            )

        for hour in range(24):
            args_dep.append(
                tuple([hod_path, tplist, np.s_[hour, :], val_on_emigrid])
            )

        print("Extracting month, day and hour profiles...")
        # List of dicts (containing 2D arrays) is not ideal for cache
        # locality (dict of 3D arrays would be better)
        res_dep = pool.starmap(extract_matrices, args_dep)
        moy_mats = res_dep[0]
        dow_mats = res_dep[1:8]
        hod_mats = res_dep[8:]
        print("... finished temporal profiles")

    lists = {
        "variables": var_list,
        "cats": catlist,
        "tps": tplist,
        "vps": vplist,
        'contribution_list': contribution_list,
    }
    datasets = {
        "emi_path": emi_path,
        "ver_path": ver_path,
    }

    stop = time.time()
    print("Finished extracting data in " + str(stop - start))
    for day_date in daterange(start_date, end_date):
        matrices = {
            "emi_mats": emi_mats,
            "hod_mats": hod_mats,
            "dow_mats": dow_mats[day_date.weekday()],
            "moy_mats": moy_mats,
            "ver_mats": ver_mats,
        }

        yield (day_date, path_template, lists, matrices, datasets, model)



def main(
    path_emi,
    output_path,
    output_name,
    prof_path,
    start_date,
    end_date,
    var_list,
    catlist,
    tplist,
    vplist,
    contribution_list,
    model
):
    point1 = time.time()

    # Make output folder
    os.makedirs(output_path, exist_ok=True)

    # Prepare arguments
    args = generate_arguments(
        start_date=start_date,
        end_date=end_date,
        emi_path=path_emi,
        ver_path=os.path.join(prof_path, "vertical_profiles.nc"),
        hod_path=os.path.join(prof_path, "hourofday.nc"),
        dow_path=os.path.join(prof_path, "dayofweek.nc"),
        moy_path=os.path.join(prof_path, "monthofyear.nc"),
        var_list=var_list,
        catlist=catlist,
        tplist=tplist,
        vplist=vplist,
        contribution_list=contribution_list,
        output_path=output_path,
        output_prefix=output_name,
        model=model
    )

    with Pool(14) as pool:  # 2 weeks in parallel
        # Use imap for smaller memory requirements (arguments are only built
        # when they are needed)
        res = pool.imap(func=process_day_one_arg, iterable=args, chunksize=1)

        # Iterate through all results to prevent the script from returning
        # before imap is done
        for r in res:
            print(r.strftime("... finished %x"))

    point2 = time.time()
    print("Overall runtime: {} s".format(point2 - point1))

