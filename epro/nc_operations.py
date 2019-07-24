"""
Operate on netcdf datasets
==========================
Creating or copying variables of netcdf-datasets with netCDF4 is a
hassle. These functions and classes aim to simplify the process.


Examples
--------
Copy a dataset and add a variable from a third one::

    with netCDF4.Dataset("a.nc") as src, netCDF4.Dataset("b.nc", "w") as dst:
        copy_dataset(src, dst)
        with netCDF4.Dataset("c.nc") as src2:
            copy_variable(src2, dst, "CO2")

Copy dimensions and rename 'lev' to 'level', copy 'CO2' variable::

    # Prepare dimension- and variable-copiers
    dim_names = [('lev', 'level'), ('lon', 'lon'), ('lat', 'lat')]
    dim_copiers = [DimensionCopier(srcn, dstn) for srcn, dstn in dim_names]

    # Have to give dimensions explicitly because of name-change
    co2_copier = VariableCopier('CO2', var_args={'dimensions': ('level',
                                                                'lon',
                                                                'lat')})

    # Apply the copiers to the datasets
    with netCDF4.Dataset("a.nc") as src, netCDF4.Dataset("b.nc", "w") as dst:
        for dc in dim_copiers:
            dc.apply_to(src, dst)
            # Also copy associated variable
            dc.copy_variable(src, dst)
        co2_copier.apply_to(src, dst)

Copy only first time-step::

    with netCDF4.Dataset("a.nc") as src:
        dim_names = list(src.dimensions.keys())
        dim_names.remove('time')
        copiers = [DimensionCopier(dim) for dim in dim_names]

        for varname in src.variables.keys():
            # Only works if all variables have the same dimensions.
            # If not, specify dimensions for each variable.
            # var_val_indices indicates which part of the variable-values
            # should be copied. Here we take the first index of the first
            # dimension ('time').
            c = VariableCopier(varname,
                               var_args={'dimensions': tuple(dim_names),
                                         'var_val_indices': np.s_[0,:]})
            copiers.append(c)

        with netCDF4.Dataset("b.nc", "w") as dst:
            for c in copiers:
                c.apply_to(src, dst)

Extract the bottom three layers in the top-left corner of the domain::

    with netCDF4.Dataset("a.nc") as src, netCDF4.Dataset("b.nc", "w") as dst:
        extract_subdomain(src, dst,
                          {'rlat': np.s_[:50],
                           'rlon': np.s_[:50],
                           'level': np.s_[-3:]})

Copying the 'lon' variable from src to dst, while changing the values
and adding an attribute::

    with netCDF4.Dataset("a.nc") as src, netCDF4.Dataset("b.nc", "w") as dst:
        vals = np.random.rand(src['lon'].shape)
        cp = VariableCreator.from_existing_var(src['lon'],
                                               var_vals=vals,
                                               var_attrs={'uncertainty':
                                                          'large'})
        cp.apply_to(dst)

You can find an application of these tools at
gitlab.empa.ch/abt503/apps/cosmo_processing_chain/blob/master/jobs/tools/mozart2int2lm.py


Design Choices
--------------
These tools don't explicitly open/close netcdf-file (notice the missing
import of netCDF4). The reason for that is that opening/closing involves
reading/writing to disk. Forcing the user to do these slow operations prevents
repeatedly opening/closing files implicitly.

The on first glance unintuitive usage of
    - What do you want to copy?  -   >>> c = VariableCopier("varname")
    - Between which files?       -   >>> c.apply_to(src_dataset, dst_dataset)
stems from limitations by the multiprocessing module. It is not possible to use
a netCDF4.Dataset as an argument to a multiprocessing-function.


Tipps
-----
Speeding up operations on netCDF4.Datasets can be mainly achieved by:

    1. Make sure work on a numpy.array.
       The default type of a dataset['varname'][:] is a numpy.ma.array, which
       checks on every element access if the element is masked out and is thus
       very slow.
    If this is not fast enough
    2. Avoid manual loops.
       If you can reformulate your operation as a matrix-multiplication or
       element-wise addition/multiplication, the optimized numpy functions
       are much faster than a explicit loop.
    If this is not fast enough
    3. Use the multiprocessing module.
       If you still need more speed, parallelizing your script will help.


Planned extensions
------------------
-   Function to combine multiple files into one
        Especially along the time-dimension to combine cosmo-output into one file
        Other dimensions would be possible but seem more complicated & less
        useful.
"""

import numpy as np


def copy_dataset(src_dataset, dst_dataset):
    """Copy all attrs, dimensions and variables from src_dataset to dst_dataset"""
    copiers = []
    for dim in src_dataset.dimensions.keys():
        copiers.append(DimensionCopier(dim))
    for var in src_dataset.variables.keys():
        copiers.append(VariableCopier(src_names=var))

    for attr in src_dataset.ncattrs():
        dst_dataset.setncattr(attr, src_dataset.getncattr(attr))
    for copier in copiers:
        copier.apply_to(src_dataset, dst_dataset)


def copy_variable(
    src_dataset,
    dst_dataset,
    src_names,
    dst_name=None,
    var_args={},
    var_val_indices=np.s_[:],
    var_attrs={},
):
    """Copy a variable from src_dataset to dst_dataset.

    Parameters
    ----------
    src_dataset : netCDF4.Dataset
    dst_dataset : netCDF4.Dataset

    For a thorough explanation of the other arguments, refer to
    VariableCopier.__init__()
    """
    (
        VariableCopier(
            src_names=src_names,
            dst_name=dst_name,
            var_args=var_args,
            var_val_indices=var_val_indices,
            var_attrs=var_attrs,
        ).apply_to(src_dataset, dst_dataset)
    )


def extract_subdomain(src_dataset, dst_dataset, subdomain):
    """Extract a subdomain of src_dataset and write it into dst_dataset

    Parameters
    ----------
    src_dataset : netCDF4.Dataset
    dst_dataset : netCDF4.Dataset
    subdomain: dict(str, numpy.lib.index_tricks.IndexExpression object)
        Specify the subdomain by giving a slice for each dimension in the
        src_dataset that you want to narrow down. Dimensions not in this
        dict will be copied in full.

    Example
    -------
    Only copy the first timepoint
    >>> extract_subdomain(src, dst, {'time': np.s_[0]})

    Extract the bottom-right corner of the domain
    >>> extract_subdomain(src, dst, {'rlat': np.s_[-50:], 'rlon': np.s_[-50:]})
    """
    copiers = []

    for dim in src_dataset.dimensions:
        if dim in subdomain:
            copiers.append(
                DimensionCopier(src_name=dim, dim_val_indices=subdomain[dim])
            )
        else:
            copiers.append(DimensionCopier(src_name=dim))

    for variable in src_dataset.variables:
        var_val_indices = []
        for dim in src_dataset[variable].dimensions:
            if dim in subdomain:
                var_val_indices.append(subdomain[dim])
            else:
                var_val_indices.append(np.s_[:])
        copiers.append(
            VariableCopier(
                src_names=variable, var_val_indices=tuple(var_val_indices)
            )
        )

    for copier in copiers:
        copier.apply_to(src_dataset, dst_dataset)


class DimensionCopier:
    """Copy netCDF-Dimensions"""

    def __init__(self, src_name, dst_name=None, dim_val_indices=np.s_[:]):
        """
        Parameters
        ----------
        src_name : str
        dst_name : str
            Defaults to None. If it is None, the name won't be changed
        dim_val_indices : numpy.lib.index_tricks.IndexExpression object
            Which subset of the dimension values should be copied. If the
            dimension size is not unlimited, it will be set to match the
            number of items specified in the slice.
            Defaults to s_[:]. If a slice is given, both start and stop have
            to be given, while the step has to be 1 (so that the number of
            elements can be computed).
        """
        self.src_name = src_name
        if dst_name is None:
            self.dst_name = src_name
        else:
            self.dst_name = dst_name

        # dim_val_indices could also be an int if we only copy a single val
        if isinstance(dim_val_indices, slice):
            assert (
                dim_val_indices.step is None or dim_val_indices.step == 1
            ), "Can only extract a closed subdomain. Step has to be 1"
        self.dim_val_indices = dim_val_indices

    def apply_to(self, src_dataset, dst_dataset):
        """Copy the specified dimension from src_dataset to dst_dataset"""
        src_dim = src_dataset.dimensions[self.src_name]

        if src_dim.size is not None and self.dim_val_indices != np.s_[:]:
            if isinstance(self.dim_val_indices, slice):
                dim_size = _zero_based_index(
                    self.dim_val_indices.stop, src_dim.size, start=False
                ) - _zero_based_index(self.dim_val_indices.start, src_dim.size)
            else:
                # Only copy a single value
                dim_size = 1
        else:
            dim_size = src_dim.size

        dst_dataset.createDimension(dimname=self.dst_name, size=dim_size)

    def copy_variable(self, src_dataset, dst_dataset):
        """Copy the variable associated with the dimension.

        If dim_val_indices is not [:], only the corresponding subset of the
        variable is copied.
        """
        copier = VariableCopier(
            src_names=[self.src_name],
            dst_name=self.dst_name,
            var_val_indices=self.dim_val_indices,
        )
        copier.apply_to(src_dataset, dst_dataset)


class VariableCopier:
    """Copy (and possibly alter) netCDF-Variables"""

    def __init__(
        self,
        src_names,
        dst_name=None,
        var_args={},
        var_val_indices=np.s_[:],
        var_attrs={},
    ):
        """
        Parameters
        ----------
        src_names : list(str)
            Names of the variables to be copied.
            If this is a string or a list of length 1, the corresponding
            netCDF-Variable will be copied into the new netCDF-file and
            named dst_name.
            If this is a list of strings, the variables will be added together
            and stored in the new netCDF-file named dst_name.
            Datatype and ncattrs will be copied from the first variable.
        dst_name : str
            Name of the variable in the destination file.
            If this is None, the name of the first variable in src_names will be
            used.
            Default: None.
        var_args: dict(str: str)
            Arguments for the netCDF4.Dataset.createVariable-function. Arguments
            given here overwrite the values obtained from the first variable in
            src_names. Useful for setting zlib, complevel, shuffle, fletcher32,
            contiguous, chunksizes and endian, as these can't be inferred from
            the source-variable.
            Default: dict()
        var_val_indices: numpy.lib.index_tricks.IndexExpression object
            Which values to copy from the source variables. By default, all
            values are copied.
            If only a subset of the values are copied, make sure the shape of
            the copied values matches the dimensions of the destination
            variable.
            Default: numpy.s_[:]
        var_attrs : dict(str: str)
            key-value pairs get turned into additional ncattrs for the var, on
            top of the ncattrs copied from the first variable in scr_names.
            Values given here overwrite values from the copied variable with
            the same attribute-name.
            Useful for adding or changing attributes, such as units.
            Default: Dict()
        """
        if not isinstance(src_names, list):
            self.src_names = [src_names]
        else:
            self.src_names = src_names

        if dst_name is None:
            self.dst_name = self.src_names[0]
        else:
            self.dst_name = dst_name
        self.var_args = var_args.copy()
        self.var_val_indices = var_val_indices
        self.var_attrs = var_attrs.copy()

    def apply_to(self, src_dataset, dst_dataset):
        """Copy the specified variable(s) from src_dataset to dst_dataset.

        If there are multiple source variables, their values will be added.
        Raise an exception if their dimensions are different.

        The datatype, least_significant_digit and the attributes will be copied
        from the first element of the src_names list.

        Constructor arguments  zlib, complevel, shuffle, fletcher32, contiguous,
        chunksizes and endian can't (easily) be obtained from the variable
        in src_dataset. If they should be set, specify them with the var_args.
        """
        ref_var = src_dataset[self.src_names[0]]

        name = self.dst_name
        # Take datatype from ref_var and not from self.var_args, since we
        # need to check if we can accumulate values based on src-dtype, not
        # dest-dtype.
        dtype = ref_var.dtype
        try:
            dims_names = self.var_args["dimensions"]
        except KeyError:
            dims_names = ref_var.dimensions
        digit = getattr(ref_var, "least_significant_digit", None)

        # Test if variable datatype is Character, if not accumulate data
        if dtype == np.dtype("S1") or dtype == np.dtype("str"):
            if len(self.src_names) > 1:
                # Characters can't be added together
                raise TypeError(
                    "Can't combine variables with {} as "
                    "datatype.".format(dtype)
                )
            vals = src_dataset[self.src_names[0]][self.var_val_indices]
        else:
            vals = np.zeros_like(ref_var[self.var_val_indices])

            for var_name in self.src_names:
                if src_dataset[var_name].dimensions != ref_var.dimensions:
                    msg = "Different dimensions:\n{}: {},\n{}: {}.".format(
                        self.src_names[0],
                        ref_var.dimensions,
                        var_name,
                        src_dataset[var_name].dimensions,
                    )
                    raise ValueError(msg)

                vals += src_dataset[var_name][self.var_val_indices]

        # Get fill value
        ncattrs = ref_var.ncattrs()
        try:
            # Remove fill value from ncattrs because setting it after
            # construction is not possible
            ncattrs.remove("_FillValue")
            fill_value = ref_var.getncattr("_FillValue")
        except ValueError:
            fill_value = None

        var_args = dict(
            varname=name,
            datatype=dtype,
            dimensions=dims_names,
            least_significant_digit=digit,
            fill_value=fill_value,
        )
        var_args.update(self.var_args)

        var_attrs = dict([(name, ref_var.getncattr(name)) for name in ncattrs])
        var_attrs.update(self.var_attrs)

        (
            VariableCreator(
                var_args=var_args, var_vals=vals, var_attrs=var_attrs
            ).apply_to(dst_dataset)
        )


class VariableCreator:
    """Creates a netCDF4 variable with the specified parameters"""

    def __init__(self, var_args, var_vals, var_attrs={}):
        """

        Parameters
        ----------
        var_args : dict
            Gets unpacked as kwargs to netCDF4.Dataset.createVariable.
            Has to contain at least 'varname' & 'datatype'.
        var_vals : np.array
            Values to assign to the variable.
        var_attrs : dict
            key-value pairs get turned into ncattrs for the var with
            netCDF4.Variable.setncattr.
        """
        assert "varname" in var_args and "datatype" in var_args, (
            "varname and datatype are required arguments for "
            "variable creation."
        )

        self.varname = var_args["varname"]
        self.var_args = var_args
        self.var_vals = var_vals
        self.var_attrs = var_attrs

    @classmethod
    def from_existing_var(
        cls, src_variable, var_args={}, var_vals=None, var_attrs={}
    ):
        """Create a VariableCreator from an existing variable.

        If none of the optional arguments are given, this essentially does
        the same as a VariableCopier.

        However, the optional arguments allow you to overwrite certain aspects
        of the src_variable.

        Parameters
        ----------
        src_variable : netCDF4.Variable
        var_args : dict
            Arguments to the variable-constructor which should be different than
            the ones copied from src_variable (name, dimensions, datatype,
            fill_value) or arguments that can't be inferred from src_variable
            (zlib, complevel, shuffle, fletcher32, contiguous, chunksizes,
            endian).
            Default: dict()
        var_vals : np.array
            Values of the variable. If this is None, the values from
            src_variable are copied.
            Default: None
        var_attrs : dict
            Attributes of the created variable. Values given here overwrite
            existing attribute-values from the src_variable

        Example
        -------
        Copying the 'lon' variable from src to dst, while changing the values
        and adding an attribute can be done like so:

        >>> src, dst = netCDF4.Dataset('src.nc'), netCDF4.Dataset('dst.nc', 'w')
        >>> vals = np.random.rand(src['lon'].shape)
        >>> cp = VariableCreator.from_existing_var(src['lon'],
                                                   var_vals=vals,
                                                   var_attrs={'uncertainty':
                                                              'large'})
        >>> cp.apply_to(dst)
        """
        copied_var_args = {
            "varname": src_variable.name,
            "datatype": src_variable.datatype,
            "dimensions": src_variable.dimensions,
        }

        try:
            copied_var_args["fill_value"] = src_variable._FillValue
        except AttributeError:
            # Not all datatypes have a default value (string)
            pass

        var_args = {**copied_var_args, **var_args}

        if var_vals is None:
            var_vals = src_variable[:]

        copied_var_attrs = dict(
            [
                (name, src_variable.getncattr(name))
                for name in src_variable.ncattrs()
                if name != "_FillValue"
            ]
        )

        var_attrs = {**copied_var_attrs, **var_attrs}

        return cls(var_args=var_args, var_vals=var_vals, var_attrs=var_attrs)

    def apply_to(self, dst_dataset, _=None):
        """Create the variable with the parameters specified in the constructor
        in the netCDF4.Dataset dst_dataset

        For consistency with the other apply_to-functions (DimensionCopier,
        VariableCopier), this function takes two arguments.

        Parameters
        ----------
        dst_dataset : netCDF4.Dataset
            The variable is created in this dataset
        _
            Ignored, defaults to None
        """
        dst_dataset.createVariable(**self.var_args)
        # https://github.com/Unidata/netcdf4-python/issues/526
        if self.var_args["datatype"] is str:
            for i, char in enumerate(self.var_vals):
                dst_dataset[self.varname][i] = char
        else:
            dst_dataset[self.varname][:] = self.var_vals
        for attrname, attrval in self.var_attrs.items():
            dst_dataset[self.varname].setncattr(name=attrname, value=attrval)


def _zero_based_index(i, l, start=True):
    """Compute the 0-based index from the slice index i in a list of length l
    Assuming step is 1.

    Examples
    --------
    >>> _concrete_index(2, 100)
    2
    >>> _concrete_index(2, 100, start=False)
    2
    >>> _concrete_index(-1, 100, start=False)
    99
    >>> _concrete_index(None, 100)
    0
    >>> _concrete_index(None, 100, start=False)
    100
    """
    # 'stolen' from https://github.com/python/cpython/blob/master/Objects/sliceobject.c#L165
    assert l >= 0, ""

    if i is None:
        if start:
            return 0
        else:
            return l

    if start:
        assert i < l, ""
    else:
        assert i <= l, ""

    if i < 0:
        return i + l

    return i
