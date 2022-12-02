#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import time
import numpy as np
import netCDF4
from scipy.interpolate import interp1d


@dataclass
class VerticalProfile:
    """Vertical profile.

    A vertical profile defines how the emission is split vertically on the
    altitude. A vertical profile is defined simply by its ratios
    and the height levels.

    You can check the conditions required on the profile in 
    :py:func:`check_valid_vertical_profile`

    :arg ratios: The proportion of emission that is in each layer.
    :arg height: The top height of the layers.
        The first layer starts at 0 meter and ends at height[0].
        The second layer starts at height[0] and ends at height[1].
        Over the last height value, there is no emission.
    """

    ratios: np.ndarray
    height: np.ndarray

@dataclass
class VerticalProfiles:
    """Vertical profiles.

    This is very similar to :py:class:`VerticalProfile` but it can store
    many ratios for the same height distributions

    :arg ratios: a (n_profiles, n_heights) array.
    :arg height: Same as :py:attr:`VerticalProfile.height` .
    """

    ratios: np.ndarray
    height: np.ndarray

    @property
    def n_profiles(self) -> int:
        return self.ratios.shape[0]
    
    def __add__(self, other: VerticalProfiles):
        assert isinstance(other, VerticalProfiles)
        assert np.allclose(other.height, self.height)

        return VerticalProfiles(
            height=self.height,
            ratios=np.concatenate([self.ratios, other.ratios], axis=0)
        )

class GroupingMethod(Enum):
    KEEP_ALL_LEVELS = auto()


def get_mid_heights(max_heights: np.ndarray) -> np.ndarray:
    """Get the mid position of the height based on emiproc convention for height levels."""
    min_height = np.roll(max_heights, 1)
    # First level start at 0
    min_height[0] = 0
    mid_heights = (max_heights + min_height) / 2
    return mid_heights

def group_vertical_profiles(
    *profiles: VerticalProfile | VerticalProfiles,
    specified_levels: np.ndarray | None
) -> VerticalProfiles:
    """Group vertical profiles into one vertical profiles object.
    
    Allows for profiles of different height levels to be groupped into one.
    Sample the profile on the heights level given.
    
    Uses baseic interpolation (np.interp) and rescaling.

    :arg specified_levels: If this is specified, you can select an arbitray 
        scale on which to reproject. If not specified, all the levels found
        in the profiles will be used.
    """

    # Find the levels we want to use
    levels = specified_levels | np.unique(np.concatenate([p.height for p in profiles]))
    
    out_ratios = []
    for p in profiles:
        interpolated_ratios = interp1d(get_mid_heights(p.height),p.ratios,bounds_error=False, fill_value=0)(get_mid_heights(levels))
        if isinstance(p, VerticalProfiles):
            out_ratios.append(
            interpolated_ratios / interpolated_ratios.sum(axis=1).reshape(-1, 1)
            ) 
        else:
            out_ratios.append(
            interpolated_ratios / interpolated_ratios.sum()
            )
        
    return VerticalProfiles(np.concatenate(out_ratios, axis=0), levels)


def check_valid_vertical_profile(vertical_profile: VerticalProfile | VerticalProfiles):
    """Check that the vertical profile meets requirements.
    
    * height must have positive values
    * height must have strictly increasing values
    * ratios must sum up to one
    * ratios must all be >= 0 
    * ratios and height must have the same len
    * no nan values in any of the arrays

    :arg veritcal_profile: The profile to check.
    :raises AssertionError: If the profile is invalid.
    """

    assert isinstance(vertical_profile, ( VerticalProfile, VerticalProfiles))

    h = vertical_profile.height
    r = vertical_profile.ratios

    assert np.all(~np.isnan(r)) and np.all(~np.isnan(h)), "Cannot contain nan values"
    
    assert np.all(h > 0)
    assert np.all(h[1:] > np.roll(h, 1)[1:]), "height must be increasing"
    if isinstance(vertical_profile, VerticalProfile):
        assert np.sum(r) == 1.
        assert len(r) == len(h)
    else:
        ratio_sum = np.sum(r, axis=1)
        assert np.allclose(ratio_sum, 1.), f"Ratios must sum to 1, but {ratio_sum=}"
        assert r.shape[1] == len(h)
    assert np.all(r >= 0)

########################################
# BELOW IS LEGACY CODE FROM EMIPROC v1 #
########################################

# dimensions:
#     category = 10 ;
#     level = 16 ;
#     char = 20 ;
# variables:
#     char category(category, nchar) ;
#           category:long_name = "name of category" ;
#     float layer_bot(level) ;
#         layer_bot:units = "m" ;
#         layer_bot:long_name = "bottom of layer above ground" ;
#     float layer_top(level) ;
#         layer_top:units = "m" ;
#         layer_top:long_name = "top of layer above ground" ;
#     float SNAP-*(level) ;
#         SNAP-*:units = "1" ;
#         SNAP-*:long_name = "vertical scale factor for sources of SNAP-* category";


def get_all_levels(levels):  # levels are the top of the layer
    layer_top = levels
    layer_bot = [0] + levels[:-1]
    layer_mid = [(i + j) / 2 for (i, j) in zip(layer_top, layer_bot)]

    return layer_bot, layer_mid, layer_top


def write_netcdf(filename, categories, cat_name, levels, scale_factors):
    layer_bot, layer_mid, layer_top = get_all_levels(levels)

    with netCDF4.Dataset(filename, "w") as nc:

        # global attributes (add input data)
        nc.setncatts(
            {
                "DESCRIPTION": "Vertical profiles for emissions",
                "DATAORIGIN": "based on profiles developed for COST-728 action",
                "CREATOR": "Jean-Matthieu Haussaire",
                "EMAIL": "jean-matthieu.haussaire@empa.ch",
                "AFFILIATION": "Empa Duebendorf, Switzerland",
                "DATE CREATED": time.ctime(time.time()),
            }
        )

        # create dimensions
        nc.createDimension("level", size=len(levels))

        # create variables and attributes
        nc_bot = nc.createVariable("layer_bot", "f4", ("level"))
        nc_bot.long_name = "bottom of layer above ground"
        nc_bot.units = "m"
        nc_bot[:] = layer_bot

        nc_mid = nc.createVariable("layer_mid", "f4", ("level"))
        nc_mid.long_name = "middle of layer above ground"
        nc_mid.units = "m"
        nc_mid[:] = layer_mid

        nc_top = nc.createVariable("layer_top", "f4", ("level"))
        nc_top.long_name = "top of layer above ground"
        nc_top.units = "m"
        nc_top[:] = layer_top

        for (i, cat) in enumerate(categories):
            nc_sca = nc.createVariable(cat_name + cat, "f4", ("level"))
            nc_sca.long_name = (
                "vertical scale factor for sources of %s category" % cat
            )
            nc_sca.units = "1"
            nc_sca[:] = scale_factors[i]


def read_profiles(filename, nlevel=16):
    levels = []
    categories = []
    profiles = []

    with open(filename) as profile_file:

        all_sevens = []

        for line in profile_file:

            # skip comments
            if line.startswith('#'):
                continue

            # read levels
            if levels == []:
                levels = [int(i) for i in line.split("\t")[1:]]
                continue

            # read profiles
            values = line.split()
            cat = values[0]
            profile = values[1:]

            categories.append(cat)
            profiles.append(profile)

    return categories, np.array(profiles, "f4"), levels


def main(output_filename, profile_filename, prefix='GNFR_'):
    categories, profiles, levels = read_profiles(profile_filename)
    write_netcdf(output_filename, categories, prefix, levels, profiles)

