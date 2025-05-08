from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from emiproc.profiles.temporal.utils import concatenate_time_profiles
from emiproc.profiles.temporal.profiles import (
    AnyProfiles,
    AnyTimeProfile,
    SpecificDayProfile,
    TemporalProfile,
)
from emiproc.profiles.temporal.specific_days import SpecificDay

logger = logging.getLogger(__name__)



def _get_type(
    t: tuple[type[SpecificDayProfile], SpecificDay] | AnyTimeProfile,
) -> type[SpecificDayProfile]:
    if isinstance(t, tuple):
        return t[0]
    return t

  
def rescale_ratios(ratios: np.ndarray) -> np.ndarray:
    """Rescale the ratios to sum up to 1.

    Profiles with ratios of only zeros will be set to constant profiles.

    :arg ratios: The ratios to rescale.
    """

    sums = ratios.sum(axis=0)

    mask_zero = sums == 0

    return_ = ratios / sums
    return_[:, mask_zero] = 1.0 / ratios.shape[0]

    return return_



class CompositeTemporalProfiles:
    """A helper class to handle mixtures of temporal profiles.

    Acts similar to a TemporalProfile

    Stores a dict for each type of profile,
    """

    _profiles: dict[
        type[AnyTimeProfile] | tuple[type[SpecificDayProfile], SpecificDay],
        AnyTimeProfile,
    ]
    # Store for each type, the indexes of the profiles
    _indexes: dict[
        type[AnyTimeProfile] | tuple[type[SpecificDayProfile], SpecificDay] | None,
        np.ndarray[int],
    ]

    def __init__(self, profiles: list[list[AnyTimeProfile]] = []) -> None:
        n = len(profiles)
        self._profiles = {}
        profiles_lists = {}
        self._indexes = {}
        # Get the unique types of profiles
        types = set(
            t if (t := type(p)) != SpecificDayProfile else (t, p.specific_day)
            for profiles_list in profiles
            for p in profiles_list
        )

        if len(types) == 0:
            # Empty profiles
            # only empty lists given
            self._indexes[None] = np.full(n, fill_value=-1, dtype=int)
            return

        # Allocate arrays
        for profile_type in types:
            if not isinstance(profile_type, tuple) and not issubclass(
                profile_type, TemporalProfile
            ):
                raise TypeError(
                    f"Profiles must be subclass of {TemporalProfile}. Not"
                    f" {profile_type=}"
                )
            profiles_lists[profile_type] = []
            self._indexes[profile_type] = np.full(n, fill_value=-1, dtype=int)
        # Construct the list and indexes based on the input
        for i, profiles_list in enumerate(profiles):
            if not isinstance(profiles_list, list):
                raise TypeError(
                    f"{profiles_list=} must be a list of {TemporalProfile}."
                )
            for profile in profiles_list:
                if profile.n_profiles != 1:
                    raise ValueError(
                        "Can only build CompositeTemporalProfiles from profiles with"
                        f" {profile.n_profiles=}, got {profile=}."
                    )
                p_type = type(profile)
                if p_type == SpecificDayProfile:
                    p_type = (p_type, profile.specific_day)
                list_this_type = profiles_lists[p_type]
                if self._indexes[p_type][i] != -1:
                    raise ValueError(
                        f"Cannot add {profile=} to {self=} as it was already added."
                    )
                self._indexes[p_type][i] = len(list_this_type)
                list_this_type.append(profile)
        # Convert the lists to arrays
        for profile_type, profiles_list in profiles_lists.items():
            ratios = np.concatenate([p.ratios for p in profiles_list])
            if isinstance(profile_type, tuple):
                profile = profile_type[0](
                    ratios=ratios,
                    specific_day=profile_type[1],
                )
            else:
                profile = profile_type(ratios=ratios)
            self._profiles[profile_type] = profile

    def __repr__(self) -> str:
        profile_name = lambda p: (
            f"{p[0].__name__}({p[1]})" if isinstance(p, tuple) else p.__name__
        )
        out = f"CompositeProfiles({len(self)} profiles "
        if len(self) < 10:
            out += f"from {[profile_name(t) for t in self.types]})"
        return out

    def __len__(self) -> int:
        indexes_len = [len(indexes) for indexes in self._indexes.values()]
        if not indexes_len:
            return 0
        # Make sure they are all equal
        if len(set(indexes_len)) != 1:
            raise ValueError(
                f"{self=} has different lengths of indexes for each profile type."
                f" {indexes_len=}"
            )
        return indexes_len[0]

    @property
    def n_profiles(self) -> int:
        """Return the number of profiles."""
        return len(self)

    def __getitem__(self, key: int) -> list[AnyTimeProfile]:
        return [
            self._profiles[p_type][index]
            for p_type, indexes in self._indexes.items()
            if p_type is not None and (index := indexes[key]) != -1
        ]

    def __setitem__(self, key: int, value: list[AnyTimeProfile]) -> None:
        self._array[key] = np.array(value, dtype=object)

    @property
    def types(self) -> list[AnyTimeProfile]:
        """Return the types of the profiles."""
        return list(set(self._profiles.keys()))

    @property
    def ratios(self) -> np.ndarray:
        """Return ratios of composite profiles.

        Idea is that we concatenate the ratio of each profile.
        nan values can be used when a profile is not defined for a given index.
        """
        # Case no profile given or only empty profiles
        if len(self.types) == 0:
            return np.empty((len(self), 0))

        return np.stack(
            [
                np.concatenate(
                    [
                        (
                            self._profiles[pt][index].ratios.reshape(-1)
                            if (index := self._indexes[pt][i]) != -1
                            else np.full(_get_type(pt).size, np.nan).reshape(-1)
                        )
                        for pt in self.types
                    ]
                )
                for i in range(len(self))
            ],
            # axis=1,
        )

    @property
    def scaling_factors(self) -> np.ndarray:
        """Return the scaling factors of the profiles."""
        # Case no profile given or only empty profiles
        if len(self.types) == 0:
            return np.empty((len(self), 0))

        return np.stack(
            [
                np.concatenate(
                    [
                        (
                            self._profiles[pt][index].ratios.reshape(-1)
                            * self._profiles[pt][index].size
                            if (index := self._indexes[pt][i]) != -1
                            else np.ones(_get_type(pt).size).reshape(-1)
                        )
                        for pt in self.types
                    ]
                )
                for i in range(len(self))
            ],
            # axis=1,
        )

    @classmethod
    def from_ratios(
        cls, ratios: np.ndarray, types: list[type], rescale: bool = False
    ) -> CompositeTemporalProfiles:
        """Create a composite profile, directly from the ratios.

        :arg ratios: The ratios of the profiles.
        :arg types: The types of the profiles, as a list of Temporal profiles types.
        :arg rescale: If True, the ratios will be rescaled to sum up to 1.

        """
        for t in types:
            # Check that the type is a subtype of TemporalProfile
            if not issubclass(t, TemporalProfile):
                raise TypeError(f"{t=} must be a {TemporalProfile}.")
        splitters = np.cumsum([0] + [t.size for t in types])
        logger.debug(f"{splitters=}")
        # Create the empty profiles
        profiles = [
            [
                t(rescale_ratios(r) if rescale else r)
                for i, t in enumerate(types)
                if not np.any(
                    np.isnan(r := profile_ratios[splitters[i] : splitters[i + 1]])
                )
            ]
            for profile_ratios in ratios
        ]
        return cls(profiles)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, CompositeTemporalProfiles):
            raise TypeError(f"{__value=} must be a {CompositeTemporalProfiles}.")
        if len(self) != len(__value):
            return False
        return (self.ratios == __value.ratios).all()

    @classmethod
    def join(cls, *profiles: CompositeTemporalProfiles) -> CompositeTemporalProfiles:
        """Join multiple composite profiles."""
        # Get the types of profiles
        _profiles = {}
        types = set(sum((p.types for p in profiles), []))
        profile_lenghts = [len(p) for p in profiles]
        total_len = sum(profile_lenghts)
        _indexes = {t: np.full(total_len, fill_value=-1, dtype=int) for t in types}
        for t in types:
            _this_type_profiles = [p._profiles[t] for p in profiles if t in p.types]
            _this_type_n_profiles = [
                len(p._profiles[t]) if t in p.types else 0 for p in profiles
            ]
            _profiles[t] = concatenate_time_profiles(_this_type_profiles)

            # offset in the indexes indexes
            curr_index = 0
            # offset in the profile indexes
            curr_profile = 0
            for i, n in enumerate(_this_type_n_profiles):
                if n == 0:
                    curr_index += profile_lenghts[i]
                    continue
                indexes = profiles[i]._indexes[t].copy()
                mask_invalid = indexes == -1
                indexes[~mask_invalid] += curr_profile
                _indexes[t][curr_index : curr_index + len(indexes)] = indexes
                curr_index += profile_lenghts[i]
                curr_profile += n

        # Get the indexes
        obj = cls([])
        obj._profiles = _profiles
        obj._indexes = _indexes

        return obj

    def copy(self) -> CompositeTemporalProfiles:
        """Return a copy of the object."""
        return CompositeTemporalProfiles.join(self)

    # define the addition to be the same as if this was a list
    def __add__(self, other: CompositeTemporalProfiles) -> CompositeTemporalProfiles:
        return self.join(self, other)

    def __radd__(self, other: CompositeTemporalProfiles) -> CompositeTemporalProfiles:
        return self.join(other, self)

    def append(self, profiles_list: list[AnyTimeProfile]) -> None:
        """Append a profile list to this."""
        new_len = len(self) + 1
        original_types = self.types
        # expend all the indexes list
        for t in self._indexes.keys():
            self._indexes[t] = np.concatenate(
                (
                    self._indexes[t],
                    np.array([-1], dtype=int),
                )
            )

        for p in profiles_list:
            t = type(p)
            if t == SpecificDayProfile:
                t = (SpecificDayProfile, p.specific_day)
            if t not in self._indexes.keys():
                if isinstance(t, tuple):
                    self._profiles[t] = t[0](ratios=p.ratios, specific_day=t[1])
                else:
                    self._profiles[t] = t(ratios=p.ratios)
                self._indexes[t] = np.full(new_len, fill_value=-1, dtype=int)
                self._indexes[t][-1] = 0
            else:
                self._indexes[t][-1] = len(self._profiles[t])
                self._profiles[t].ratios = np.concatenate(
                    (self._profiles[t].ratios, p.ratios)
                )

    @property
    def size(self) -> int:
        """Return the size of the profiles."""
        return sum(p.size for p in self._profiles.values())

    def broadcast(self, types: list[TemporalProfile]) -> CompositeTemporalProfiles:
        """Create a new composite profile with the given types.

        The non specified profiles will be set to constant profiles.
        """
        all_ratios = []
        for t in types:
            # Get constant ratios
            composite_ratios = np.ones((len(self), t.size)) / t.size
            if t in self.types:
                ratios = self._profiles[t].ratios
                # Need to scale with the indexes
                indexes = self._indexes[t]
                # Fill the ratios with the existing profiles
                mask_valid = indexes != -1
                composite_ratios[mask_valid, :] = ratios[indexes[mask_valid], :]

            all_ratios.append(composite_ratios)

        return CompositeTemporalProfiles.from_ratios(
            np.concatenate(all_ratios, axis=1), types
        )


def make_composite_profiles(
    profiles: AnyProfiles,
    indexes: xr.DataArray,
) -> tuple[CompositeTemporalProfiles, xr.DataArray]:
    """Create a composite temporal profiles from a list of profiles and indexes.

    :arg profiles: The profiles to use.
    :arg indexes: The indexes to use.
        The indexes must have a dim called "profile" with the name of the profile type.

    """

    if not isinstance(profiles, AnyProfiles):
        raise TypeError(f"{profiles=} must be an {AnyProfiles}.")

    logger.debug(f"making composite profiles from {profiles=}, {indexes=}")

    if "profile" not in indexes.dims:
        raise ValueError(f"{indexes=} must have a dim called 'profile'.")
    # If size of the profiles is 1, then we can simply return the profiles
    if indexes.profile.size == 1:
        # It is only one type of profile
        return CompositeTemporalProfiles([[p] for p in profiles]), indexes.squeeze(
            "profile"
        )

    # Stack the arrays to keep only the profiles dimension and the new stacked dim
    dims = list(indexes.dims)
    dims.remove("profile")
    stacked = indexes.stack(z=dims)

    str_array = np.array(
        [
            str(array.values.reshape(-1))
            for lab, array in stacked.groupby(group="z", squeeze=False)
        ]
    )
    logger.debug(f"{str_array=}")
    u, inv = np.unique(str_array, return_inverse=True)

    extracted_profiles = [
        [
            profiles[i]
            # Unpakc the profiles from the str
            for i in np.fromstring(array_str[1:-1], sep=" ", dtype=int)
            if i != -1
        ]
        # Loop over each unique profile found
        for array_str in u
    ]
    logger.debug(f"{extracted_profiles=}")
    new_indexes = xr.DataArray(inv, dims=["z"], coords={"z": stacked.z})

    # Remove the z dimension from the profiles
    out_indexes = new_indexes.unstack("z")

    return CompositeTemporalProfiles(extracted_profiles), out_indexes
