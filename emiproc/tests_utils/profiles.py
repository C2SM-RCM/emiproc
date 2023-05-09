import xarray as xr
import numpy as np


da_profiles_indexes_catsub = xr.DataArray(
    data=np.array([[0, 1, 1], [2, 0, 3]]),
    coords=[["a", "b"], ["CO2", "CH4", "NOx"]],
    dims=["category", "substance"],
)
da_profiles_indexes_sub = xr.DataArray(
    data=np.array([0, 1, 1]),
    coords=[["CO2", "CH4", "NOx"]],
    dims=["substance"],
)

if __name__ == "__main__":
    from emiproc.profiles.utils import get_desired_profile_index

    print(da_profiles_indexes_catsub)

    print(get_desired_profile_index(da_profiles_indexes_catsub, cat="a", sub="CH4"))
    print(
        get_desired_profile_index(
            da_profiles_indexes_sub,
            cat="a",
            # sub="CH4"
        )
    )
    print(
        get_desired_profile_index(
            da_profiles_indexes_catsub, cell=43, cat="a", sub="CH4"
        )
    )
