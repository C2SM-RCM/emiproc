"""Test for the vprm module.

Tests are there moslty to check that things run problerly and that the inputs/
outputs expected are there. The actual values are not checked.
"""

import pytest
import pandas as pd

import numpy as np
from emiproc.profiles.vprm import (
    calculate_vprm_emissions,
    calculate_vegetation_indices,
    interpolate_satellite_index_series_kalman,
    interpolate_satellite_index_series_lowess,
    interpolate_satellite_index_series,
    interpolate_satellite_indices,
)


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            ("T", "global"): [25, 30, 28, 27],
            ("vegetation_type_1", "lswi"): [0.5, 0.6, 0.7, 0.8],
            ("vegetation_type_1", "evi"): [0.2, 0.3, 0.4, 0.5],
            ("vegetation_type_2", "lswi"): [0.3, 0.4, 0.5, 0.6],
            ("vegetation_type_2", "evi"): [0.1, 0.2, 0.3, 0.4],
            ("RAD", ""): [100, 200, 150, 180],
        }
    )

    df_vprm = pd.DataFrame(
        {
            "alpha": [0.1, 0.2],
            "beta": [0.5, 0.6],
            "lambda": [0.8, 0.9],
            "Tmin": [20, 22],
            "Topt": [25, 27],
            "Tmax": [30, 32],
            "Tlow": [18, 20],
            "PAR0": [50, 60],
        },
        index=["vegetation_type_1", "vegetation_type_2"],
    )

    return df, df_vprm


def test_calculate_vprm_emissions(sample_data):
    df, df_vprm = sample_data
    result = calculate_vprm_emissions(df, df_vprm)

    assert ("PAR", "") in result.columns
    assert ("vegetation_type_1", "Tscale") in result.columns
    assert ("vegetation_type_2", "Tscale") in result.columns
    assert ("vegetation_type_1", "Wscale") in result.columns
    assert ("vegetation_type_2", "Wscale") in result.columns
    assert ("vegetation_type_1", "Pscale") in result.columns
    assert ("vegetation_type_2", "Pscale") in result.columns
    assert ("vegetation_type_1", "resp") in result.columns
    assert ("vegetation_type_2", "resp") in result.columns
    assert ("vegetation_type_1", "gee") in result.columns
    assert ("vegetation_type_2", "gee") in result.columns
    assert ("vegetation_type_1", "nee") in result.columns
    assert ("vegetation_type_2", "nee") in result.columns


def test_bad_model(sample_data):
    df, df_vprm = sample_data
    with pytest.raises(ValueError):
        calculate_vprm_emissions(df, df_vprm, model="bad_model")


def test_urban_model(sample_data):
    df, df_vprm = sample_data

    df[("T", "urban")] = [28, 32, 30, 29]
    df[("vegetation_type_1", "evi_ref")] = df[("vegetation_type_1", "evi")]
    df[("vegetation_type_2", "evi_ref")] = df[("vegetation_type_1", "evi")]
    df_vprm["isa"] = 0.5

    result = calculate_vprm_emissions(df, df_vprm, model="urban")
    assert ("vegetation_type_1", "nee") in result.columns
    assert ("vegetation_type_2", "nee") in result.columns


def test_urban_winbourne_data(sample_data):
    df, df_vprm = sample_data

    df[("T", "urban")] = [28, 32, 30, 29]
    df[("vegetation_type_1", "evi_ref")] = df[("vegetation_type_1", "evi")]
    df[("vegetation_type_2", "evi_ref")] = df[("vegetation_type_1", "evi")]
    df_vprm["isa"] = 0.5

    result = calculate_vprm_emissions(df, df_vprm, model="urban_winbourne")
    assert ("vegetation_type_1", "nee") in result.columns
    assert ("vegetation_type_2", "nee") in result.columns


def test_modified_vprm_model(sample_data):
    df, df_vprm = sample_data

    df_vprm["alpha1"] = 0.065
    df_vprm["alpha2"] = 0.0024

    df_vprm["theta1"] = 0.116
    df_vprm["theta2"] = -0.0005
    df_vprm["theta3"] = 0.0009

    df_vprm["gamma"] = 4.61

    df_vprm["Tcrit"] = -15.0
    df_vprm["Tmult"] = 0.55

    calculate_vprm_emissions(df, df_vprm, model="modified_groudji")


def test_missing_urban_temperature(sample_data):
    with pytest.raises(KeyError):
        calculate_vprm_emissions(*sample_data, model="urban")


def test_calculate_vegetation_indices():
    nir = np.array([0.8, 0.9, 0.7, 0.6])
    swir = np.array([0.4, 0.5, 0.3, 0.2])
    red = np.array([0.6, 0.7, 0.5, 0.4])
    blue = np.array([0.2, 0.3, 0.1, 0.0])

    evi, lswi, ndvi = calculate_vegetation_indices(nir, swir, red, blue)

    # Check that the arrays have the right shape
    assert evi.shape == (4,)
    assert lswi.shape == (4,)
    assert ndvi.shape == (4,)


@pytest.fixture
def sparse_satellite_series():
    index = pd.date_range("2023-01-01", periods=12, freq="h")
    return pd.Series(
        [0.22, 0.24, 0.25, 0.40, 0.27, 0.29, np.nan, 0.31, 0.32, np.nan, 0.34, 0.35],
        index=index,
    )


@pytest.fixture
def sparse_deciduous_df():
    index = pd.date_range("2023-01-01", periods=12, freq="h")
    return pd.DataFrame(
        {
            ("Deciduous", "evi"): [0.10, 0.12, np.nan, 0.35, 0.17, 0.20, np.nan, 0.23, np.nan, 0.24, 0.25, np.nan],
            ("Deciduous", "lswi"): [0.05, np.nan, 0.06, 0.07, np.nan, 0.09, 0.10, np.nan, 0.11, np.nan, 0.12, 0.13],
            ("RAD", ""): np.linspace(100, 200, 12),
        },
        index=index,
    )


def test_interpolate_satellite_index_series_filters_outlier(sparse_satellite_series):
    interpolated = interpolate_satellite_index_series(
        sparse_satellite_series,
        outlier_threshold=0.25,
    )

    assert interpolated.notna().all()
    # Outlier should be strongly corrected toward neighborhood values.
    assert interpolated.iloc[3] < 0.35


def test_interpolate_satellite_indices_dataframe(sparse_deciduous_df):
    out = interpolate_satellite_indices(
        sparse_deciduous_df,
        vegetation_types=["Deciduous"],
    )

    assert out[("Deciduous", "evi")].notna().all()
    assert out[("Deciduous", "lswi")].notna().all()
    assert ("Deciduous", "evi_mask") in out.columns
    assert ("Deciduous", "evi_extracted") in out.columns


@pytest.mark.slow
def test_interpolate_satellite_index_series_lowess(sparse_satellite_series):
    out = interpolate_satellite_index_series_lowess(
        sparse_satellite_series,
        frac=0.4,
        it=2,
    )

    assert out.notna().all()
    assert out.index.equals(sparse_satellite_series.index)


def test_interpolate_satellite_index_series_kalman(sparse_satellite_series):
    out = interpolate_satellite_index_series_kalman(
        sparse_satellite_series,
        transition_covariance=0.02,
        observation_covariance=0.05,
    )

    assert out.notna().all()
    assert out.index.equals(sparse_satellite_series.index)


@pytest.mark.slow
def test_interpolate_satellite_indices_with_lowess(sparse_deciduous_df):
    out_lowess = interpolate_satellite_indices(
        sparse_deciduous_df,
        vegetation_types=["Deciduous"],
        interpolation_method="lowess",
        frac=0.5,
        it=2,
    )

    assert out_lowess[("Deciduous", "evi")].notna().all()
    assert out_lowess[("Deciduous", "lswi")].notna().all()


def test_interpolate_satellite_indices_with_kalman(sparse_deciduous_df):
    out_kalman = interpolate_satellite_indices(
        sparse_deciduous_df,
        vegetation_types=["Deciduous"],
        interpolation_method="kalman",
        transition_covariance=0.02,
        observation_covariance=0.05,
    )
    assert out_kalman[("Deciduous", "evi")].notna().all()
    assert out_kalman[("Deciduous", "lswi")].notna().all()


def test_interpolate_satellite_indices_unknown_method_raises(sparse_deciduous_df):
    with pytest.raises(ValueError, match="Unknown interpolation method"):
        interpolate_satellite_indices(
            sparse_deciduous_df,
            vegetation_types=["Deciduous"],
            interpolation_method="does_not_exist",
        )
