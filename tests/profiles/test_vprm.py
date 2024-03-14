"""Test for the vprm module.

Tests are there moslty to check that things run problerly and that the inputs/
outputs expected are there. The actual values are not checked.
"""
import pytest
import pandas as pd

import numpy as np
from emiproc.profiles.vprm import calculate_vprm_emissions, calculate_vegetation_indices


@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        ('T', 'global'): [25, 30, 28, 27],
        ('T', 'urban'): [28, 32, 30, 29],
        ('vegetation_type_1', 'lswi'): [0.5, 0.6, 0.7, 0.8],
        ('vegetation_type_1', 'evi'): [0.2, 0.3, 0.4, 0.5],
        ('vegetation_type_2', 'lswi'): [0.3, 0.4, 0.5, 0.6],
        ('vegetation_type_2', 'evi'): [0.1, 0.2, 0.3, 0.4],
        ('RAD', ''): [100, 200, 150, 180]
    })
    
    df_vprm = pd.DataFrame({
        'alpha': [0.1, 0.2],
        'beta': [0.5, 0.6],
        'lambda': [0.8, 0.9],
        'Tmin': [20, 22],
        'Topt': [25, 27],
        'Tmax': [30, 32],
        'Tlow': [18, 20],
        'PAR0': [50, 60],
        'is_urban': [False, True]
    }, index=['vegetation_type_1', 'vegetation_type_2'])
    
    return df, df_vprm

def test_calculate_vprm_emissions(sample_data):
    df, df_vprm = sample_data
    result = calculate_vprm_emissions(df, df_vprm)
    
    assert ('PAR', '') in result.columns
    assert ('vegetation_type_1', 'Tscale') in result.columns
    assert ('vegetation_type_2', 'Tscale') in result.columns
    assert ('vegetation_type_1', 'Wscale') in result.columns
    assert ('vegetation_type_2', 'Wscale') in result.columns
    assert ('vegetation_type_1', 'Pscale') in result.columns
    assert ('vegetation_type_2', 'Pscale') in result.columns
    assert ('vegetation_type_1', 'resp') in result.columns
    assert ('vegetation_type_2', 'resp') in result.columns
    assert ('vegetation_type_1', 'gee') in result.columns
    assert ('vegetation_type_2', 'gee') in result.columns
    assert ('vegetation_type_1', 'nee') in result.columns
    assert ('vegetation_type_2', 'nee') in result.columns




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
