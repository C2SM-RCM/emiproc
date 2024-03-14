import pytest
from emiproc.utilities import total_emissions_almost_equal

def test_total_emissions_almost_equal():
    # Create a reference dictionary
    ref_dict = {
        'sub1': {'cat1': 10.0, 'cat2': 20.0},
        'sub2': {'cat1': 30.0, 'cat2': 40.0}
    }

    # Test case 1: Total emissions are exactly equal
    assert total_emissions_almost_equal(ref_dict, ref_dict) == True

    # Test case 2: Total emissions are almost equal within the default tolerance
    total_dict_2 = {
        'sub1': {'cat1': 10.00001, 'cat2': 19.99999},
        'sub2': {'cat1': 29.99999, 'cat2': 40.00001}
    }
    assert total_emissions_almost_equal(ref_dict, total_dict_2) == True

    # Test case 3: Total emissions are not almost equal within the default tolerance
    total_dict_3 = {
        'sub1': {'cat1': 10.0001, 'cat2': 19.99},
        'sub2': {'cat1': 29.9999, 'cat2': 40.01}
    }
    assert (total_emissions_almost_equal(ref_dict, total_dict_3) == False)

    # Test case 4: Total emissions have different subcategories
    total_dict_4 = {
        'sub1': {'cat1': 10.0, 'cat2': 20.0},
        'sub3': {'cat1': 30.0, 'cat2': 40.0}
    }
    pytest.raises(ValueError, total_emissions_almost_equal, ref_dict, total_dict_4)

    # Test case 5: Total emissions have different categories
    total_dict_5 = {
        'sub1': {'cat1': 10.0, 'cat3': 20.0},
        'sub2': {'cat1': 30.0, 'cat2': 40.0}
    }
    pytest.raises(ValueError, total_emissions_almost_equal, ref_dict, total_dict_5)
