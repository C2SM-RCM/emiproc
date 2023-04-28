import pytest 
from emiproc.inventories import  EmissionInfo
from emiproc.tests_utils.test_inventories import inv




def test_emission_info():

    info = EmissionInfo()

    info2 = EmissionInfo(height=1.0, height_over_buildings=False)

def test_inventory_with_emission_info():

    info = EmissionInfo()

    def set_missing_info():
        # 2 Catergories are missing
        inv.emission_infos = {'adf': info}

    def set_correct_info():
        inv.emission_infos = {'adf': info, 'liku': info, 'test': info }
        
        assert inv.emission_infos['adf'].height == info.height

    # If a category is missing, an error should be raised
    pytest.raises(ValueError, set_missing_info )

    set_correct_info()


   

