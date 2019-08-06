
import datetime
import os

from epro.hourly_emissions import speciation as spec

path_emi = 'all_outputs/final/offline/All_emissions.nc'

output_path = 'all_outputs/final/offline/hourly/'
output_name = "emis_"
prof_path = 'all_outputs/final/profiles/'

start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2019, 1, 8)  # included


var_list = ['CO2_ALL_E']

contribution_list = None

catlist = [
    [
        'CO2_A_TNO',
        'CO2_A_TNO',
        'CO2_B_TNO',
        'CO2_B_TNO',
        'CO2_C_TNO',
        'CO2_F1_TNO',
        'CO2_F2_TNO',
        'CO2_F3_TNO',
        'CO2_D_TNO',
        'CO2_D_TNO',
        'CO2_E_TNO',
        'CO2_G_TNO',
        'CO2_H_TNO',
        'CO2_H_TNO',
        'CO2_I_TNO',
        'CO2_J_TNO',
        'CO2_J_TNO',
        'CO2_K_TNO',
        'CO2_L_TNO',
        'CO2_B_Carbocount',
        'CO2_C_Carbocount',
        'CO2_J_Carbocount',
        'CO2_L_Carbocount',
        'CO2_F_Carbocount',
    ],  
]

tplist = [
    [
        'GNFR_A_TNO',
        'GNFR_B_TNO',
        'GNFR_C_TNO',
        'GNFR_F_TNO',
        'GNFR_F_TNO',
        'GNFR_F_TNO',
        'GNFR_D_TNO',
        'GNFR_E_TNO',
        'GNFR_G_TNO',
        'GNFR_H_TNO',
        'GNFR_I_TNO',
        'GNFR_J_TNO',
        'GNFR_K_TNO',
        'GNFR_L_TNO',
        'GNFR_B_Carbocount',
        'GNFR_C_Carbocount',
        'GNFR_J_Carbocount',
        'GNFR_L_Carbocount',
        'GNFR_F_Carbocount',
    ],  
]

""" All emissions are applied on the floor in fact"""
vplist = [['GNFR_area_sources']]
vplist[0] *= len(tplist[0])

