
import datetime
import os


path_emi = os.path.join('TNO-test', 'offline', 'test-tno.nc')

output_path = os.path.join('TNO-test', 'offline', 'hourly')
output_name = "Oae_paper_"
prof_path = os.path.join("TNO-test", 'profiles')

start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2019, 1, 1)  # included


var_list = ['CO2']

catlist = [
    ['CO2_A_AREA', 'CO2_A_POINT', 'CO2_F_AREA']
]

tplist  = [
    ['GNFR_A', 'GNFR_A', 'GNFR_F']
]
vplist  = [
    ['GNFR_area_sources', 'GNFR_A', 'GNFR_area_sources']
]
