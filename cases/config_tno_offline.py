
import datetime
import os


path_emi = (
    "../testdata/oae_paper/offline/All_emissions.nc"
) 

output_path = "./output_example/"
output_name = "Oae_paper_"
prof_path = "../profiles/example_output/"

start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2019, 1, 9)  # included
