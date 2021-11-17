import subprocess
import netCDF4 as nc
import amrs.nc

def test_performance_at_rlon():
    # Execute processing case which has rlon==0. Hard-coded (in shell) to save effort of sending in all correct arguments otherwise...
    out_filename = "outgrid.nc"
    subprocess.run(["python -m ../emiproc grid --case-file tno_test.py --output_path output-path . --output-name {}".format(out_filename)])
    
    # Reference file in tests/ folder
    out_filename_ref = "tno_ref.nc"

    with nc.Dataset(out_filename) as own, nc.Dataset(out_filename_ref) as ref:
        assert amrs.nc.compare.datasets_equal(own, ref, [])
