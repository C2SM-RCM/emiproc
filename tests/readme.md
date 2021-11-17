The current test is a bit clunky.

What you need to do is:
* do `python -m pip install .` for `emiproc` to be accessible from anywhere
* always clear the folder `tests/outputs/online`
* it also requires the `amrs` package to be installed, to compare the created netcdf files

Then, the testcase can be run simply by browsing to this `tests/` folder, and running `python performance_at_rlon.py`. It already contains the 'correct' reference output, `tno_ref.nc`, which contains no stripe at `rlon=0`.
