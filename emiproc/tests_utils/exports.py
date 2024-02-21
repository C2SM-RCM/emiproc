from emiproc.exports.netcdf import nc_cf_attributes

test_nc_metadata = nc_cf_attributes(
    author="Emiproc Test",
    contact="name@domain.com",
    title="Hourly Emissions for ...",
    source="emiproc tests",
    institution="emiproc, tests",
)
