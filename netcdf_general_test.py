import numpy as np
import pathlib
import netCDF4 as nc

dummydata = np.arange(0, 2500, 1).astype(np.int32).reshape(100, 5, 5)
print(dummydata.shape)

print(dummydata[0, :, :])

outputpath = pathlib.Path("/Users/kanchana/Documents/CURRENTRESEARCH/MIGRATORYCROSSROADS/OUTPUTDATA/test")
outputfile = outputpath / "test.nc"
print(outputpath.exists())

test_ds = nc.Dataset(outputfile, "w", format = "NETCDF4_CLASSIC")
test_ds.title = "test data for PASCAL v4"
test_ds.subtitle = "test 1"
test_ds.author = "Kanchana Bandara"
test_ds.administrator = "Akvaplan niva AS"

print(test_ds)

timedim = test_ds.createDimension("time", None)
latdim = test_ds.createDimension("lat", 5)
londim = test_ds.createDimension("lon", 5)

timevar = test_ds.createVariable("time", np.float32, ("time", ))
timevar.units = "time in 6hrs"
timevar.longname = "time in 6 hr units from 1 January to 31 December in an arbitary calendar year"

latvar = test_ds.createVariable("latdim", np.float32, ("lat", ))
latvar.units = "decimal degrees North"
latvar.longname = "latitude"

lonvar = test_ds.createVariable("londim", np.float32, ("lon", ))
lonvar.units = "decimal degrees East"
lonvar.longname = "longitude"

datavar = test_ds.createVariable("value", np.int32, ("time", "lat", "lon", ))
datavar.units = "ind/m3"
datavar.standard_name = "population density"

timevar[:] = np.arange(0, 100, 1)
latvar[:] = np.arange(0, 5, 1)
lonvar[:] = np.arange(0, 5, 1)
datavar[:] = dummydata

print(datavar.shape)







