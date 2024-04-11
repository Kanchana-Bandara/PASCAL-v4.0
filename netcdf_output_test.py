import numpy as np
import pathlib
import os
import netCDF4 as nc

variables
currentsubpopulation = 12
ndevelopmentalstage = 14
ntime = 1460
longitudegrade = np.arange(start = 2.0000, stop = 2.7500, step = 0.083344, dtype = np.float32)
longitudegrade = np.append(longitudegrade, 2.7500)
latitudegrade = np.arange(start = 68.0000, stop = 68.7500, step = 0.083344, dtype = np.float32)
latitudegrade = np.append(latitudegrade, 68.7500)
depthgrade = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])

spatialdistribution_ps = np.repeat(0,  ndevelopmentalstage * longitudegrade.size * latitudegrade.size * depthgrade.size * ntime).astype(np.int32).reshape(ndevelopmentalstage, longitudegrade.size, latitudegrade.size, depthgrade.size, ntime)


outputpath = pathlib.Path("/Users/kanchana/Documents/CURRENTRESEARCH/MIGRATORYCROSSROADS/OUTPUTDATA/")
outputfolder = None


outputfolder = input("TYPE-IN HERE AND PRESS ENTER: ")

os.mkdir(outputpath / outputfolder)

#assembly and writing netcdf files

#file1: space-, time-, and stage-specific population size (datatype = np.int32)
#nb: dimensions: <stage> <longitude> <latitude> <depth> <time>
#datafile creation
ncfilename = "populationsize_" + "sbp_" + str(currentsubpopulation) + ".nc"
outputfile = outputpath / outputfolder / ncfilename
populationsize_ds = nc.Dataset(outputfile, "w", format = "NETCDF4_CLASSIC")

#writing datafile attributes (add as needed)
populationsize_ds.title = "PASCALv4 output datafile: population size"
populationsize_ds.subtitle = "subpopulation ID: " + str(currentsubpopulation)
populationsize_ds.project = "NFR Migratory Crossroads"
populationsize_ds.author = "Kanchana Bandra"
populationsize_ds.nb = "evaluation output - do not use for analyses"

#creating dataset dimensions
stagedim = populationsize_ds.createDimension("devstage", ndevelopmentalstage)
londim = populationsize_ds.createDimension("lon", longitudegrade.size)
latdim = populationsize_ds.createDimension("lat", latitudegrade.size)
depthdim = populationsize_ds.createDimension("depth", depthgrade.size)
timedim = populationsize_ds.createDimension("time", ntime)

#creating dimensionality variabels & data variables
stagevar = populationsize_ds.createVariable("devstage", np.int32, ("devstage", ))
stagevar.units = "dim.less"
stagevar.longname = "developmental stage"

lonvar = populationsize_ds.createVariable("lon", np.float32, ("lon", ))
lonvar.units = "degrees east"
lonvar.longname = "longitude"

latvar = populationsize_ds.createVariable("lat", np.float32, ("lat", ))
latvar.units = "degrees north"
latvar.longname = "latitude"

depthvar = populationsize_ds.createVariable("depth", np.int32, ("depth", ))
depthvar.units = "m"
depthvar.longname = "depth levels"

timevar = populationsize_ds.createVariable("time", np.int32, ("time", ))
timevar.units = "6 h"
timevar.longname = "time of year in 6h intervals"

datavar = populationsize_ds.createVariable("popsize", np.int32, ("devstage", "lon", "lat", "depth", "time", ))
datavar.units = "no. of individuals"
datavar.longname = "estimated stage-, time- and space-specific population size"

stagevar[:] = np.arange(0, 14, 1)
lonvar[:] = longitudegrade
latvar[:] = latitudegrade
depthvar[:] = depthgrade
timevar[:] = np.arange(0, 1460, 1)
datavar[:] = spatialdistribution_ps

