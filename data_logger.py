import numpy as np
import netCDF4 as nc
import pandas as pd
import os
import termcolor

DEFAULT_SAVE = {'population_size':{"units":"no. of individuals", "longname":"estimated stage-, time- and space-specific population size of Calanus finmarchicus"},
                'biomass':{"units":"gC", "longname":"estimated stage-, time- and space-specific biomass of Calanus finmarchicus"}}

class OutputLogger(object):

    def __init__(self, outputfolder, total_timesteps, output_grid, devstages = 13, lifestrategies_size=[5,8], no_evolvable=8, save_spatial=DEFAULT_SAVE):
        # These are split by developmental stages (up to 13 with 12 (13) being female (male) at stage zz)
        self.currentsubpopulation = 'all' # At the moment not logged by subpopulation
        self.total_mass = [] 
        self.spatial_atts = save_spatial
        self.spatial_output = {}
        for save_var in save_spatial.keys():
            self.spatial_output[save_var] = []
        self.devstages = devstages
        self.lifestrategies_size = lifestrategies_size
        self.total_timesteps = total_timesteps
        self.no_evolvable = no_evolvable
        self.output_grid = output_grid
        self.prep_grid()

        # These will have columns for each of the different life strategies and a row for each timestep
        self.lifecycle_i = np.zeros([self.total_timesteps, self.lifestrategies_size[0]])
        self.lifecycle_f = np.zeros([self.total_timesteps, self.lifestrategies_size[1]])

        # Log the changing genomes
        self.genome_log = np.zeros([self.total_timesteps, self.no_evolvable, 2])

        # Setup the output
        self.outputfolder = outputfolder
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)

        self.outputfile = f'./{outputfolder}/output_ps.nc'

    def prep_grid(self):
        self.min_lon = np.min(self.output_grid['lon'])
        self.max_lon = np.max(self.output_grid['lon'])
        self.min_lat = np.min(self.output_grid['lat'])
        self.max_lat = np.max(self.output_grid['lat'])
        if self.min_lon == self.max_lon:
            self.lon_res = 1
        else:
            self.lon_res = self.output_grid['lon'][1] - self.output_grid['lon'][0]

        if self.min_lat == self.max_lat:
            self.lat_res = 1
        else:
            self.lat_res = self.output_grid['lat'][1] - self.output_grid['lat'][0]

    # This doesn't feel very pythonesque
    def add_lifecycle_i(self, val, col, timestep):
        self.lifecycle_i[timestep, col] += val

    def add_lifecyle_f(self, val, col, timestep):
        self.lifecycle_f[timestep, col] += val

    def log_spatial(self, cxyz, data_dict):
        # This is done at the coupler level as it varies based on the forcing (1-D or spatially resolved)
        # TODO - write at each timestep rather than dumping to a dict
        resolved_dict = self.resolve_spatial(cxyz, data_dict)
        for varname, data in self.spatial_output.items():
           data.append(resolved_dict[varname])

    def log_evolvable(self, genomes, timestep):
        self.genome_log[timestep, :, 0] = np.mean(genomes)
        self.genome_log[timestep, :, 1] = np.std(genomes)

    def resolve_spatial(self, cxyz, data_dict):
        gridded_data = {}
        for var in data_dict.keys():
            gridded_data[var] = np.zeros([self.devstages, len(self.output_grid['lon']), len(self.output_grid['lat']), len(self.output_grid['depth'])])
        
        cxyz[cxyz[:,1] > self.max_lon,1] = self.max_lon
        cxyz[cxyz[:,1] < self.min_lon,1] = self.min_lon
        cxyz[cxyz[:,2] > self.max_lat,2] = self.max_lat
        cxyz[cxyz[:,2] < self.min_lat,2] = self.min_lat

        for d in np.arange(0, self.devstages):
            if np.any(cxyz[:,0] == d):
                # Pretty sure there is a more efficient version of this but use for now
                for i, this_cxyz in enumerate(cxyz):
                    lon_ind = int(np.floor((this_cxyz[1] - self.min_lon)/ self.lon_res))
                    lat_ind = int(np.floor((this_cxyz[2] - self.min_lat)/ self.lat_res))
                    
                    for var, data in data_dict.items():
                        gridded_data[var][int(this_cxyz[0]), lon_ind, lat_ind, int(this_cxyz[3])] += data[i]

        return gridded_data

    def write_spatial(self):
        #file1: space-, time-, and tage-specific population size (datatype = np.int32)
        #-----------------------------------------------------------------------------
        #nb: dimensions: <stage> <longitude> <latitude> <depth> <time>
        #datafile creation
        outputfile = self.outputfile
        populationsize_ds = nc.Dataset(outputfile, "w", format = "NETCDF4_CLASSIC")

        #writing datafile attributes (add as needed)
        populationsize_ds.title = "PASCALv4 output datafile: population size"
        populationsize_ds.subtitle = f"subpopulation ID: {self.currentsubpopulation}" 
        populationsize_ds.project = "NFR Migratory Crossroads"
        populationsize_ds.author = "Kanchana Bandra"
        populationsize_ds.warning = "evaluation output - do not use for analyses"

        #creating dataset dimensions
        stagedim = populationsize_ds.createDimension("devstage", self.devstages)
        londim = populationsize_ds.createDimension("lon", len(self.output_grid['lon']))
        latdim = populationsize_ds.createDimension("lat", len(self.output_grid['lat']))
        depthdim = populationsize_ds.createDimension("depth", len(self.output_grid['depth']))
        timedim = populationsize_ds.createDimension("time", self.total_timesteps)

        #creating dimensionality variabels & data variables
        stagevar = populationsize_ds.createVariable("devstage", np.int32, ("devstage", ))
        stagevar.units = "dim.less"
        stagevar.longname = "developmental stage"

        lonvar = populationsize_ds.createVariable("lon", np.float32, ("lon", ))
        lonvar.units = "degrees east"
        lonvar.longname = "longitude"
        lonvar[:] = self.output_grid['lon']

        latvar = populationsize_ds.createVariable("lat", np.float32, ("lat", ))
        latvar.units = "degrees north"
        latvar.longname = "latitude"
        latvar[:] = self.output_grid['lat']

        depthvar = populationsize_ds.createVariable("depth", np.int32, ("depth", ))
        depthvar.units = "m"
        depthvar.longname = "depth levels"
        depthvar[:] = self.output_grid['depth']

        timevar = populationsize_ds.createVariable("time", np.int32, ("time", ))
        timevar.units = "6 h"
        timevar.longname = "time of year in 6h intervals"
        timevar[:] = self.output_grid['time']

        for var, atts in self.spatial_atts.items():
            populationsize_ds = self._write_4d_var(populationsize_ds, var, self.spatial_output[var], attributes = atts)
        
        populationsize_ds.close()


    def _write_4d_var(self, ds, varname, data, attributes={}, dtype=np.int32):
        dv1 = ds.createVariable(varname, dtype, ("time", "devstage", "lon", "lat", "depth",))
        for att_name, att_val in attributes.items():
            setattr(dv1,att_name,att_val)
        dv1[:] = self.pad_data(np.asarray(data))
        return ds

    def write_lifestrategies(self):
        lcstrategies = np.hstack(tup = (self.lifecycle_i, self.lifecycle_f), dtype = np.float32)
        lcstrategies_pd = pd.DataFrame(data = lcstrategies,
                                    columns = ["nddev", "nden_c4", "nden_c5", "ndex_c4", "ndex_c5", "strm_ddev", "stom_ddev", "strm_den_c4", "stom_den_c4", "strm_den_c5", "stom_den_c5", "stom_dex_civ", "stom_dex_cv"])

        #auto-generated path and filename
        txtfilename = "lifestrategies_" + "sbp_" + str(self.currentsubpopulation) + ".csv"
        outputfile = f"{self.outputfolder}/{txtfilename}"

        #writing csv
        lcstrategies_pd.to_csv(path_or_buf = outputfile, index = True, header = True)

        #file-write status print
        termcolor.cprint(text = "[FILE WRITING COMPLETED]", color = "light_red")


    def pad_data(self, data):
        target_shape =  (self.total_timesteps, self.devstages, len(self.output_grid['lon']), len(self.output_grid['lat']),
                            len(self.output_grid['depth']))
        if data.shape != target_shape:
            padded = np.full(target_shape, np.nan, dtype=data.dtype)
            padded[:data.shape[0], ...] = data
            return padded
        else:
            return data
