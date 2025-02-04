#from pascal_drift import PascalDrift
from individual import SuperIndividual
from functools import reduce
from data_logger import OutputLogger
from pascal_drift import PascalDrift


import datetime as dt
import numpy as np
import termcolor
import sys
from time import sleep
from time import gmtime, strftime

class dotdict(dict): # Move to utils
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def flatten_list(xss): # Move to utils
    return [x for xs in xss for x in xs]

class PascalSimulation(object):
    def __init__(self, nsupindividuals, nvindividualspersupindividual, global_settings, reader, timestep, start_date, duration, seeding_rate, diapause_depth = 500, outputgrid=None, debug=None, opendriftoutfile=None):
        print("")
        termcolor.cprint(text = "Pan-Arctic Behavioural and Life-history Simulator for Calanus, PASCAL version 4.00", color = "cyan")
        termcolor.cprint(text = "Kanchana Bandara et al. | NFR Migratory Crossroads 2024-2027", color = "cyan")
        termcolor.cprint(text = "Evaluation execution for functionality testing and debugging", color = "cyan")
        termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")
        print("")
        termcolor.cprint(text = "enter a unique identifier for the execution (e.g., pascalv4_r001):", color = "light_red")
        self.outputfolder = input("TYPE ID HERE AND PRESS ENTER: ")
        termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")
        print("")
        execstarttime_rec = dt.datetime.now()
        execstarttime_prt = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        termcolor.cprint(text = f"\nexecution started at: {execstarttime_prt} GMT", color = "light_blue")
        termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")

        self.nsup = nsupindividuals
        self.supindividuals = [None for j in np.arange(0, nsupindividuals)]
        self.ni_per_sup = nvindividualspersupindividual
        self.global_settings = global_settings
        if 'depthrange' not in global_settings.keys():
            self.global_settings['depthrange'] = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])

        self.diapause_depth = diapause_depth

        self.start_time = start_date
        self.end_time = start_date + dt.timedelta(days = 365*duration) # Guess this doesn't really deal with leap years...
        self.current_time = start_date
        self.timestep = timestep

        total_tsteps = int((self.end_time - self.start_time)/self.timestep)        
        self.all_steps = np.arange(0,total_tsteps)

        self.seeding_rate = seeding_rate

        # Setup the environment, logger, and the super individuals
        self.prep_environment(reader)
        self.prep_outputgrid(outputgrid)
        self.datalogger = OutputLogger(self.outputfolder, len(self.all_steps), self.outputgrid)

        self.opendriftout = opendriftoutfile
        self.debug = debug
        if self.debug is not None:
            self.debug_output = {}
            for this_var in self.debug:
                self.debug_output[this_var] = []

    def run(self):
        termcolor.cprint(text = "[SIMULATION IN PROGRESS]", color = "light_red")

        # Setup initial individuals
        self.seed(self.seeding_rate)

        try:
            for this_step in self.all_steps:
                self.update_environment()
                self.update_lifestage()
                # Maybe some of these happen at a slower timestep
                self.log_spatial()
                self.gene_hunt()
                self.clean_dead() # Need to add to log file, reorder superindividual dictionary and sort environment index(?) 
                self.respawn()
                if self.current_time.day == 1 and self.current_time.hour == 0 and self.current_time.minute == 0:
                    self.report()

                if self.debug is not None:
                    self.debug_out()

                # Increment datetime
                self.current_time += self.timestep
        except:
            print('Error!')

        # Tidy up
        self.finish_run()


    def update_lifestage(self):
        for this_individual in self.supindividuals:
            if this_individual is not None:
                this_individual.update_lifestage()

    def log_spatial(self):
        spatial_data = np.asarray([si.get_spatial_log_data() for si in self.active_supindividuals()]) #[col, self.zidx, self.nvindividuals, self.structuralmass + self.reservemass] 
        cxyz = np.stack([spatial_data[:,0], self.tracker.elements.lon[self.environment_indices()], self.tracker.elements.lat[self.environment_indices()], spatial_data[:,1]]).T
        self.datalogger.log_spatial(cxyz, spatial_data[:,2], spatial_data[:,3])

    def respawn(self):
        nspaces = np.sum(np.asarray(self.supindividuals) == None)
        if nspaces == 0:
            # If nspaces is 0 realised and potential fecundity should go to 0
            for si in self.supindividuals:
                if si is not None:
                    si.potentialfecundity = 0
                    si.realisedfecundity = 0
        else:
            nseeds = self.seeding_rate if self.current_time.year == self.start_time.year else 0 # Does this need to cover the first year in duration in case a run doesn't start Jan 1st?
            #nspawns = reduce(lambda x,y : x + y, [si.potentialfecundity for si in np.asarray(self.supindividuals)[self.active]]) # Not sure how slow the list comprehension below is; could do this first and only get genome if nspawns > 0
            
            inherited_genome = flatten_list([[si.get_child_genome() for k in np.arange(0,si.potentialfecundity)] for si in self.active_supindividuals()])
            nspawns = len(inherited_genome)

            if nspawns + nseeds <= nspaces:
                self.seed(nseeds, genome=None)
                self.seed(nspawns, genome=inherited_genome)

            else: #nseeds is now implicitly 0 for this iteration but should it be zeroed for the rest of this year?

                if nspaces > nspawns:
                    self.seed(nspawns, genome=inherited_genome)
                else:
                    # not enough spaces so females compete for egg-placement via a fecundity-proportional selection process

                    #competition for egg placement in the new generation - spawning proceeds with constraints
                    #fecundity-proportional selection (FPS)
                    #this writes the FPS output into the realized fecundity state variable
                    #nb:the np.floor() is taken instead of np.round() beacuse the latter bares the risk of the nspawns (i.e., sum(realizedfecundity)) becoming higher than the available empty spaces ('nspaces')
        
                    # Need to fix
                    #realizedfecundity[:, currentsubpopulation] = np.floor((potentialfecundity[:, currentsubpopulation] / nspawns) * nspaces).astype(np.int32)
                    #realized_fecundity = 

                    adjusted_nspawn = nspaces
                    self.seed(adjusted_nspawn, genome=inherited_genome)


    def seed(self, nseeds, environment_indices=None, genome=None):
        if nseeds > 0:
            # Empty spaces are always shuffled to the end of the array so just start from the first None
            firstNone = np.min(np.where(np.isin(self.supindividuals, None)))
           
            if environment_indices is None:
                environment_indices = np.zeros(nseeds, dtype=int)
            if genome is None:
                genome = [None for i in np.arange(0,nseeds)]
        
            for i in np.arange(0, nseeds):
                self.supindividuals[i + firstNone] = SuperIndividual(self.global_settings, self.diapause_depth, self.tracker.environment, self.tracker.environment_profiles, environment_indices[i], nindividuals=self.ni_per_sup, genes=genome[i]) # Should diapause depth be random?


    def clean_dead(self):
        remove = [j for j, si in enumerate(self.active_supindividuals()) if si.lifestatus==0] # We can use active inidivuals because Nones should always be at the end of the array
        if len(remove) > 0:
            [self.supindividuals.pop(i) for i in remove]
            self.supindividuals = flatten_list([self.supindividuals, [None for i in remove]])

    def finish_run(self):
        if self.debug is not None:
            np.save('debug_output.npy', self.debug_output)
        self.datalogger.write_lifestrategies()
        self.datalogger.write_spatial()

    def report(self):
        termcolor.cprint(text = f"[PROG:{f'{self.progress():.0f}':>8}%] [MO: {self.current_time.strftime('%b')[0].capitalize()}] [YR: {self.current_time.year}] [ESTIMATED POPULATION SIZE: {self.population_size()}] No si = {len(self.active_supindividuals())}")
        if self.current_time.month == 12:
            print("")
        

    def progress(self):
        return (self.time_ind/len(self.all_steps))*100

    def population_size(self):
        return np.sum([si.nvindividuals for si in self.active_supindividuals()])

    def active_supindividuals(self):
        return np.asarray(self.supindividuals)[~np.isin(self.supindividuals, None)]

    def environment_indices(self):
        return [si.environment_index for si in self.active_supindividuals()]

    def debug_out(self):
        for this_var in self.debug:
            self.debug_output[this_var].append(getattr(self.supindividuals[0], this_var))
            
        

class Pascal1D(PascalSimulation):
    def prep_environment(self, reader):
        self.all_data = reader
        self.time_ind = -1
        self.update_environment()

    def prep_outputgrid(self, outputgrid):
        self.outputgrid = {'lon':[0], 'lat':[0], 'depth':self.global_settings['depthrange'], 'time':self.all_steps}

    def update_environment(self):
        self.time_ind+= 1
        init_dict = {}
        for k,v in self.all_data.items():
            if k == 'z':
                init_dict['z'] = v
            else:
                init_dict[k] = v[self.time_ind,...]
        init_dict['lon'] = np.asarray([0])
        init_dict['lat'] = np.asarray([0])
        self.tracker = dotdict({'environment':dotdict(init_dict), 'environment_profiles':dotdict(init_dict), 'elements':dotdict(init_dict)})
        

    def gene_hunt(self):
        # In 1-D all the animals are near to each other so all females are considered near to all males, therefore just pick a random male
        noninseminated_females = [i for i in self.active_supindividuals() if i.sex == 'F' and i.inseminationstate == 0]
        males = [i for i in self.active_supindividuals() if i.sex == 'M']
        
        if len(males) > 0:
            for this_f in noninseminated_females:
                selected_male = random.choice(males)
                this_f.malegenome = selected_male.genome
        

class PascalAdvection(PascalSimulation):

    def prep_environment(self, reader):
        self.time_ind = 0
        self.tracker = PascalDrift(loglevel=100)
        self.tracker.add_reader(reader)
        self.tracker.set_config('general:use_auto_landmask', False)
        self.tracker.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Sundby1983')
        self.tracker.seed_elements(lon=3, lat=60.5, z=-10, number=self.nsup, radius=30000, time=self.start_time - self.timestep) # Need to change start times to fit sequential seeding in original pascal
        self.tracker.run_prep(time_step=self.timestep.seconds,
                steps=None,
                time_step_output=None,
                duration=None,
                end_time=self.end_time,
                stop_on_error=True)
        self.tracker.run_1step() # Need this to populate the environment

    def prep_outputgrid(self, outputgrid):
        outputgrid['time'] = self.all_steps
        outputgrid['depth'] = self.global_settings['depthrange']
        self.outputgrid = outputgrid

    def update_environment(self):
        self.time_ind+= 1
        self.tracker.run_1step()

    def gene_hunt(self):
        #!!!!!!! Temporary code !!!!!!!!!
        # This just applies from any male, but should do a search for a max distance first. This should be fairly straightforward with
        # self.tracer.elements['x'] (/y) and then a self.max_dist
        noninseminated_females = [i for i in self.active_supindividuals() if i.sex == 'F' and i.inseminationstate == 0]
        males = [i for i in self.active_supindividuals() if i.sex == 'M']

        if len(males) > 0:
            for this_f in noninseminated_females:
                selected_male = random.choice(males)
                this_f.malegenome = selected_male.genome

    def finish_run(self):
        self.tracker.run_end(outfile=self.opendriftout, export_variables=['x', 'y', 'temperature'])
    
        super(PascalAdvection, self).finish_run()



