from math import e
from individual import SuperIndividual
from data_logger import OutputLogger
from pascal_drift import PascalDrift
from utils import dotdict, flatten_list, flatten_dict, points_within_distance

import datetime as dt
import numpy as np
import pandas as pd
import termcolor
from time import gmtime, strftime

DEFAULT_DEPTHRANGE = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30,
                               35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223,
                               267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])

class PascalSimulation(object):
    def __init__(
        self, nsupindividuals, nvindividualspersupindividual,
        global_settings, reader, timestep, start_date, duration,
        seeding_rate, timestep_isplit=1, diapause_depth=500, outputgrid=None,
        debug=None, opendriftoutfile=None, verbose=False, start_locations=None,
        tracker_config=None, headless=None
    ):
        print("")
        termcolor.cprint(
            text="Pan-Arctic Behavioural and Life-history Simulator for Calanus, "
                 "PASCAL version 4.20",
            color="cyan"
        )
        termcolor.cprint(
            text="Kanchana Bandara et al. | NFR Migratory Crossroads 2024-2027",
            color="cyan"
        )
        termcolor.cprint(
            text="Evaluation execution for functionality testing and debugging",
            color="cyan"
        )
        termcolor.cprint(
            text="_" * 84,
            color="light_blue"
        )
        if headless is None:
            print("")
            termcolor.cprint(
                text="enter a unique identifier for the execution (e.g., pascalv42_r001):",
                color="light_red"
            )
            self.outputfolder = input("TYPE ID HERE AND PRESS ENTER: ")
            termcolor.cprint(text="_" * 84, color="light_blue")
            print("")
        else:
            print("")
            self.outputfolder = headless
            termcolor.cprint(
                text=f"Run output folder {headless}",
                color="light_red"
            )
            termcolor.cprint(text="_" * 84, color="light_blue")
            print("")


        execstarttime_prt = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        termcolor.cprint(
            text=f"\nexecution started at: {execstarttime_prt} GMT",
            color="light_blue"
        )
        termcolor.cprint(text="_" * 84, color="light_blue")

        self.nsup = nsupindividuals
        self.supindividuals = [None for j in np.arange(0, nsupindividuals)]
        self.ni_per_sup = nvindividualspersupindividual
        self.global_settings = global_settings
        if 'depthrange' not in global_settings.keys():
            self.global_settings['depthrange'] = DEFAULT_DEPTHRANGE
        self.diapause_depth = diapause_depth

        self.start_time = start_date
        # Guess this doesn't really deal with leap years...
        self.end_time = start_date + dt.timedelta(days=365 * duration)
        self.current_time = start_date
        self.timestep = timestep
        self.isplit = timestep_isplit

        total_tsteps = int(np.ceil((self.end_time - self.start_time) / self.timestep))
        # Ensure total_tsteps is a multiple of self.isplit; if not, reduce it
        rem = total_tsteps % self.isplit
        if rem != 0:
            total_tsteps -= (self.isplit - rem)

        self.all_steps = np.arange(0,total_tsteps,self.isplit)

        self.seeding_rate = seeding_rate

        # Setup the environment, logger, and the super individuals
        if start_locations is None:
            start_locations = [[0,0]]
        elif isinstance(start_locations, np.ndarray):
            start_locations = [tuple(loc) for loc in start_locations]

        self.start_locations = start_locations
        self.opendriftout = opendriftoutfile
        self.tracker_config = tracker_config
        self.prep_environment(reader)

        self.prep_outputgrid(outputgrid)
        self.datalogger = OutputLogger(
            self.outputfolder, len(self.all_steps), self.outputgrid
        )

        self.debug = debug
        if self.debug is not None:
            self.debug_output = {}
            for this_var in self.debug:
                self.debug_output[this_var] = []

        self.individual_stats = {}
        self.next_unique_id = 0

        self.verbose = verbose

    def run(self):
        termcolor.cprint(text = "[SIMULATION IN PROGRESS]", color = "light_red")

        # Setup initial individuals
        seed_locations_ind = np.random.choice(len(self.start_locations), 
                    size=self.seeding_rate, 
                    replace=True
                )
        seed_locations = [self.start_locations[i] for i in seed_locations_ind]      
        self.seed(self.seeding_rate, seed_locations, genome=None)

        for this_step in self.all_steps:
            self.update_environment()
            for isplit in np.arange(0, self.isplit):
                self.update_lifestage()

            self.log_spatial()
            self.gene_hunt()
            self.clean_dead()
            self.respawn()
            if self.current_time.day == 1 and self.current_time.hour == 0:
                self.report()

            if self.debug is not None:
                self.debug_out()

            # Increment datetime
            self.current_time += self.timestep

        # Tidy up
        self.finish_run()

    def update_lifestage(self):
        for this_individual in self.supindividuals:
            if this_individual is not None:
                this_individual.update_lifestage()

    def log_spatial(self):
        varlist = self.datalogger.spatial_var_list
        data_dict = {}
        for this_var in varlist:
            data_dict[this_var] = []

        c = []
        z = []

        for si in self.active_supindividuals():
            c_add, z_add, d_add = si.get_spatial_log_data(
                self.datalogger.spatial_var_list
            )
            c.append(c_add)
            z.append(z_add)
            for k,v in data_dict.items():
                v.append(d_add[k])

        for k,v in data_dict.items():
            data_dict[k] = np.asarray(v)

        env_indices = self.environment_indices()
        cxyz = np.stack([
            np.asarray(c),
            self.tracker.elements.lon[env_indices],
            self.tracker.elements.lat[env_indices],
            np.asarray(z)
        ]).T

        self.datalogger.log_spatial(cxyz, data_dict)

    def respawn(self):
        nspaces = np.sum(np.asarray(self.supindividuals) == None)
        if nspaces > 0:  # Skip if there ain't no space
            if self.current_time.year == self.start_time.year:
                nseeds = self.seeding_rate
                seed_locations_ind = np.random.choice(len(self.start_locations), 
                    size=nseeds, 
                    replace=True
                )
                seed_locations = [self.start_locations[i] for i in seed_locations_ind]
            else:
                nseeds = 0
            # The blendrn/threshold process is individual based
            inherited_genome = flatten_list([
                [si.get_child_genome() for k in np.arange(0, si.potentialfecundity)]
                for si in self.active_supindividuals()
            ])
            inherited_locations = flatten_list([
                [self.tracker.elements.lon[si.environment_index],
                 self.tracker.elements.lat[si.environment_index]]
                for si in self.active_supindividuals()
                for k in np.arange(0, si.potentialfecundity)
            ])

            nspawns = len(inherited_genome)

            # This writes out the logic from the decision tree, could probably
            # be simplified but might reduce readibility
            if nseeds > 0 and nspawns > 0:
                if nspawns + nseeds <= nspaces:
                    self.seed(nseeds, seed_locations, genome=None)
                    self.seed(nspawns, inherited_locations, genome=inherited_genome)
                    if self.verbose:
                        print(f'__respawn__ Seeding {nseeds} and spawning {nspawns}')
                else:
                    if nspaces > nspawns:
                        self.seed(nspawns, inherited_locations, genome=inherited_genome)
                        if self.verbose:
                            print(f'__respawn__ Spawning {nspawns}')
                    else:
                        adjusted_genome, adjusted_locations = self.fecundity_proportional_selection(
                            nspawns
                        )
                        self.seed(nspaces, adjusted_locations, genome=adjusted_genome)
                        if self.verbose:
                            print(
                                f'__respawn__ Spawning {len(adjusted_genome)} '
                                'through fecundity proportional selection'
                            )

            elif nspawns > 0 and nseeds == 0:
                if nspaces > nspawns:
                    self.seed(nspawns, inherited_locations, genome=inherited_genome)
                else:
                    adjusted_genome, adjusted_locations = self.fecundity_proportional_selection(
                        nspawns
                    )
                    if self.verbose:
                        print(
                            f'__respawn__ Spawning {len(adjusted_genome)} '
                            'through fecundity proportional selection'
                        )
                    self.seed(nspaces, adjusted_locations, genome=adjusted_genome)

            elif nseeds > 0 and nspawns == 0:
                if nspaces > nseeds:
                    # Not sure why we don't just seed all available spaces?
                    self.seed(nseeds, seed_locations, genome=None)
                    if self.verbose:
                        print(f'__respawn__ Seeding {nseeds}')

        # Reset potential fecundity in individuals
        for si in self.active_supindividuals():
            si.potentialfecundity = 0

    def seed(self, nseeds, locations, genome=None):
        # Empty spaces are always shuffled to the end of the array
        # so just start from the first None
        firstNone = np.min(np.where(np.isin(self.supindividuals, None)))

        if genome is None:
            genome = [None for i in np.arange(0,nseeds)]

        for i in np.arange(0, nseeds):
            environment_index = self.free_env_indices.pop(0)
            self.set_tracker(environment_index, locations[i])

            # Should diapause depth be random?
            self.supindividuals[i + firstNone] = SuperIndividual(
                self.global_settings,
                self.diapause_depth,
                self.tracker.environment,
                self.tracker.environment_profiles,
                environment_index,
                nindividuals=self.ni_per_sup,
                genes=genome[i],
                unique_id=self.next_unique_id
            )
            self.individual_stats[self.next_unique_id] = {
                'start_step': self.current_time
            }
            self.next_unique_id += 1

    def fecundity_proportional_selection(self, nspaces):
        potentialfecundity = [
            si.potentialfecundity for si in self.active_supindividuals()
        ]
        nspawns = np.sum(potentialfecundity)
        realizedfecundity = np.round(
            potentialfecundity / nspawns * nspaces, decimals=0
        ).astype(np.int32)
        diff = nspaces - np.sum(realizedfecundity)

        indices = np.argsort(potentialfecundity)
        if diff > 0:
            for i in range(diff):
                realizedfecundity[indices[-(i + 1)]] += 1
        elif diff < 0:
            for i in range(abs(diff)):
                # Remove fecundity from the bottom up
                realizedfecundity[indices[-(i + 1)]] -= 1

        adjusted_genome = flatten_list([
            [si.get_child_genome() for k in np.arange(0, rf)]
            for si, rf in zip(self.active_supindividuals(), realizedfecundity)
        ])

        adjusted_locations = flatten_list([
            [self.tracker.elements.lon[si.environment_index],
             self.tracker.elements.lat[si.environment_index]]
            for si, rf in zip(self.active_supindividuals(), realizedfecundity)
            for k in np.arange(0, rf)
        ])

        return adjusted_genome, adjusted_locations

    def clean_dead(self):
        # We can use active individuals because Nones should always be
        # at the end of the array
        remove = [
            j for j, si in enumerate(self.active_supindividuals())
            if si.lifestatus == 0
        ]
        for i in remove:
            self.record_lifestats(self.active_supindividuals()[i])

        # Keep track of free slots in the particle tracker
        for i in remove:
            self.free_env_indices.append(self.active_supindividuals()[i].environment_index)

        if len(remove) > 0:
            [self.supindividuals.pop(i - j) for j, i in enumerate(remove)]
            self.supindividuals = flatten_list([
                self.supindividuals, [None for i in remove]
            ])
            if self.verbose:
                print(
                    f'__clean_dead__ removed {len(remove)} - {remove} si, '
                    f'len array {len(self.supindividuals)}'
                )

    def finish_run(self):
        if self.debug is not None:
            np.save(f'{self.outputfolder}/debug_output.npy', self.debug_output)
        self.write_lifestats()
        self.datalogger.write_spatial()

    def report(self):
        progress = f'{self.progress():.0f}'
        month = self.current_time.strftime('%b')[0].capitalize()
        year = self.current_time.year
        pop_size = self.population_size()
        n_si = len(self.active_supindividuals())
        termcolor.cprint(
            text=f"[PROG:{progress:>8}%] [MO: {month}] [YR: {year}] "
                 f"[ESTIMATED POPULATION SIZE: {pop_size}] No si = {n_si}"
        )
        if self.current_time.month == 12 and self.current_time.hour == 0:
            print("")

    def record_lifestats(self, individ):
        self.individual_stats[individ.unique_id].update({
            'end_age': individ.age,
            'end_individuals': individ.nvindividuals,
            'sex': individ.sex,
            'end_step': self.current_time,
            'total_fecundity': individ.totalfecundity,
            'end_stage': individ.developmentalstage,
            'genes': individ.genome,
            'end_structmass': individ.structuralmass,
            'end_cmm': individ.get_currentcmm(),
            'end_diapause_state': individ.diapausestate,
            'end_diapause_strategy': individ.diapausestrategy
        })

    def write_lifestats(self):
        df = pd.DataFrame(flatten_dict(self.individual_stats))
        death_cause = np.zeros(len(df))
        fecundity_mask = (
            df['total_fecundity'] >= self.global_settings['fecundityceiling']
        )
        death_cause[fecundity_mask] = 3
        age_mask = df['end_age'] >= self.global_settings['ageceiling']
        death_cause[age_mask] = 2
        threshold_mask = (
            df['end_individuals'] <=
            self.global_settings['virtualindividualthrehold']
        )
        death_cause[threshold_mask] = 1
        df['death_cause'] = death_cause
        df.to_csv(f'{self.outputfolder}/lifestats.csv')

    def progress(self):
        return (self.time_ind/len(self.all_steps))*100

    def population_size(self):
        return np.sum([si.nvindividuals for si in self.active_supindividuals()])

    def active_supindividuals(self):
        mask = ~np.isin(self.supindividuals, None)
        return np.asarray(self.supindividuals)[mask]

    def environment_indices(self):
        return [si.environment_index for si in self.active_supindividuals()]

    def debug_out(self):
        if self.supindividuals[0] is not None:
            for this_var in self.debug:
                if this_var.split('_')[0] == 'env':
                    var_name = this_var.split('_')[1]
                    self.debug_output[this_var].append(
                        self.supindividuals[0].get_zi(var_name)
                    )
                else:
                    self.debug_output[this_var].append(
                        getattr(self.supindividuals[0], this_var)
                    )

class Pascal1D(PascalSimulation):

    def prep_environment(self, reader):
        self.free_env_indices = list(np.zeros(self.nsup, dtype=int))
        self.all_data = reader
        self.time_ind = -1
        self.update_environment()

    def prep_outputgrid(self, outputgrid):
        self.outputgrid = {
            'lon': [0],
            'lat': [0],
            'depth': self.global_settings['depthrange'],
            'time': self.all_steps
        }

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
        self.tracker = dotdict({
            'environment': dotdict(init_dict),
            'environment_profiles': dotdict(init_dict),
            'elements': dotdict(init_dict)
        })
    
    def set_tracker(self, environment_index, location):
        pass

    def gene_hunt(self):
        # In 1-D all the animals are near to each other so all females are
        # considered near to all males, therefore just pick a random male
        noninseminated_females = [
            i for i in self.active_supindividuals()
            if i.sex == 'F' and i.inseminationstate == 0
        ]
        males = [i for i in self.active_supindividuals() if i.sex == 'M']
        
        if len(males) > 0:
            for this_f in noninseminated_females:
                selected_male = np.random.choice(males)
                this_f.malegenome = selected_male.genome
                this_f.inseminationstate = 1
        

class PascalAdvection(PascalSimulation):

    def prep_environment(self, reader):
        self.free_env_indices = list(np.arange(0, self.nsup))
        self.time_ind = 0
        self.tracker = PascalDrift(loglevel=100)
        if isinstance(reader, (list, tuple)):
            for r in reader:
                self.tracker.add_reader(r)
        else:
            self.tracker.add_reader(reader)

        self.tracker.set_config('general:use_auto_landmask', False)
        self.tracker.set_config('general:seafloor_action', 'lift_to_seafloor')

        if self.tracker_config is not None:
            for k,v in self.tracker_config.items():
                self.tracker.set_config(k, v)
       
        start_loc = self.start_locations[0]
        self.tracker.seed_elements(
                lon=start_loc[0], lat=start_loc[1], z=-10, number=self.nsup, radius=10,
                time=self.start_time - self.timestep
            )
        self.tracker.run_prep(
            time_step=self.timestep.seconds*self.isplit,
            steps=None,
            time_step_output=None,
            duration=None,
            end_time=self.end_time + dt.timedelta(days=365),
            stop_on_error=True,
            outfile=self.opendriftout,
            export_variables=['x', 'y', 'temperature']
        )
        # Need this to populate the environment
        self.tracker.run_1step()

    def prep_outputgrid(self, outputgrid):
        outputgrid['time'] = self.all_steps
        outputgrid['depth'] = self.global_settings['depthrange']
        self.outputgrid = outputgrid

    def set_tracker(self, environment_index, location):
        self.tracker.elements.lon[environment_index] = location[0]
        self.tracker.elements.lat[environment_index] = location[1]

    def update_environment(self):
        self.time_ind+= 1
        self.tracker.run_1step()

    def gene_hunt(self):
        # Get genes from nearby males
        noninseminated_females = [
            i for i in self.active_supindividuals()
            if i.sex == 'F' and i.inseminationstate == 0
        ]
        
        all_males = [
            i for i in self.active_supindividuals() if i.sex == 'M']
        
        all_males_ll = np.stack([
            self.tracker.elements.lon[
                [j.environment_index for j in all_males]
            ],
            self.tracker.elements.lat[
                [j.environment_index for j in all_males] ]
        ]).T

        for this_f in noninseminated_females:
            this_f_ll = [self.tracker.elements.lon[
                this_f.environment_index],
                self.tracker.elements.lat[this_f.environment_index]]
            near_males = points_within_distance(
                this_f_ll, all_males_ll,
                self.global_settings['maxmatingdistance'])
            if np.sum(near_males) > 0:
                selected_male = np.random.choice(all_males[near_males])
                this_f.malegenome = selected_male.genome
                this_f.inseminationstate = 1

    def finish_run(self):
        self.tracker.run_end()
    
        super(PascalAdvection, self).finish_run()



