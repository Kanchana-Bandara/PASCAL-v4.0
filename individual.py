########################################################################
#pan arctic behavioural and life-history simulator for calanus (pascal)#
########################################################################

#version 4.00 :: python development :: temporary mbpro14 edition :: merge later
#super-individual-based model for simulating behavioural and life-history strategies of the north atlantic copepod, calanus finmarchicus
#evaluation file on linux mint

#modules
#=========================================================================================================================================================================================================================================================
import sys
#import math
import os
import numpy as np
import pandas as pd
import pathlib
import netCDF4 as nc
#from tqdm import tqdm
import termcolor
from time import sleep
from time import gmtime, strftime
from datetime import datetime
import pascalv4_mod_verticalmigration as vm
import pascalv4_mod_growthanddevelopment as gd
import pascalv4_mod_survival as sv
import pascalv4_mod_reproduction as rp

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class SuperIndividual(object):
    def __init__(self, global_settings, diapausedepth, environment, environment_index, eggmass=0.23, nindividuals=10000, genes=None, cxthreshold=0.7, muthreshold=0.2, datalogger=None):
        #these are reflective of individual states and vary during the lifespan of super individuals depending on the individual-environment interactions and internal processes (e.g., hardcoded strategies)
        self.global_settings = global_settings
        self.eggmass = eggmass

        #defines the living (1) or dead (0) state of super individuals
        self.lifestatus = 1
        #defines the no. of virtual individuals contained in a super individual - this no. is defined by the constant, 'nvindividualspersupindividual' above
        self.nvindividuals = nindividuals
        #defines the developmental stage of super individuals:
        #0:Egg, 1:NI, 2:NII, 3:NIII, 4:NIV, 5:NV, 6:NVI, 7:CI, 8:CII, 9:CIII, 10:CIV, 11:CV, 12:CVI-F, 13:CVI-M
        self.developmentalstage = 0
        #defines the mean temperature trajectory encountered during the early lifestages
        #nb:this is obsolete beyond non-feeding stages, whose development is estimated as a function of growth
        self.thermalhistory = 0.00
        #defines the structural body mass of the super individual (min = 0.23 ugC at embryonic stage), initializes with 0.00
        self.structuralmass = eggmass
        #defines the maximum lifetime structural mass of a super individual (used for starvation risk estimation)
        self.maxstructuralmass = eggmass
        #defines the energy reserve mass of the super individual (max = 0.70 x structural mass), initializes with 0.00
        self.reservemass = 0.00
        #defines the age of the super individual
        self.age = 0
        #defines the sex of the super individual (M:male, F:female, U:undefined)
        self.sex = "U"
        #defines the time of diapause entry of the super individual
        self.timeofdiapauseentry = 0
        #defines the time of diapause exit of the super individual
        self.timeofdiapauseexit = 0
        #defines the structural body mass at diapause entry
        self.structuralmassatdiapauseentry = 0.00
        #defines the energy reserve mass at diapause entry
        self.reservemassatdiapauseentry = 0.00
        #defines the developmental stage at diapause entry
        self.developmentalstageatdiapauseentry = 0
        #defines the depth of diapause
        self.diapausedepth = diapausedepth
        #defines the state of diapause of CIV and CV individuals ("A": active, "E": entry, "D":diapause, "X":exit, "P": post, "U": undefined)
        #the "U1" datatype stores one unicode character in each slot (index position)
        self.diapausestate = "A"
        #defines the energy reserve mass at diapause exit
        #defines whether the super individual will enter diapause and then potentially molt to the adult or potentilly develop directly to adulthood without diapause
        self.diapausestrategy = -1
        #nb:no need to track structural mass or developmental stage at diapsue exit as those do not change
        self.adultsize = 0
    
        self.reservemassatdiapauseexit = 0.00
        #defines the insemination state of females (0: not inseminated, 1: inseminated)
        self.inseminationstate = 0
        #defines the energy allocated to reproductive output
        self.reproductiveallocation = 0.00
        #defines the male genome copied to a female after mating (this is a 3D array!)
        self.malegenome = None
        #defines the total no. of eggs produced by a female during its lifespan
        #nb:not all of these eggs are spawned into the super individual pool - only the reward-based ones (see below)
        self.totalfecundity = 0
        #defines the no. of eggs procuced by a female at each timepoint
        self.potentialfecundity = 0

        #evolvable attributes ('genes')
        #______________________________

        #these are attributes whose values freely evolve across time and space as the model is iteratively computed
        #no artificial forcing is applied to optimize the free attribute combination - it is dependent on the 'simulated natural selection' that happens within the model
#all evolvable attributes range from 0 - 1 in floating point designation
        #defines the body size trajectory that a super individual follows during its lifespan
        #nb: pascalv4 does not support p2sensitivity or p2reactivity attributes - these can be included in future developments
        if genes == None:
            genes = np.random.rand(8)

        self.genome = dotdict({'a1_bodysize':genes[0],
                              'a2_irradiancesensitivity':genes[1], #defines the spectral sensitivity of a given super individual
                              'a3_pred1sensitivity':genes[2], #defines the visual predator sensitivity (i.e., the ability of a super individual to percieve a visual predator in its environment)
                              'a4_pred1reactivity':genes[3], #defines the reactivity to visual predators
                              'a5_energyallocation':genes[4], #defines the energy allocation pattern of a given super individual
                              'a6_diapauseprobability':genes[5], #defines the probability of diapause entry of a given super individual (higher: likely to diapause, lower: less likely to diapause and more likely to develop directly to adulthood)
                              'a7_diapauseentry':genes[6], #defines the timing of diapause entry of a given super individual
                              'a8_diapauseexit':genes[7]}) #defines the timing of diapause exit of a given super individual}

        # The thresholds for mutation/crossover of genes during reproduction
        self.cxthreshold = cxthreshold        
        self.muthreshold = muthreshold

        # Its alive to start with
        self.alive = True

        # Environment variables
        self.environment = environment
        self.environment_index = environment_index
        #this estimates the normalized and range-scaled 0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
        self.zidx = None
        self.zpos = None

        self.time = 0 # For logging

        # Run variables
        self.modelres = 6

        # Data logging only happens if a logger object is passed to the individual
        self.datalogger = datalogger

    def update_vert(self):
        self.zidx = np.argmin(abs(self.global_settings['depthrange'] - self.zpos))

    def temperature_zi(self):
        return self.environment.temperature[self.environment_index, self.zidx]

    def food1concentration_zi(self):
        return self.environment.food1concentration[self.environment_index, self.zidx]

    def pred1dens_zi(self):
        return self.environment.pred1dens[self.environment_index, self.zidx]

    def pred1lightdep_zi(self):
        return self.environment.pred1lightdep[self.environment_index, self.zidx]

    def update_lifestage(self):
        #the growth & development, survival and reproductive simulation happens within this if() condition based on developmental stage
        #no else() condition is written, as the loop skips if a super indivdual is dead or unseeded/uninitialized

        #this structures the simulation into following developmental stage categories:
        #1. non-feeding egg, NI and NII stages (index: 0, 1, 2)
        #2. feeding but non-energy-storing NIII-NVI,CI-CIII stages (index: 3, 4, 5, 6, 7, 8, 9)
        #3. feeding and energy-storing CIV and CV (diapausing) stages (index: 10, 11)
        #4. adult females (index: 12)
        #5. adult males (index: 13)
        #nb:these stage groupings are for C. finmarchicus only - for C.glacialis and C.hyperboreus, stage compositions of some categories may vary

        #simulation of de-growth, development and survival of non-feeding stages (egg, NI, NII): dsc-I
        #______________________________________________________________________________________________

        if self.developmentalstage <= 2:
            self.stage_lt_2()
            
        elif self.developmentalstage >= 3 and self.developmentalstage < 10:
            self.stage_3_9()

        elif self.developmentalstage == 10 or self.developmentalstage == 11:
            self.stage_10_11()

        else:
            self.stage_12_13()
   
        # Spatial logging is done at the coupler level as it is different depending on the domain      
        """
        if self.datalogger is not None:
            self.datalogger.log_spatial()
        """
        #post-developmental-stage processing
        #___________________________________

        #simulation of death of super individuals (i.e., when all virtual individual dies, a super individual also dies)
        #all state variables, gene values, loggers etc. are reset for a new super individual to take its place (these do not need to be re-initialized at seeding/spawning; only the 'gene' values do)

        if self.nvindividuals <= 0 or self.age >= self.global_settings['ageceiling'] or self.totalfecundity >= self.global_settings['fecundityceiling']: #!!!!! This should be done by the subpopulation
            #print('DEATH !!!!!!!!!!!!!!!!!!!')
            self.lifestatus = 0



    def stage_lt_2(self):
        #this is developmental stagegg category - I (dsc-I)
        #these developmental stages do not feed, so a general de-growth takes place over time
        #their development is temperature dependent but food-independent

        #dsc-I :: diel and seasonal vertical migration submodel
        #------------------------------------------------------
        #this uses a module-driven function to estimate the relative (index) and absolute vertical positions of the super individual based on the developmental stage and environmental variables
        #for non-feeding stages, the model assumes no individual vertical swimming capability, e.g., see: https://doi.org/10.1016/1054-3139(95)80062-X 
        #their vertical position is thus assumed to be vary uniformly randomly within the surface mixed layer (but with some modification, see the "vm" module)
        
        #nb:the mixed layer depth data are in np.float32 type, which needs to be converted to integers before proceeding further
        #calls the vertical migration estimator function of the developmental stage category 1 (dsc1: egg, NI, NII)
        self.zpos, self.zidx = vm.verticalmigration_dsc1(smld = self.environment.mld[self.environment_index])

        #dsc-I:growth and development submodel
        #-------------------------------------
        #update the current thermal history (this is an arithmetic mean)
        currentthermalhistory = (self.thermalhistory + self.temperature_zi()) / 2.00 #!!!!!!!!!!!!!!

        #this is the parameter "a" in Belehrádek’s (1935) temperature function, adopted from Campbel et al. (2001), see: https://doi.org/10.3354/meps221161
        currentdevelopmentalcoefficient = self.global_settings['developmentalcoefficient'][self.developmentalstage]
        #this uses a module-driven function to estimate the growth and development rate of the super individual based on internal states and environmental variables 
        #at egg, nauplius I & II only degrowth occurs (due to no feeding)
        #the developmental and growth rates are solely dependent on ambient temperatures
        #call the growth and development function for the dsc#1, which returns two outputs
        #output units: 6h pings for developmental time; 6h estimate for growth rate - but it is negative, signifying degrowth
        #nb:this degrowth rate is reduced the basal metabolic rate (= total metabolic rate at dsc-I)
        currentdevelopmentaltime, currentgrowthrate = gd.growthanddevelopment_dsc1(temperature = self.temperature_zi(),
                                                                                devcoef = currentdevelopmentalcoefficient,
                                                                                thist = currentthermalhistory,
                                                                                strmass = self.structuralmass, modelres = self.modelres)

        #this updates the structural mass of the super individual by adding the growth accumulated at 'currenttime'
        #nb:despite the addition, this is effectively a substraction because of the negative growth output by the dsc-I growth function above
        #nb:the "maxstructuralmass" does not need an update, because of structural degrowth during egg, and nauplii I & II
        self.structuralmass = self.structuralmass + currentgrowthrate

        #dsc-I: age and developmental stage advancement
        #----------------------------------------------
        #this updates the age of the super individual
        #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
        self.age += 1
        #if the currentage is greater than or equals to the mean developmental time sustained across the lifespan of the super individual, then the developmental stage advances
        if self.age >= currentdevelopmentaltime: #!!!!! Needs checking
            self.developmentalstage += 1
        #end if

        #dsc-I: survival submodel
        #------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        currentmortalityrisk = sv.mortalityrisk_dsc1(strmass = self.structuralmass,
                                                    maxstrmass = self.maxstructuralmass,
                                                    devstage = self.developmentalstage,
                                                    p1dens = self.pred1dens_zi(), #this estimates the visual predator density ("pred1dens") as a probability of death 
                                                    p1lightdp = self.pred1lightdep_zi(), #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                                                    p2risk = self.global_settings['pred2risk'],
                                                    bgmrisk = self.global_settings['bgmortalityrisk'])
        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        #this death is simulated after the stage-specific processes
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * currentmortalityrisk)

        #update the non-conditional state variables (conditionally state changed state variables are update in-place, e.g., developmental stage - see above), note maxstructural mass is not updated due to structural degrowth, see comment line 210
        self.thermalhistory = currentthermalhistory


    def stage_3_9(self):
        #these developmental stages feed, but do not channel the assimilated energy into an energy reserve
        #their development is temperature and food dependent
        #when they feed, it results in a propotional reduction in the food concentration (individual-environment feedbacks) :: <to be built>
        #opening state variable queries: extraction of 'current' internal states (from apropriate state variables) relevant to the stage category
        #nb:the cumulative development time state variable is discontinued from this stage category onwards (because henceforth, development is taken as a function of somatic growth)

        #dsc-II: diel and seasonal vertical migration submodel
        #-----------------------------------------------------
        #this uses a modular function to estimate the relative (index) and absolute vertical positions of the super individual based on the developmental stage and environmental variables
        #for feeding but non-energy-storing stages (dsc-III: NIII-CIII), the vertical position is estimated as a function of environmental variables (resource & risks) and super-individual-specific attribute values ('genes')

        #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
        #calling the vertical position estimation function from the module
        #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
        self.zpos, self.zidx, self.maxzdistance, self.actualzdistance = vm.verticalmigration_dsc2(temprange = self.environment.temperature[self.environment_index, :],
                                                                                                        f1conrange = self.environment.food1concentration[self.environment_index, :],
                                                                                                        iradrange = self.environment.irradiance[self.environment_index,:],
                                                                                                        maxirad = self.global_settings['maxirradiance'],
                                                                                                        p1dnsrange = self.environment.pred1dens[self.environment_index,:],
                                                                                                        a2 = self.genome.a2_irradiancesensitivity,
                                                                                                        a3 = self.genome.a3_pred1sensitivity,
                                                                                                        a4 = self.genome.a4_pred1reactivity,
                                                                                                        pvp = self.zpos,
                                                                                                        pvi = self.zidx,
                                                                                                        strmass = self.structuralmass,
                                                                                                        modelres = self.modelres)

        #dsc-II: growth and development submodel
        #---------------------------------------
        #this function estimates the somatic growth rate, which is used in the calculation of development rate (= 1 / developmental time)
        #only somatic growth (structural growth) occurs at this stage, no energy reserves are maintained
        #the function takes ambient temperature and food concentration as environmental inputs and current structural mass and developmental stage as internal state inputs
        currentgrowthrate = gd.growthanddevelopment_dsc2(temperature = self.temperature_zi(),
                                                        f1con = self.food1concentration_zi(),
                                                        strmass = self.structuralmass,
                                                        maxzd = self.maxzdistance,
                                                        actzd = self.actualzdistance,
                                                        modelres = self.modelres)

        #this updates the structural mass after growth no growth or degrowth
        #despite the "+" operator, negative growth results in subtraction (signifying degrowth)
        self.structuralmass = self.structuralmass + currentgrowthrate

        #this updates the maximum lifetime structural mass (for starvation risk estimation)  !!!!!!!!!!! Is this the right way round.....?
        if self.structuralmass > self.maxstructuralmass:
            self.maxstructuralmass = self.structuralmass
        #end if

        #dsc-II: age and developmental stage advancement
        #-----------------------------------------------
        #this updates the age of the super individual
        #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
        self.age += 1

        #the development rate or development time is not estimated by the above growth and development function
        #however, it is calculated based on the somatic growth rate
        #to estimate the molting state (i.e., whether a super individual is ready to molt to the next stage or not) of a super individual, the stage- and super-individual-specific critical molting mass is required
        #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
        #this estimates the stage-specific critical molting mass based on the 'gene' a1:
        currentcmm = self.global_settings['cmm_lower'][self.developmentalstage] + (self.global_settings['cmm_upper'][self.developmentalstage] - self.global_settings['cmm_lower'][self.developmentalstage]) * self.genome.a1_bodysize

        #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
        #the else condition is not mentioned here, as no stage increment occurs if the if() condition is invalid
        if self.structuralmass >= currentcmm:
            #molting occurs and stage is updated in-place(due to conditional state change)
            self.developmentalstage += 1
        #end if

        #dsc-II: survival submodel
        #-------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        currentmortalityrisk = sv.mortalityrisk_dsc2(strmass = self.structuralmass, maxstrmass = self.maxstructuralmass, p1dens = self.pred1dens_zi(), p1lightdp = self.pred1lightdep_zi(), p2risk = self.global_settings['pred2risk'], bgmrisk = self.global_settings['bgmortalityrisk'])

        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * currentmortalityrisk)

        #simulation of growth, development and survival of feeding and energy-storing stages (CIV, CV): dsc-III
        #______________________________________________________________________________________________________


    def stage_10_11(self):
        #this is developmental stage category-IIII (dsc-III that includes energy-storing civ and cv stages)
        #these are feeding stages that actively maintains an energy reserve
        #the energy reserve is used for countering starvation mortality risk and for spending the unproductive part of the year in a diapause state
        #depending on the state of diapause, this stage is split into 5 subcategories
        #dsc-IIIA: active pre-diapause state; dsc-IIIE: active diapause entry state; dsc-IIID: diapause state; dsc-IIIX: active exit state; dscIII-P: active post-diapause state
        #these subcategories have to be coded separately due to their intricate changes of physiology and behaviour 

        #if the diapause strategy is undefined (-1: typical for newly seeded/spawned super individual arriving at civ/cv for the first time), define the diapause strategy (0, 1)
        #nb:the diapause strategy is linked to the diapause probability 'gene'
        if self.diapausestrategy == -1:
            #random number for diapause strategy determination
            dsdet = np.random.rand(1).squeeze()
            #falls into 0 (direct development, no diapause) or 1 (diapause) depending on the diapause probability 'gene' value
            self.diapausestrategy = 1 if dsdet < self.genome.a6_diapauseprobability else 0
        #end if

        #this splits the dsc-III into subcategory-specific processing drives
        #nb:no else condition is written due to the presence of "U" undefined diapause state, which is not processed
        if self.diapausestate == "A":
            self.diapauseA()
        elif self.diapausestate == "E":
            self.diapauseE()
        elif self.diapausestate == "D":
            self.diapauseD()
        elif self.diapausestate == "X":
            self.diapauseX()
        elif self.diapausestate == "P":
            self.diapauseP()
        #end if

    #simulation of growth, development and survival of adult male and female stages (CVI-F, CVI-M): dsc-IV
    #_____________________________________________________________________________________________________

    def stage_12_13(self):
        #these are adult stages (dsc-IV) that can be males or females
        #the sex should be determined at the first entry to this stage (not initialized in the seeding/spawning)
        #sex is determined randomly at ca. 0.5:0.5 M:F probability
        if self.sex == "U":
            #drawing a uniform random number to compare with the threshold of 0.50
            sexdet = np.random.rand(1).squeeze()
            self.sex = "M" if sexdet < 0.5 else "F"
        #end if

        #they feed, grow but do not maintain an active energy reserve - instead they use stored energy for fulfilling metabolic demands
        #reproduction and spawning happens during the adult stage, where reproduction includes mate finding, insemination, recombination and mutation - and eventually, spawning (egg production)
        #adult males do not feed - but use whatever the energy reserves they possess for survival (hence, they are short-lived and maintians 'genetic' diversity within the subpopulation)
        #both sexes engage in shorter term vertical behavior because it improves the chances of male-female encounter
        #nb:the "sex" and "maxstructuralmass" are inquired but not updated within the stage - so there is no updates therein at the closure (see below)
        #the "malegenome" is not inquired but updated in-place if a female is inseminated (see below)

        #dsc-IV: diel and seasonal vertical migration submodel
        #-----------------------------------------------------
        #this uses a module-driven function to estimate the vertical position & index ("currentzpos", "currentzidx") of the super individual based on the developmental stage and environmental variables
        #for adult male and female stages (CVI:subcategories M & F), the vertical position is estimated as a function of environmental variables (resource & risks) and super-individual-specific attribute values ('genes')

        #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
        #calling the vertical position estimation function from the module
        #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
        self.zpos, self.zidx, self.maxzdistance, self.actualzdistance = vm.verticalmigration_dsc4(temprange = self.environment.temperature[self.environment_index, :],
                                                                                                        f1conrange = self.environment.food1concentration[self.environment_index, :],
                                                                                                        iradrange = self.environment.irradiance[self.environment_index,:],
                                                                                                        maxirad = self.global_settings['maxirradiance'],
                                                                                                        p1dnsrange = self.environment.pred1dens[self.environment_index,:],
                                                                                                        a2 = self.genome.a2_irradiancesensitivity,
                                                                                                        a3 = self.genome.a3_pred1sensitivity,
                                                                                                        a4 = self.genome.a4_pred1reactivity,
                                                                                                        pvp = self.zpos,
                                                                                                        pvi = self.zidx,
                                                                                                        strmass = self.structuralmass,
                                                                                                        resmass = self.reservemass,
                                                                                                        modelres = self.modelres)

        #dsc-IV: growth and development submodel
        #---------------------------------------
        #this is male and female specific because adult males do not feed

        #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
        #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
        #obligatory negative growth (degrowth; structural or energy reserve) for adult male and growth/degrowth for adult female
        if self.sex == "F":
            #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
            #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
            #this is female-specific (growth/degrowth both possible)
            currentgrowthrate = gd.growthanddevelopment_dsc4f(temperature = self.temperature_zi(),
                                                            f1con = self.food1concentration_zi(),
                                                            strmass = self.structuralmass,
                                                            resmass = self.reservemass,
                                                            maxzd = self.maxzdistance,
                                                            actzd = self.actualzdistance,
                                                            modelres = self.modelres)

        else:

            #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
            #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
            #this is male-specific (degrowth is only possible)
            currentgrowthrate = gd.growthanddevelopment_dsc4m(temperature = self.temperature_zi(),
                                                            f1con = self.food1concentration_zi(),
                                                            strmass = self.structuralmass,
                                                            resmass = self.reservemass,
                                                            maxzd = self.maxzdistance,
                                                            actzd = self.actualzdistance,
                                                            modelres = self.modelres)

        #end if

        #dsc-IV:age advancement
        #------------------------
        #this updates the age of the super individual
        #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
        #there is no developmental stage advancement after reaching the adulthood (CVI-F/M)
        #the structural mass also reaches a maximum after reaching adulthood (check with v3.1 - check, ok!)
        self.age += 1

        #dsc-IV:reproduction and spawning submodel - part-A: energy allocation
        #---------------------------------------------------------------------
        #this submodel is sex-specific
        #nearby males and females mate, and a given female gets inseminated by one (1) male only
        #a given male can mate with many females during its short lifespan
        #the mate choice is assumed to be random

        if self.sex == "M":

            #for the male, only the structural and reserve masses are updated
            #nb:only degrowth is possible because males do not feed
            self.update_mass(currentgrowthrate)

        else:

            #for the female, growth can be positive or negative (depending on food conditions and vertical behaviour)
            #in phases of positive growth, the surplus energy is allocated to egg production and/or structural growth (only if a female sustained degrowth in the past)
            #in case of past degrowth, the female channels 50% of the surplus energy for structural growth and 50% to egg production (a sensible estimate)

            #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
            self.adultsize = self.global_settings['cmm_lower'][12] + (self.global_settings['cmm_upper'][12] - self.global_settings['cmm_lower'][12]) * self.genome.a1_bodysize

            if currentgrowthrate >= 0.00:

                #here, the growth rate is positive (i.e., there is surplus assimilation)
                #surplus assimilation is fully or partly channeled to reproductive output
                if self.structuralmass >= self.adultsize:

                    #in this case, the female is healthy (i.e., havent sustained structural degrowth)
                    #surplus assimilation is fully allocated to reproductive output (this can be fitted with a conversion factor, which is ca. 80% in some models)
                    #but only if the female is inseminated
                    #if so or else, no change to structural mass; no change to reserve mass - the surplus assimilation is discounted for non-inseminated females at structural mass ceiling

                    if self.inseminationstate == 1:
                        #reproductive allocation only if the female is inseminated - otherwise, surplus assimilation is discounted (no else condition is written)
                        self.reproductiveallocation = self.reproductiveallocation + currentgrowthrate
                    #end if

                else:

                    #in this case, the female has sustained structural degrowth and needs recovery
                    #surplus assimilation is only partly allocated to reproductive output (50%) if the female is inseminated: no change to energy reserve mass
                    #if the female is not inseminated, all the surplus assimilation is channeled to structural growth
                    #no need to cap the structural mass, as it does not grow beyond (apart from a very small amount) the limit ("currentadultsize") because of the <if> condition above
                    #this means, as the female grows beyond the maximum structural mass the if condition above is activated and it does not have a structural growth allocation routine

                    if self.inseminationstate == 1:

                        #surplus assimilation is equally channeled between reproduction and structural growth for inseminated females
                        self.reproductiveallocation = self.reproductiveallocation + 0.50 * currentgrowthrate
                        self.structuralmass = self.structuralmass + 0.50 * currentgrowthrate

                    else:

                        #for non-inseminated females, all surplus assimilation is allocated to structural growth
                        self.structuralmass = self.structuralmass + currentgrowthrate

                    #end if
                #end if

            else:
                self.update_mass(currentgrowthrate)

            #end if

        #end if


        #dsc-IV:reproduction and spawning submodel - part-B: mate selection and spawning
        #-------------------------------------------------------------------------------
        #this applies to the adult female only
        #if the female is non-inseminated, it finds a male (randomly) in the proximity and mates
        #during mating the male 'genome' is copied to the female into a state variable 
        #if the female is inseminated, it produces eggs using the reproductive allocation upadted above
        #the else condition ('currentsex == male') is not written as it returns nothing
        #
        #in the particle tracking version the mate finding needs knowledge of the other particles so 
        # occurs outside of the individual class and the genome is copied into the individual so the male finding
        #code is removed and if there is a near male its genome is in self.malegenome (otherwise it is None)

        if self.sex == "F":
            if self.inseminationstate == 0 and self.malegenome is not None:
                self.inseminationstate = 1      
 
            elif self.inseminationstate == 1: 
                #for inseminated females, egg production may occur depending on the reproductive allocation
                #the else condition ('currentreproductiveallocation < eggmass') is not written as it returns nothing 

                if self.reproductiveallocation >= self.eggmass:

                    #spawning can occur since there is a surplus allocation for this purpose
                    #nb:int does not round up or down - it takes the significant digit: therefore, a d0 rounded value is converted to int
                    #assimilated energy cannot vanish: so, the reproductive allocation is re-updated with the remainder
                    self.potentialfecundity = self.reproductiveallocation / self.eggmass
                    self.reproductiveallocation = self.reproductiveallocation - (self.eggmass * self.potentialfecundity)
                    self.potentialfecundity = int(np.round(self.potentialfecundity, decimals = 0))
                    self.totalfecundity = self.totalfecundity + self.potentialfecundity

                #end if
        #end if

        #dsc-IV:survival submodel
        #------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        currentmortalityrisk = sv.mortalityrisk_dsc4(strmass = self.structuralmass,
                                                    maxstrmass = self.maxstructuralmass,
                                                    resmass = self.reservemass,
                                                    p1dens = self.pred1dens_zi(),
                                                    p1lightdp = self.pred1lightdep_zi(),
                                                    p2risk = self.global_settings['pred2risk'],
                                                    bgmrisk = self.global_settings['bgmortalityrisk'])

        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        #this death is simulated after stage-specific processes 
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * currentmortalityrisk)


    def diapauseA(self):
        #this is the active pre-diapausing super individuals belonging to civ and cv stages
        #they actively feed, grow and develop (civ-cv-adult) with (1-year life cycle)or without diapause (< 1 year life cycle)
        #actively maintain an energy reserve for starvation compensation and diapause (if they diapause)
        #they are subjected to two end pathways: (i)direct development into civ->cv->adult and onwards or (ii)become diapause entry stage (dsc-IIIE) and onwards
        #ranges are such that it includes data across entire depth range (nb: irregular intervals)

        #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
        #calling the vertical position estimation function from the module
        #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
        self.zpos, self.zidx, self.maxzdistance, self.actualzdistance = vm.verticalmigration_dsc3a(temprange = self.environment.temperature[self.environment_index,:],
                                                                                                        f1conrange = self.environment.food1concentration[self.environment_index,:],
                                                                                                        iradrange = self.environment.irradiance[self.environment_index,:],
                                                                                                        maxirad = self.global_settings['maxirradiance'],
                                                                                                        p1dnsrange = self.environment.pred1dens[self.environment_index,:],
                                                                                                        a2 = self.genome.a2_irradiancesensitivity,
                                                                                                        a3 = self.genome.a3_pred1sensitivity,
                                                                                                        a4 = self.genome.a4_pred1reactivity,
                                                                                                        pvp = self.zpos,
                                                                                                        pvi = self.zidx,
                                                                                                        strmass = self.structuralmass,
                                                                                                        resmass = self.reservemass,
                                                                                                        modelres = self.modelres)

        #dsc-IIIA: growth and development submodel
        #-----------------------------------------
        #extraction of apropriate environmental variables based on the current zidx
        #this function estimates the somatic growth rate, which is used in calculating the development rate (= 1 / development time)
        #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
        currentgrowthrate = gd.growthanddevelopment_dsc3a(temperature = self.temperature_zi(),
                                                        f1con = self.food1concentration_zi(),
                                                        strmass = self.structuralmass,
                                                        resmass = self.reservemass,
                                                        maxzd = self.maxzdistance,
                                                        actzd = self.actualzdistance,
                                                        modelres = self.modelres)

        #whether the super individual enters diapause or develop directly towards adulthood without diapause is governed by the 'gene' "a6_diapauseprobability"
        #and this binary state is defined in the seeding and/or spawning submodel and updated in the state variable "diapausestrategy", which is estimated above at stage civ/cv entry

        #this estimates the maximum adult size reachable by the super individual (for establishing a structuralmass ceiling for diapausing individuals until energy reserves are sufficiently accumulated for diapause)
        #to estimate the molting state of a super individual (i.e., whether a super individual is ready to molt to the next stage or not), the stage- and super-individual-specific critical molting mass is required
        #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
        #the index position 11 is the CV->CVI(M/F) molting mass, which is the maximum structural mass reachaable by a super individual in a given environment and given 'gene' value of "a1_bodysize"
        self.adultsize = self.global_settings['cmm_lower'][11] + (self.global_settings['cmm_upper'][11] - self.global_settings['cmm_lower'][11]) * self.genome.a1_bodysize

        #based on the growth rate, this updates the structuralmass and energy reserve mass
        #however, the surplus assimilation allocation patterns for somatic growth & reserve buildup are markedly different for directly developing super individuals and diapausing individuals
        #nb:this <if> condition is not written for currentgrowthrate == 0.00 because it doesnt update neither the structural nor reserve masses

        if currentgrowthrate > 0:

            if self.diapausestrategy == 0:

                #this is the surplus assimilation allocation for non-diapausing super individuals
                #all surplus assimilation is channeled to structural growth (somatic growth); no reserves are maintained (this is slightly differnt from pascal v.3.1)
                #in other words, the "a5_energyallocation" 'gene' is effectively supressed (thats why these 'genes' are called dynamic evolvable attributes)
                #therefore, no update in the "currentreservemass"
                #since there is no reserve allocation, no control is needed to regulate the reserve:structure ratio (max = 1.00)
                self.structuralmass = self.structuralmass + currentgrowthrate

            else:

                #this is the surplus assimilation allocation for diapausing super individuals for which, the 'gene' "a5_energyallocation" is not supressed
                #however, the surplus assimilation allocation depends on the structural mass of the super individual with respect to the maximum reachable body mass ("currentadultsize")
                #the structural mass of the super individual must be held static, if it is at the "currentadultsize" until the reserve/structure ratio is satisfied for diapause entry
                if self.structuralmass >= self.adultsize:

                    #this blocks further allocation of surplus assimilation to structural (or somatic) growth
                    #and all assimilation is channeld to reserve build up
                    #as a result, no update in the structural mass
                    self.reservemass = self.reservemass + currentgrowthrate

                else:

                    #this allocates the surplus assimilation to both structural (somatic) growth and reserve build up based on proportions defined by the 'gene' "a5_energyallocation"
                    #here, a fraction defined by the evolvable dynamic attribute "a5_energyallocation" is channeled to structural growth
                    self.structuralmass = self.structuralmass + currentgrowthrate * (1.00 - self.genome.a5_energyallocation)
                    #the rest is channel to reserve build up
                    #nb:here, there is no explicit reserve mass limitation applied, but self-limitation occur at the diapause entry condition, which has a reservemass/structuralmass ceiling of 1.00
                    self.reservemass = self.reservemass + currentgrowthrate * self.genome.a5_energyallocation

                #end if
            #end if

        else:
            self.update_mass(currentgrowthrate)

        #end if

        #this conditionally updates the maximum lifetime structural mass (for starvation risk estimation)
        #only potentially valid for positive structural growth
        if self.structuralmass > self.maxstructuralmass:
            self.maxstructuralmass = self.structuralmass

        #end if

        #dsc-IIIA:age, developmental stage and/or diapause state advancement
        #-------------------------------------------------------------------
        #this extracts the diapause entry evolvable dynamic attribute for the super individual
        #this updates the age of the super individual
        #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
        self.age += 1

        #the development rate or development time is a function of growth rate calculated by the modular function above
        #to estimate the molting state (i.e., whether a super individual is ready to molt to the next stage or not) of a super individual, the stage- and super-individual-specific critical molting mass is required
        #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
        currentcmm = self.global_settings['cmm_lower'][self.developmentalstage] + (self.global_settings['cmm_upper'][self.developmentalstage] - self.global_settings['cmm_lower'][self.developmentalstage]) * self.genome.a1_bodysize
        #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass

        if self.diapausestrategy == 0:

            #for super individuals that do not undergo diapause, an attempt is made to develop directly from civ - cv - adult
            #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
            #the else condition is not mentioned here, as no stage increment occurs if the if() condition is invalid
            if self.structuralmass >= currentcmm:

                #molting occurs and stage is updated
                #the diapause state does not change and remains at "A" (active)
                currentdevelopmentalstage += 1

                #update the state variable in-place (due to conditional state change)
                developmentalstage[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage

                #data logging
                #------------
                if self.datalogger is not None:
                    self.datalogger.add_lifecycle_i(self.nvindividuals, 0)
                    self.datalogger.add_lifcycle_f((self.structuralmass * self.tnvindividuals) / 1e6, 0)
                    self.datalogger.add_lifcycle_f((self.structuralmass * self.nvindividuals) / 1e6, 1)

                #end if

            #end if

            #for super individuals that undergo diapause, development from civ-cv is allowed (not obligatory) but that from cv-adult is blocked

            if self.developmentalstage == 10:

                #these are civ stages, and they are allowed to develop into cv stages if the diapause entry condition (structural to reserve mass ratio) allows
                #first, the molting condition is checked:
                #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
                if self.structuralmass >= currentcmm:

                    #first, the molting condition is checked
                    #molting occurs and stage is updated
                    #no change occur in the diapause state ("A":active)
                    self.developmentalstage += 1

                #second, the diapause entry condition is checked, if this isn't met nothing happens:
                else:
                    self.check_diapauseentry()

                #end if
            else:

                #these are cv stages, and they are not allowed to develop into adults before entering diapause
                #therefore, they are not checked for the developmental stage condition; also, in the allocation section above, a structural mass ceiling is established
                #this holds their structural mass maximally at right below the adult size without development to adult
                #this checks for the diapause entry condition:
                self.check_diapauseentry()

            #end if

        #end if
        #dsc-IIIA:survival submodel
        #--------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        currentmortalityrisk = sv.mortalityrisk_dsc3a(strmass = self.structuralmass,
                                                    maxstrmass = self.maxstructuralmass,
                                                    resmass = self.reservemass,
                                                    p1dens = self.pred1dens_zi(),
                                                    p1lightdp = self.pred1lightdep_zi(),
                                                    p2risk = self.global_settings['pred2risk'],
                                                    bgmrisk = self.global_settings['bgmortalityrisk'])

        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        #this death is simulated after stage-specific processes
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * currentmortalityrisk)


    def diapauseE(self):
        #this developmental stage group subcategory includes super individuals that are at the diapause entry
        #at this point, they do not feed but seek a preferred diapause depth
        #despite not feeding, their metabolism occurrs at the regular rate (basal and active metabolic rates)
        #for this, they are not in diapause yet - until they find the preferred diapause depth and 'settle' therein at a diapause state ("D":diapause)

        #this uses a modular function to estimate the vertical position (i.e., seasonal vertical migration to diapause depths)
        self.zpos, self.zidx, self.maxzdistance, self.actualzdistance = vm.verticalmigration_dsc3e(diapdepth = self.diapausedepth,
                                                                                                        strmass = self.structuralmass,
                                                                                                        pvp = self.zpos,
                                                                                                        modelres = self.modelres)

        #dsc-IIIE: growth and development submodel
        #-----------------------------------------
        #nb:somatic growth do not occur at dsc-IIIE because super individuals stop feeding and begins to use the energy reserve for survival
        #however, until reaching the diapause depth, the metabolic rate occurs at a regular rate (i.e., reserves are burnt faster than at diapause)
        #the potential degrowth and/or reserve utilization is therefore, depndent on the ambient temperature and the total bodymass of the super individual

        #this uses a modular function to estimate the potential degrowth and/or reserve utilization of super individuals
        currentgrowthrate = gd.growthanddevelopment_dsc3e(temperature = self.temperature_zi(),
                                                        strmass = self.structuralmass,
                                                        resmass = self.reservemass,
                                                        actzd = self.actualzdistance,
                                                        maxzd = self.maxzdistance,
                                                        modelres = self.modelres)

        #super individuals with limited energy reserves can also enter the diapause entry stage (dsc-IIIE: cf. lower values of the a7_diapauseentry 'gene')
        #therefore, the reserve exhaustion and/or structural catabolization is done as follows:


        #dsc-IIIE: age and diapause state advancement
        #-----------------------------------------------
        #nb:no developmental stage advancement at this level (due to no feeding & growth)
        #this advances the current age by 1 ping (= 6 hours)
        self.age += 1

        #this advances the diapause state (to "D":diapause) if the super individual had reached the diapause depth
        if self.zpos == self.diapausedepth:
            self.diapausestate = "D"
        #end if

        #dsc-IIIE: survival submodel
        #---------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        currentmortalityrisk = sv.mortalityrisk_dsc3e(strmass = self.structuralmass,
                                                    maxstrmass = self.maxstructuralmass,
                                                    resmass = self.reservemass,
                                                    p1dens = self.pred1dens_zi(),
                                                    p1lightdp = self.pred1lightdep_zi(),
                                                    p2risk = self.global_settings['pred2risk'],
                                                    bgmrisk = self.global_settings['bgmortalityrisk'])

        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * currentmortalityrisk)


    def diapauseD(self):
        #this developmental stage subcategory refers to diapausing individuals at their preferred diapause depths
        #they dont feed, dont grow or develop
        #their basal metabolic rate occurrs at 25% of the regular basal metabolic rate, see: Maps et al.(2012): https://doi.org/10.1093/plankt/fbt100 
        #there is no active metabolic costs during diapause, which means no active movements - only passive drifts brought about by water currents
        #therefore, the diel and seasonal vertical migration submodel is not called

        #dsc-IIID: growth and development submodel
        #-----------------------------------------
        #nb:no somatic growth occurs at dsc-IIID because super individuals are at diapause
        #the metabolic rate occurs at a reduced rate (i.e., 25% of the regular metabolic rate)
        #the potential degrowth and/or reserve utilization are depndent on the ambient temperature and the total bodymass of the super individual

        #this uses a modular function to estimate the potential degrowth and/or reserve utilization of super individuals
        currentgrowthrate = gd.growthanddevelopment_dsc3d(temperature = self.temperature_zi(),
                                                        strmass = self.structuralmass,
                                                        resmass = self.reservemass,
                                                        modelres = self.modelres)

        #super individuals with limited energy reserves can also enter diapause (dsc-IIID: cf. lower values of the "a7_diapauseentry" 'gene')
        #nb:however, their diapause state changes from "D" to "X" if the reserves run out (cf. "a8_diapauseexit" 'gene')
        #therefore, the reserve exhaustion and/or structural catabolization is done as follows:
        self.update_mass(currentgrowthrate)

        #end if
        #dsc-IIID:age and diapause state advancement
        #-------------------------------------------
        #nb:no developmental stage advancement at this level (due to no feeding & growth)
        #this advances the current age by 1 ping (= 6 hours)
        self.age += 1

        #if the diapause entry reserve mass is zero (due to diapause entry 'gene' value being 0.00), a separate condition is used to evaluate the exit-state
        #this is to avoid errors emerged from dividing by zero

        if self.reservemassatdiapauseentry <= 0:

            #the diapsue state changes to diapause exit or "X"
            self.diapausestate = "X"
            self.timeofdiapauseexit = self.time

        else:

            if 1.00 - self.reservemass / self.reservemassatdiapauseentry >= self.genome.a8_diapauseexit:

                #the diapsue state changes to diapause exit or "X"
                self.diapausestate = "X"

                self.reservemassatdiapauseexit = self.reservemass
                self.timeofdiapauseexit = self.time

            #end if

        #end if

        #dsc-IIID: survival submodel
        #---------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        self.mortalityrisk = sv.mortalityrisk_dsc3d(strmass = self.structuralmass,
                                                    maxstrmass = self.maxstructuralmass,
                                                    resmass = self.reservemass,
                                                    p1dens = self.pred1dens_zi(),
                                                    p1lightdp = self.pred1lightdep_zi(),
                                                    p2risk = self.global_settings['pred2risk'],
                                                    bgmrisk = self.global_settings['bgmortalityrisk'])

        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        #this death is simulated after stage-specific processes
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * self.mortalityrisk)


    def diapauseX(self):
        #these include the super individuals belonging to the developmental stage subcategory that are exitting diapause
        #they seek to move out of the diapause habitat (depth) and ascend to the surface (a random photic zone depth zidx = 0:25)
        #their basal and active metabolic rates are back to regular levels
        #however, they do not feed yet - do so only after completing the seasonal ascent, i.e., at diapause state "P" (dsc-IIIP)

        #this uses a modular function to estimate the vertical position (i.e., seasonal vertical migration out of diapause depths)
        self.zpos, self.zidx, self.maxzdistance, self.actualzdistance = vm.verticalmigration_dsc3x(pvp = self.zpos,
                                                                                                        pvi = self.zidx,
                                                                                                        strmass = self.structuralmass,
                                                                                                        modelres = self.modelres)

        #dsc-IIIX: growth and development submodel
        #-----------------------------------------
        #nb:no somatic growth occurs at dsc-IIIX because super individuals stop feeding and still continue to use the energy reserve for survival
        #nb:feeding, growth and development resumes only at the dsc-IIIP stage subcategory (see below)
        #however, the metabolic rate occurs at a regular rate (i.e., reserves are burnt faster than at diapause)
        #the potential degrowth and/or reserve utilization is therefore, depndent on the ambient temperature and the total bodymass of the super individual

        #this uses a modular function to estimate the potential degrowth and/or reserve utilization of super individuals
        currentgrowthrate = gd.growthanddevelopment_dsc3x(temperature = self.temperature_zi(),
                                                        strmass = self.structuralmass,
                                                        resmass = self.reservemass,
                                                        actzd = self.actualzdistance,
                                                        maxzd = self.maxzdistance,
                                                        modelres = self.modelres)

        #super individuals with limited energy reserves can also enter the diapause entry stage (dsc-IIIE: cf. lower values of the a7_diapauseentry 'gene')
        #therefore, the reserve exhaustion and/or structural catabolization is done as follows:
        self.update_mass(currentgrowthrate)

        #end if
        #dsc-IIIX: age and diapause state advancement
        #--------------------------------------------
        #nb:no developmental stage advancement at this level (due to no feeding & growth)
        #this advances the current age by 1 ping (= 6 hours)
        self.age += 1

        #this advances the diapause state (to "P":post-diapause) if the super individual had reached the upper pelagial (<= 100 m)
        if self.zpos <= 100:

            self.diapausestate = "P"

            #data logging
            #------------
            if self.datalogger is not None:
                if self.developmentalstage == 10:
                    #for civ diapause-exits
                    self.datalogger.add_lifecycle_i(self.nvindividuals,3)
                    self.datalogger.add_lifecycle_f((self.reservemass * self.nvindividuals) / 1e6, 6)
                elif self.developmentalstage == 11:
                    #for cv diapause-exits
                    self.datalogger.add_lifecycle_i(self.nvindividuals,4)
                    self.datalogger.add_lifecycle_f((self.reservemass * self.nvindividuals) / 1e6, 7)

        #end if

        #dsc-IIIX: survival submodel
        #---------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        currentmortalityrisk = sv.mortalityrisk_dsc3x(strmass = self.structuralmass,
                                                    maxstrmass = self.maxstructuralmass,
                                                    resmass = self.reservemass,
                                                    p1dens = self.pred1dens_zi(),
                                                    p1lightdp = self.pred1lightdep_zi(),
                                                    p2risk = self.global_settings['pred2risk'],
                                                    bgmrisk = self.global_settings['bgmortalityrisk'])

        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * currentmortalityrisk)


    def diapauseP(self):
        #these include super individuals belonging to the post diapause stages ("P") at the developmental stage category III (dsc-IIIP)
        #they feed, grow and develop as usual (this is the first state after which regular metabolism remains after diapause exit)
        #they do not actively maintain an energy reserve - but may use the remaining reserves for managing starvation risk and channeling into structural growth during food shortage
        #they develop directly to adults
        #nb:no else condition is written due to the presence of "U" undefined diapause state, which is not processed

        #dsc-IIIP: diel and seasonal vertical migration submodel
        #-----------------------------------------------------
        #this uses a module-driven function to estimate the vertical position & index ("currentzpos", "currentzidx") of the super individual based on the developmental stage and environmental variables
        #for feeding and energy-storing stages (CIV-CV:subcategory-P), the vertical position is estimated as a function of environmental variables (resource & risks) and super-individual-specific attribute values ('genes')

        #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
        #calling the vertical position estimation function from the module
        #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
        self.update
        self.zpos, self.zidx, self.maxzdistance, self.actualzdistance = vm.verticalmigration_dsc3p(temprange = self.environment.temperature[self.environment_index, :],
                                                                                                        f1conrange = self.environment.food1concentration[self.environment_index, :],
                                                                                                        iradrange = self.environment.irradiance[self.environment_index,:],
                                                                                                        maxirad = self.global_settings['maxirradiance'],
                                                                                                        p1dnsrange = self.environment.pred1dens[self.environment_index,:],
                                                                                                        a2 = self.genome.a2_irradiancesensitivity,
                                                                                                        a3 = self.genome.a3_pred1sensitivity,
                                                                                                        a4 = self.genome.a4_pred1reactivity,
                                                                                                        pvp = self.zpos,
                                                                                                        pvi = self.zidx,
                                                                                                        strmass = self.structuralmass,
                                                                                                        resmass = self.reservemass,
                                                                                                        modelres = self.modelres)

        #dsc-IIIP: growth and development submodel
        #-----------------------------------------
        #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
        #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
        currentgrowthrate = gd.growthanddevelopment_dsc3p(temperature = self.temperature_zi(),
                                                        f1con = self.food1concentration_zi(),
                                                        strmass = self.structuralmass,
                                                        resmass = self.reservemass,
                                                        maxzd = self.maxzdistance,
                                                        actzd = self.actualzdistance,
                                                        modelres = self.modelres)

        #all surplus assimilations are channeled to structural growth: no energy reserves are maintained or replinshed
        #however, reserves may be used for balancing degrowth and starvation risk therein
        #nb:the no growth condition (currentgrowthrate == 0) is not written as it does not affect neither structural mass nor reserve mass
        self.update_mass(currentgrowthrate)

        #dsc-IIIP:age and developmental stage advancement
        #------------------------------------------------
        #this updates the age of the super individual
        #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
        self.age += 1

        #the developmental stage advancement can be from civ-cv and cv-cvi(F/M) depending on the diapause stage
        #to estimate the molting state (i.e., whether a super individual is ready to molt to the next stage or not) of a super individual, the stage- and super-individual-specific critical molting mass is required
        currentcmm = self.global_settings['cmm_lower'][self.developmentalstage] + (self.global_settings['cmm_upper'][self.developmentalstage] - self.global_settings['cmm_lower'][self.developmentalstage]) * self.genome.a1_bodysize

        #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
        #the else condition is not mentioned here, as no stage increment occurs if the if() condition is invalid
        if self.structuralmass >= currentcmm:

            #molting occurs and stage is updated
            #the diapause state does not change and remains at "A" (active)
            self.developmentalstage += 1

        #end if

        #dsc-IIIP:survival submodel
        #--------------------------
        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
        currentmortalityrisk = sv.mortalityrisk_dsc3p(strmass = self.structuralmass,
                                                    maxstrmass = self.maxstructuralmass,
                                                    resmass = self.reservemass,
                                                    p1dens = self.pred1dens_zi(),
                                                    p1lightdp = self.pred1lightdep_zi(),
                                                    p2risk = self.global_settings['pred2risk'],
                                                    bgmrisk = self.global_settings['bgmortalityrisk'])

        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
        #when all virtual individuals contained in a super individual dies, then the super individual also dies
        #this death is simulated after stage-specific processes
        self.nvindividuals = self.nvindividuals - int(self.nvindividuals * currentmortalityrisk)

    def check_diapauseentry(self):
        if self.reservemass / self.structuralmass >= self.genome.a7_diapauseentry:
            #if this condition is satisfied, the super individual is in the "diapause entry mode" ("E")
            self.diapausestate = "E"

            #data logging
            #------------
            if self.datalogger is not None:
                self.datalogger.add_lifecycle_i(self.nvindividuals, 2)
                self.datalogger.add_lifcycle_f((self.structuralmass * self.nvindividuals) / 1e6, 4)
                self.datalogger.add_lifcycle_f((self.structuralmass * self.nvindividuals) / 1e6, 5)


    def update_mass(self, currentgrowthrate): 
        if currentgrowthrate > 0:
            #if the net growth rate is positive, all the surplus assimilation is channeled to somatic growth
            #no changes in the reserve mass
            self.structuralmass = self.structuralmass + currentgrowthrate

            #this updates the maximum structural mass if necessary:
            if self.structuralmass >= self.maxstructuralmass:
                self.maxstructuralmass = self.structuralmass

        elif self.reservemass >= abs(currentgrowthrate):
            #if the reserve mass is sufficient to balance the degrowth (i.e., metabolic demands of diapause entry):
            #reserves are proportionally mobilized and no change occurs in the structural mass (despite the "+" operator, it is effectively a subtraction as growt rate is negative)
            self.reservemass = self.reservemass + currentgrowthrate

        else:
            #if reserves are not sufficient to balance the degrowth:
            #the structural mass is proportionally catabolized (despite the + operator, it is effectively a subtraction as growth rate is negative)
            #no change to the reserve mass
            self.structuralmass = self.structuralmass + currentgrowthrate

    def get_child_genome(self):
        spawning_f = self.genome
        spawning_m = self.malegenome
        spawning_n = {}
        for attr_name in spawning_f.keys():
            #this is the crossover probability per-gene (if this value is lower than crossover threshold, then crossover occurs)
            cxprob = np.random.rand(1).squeeze()
            #this is mutation probability per-gene (if this value is lower than the mutation threshold, then mutation occurs)
            muprob = np.random.rand(1).squeeze()
            #this is the blend value per-gene in BLX-alpha algorithm
            #nb: see Tkahashi et al, (2001) 10.1109/CEC.2001.934452
            cxval = np.random.rand(1).squeeze()

            #crossover algorithm (BLX-alpha)
            if cxprob < self.cxthreshold:
                #if the crossover threshold is met, then male and female genomes are blended with BLX-alpha crossover 
                spawning_n[attr_name] = (cxval * spawning_f[attr_name]) + ((1.00 - cxval) * spawning_m[attr_name])
            else:
                #otherwise, the female genome is inherited without blending
                spawning_n[attr_name] = spawning_f[attr_name]
            #end if
            #mutation algorithm (random replacement)
            #nb:the <else> condition is not written because, no mutation does not change the genome 
            if muprob < self.muthreshold:
                spawning_n[attr_name] = np.random.rand(1)[0]

        return spawning_n

    def get_spatial_log_data(self):
        if self.developmentalstage == 12 and self.sex == 'M':
            col = 13
        else:
            col = self.developmentalstage

        return [col, self.environment.x[self.environment_index], self.environment.y[self.environment_index], self.zidx, self.nvindividuals, self.structuralmass + self.reservemass] #Check this is the correct Z
    
