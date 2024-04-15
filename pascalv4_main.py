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

#variable definitions
#=========================================================================================================================================================================================================================================================

#core variables and constants (in python, constants are not explicitly defined)
#______________________________________________________________________________

#defines the no. of subpopulations in the simulated population
#each subpopulation is processed independently (unless interaction is simulated) in its own processor node or thread
#the no. of available processor nodes thus defines the total no. of subpopulations simulated in the model
nsubpopulations = 1
#defines the no. of super-individuals in a parallel processing cluster
#super individuals represent an scaled entity of individuals - which, in simulation terms, 'behaves' similarly
#for more info. read: Scheffer et al., (1995) :: https://doi.org/10.1016/0304-3800(94)00055-M
nsupindividuals = 100000
#defines the maximum no. of timepoints in the annual cycle (varies on the temporal resolution of the model)
#here, a 6-hour temporal resolution is modelled (1460 x 6 = 8760) to speed things up - which gives room for simulate diel vertical migrations (dvm) at a 1/6 of the computational cost
#python is a 0-index program - so, this value should be trimmed to 1459 when running in a while loop
ntime = 1460
#defines the no. of calendar years that the simulation is run
nyears = 2
#defines the maximum simulated depth in the model environment
#here, a 1246-m water column is simulated as that has been known to contain the calanus finmarchicus during much of its life cycle (including the deep overwintering) in the norwegian sea
#for more info. read: Østvedt, (1955) :: https://imr.brage.unit.no/imr-xmlui/bitstream/handle/11250/109383/zooplankton_1955_ostvedt.pdf?sequence=3
ndepth = 1246
#defines the no. of individuals contained in a super individual (in some papers, this is termed the 'internal number': see https://doi.org/10.1080/17451000.2011.642805)
#at birth (i.e., seeding or spawning) this no is provided to a super individual by default - no selection bias as in the paper linked above
#the no of virtual individuals contained in a super individual cannot be increased at will, beacuse it will have drastic consequences in the population management (resource competition, poor genetic diversity, consumption biases etc.)
#the no. of virtual individuals can decrease due to mortality risk - as consumptive mortality directly operates on virtual individuals
nvindividualspersupindividual = 10000
#defines the no. of dynamic evolvable attributes in the model ('genes')
#nb:change this value when adding or removing 'genes' - otherwise the reproductive submodel will malfunction!!!
ndeattr = 8
#defines the no. of developmental stages in the model (including both sexes)
ndevelopmentalstage = 14
#defines the no of dimensions and spatio-temporal breaks of the model (= spatio-temporal resolution)
#the order is time(t, 6h), depth (z, var.m) longitude(x, 9.km), latitude(y, 9.km)
#nb: 3d environments are depth-integrated environments, such as the surface mixed layer depth
nresolution4D = "(1460, 37, 10, 10)"
nresolution3DA = "(1460, 10, 10)"
nresolution3DB = "(13, 23, 23)"
#defines the longitude, latitude and depth ratings of the x, y & z dimensions
#nb:the vertical resolution of the model is not uniform - but shaped by the depth grading of the environmental arrays
#there are 97 longitude and latitude grades and 37 depth grades, which are defined as:
longitudegrade = np.arange(start = 2.0000, stop = 2.7500, step = 0.083344, dtype = np.float32)
longitudegrade = np.append(longitudegrade, 2.7500)
latitudegrade = np.arange(start = 68.0000, stop = 68.7500, step = 0.083344, dtype = np.float32)
latitudegrade = np.append(latitudegrade, 68.7500)
depthgrade = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
#defines the developmental stage grading (n = 13), category#1 food concentration grading (n = 23) and temperature grading (n = 23) for slicing environmental-specific criticalmolting masses (<stage>, <f1con>, <temp>)
developmentalstagegrade = np.arange(start = 0, stop = 13, step = 1)
food1concentrationgrade = np.arange(start = 1, stop = 24, step = 1)
temperaturegrade = np.arange(start = -2, stop = 21, step = 1)

#state variables
#________________

#these are reflective of individual states and vary during the lifespan of super individuals depending on the individual-environment interactions and internal processes (e.g., hardcoded strategies)
#defines the living (1) or dead (0) state of super individuals
lifestatus = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the no. of virtual individuals contained in a super individual - this no. is defined by the constant, 'nvindividualspersupindividual' above
nvindividuals = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the developmental stage of super individuals:
#0:Egg, 1:NI, 2:NII, 3:NIII, 4:NIV, 5:NV, 6:NVI, 7:CI, 8:CII, 9:CIII, 10:CIV, 11:CV, 12:CVI-F, 13:CVI-M
developmentalstage = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the mean temperature trajectory encountered during the early lifestages
#nb:this is obsolete beyond non-feeding stages, whose development is estimated as a function of growth
thermalhistory = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the structural body mass of the super individual (min = 0.23 ugC at embryonic stage), initializes with 0.00
structuralmass = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the maximum lifetime structural mass of a super individual (used for starvation risk estimation)
maxstructuralmass = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the energy reserve mass of the super individual (max = 0.70 x structural mass), initializes with 0.00
reservemass = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the age of the super individual
age = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the sex of the super individual (M:male, F:female, U:undefined)
sex = np.repeat("U", nsubpopulations * nsupindividuals).astype("U1").reshape(nsupindividuals, nsubpopulations)
#defines the time of diapause entry of the super individual
timeofdiapauseentry = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the time of diapause exit of the super individual
timeofdiapauseexit = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the structural body mass at diapause entry
structuralmassatdiapauseentry = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the energy reserve mass at diapause entry
reservemassatdiapauseentry = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the developmental stage at diapause entry
developmentalstageatdiapauseentry = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the depth of diapause
diapausedepth = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the state of diapause of CIV and CV individuals ("A": active, "E": entry, "D":diapause, "X":exit, "P": post, "U": undefined)
#the "U1" datatype stores one unicode character in each slot (index position)
diapausestate = np.repeat("U", nsubpopulations * nsupindividuals).astype("U1").reshape(nsupindividuals, nsubpopulations)
#defines the energy reserve mass at diapause exit
#defines whether the super individual will enter diapause and then potentially molt to the adult or potentilly develop directly to adulthood without diapause
diapausestrategy = np.repeat(-1, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#nb:no need to track structural mass or developmental stage at diapsue exit as those do not change
reservemassatdiapauseexit = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the insemination state of females (0: not inseminated, 1: inseminated)
inseminationstate = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the energy allocated to reproductive output
reproductiveallocation = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the male genome copied to a female after mating (this is a 3D array!)
malegenome = np.repeat(0.00, nsubpopulations * nsupindividuals * ndeattr).astype(np.float32).reshape(nsupindividuals, nsubpopulations, ndeattr)
#defines the total no. of eggs produced by a female during its lifespan
#nb:not all of these eggs are spawned into the super individual pool - only the reward-based ones (see below)
totalfecundity = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the no. of eggs procuced by a female at each timepoint
potentialfecundity = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#however, not all eggs may be spawned in the subpopulation because, there may be a competition for empty spaces
#if that is the case, a fecundity-propotional-selection further trims down this potential fecundity
realizedfecundity = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)
#defines the absolute x-position or longitudinal position of the inidividual (based on a relative rectangular coordinate grid - change datatype to 'float32' for absolute coordinate grid)
xpos = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the absolute y-position or longitudinal position of the inidividual (based on a relative rectangular coordinate grid - change datatype to 'float32' for absolute coordinate grid)
ypos = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the absolute z-position or vertical position of the individual, min = 1 m, max = ndepth, initializes with 0
zpos = np.repeat(0, nsubpopulations * nsupindividuals).astype(np.int32).reshape(nsupindividuals, nsubpopulations)

#these are state variable placeholders used to store the 'current' values of the above state variables
#additionally, there are three relative positional placeholders (indices) for xpos, ypos and zpos
currentlifestatus = 0; currentdevelopmentalstage = 0; currentthermalhistory = 0.00; currentnvindividuals = 0; currentstructuralmass = 0.00; currentmaxstructuralmass = 0.00; currentreservemass = 0.00; currentage = 0
currentsex = "U"; currentdiapausedepth = 0; currentdiapausestate = "U"; currentdiapausestrategy = 0; currentxpos = 1; currentypos = 1; currentzpos = 1; currentxidx = 0; currentyidx = 0; currentzidx = 0
currentreservemassatdiapauseentry = 0.00; currentinseminationstate = 0; currentreproductiveallocation = 0.00; currentpotentialfecundity = 0.00; currenttotalfecundity = 0; currenttotalmass = 0.00

#evolvable attributes ('genes')
#______________________________

#these are attributes whose values freely evolve across time and space as the model is iteratively computed
#no artificial forcing is applied to optimize the free attribute combination - it is dependent on the 'simulated natural selection' that happens within the model
#all evolvable attributes range from 0 - 1 in floating point designation
#defines the body size trajectory that a super individual follows during its lifespan
#nb: pascalv4 does not support p2sensitivity or p2reactivity attributes - these can be included in future developments
a1_bodysize = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the spectral sensitivity of a given super individual
a2_irradiancesensitivity = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the visual predator sensitivity (i.e., the ability of a super individual to percieve a visual predator in its environment)
a3_pred1sensitivity = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the reactivity to visual predators
a4_pred1reactivity = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the energy allocation pattern of a given super individual
a5_energyallocation = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the probability of diapause entry of a given super individual (higher: likely to diapause, lower: less likely to diapause and more likely to develop directly to adulthood)
a6_diapauseprobability = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the timing of diapause entry of a given super individual
a7_diapauseentry = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)
#defines the timing of diapause exit of a given super individual
a8_diapauseexit = np.repeat(0.00, nsubpopulations * nsupindividuals).astype(np.float32).reshape(nsupindividuals, nsubpopulations)

#these are variable placeholders used to store the 'current' attribute values of the above 'genome'
current_a1_bodysize = 0.00; current_a2_irradiancesensitivity = 0.00; current_a3_pred1sensitivity = 0.00; current_a4_pred1reactivity = 0.00; current_a5_energyallocation = 0.00
current_a6_diapauseprobability = 0.00; current_a7_diapauseentry = 0.00; current_a8_diapauseexit = 0.00

#submodel-specific variables
#___________________________

#variables related to environmental submodel
#-------------------------------------------
#nb: most are initialized with "null" values, as those are placeholders for differnt data types
#nb: from these, the nc metadata holders and nc array slice holders are deleted immidiately after numpy nd array conversions to preserve system memory!
inputdatapath = None
#variables for storing nc file metadata
temperature_data_nc = None; food1concentration_data_nc = None; nsv_data_nc = None; esv_data_nc = None; smld_data_nc = None; irradiance_data_nc = None; pred1lightdep_data_nc = None; pred1dens_data_nc = None; cmm_data_nc = None
#variables for storing data arrays sliced from the nc files
temperature_nc = None; food1concentration_nc = None; nsv_nc = None; esv_nc = None; smld_nc = None; irradiance_nc = None; pred1lightdep_nc = None; pred1dens_nc = None; cmm_nc = None
#variable for storing the numpy nd array converts from the above
temperature = None; food1concentration = None; nsv = None; esv = None; smld = None; irradiance = None; pred1lightdep = None; pred1dens = None; cmm = None 
#these are environmental variable placeholders that stores the 'current' values of environmental data
#placeholders are initialized using their theoreitical minimum values
currenttemperature = -2.00; currentfood1concentration = 0.00; currentirradiance = 0.00; currentpred1dens = 0.00; currentesv = 0; currentnsv = 0; currentsmld = 1; currentpred1lightdep = 0
#these are environmental variable placeholders that stores 'potential' values of environmental data used in vertical position estimation
#nb: only temperature, foodconcentration category 1, irradiance and predationrisk category 1 is required in the vertical position estimation
currenttemperature_range = -2.00; currentfood1concentration_range = 0.00; currentirradiance_range = 0.00; currentpred1dens_range = 0.00
#accessory environmental variables for storing edge values
maxirradiance = 0.00; mintemperature = 0.00; maxtemperature = 0.00; maxfood1concentration = 0.00; mintemperature_idx = 0; maxtemperature_idx = 0; maxfood1concentration = 0
cmm_lower = None; cmm_upper = None

#variables related to the seeding, spawning and population management submodel
#-----------------------------------------------------------------------------
#defines the no. of empty slots in the subpopulation to seed or spawn new super individuals into
nspaces = 0
#this is a vector containing the slot identities (row nos.) of empty spaces in the subpopulation
spacesid = None
#defines the slot identities (row nos.) of empty spaces of the subpopulation that are randomly selected to seed or spawn new super individuals
replacementid = None
#defines the identities (index positions, row nos.) of empty spaces of the subpopulation allocated for seeding (subset of replacementid)
seedingid = None
#defines the identities (index positions, row nos.) of empty spaces of the subpopulation allocated for spawning (subset of replacementid)
spawningid = None
#defines the identities - with repeated entities (index positions, row nos.) of the females that have spawned
parentid = None
#defnes the unique parent identities of the females that have spawned
parentid_unique = None
#this is the seeding rate
seedingrate = 50
#this is the placeholder for the seeding rate
nseeds = 0
#this is the iterator for spawns (for the spawning loop)
currentspawn = 0
#this is the iterator for genes (for the spawning loop)
currentdeattr = 0
#this is the total no. of super individuals spawned at a given time
#i.e., the sum of the realized fecundity #1 of the subpopulation at a given time
nspawns = 0
#this is the index positions of those in the subpopulation (a numpy array; tupel -> np.array conversion occurs in place: see below)
livingsupindividuals = None
#this is the number of living super individuals (i.e., the sum of 'livingsupindividuals')
nlivingsupindividuals = 0
#crossover threshold
cxthreshold = 0.70
#random crossover probability
cxprob = 0.00
#random crossover value
cxval = 0.00
#mutation threshold
muthreshold = 0.20
#mutation probability
muprob = 0.00
#these are the male, female and nexgen genomes that recombine and mutate during spawning
spawngenome_m = None
spawngenome_f = None
spawngenome_n = None
#this is the spatially-integrated total subpopulation size at a given time
totalsubpopulationsize = 0

#variables related to growth and development submodel
#----------------------------------------------------
currentdevelopmentaltime = 0.00; currentmeandevelopmentaltime = 0.00
developmentalcoefficient = np.array([595.00, 388.00, 581.00]).astype(np.float32)
currentdevelopmentalcoefficient = 0.00
currentgrowthrate = 0.00
currentmaxzdistance = 0; currentactualzdistance = 0
currentadultsize = 0.00

#variables related to the survival submodel
#------------------------------------------
bgmortalityrisk = 0.00
pred2risk = 0.00
currentmortalityrisk = 0.00

#variables related to reproduction and spawning submodel
#-------------------------------------------------------
nmales = 0
maleids = None
selectedmale = 0
eggmass = 0.23
sexdet = None
dsdet = None

#variables related to time-tracking and book-keeping
#___________________________________________________
#these are start and end times of the model execution, respectively (in GMT)
execstarttime_rec = None
execstarttime_prt_prt = None
execendtime_rec = None
execendtime_prt = None

#these are the time intervals (1st of each month in the calendar year) and their labels that a progress report of the simulation is produced
reportingintervals = np.array([0, 124, 236, 360, 480, 604, 724, 848, 972, 1216, 1336, 1459]).astype(np.int32)
reportinlabels = np.array(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]).astype("U1")

#these are the total no. of reporting intervals, its current value and % progress for progress reporting
nreportingintervals = reportingintervals.size * (nyears + 1)
currentreportinginterval = 0
currentreportinglabel = "U"
currentprogress = 0

#this is the stage-, time- and space-specific population size tracker (developmental stage order: E, NI-NVI, CI-CV, AF, AM)
#obs: very large array - beware of system memory overload (e.g.,  302 Mb for 14 x 10 x 10 x 37 x 1460)
spatialdistribution_ps = np.repeat(0,  ndevelopmentalstage * longitudegrade.size * latitudegrade.size * depthgrade.size * ntime).astype(np.int32).reshape(ndevelopmentalstage, longitudegrade.size, latitudegrade.size, depthgrade.size, ntime)

#this is the stage-, time- and space-specific biomass tracker (developmental stage order: E, NI-NVI, CI-CV, AF, AM)
#obs: very large array - beware of system memory overload (e.g.,  302 Mb for 14 x 10 x 10 x 37 x 1460)
spatialdistribution_bm = np.repeat(0.00,  ndevelopmentalstage * longitudegrade.size * latitudegrade.size * depthgrade.size * ntime).astype(np.float32).reshape(ndevelopmentalstage, longitudegrade.size, latitudegrade.size, depthgrade.size, ntime)

#this is the stage-, time- and space-specific life cycle strategy tracker for integer parameters (3 parameters: parameter order: no.ddev, no.den, no.dex)
#obs: very large array - beware of system memory overload (e.g.,  302 Mb for 14 x 10 x 10 x 37 x 1460)
#spatialdistribution_lsi = np.repeat(0,  3 * longitudegrade.size * latitudegrade.size * depthgrade.size * ntime).astype(np.int32).reshape(3, longitudegrade.size, latitudegrade.size, depthgrade.size, ntime)

#this is the stage-, time- and space-specific life cycle strategy tracker for float parameters (5 parameters: parameter order: strmass.ddev, stomass.ddev, strmass.den, stomass.den, stomass.dex)
#obs: very large array - beware of system memory overload (e.g.,  302 Mb for 14 x 10 x 10 x 37 x 1460)
#spatialdistribution_lsf = np.repeat(0.00,  5 * longitudegrade.size * latitudegrade.size * depthgrade.size * ntime).astype(np.float32).reshape(5, longitudegrade.size, latitudegrade.size, depthgrade.size, ntime)

#space-integrated tracking of life-cycle strategies - integers only
#nb:this has five trackers (nddev, nden.civ, nden.cv, ndex.civ, ndex.cv)
lcstrategies_i = np.repeat(0, ntime * 5).astype(np.int32).reshape(ntime, 5)

#space-integrated tracking of life-cycle strategies - floating point numbers only
#nb:this has five trackers (strmass.ddev, stomasss.ddev, strmass.den.civ, strmass.den.cv, stomass.den.civ, stomass.den.cv, stomass.dex.civ, stomass.dex.cv)
lcstrategies_f = np.repeat(0.00, ntime * 8).astype(np.float32).reshape(ntime, 8)

#placeholder for h-stacking for output writing
lcstrategies = None
#placeholder for h-stacked output convertable to a pandas dataframe for easier output processing
lcstrategies_pd = None

#path for writing datafiles
#outputpath = pathlib.Path("/Users/kanchana/Documents/CURRENTRESEARCH/MIGRATORYCROSSROADS/OUTPUTDATA/")
outputpath = pathlib.Path("/Users/kanchana/Documents/CURRENTRESEARCH/MIGRATORYCROSSROADS/OUTPUTDATA/")
outputfolder = None

#iterators
#_________

#iterators are placeholders to store the currently iterating value updates to variables
#iterators inside closed loops [for()] are initialized as 0s, whereas those inside open loops [while()] are initialized as 1
currentsubpopulation = 0; currentsupindividual = 0; currenttime = -1; currentyear = 0

#boolean looping conditions
#__________________________

#these are switchable boolean states that defines the duration that a while() loop is run in the model
#timecondition defines the state that the while loop is run under; when 'False', the loop exits
timecondition = True
usercondition = True
ageceiling = 2180
fecundityceiling = 1000
datalogger = 0

#welcome text
#=========================================================================================================================================================================================================================================================

print("")
termcolor.cprint(text = "Pan-Arctic Behavioural and Life-history Simulator for Calanus, PASCAL version 4.00", color = "cyan")
termcolor.cprint(text = "Kanchana Bandara et al. | NFR Migratory Crossroads 2024-2027", color = "cyan")
termcolor.cprint(text = "Evaluation execution for functionality testing and debugging", color = "cyan")
termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")
print("")
termcolor.cprint(text = "enter a unique identifier for the execution (e.g., pascalv4_r001):", color = "light_red")
outputfolder = input("TYPE ID HERE AND PRESS ENTER: ")
termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")
print("")
execstarttime_rec = datetime.now()
execstarttime_prt = strftime("%Y-%m-%d %H:%M:%S", gmtime())
termcolor.cprint(text = f"\nexecution started at: {execstarttime_prt} GMT", color = "light_blue")
termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")

#model initialization
#=========================================================================================================================================================================================================================================================

#4D environmental submodel
#_________________________

#nb:for a model running on a singular annual circuit, it is possible to nest the environmental submodel outside the itreative structure
#nb:for models running a timeseries, nest the environmental submodel within the time loop, where a new annual circuit is loaded at the onset of each calendar year 
#imports 4D datafiles containing model environments (temperature, seawater velocity & concentration of food source #1) as netcdf4 files
#the surface mixed layer depth datafile lacks a z-dimension (hence 3D) 
#the stage-specific critical molting mass datafile has different dimensions (<stage>, <food1con>, <temperature>)

#defines the path to the data files (switch on or off the desired path depending on the workstation - i.e., mac or linux)
inputdatapath = pathlib.Path("/Users/kanchana/Documents/CURRENTRESEARCH/MIGRATORYCROSSROADS/INPUTDATA/4DDF_12022024")
#inputdatapath = pathlib.Path("/home/kanchana/Research/MigratoryCrossroads/INPUTDATA/4DDF_12022024")

#executes the model only if the path for input data is valid
if inputdatapath.exists():

    print("")
    print("checking for input data path...")
    termcolor.cprint(text = "the input datapath exists", color = "light_green")
    print("____________________________________________________________________________________")

else:

    print("")
    print("checking for input data path...")
    termcolor.cprint(text = "referencing error - check input data path!", color = "light_red")
    termcolor.cprint(text = "the model execution is terminated!", color = "light_red")
    sys.exit(1)

#end if

#execute the model only if the path for output data is valid
if outputpath.exists():

    print("")
    print("checking for output data path...")
    termcolor.cprint(text = "the output datapath exists", color = "light_green")
    print("____________________________________________________________________________________")

else:

    print("")
    print("checking for output data path...")
    termcolor.cprint(text = "referencing error - check output data path!", color = "light_red")
    termcolor.cprint(text = "the model execution is terminated!", color = "light_red")
    sys.exit(1)

#end if

#loads the netcdf metadata for each environmental variable
temperature_data_nc = nc.Dataset(inputdatapath / "temperature4di.nc", mode = "r")
food1concentration_data_nc = nc.Dataset(inputdatapath / "chl4di.nc", mode = "r")
nsv_data_nc = nc.Dataset(inputdatapath / "vo4di.nc", mode = "r")
esv_data_nc = nc.Dataset(inputdatapath / "uo4di.nc", mode = "r")
smld_data_nc = nc.Dataset(inputdatapath / "smld3di_v2.nc", mode = "r")
irradiance_data_nc = nc.Dataset(inputdatapath / "irradiance4di.nc")
pred1lightdep_data_nc = nc.Dataset(inputdatapath / "pred1lightdep4di.nc")
pred1dens_data_nc = nc.Dataset(inputdatapath / "pred1dens4di.nc")
cmm_data_nc = nc.Dataset(inputdatapath / "cmm3di.nc", mode = "r")

#extracts the data variable in each netcdf file
#converts the nc files into numpy ndarrays
#there is a dimension movement happening after importing to python (i.e., lat, lon, depth, time -> time, depth, lon, lat)
#the netcdf datafile is deleted immidiately for better memory management
#the surface mixed layer depth (smld) has a type conversion from float32 to np.int32 for easier indexing (decimals are truncated not rounded up or down!)
temperature_nc = temperature_data_nc.variables["temperature"][:, :, 0:10, 0:10]
temperature = np.array(temperature_nc, dtype = np.float32)
del temperature_nc
del temperature_data_nc

food1concentration_nc = food1concentration_data_nc.variables["chl"][:, :, 0:10, 0:10]
food1concentration = np.array(food1concentration_nc, dtype = np.float32)
del food1concentration_nc
del food1concentration_data_nc

nsv_nc = nsv_data_nc.variables["vo"][:, :, 0:10, 0:10]
nsv = np.array(nsv_nc, dtype = np.float32)
del nsv_nc
del nsv_data_nc

esv_nc = esv_data_nc.variables["uo"][:, :, 0:10, 0:10]
esv = np.array(esv_nc, dtype = np.float32)
del esv_nc
del esv_data_nc

smld_nc = smld_data_nc.variables["smld"][:, 0:10, 0:10]
smld = np.array(smld_nc, dtype = np.float32)
del smld_nc
del smld_data_nc

irradiance_nc = irradiance_data_nc.variables["par"][:, :, 0:10, 0:10]
irradiance = np.array(irradiance_nc, dtype = np.float32)
del irradiance_nc
del irradiance_data_nc

pred1lightdep_nc = pred1lightdep_data_nc.variables["parrs"][:, :, 0:10, 0:10]
pred1lightdep = np.array(pred1lightdep_nc, dtype = np.float32)
del pred1lightdep_nc
del pred1lightdep_data_nc

pred1dens_nc = pred1dens_data_nc.variables["p1risk"][:, :, 0:10, 0:10]
pred1dens = np.array(pred1dens_nc, dtype = np.float32)
del pred1dens_nc
del pred1dens_data_nc

cmm_nc = cmm_data_nc.variables["cmmu"][:]
cmm = np.array(cmm_nc, dtype = np.float32)
del cmm_nc
del cmm_data_nc

#execute the model only if environmental arrays are in correct dimension
#the execution is a user decision
print("")
print("checking dataset dimensions...")
print(f"\ncheck the dimensions of the 4D model environment <TIME, DEPTH, LONGITUDE, LATITUDE>: ")
print(f"envdim4D temperature       : {temperature.shape}")
print(f"envdim4D food1concentration: {food1concentration.shape}")
print(f"envdim4D nsv               : {nsv.shape}")
print(f"envdim4D esv               : {esv.shape}")
print(f"envdim4D irradiance        : {irradiance.shape}")
print(f"envdim4D pred1lightep      : {pred1lightdep.shape}")
print(f"envdim4D pred1dens         : {pred1dens.shape}")
termcolor.cprint(text = f"envdim4D expected          : {nresolution4D}", color = "light_red")
print(f"\ncheck the dimensions of the 3D model environment - Class A <TIME, LONGITUDE, LATITUDE>")
print(f"envdim3DA smld             : {smld.shape}")
termcolor.cprint(text = f"envdim3DA expected         : {nresolution3DA}", color = "light_red")
print(f"\ncheck the dimensions of the 3D model environment - Class B <DEVSTAGE, F1CONRANGE, TEMPRANGE>")
print(f"envdimD3B cmm              : {cmm.shape}")
termcolor.cprint(text = f"envdim3DB expected         : {nresolution3DB}\n", color = "light_red")

while usercondition:

    termcolor.cprint(text = "check for dimensional mismatches and decide:", color = "light_red")
    userinput = input("PROCEED? Y/N: ")

    if userinput == "Y" or userinput == "y" or userinput == "1":
        
        print("")
        print("model execution proceeds...")
        print("____________________________________________________________________________________")
        print("")
        termcolor.cprint(text = "[MODEL RESOLUTION]", color = "light_red")
        termcolor.cprint(text = f" nxcoords: {longitudegrade.size}")
        termcolor.cprint(text = f" nycoords: {latitudegrade.size}")
        termcolor.cprint(text = f" nzcoords: {depthgrade.size}")
        termcolor.cprint(text = f" ntcoords: {ntime}")
        termcolor.cprint(text = f" nsupinds: {nsupindividuals}")
        termcolor.cprint(text = f" nvirinds: {nvindividualspersupindividual}")
        termcolor.cprint(text = f" nsubpops: {nsubpopulations}")
        print("____________________________________________________________________________________")
        print("")
        termcolor.cprint(text = "[SIMULATION IN PROGRESS]", color = "light_red")
        print(" ")
        usercondition = False

    elif userinput == "N" or userinput == "n" or userinput == "0":

        print("____________________________________________________________________________________")
        print("")
        termcolor.cprint(text = "model execution terminated by the user!", color = "light_red")
        sys.exit(1)

    #end if
        
#end while

#this estimates the maximum ambient shortwave irradiance of the model environment (used in the vertical position estimation)
maxirradiance = np.nanmax(irradiance)
#this estimates the ceiling of the minimum temperature across the upper pelagial (where development of super individuals usually takes place): depth cutoff point is 200 m (depthgrade index = 25) as an integer
mintemperature = int(np.ceil(np.nanmin(temperature[:,0:25,:,:])))
#this estimates the ceiling of the maximum temperature across the upper pelagial (where development of super individuals usually takes place): depth cutoff point is 200 m (depthgrade index = 25) as an integer
maxtemperature = int(np.ceil(np.nanmax(temperature[:,0:25,:,:])))
#this estimates the ceiling maximum food1concentration across the upper pelagial (where development of super individuals usually takes place): depth cutoff point is 200 m (depthgrade index = 25) as an integer
maxfood1concentration = int(np.ceil(np.nanmax(food1concentration[:,0:25,:,:])))
#index position of the minimum temperature (with respect to the temperaturegrade numpy array) as a minimum absolute error function (mae)
mintemperature_idx = np.argmin(abs(temperaturegrade - mintemperature))
#index position of the maximum temperature (with respect to the temperaturegrade numpy array) as a minimum absolute error function (mae)
maxtemperature_idx = np.argmin(abs(temperaturegrade - maxtemperature))
#index position of maximum category#1 food concentration (with respect to the food1concentrationgrade numpy array) as a minimum absolute error function (mae)
maxfood1concentration_idx = np.argmin(abs(food1concentrationgrade - maxfood1concentration))

#extracting the stage-specific critical molting masses bounds for the given model environment (slicing: <devstage, food1con, temp>)
#nb:higher body masses are prevalent at lower temperatures, due to the temperature-size rule
cmm_lower = cmm[:, maxfood1concentration_idx, maxtemperature_idx]
cmm_upper = cmm[:, maxfood1concentration_idx, mintemperature_idx]

#three-tier iterative computing
#=========================================================================================================================================================================================================================================================

#simulation of behavioural and life-history strategies of C. finmarchicus using open-ended simulations
#simulation has three tiers: (i) sub-populations (ii) time and (iii) super individuals - from which the outermost tier is parallelly-processed in a work-sharing construct
#this is the outermost subpopulation loop where each subpopulation is processed in a unique processor node
#the no. of subpopulations depends on the no. of processor nodes available for the simulation
for currentsubpopulation in range(nsubpopulations):

    #time advances 1 ping at a time, where 1 ping is 6 hours in the model´s internal clock
    #for the ease of logging, time loop it is kept as an open-ended 'while()' loop
    #the open-ended loop runs for 1460 pings - then flips the calendar year, but the endpoint is trimmed because of 0-indexing (otherwise this loop runs for 1461 pings)
    while timecondition:

        #time advancing: initializes at -1, and becomes 0 at t = 0, runs upto 1459 pings
        #use a conditional on currenttime == ntime - 1 and currentyear == nyear to switch timecondition to 'false' and to quit out of the timeloop
        currenttime += 1

        if currenttime > ntime - 1:
        
            currenttime = 0
            currentyear += 1
            
            #activate data logging capability at the final calendar year of the simulation
            #nb:this needs to be changed when running timeseries or in real-time 
            datalogger = 1 if currentyear == nyears else 0

        #end if

        #progress reporting
        #__________________
        #prints a population size summary at the onset of each month
        if np.any(reportingintervals == currenttime):
            
            currentreportinginterval += 1
            currentreportinglabel = reportinlabels[np.array(np.where(reportingintervals == currenttime)).squeeze()]
            currentprogress = int(round(currentreportinginterval / nreportingintervals * 100, ndigits = 0))
            
            sleep(0.10)
            totalsubpopulationsize = np.sum(nvindividuals[:, currentsubpopulation])
            
            if currentprogress < 10:
                
                termcolor.cprint(text = f"[PROG:   {currentprogress}%] [MO: {currentreportinglabel}] [YR: {currentyear}] [ESTIMATED POPULATION SIZE: {totalsubpopulationsize}]")
                
            elif currentprogress >= 10 and currentprogress < 100:
                
                termcolor.cprint(text = f"[PROG:  {currentprogress}%] [MO: {currentreportinglabel}] [YR: {currentyear}] [ESTIMATED POPULATION SIZE: {totalsubpopulationsize}]")
            
            else:
                
                termcolor.cprint(text = f"[PROG: {currentprogress}%] [MO: {currentreportinglabel}] [YR: {currentyear}] [ESTIMATED POPULATION SIZE: {totalsubpopulationsize}]")
                
            #end if
            
            if currenttime == reportingintervals[-1]:
                
                print(" ")
                
            #end if
            
        #end if

        #within the time loop, the super individuals processed in a loop
        #before advancing into super individual loop, seeding and/or spawning happen(s)
        
        #pascal v4 :: seeding, spawning and population management submodel
        #_________________________________________________________________
        
        #seeding happens only during the 1st calendar year of the simulation (i.e., currentyear == 0)
        #spawning may occur at any given calendar year but generally absent during initial part of the first year (needs time to grow, develop, reproduce & spawn)
        #seeding, spawning and population management submodel cannot be modular because of heavy dependency on global variables
        #the model doesnt initialize without the seeding, spawning and population management submodel (goes on empty circuits, defined by the '0' lifestatus)
        
        #this calculates the no. of seeds to be released per time point (nb: seeding happens only in the 1st calendar year)
        nseeds = seedingrate if currentyear == 0 else 0
        #this calculates the no. of eggs to be spawned at this time point (nb: can happen at any given calendar year)
        nspawns = np.sum(potentialfecundity[:, currentsubpopulation])
        #this calculates the no. of empty spaces in the subpopulation at a given time
        nspaces = nsupindividuals - np.sum(lifestatus[:, currentsubpopulation])

        #this section seeds and/or spawns depending on the empty spaces available in the subpopulation
        if nspaces <= 0:

            #no empty spaces in the subpopulation to perform seeding or spawning
            #none of these eggs are seeded (nb: perhaps log this number!)
            nspawns = 0; nseeds = 0
            #nullify the potential fecundity and realized fecundity
            #nb:the super-individual-specific total fecundity calculates these unspawned eggs - and are partially involved in the breaking condition decision (see below)
            #nb:these can be logged but remember to factor in the eggs unspawned in the fps below
            potentialfecundity[:, currentsubpopulation] = 0
            realizedfecundity[:, currentsubpopulation] = 0
        
        else:

            #there are space(s) for seeding and/or spawning to occur
            #seeding and/or spawning may occur
            if nspaces >= nseeds + nspawns:
            
                #seeding and spawning may both occur without limitations
                #these are the identities (index positions) of the empty spaces in the subpopulation
                spacesid = np.array(np.where(lifestatus[:, currentsubpopulation] == 0)).squeeze()
                #these are the identities (index positions) replaced by newly seeded and/or spawned super individuals
                #this returns an empty array if nseed = 0 and nspawns = 0
                replacementid = np.random.choice(a = spacesid, size = nseeds + nspawns, replace = False)
                
                #seeding happens, if there are seeds to 'saw'
                #whether seeding does or doesn't happen has no impact on the spawning (the code relies on 'nseeds' but it is written to have no impact on spawning even when 'nseeds' = 0)
                if nseeds > 0:
                
                    #seeding
                    #-------
                    #this seeds uniformly random values for dynamic evolvable attributes ('genes')
                    #nb:this code works even when nseeds = 0 and/or nspawns = 0; only the replacement becomes null (i.e., [])
                    #these are the index positions of super individual slots that will be filled-in during seeding (i.e., identities of newly seeded super individuals)
                    seedingid = replacementid[0:nseeds]
                    #random seeding of dynamic evolvable attributes (this works even of the 'nseeds = 0')
                    a1_bodysize[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                    a2_irradiancesensitivity[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                    a3_pred1sensitivity[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                    a4_pred1reactivity[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                    a5_energyallocation[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                    a6_diapauseprobability[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                    a7_diapauseentry[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                    a8_diapauseexit[seedingid, currentsubpopulation] = np.random.rand(nseeds)
                
                #end if

                if nspawns > 0:
                
                    #spawning
                    #--------                  
                    #this is complicated than random seeding, because male and female genomes should interact (recombine) and shift (mutation)
                    #these are the index positions of super individual slots that will be filled-in during spawning
                    spawningid = replacementid[nseeds:nseeds + nspawns]
                    
                    #these are the identities (index positions) of the inseminated females that are producing eggs (unique and repeated - see comments below)
                    #obs:one female can produce several eggs, and thus its parent id must be replicated to reflect the no. of eggs produced
                    parentid_unique = np.array(np.where(potentialfecundity[:, currentsubpopulation] > 0)).squeeze()
                    parentid = np.repeat(parentid_unique, potentialfecundity[parentid_unique, currentsubpopulation])

                    #extracting the genome of spawning males ('gene' values stored during insemination)
                    #nb:this is a 2d numpy array (no hanging index on the 1st dimension: 0-1-2)
                    spawngenome_m = malegenome[parentid, currentsubpopulation, :]

                    #extracting the genome of spawning females ('gene' values are stored in their respective 'genes')
                    #the shape of the numpy nd array must be defined first before storing the female genome
                    spawngenome_f = np.repeat(0.00, nspawns * ndeattr).reshape(nspawns, ndeattr)
                    
                    #filling in the extracted female 'gene' values in the numpy nd array
                    spawngenome_f[:, 0] = a1_bodysize[parentid, currentsubpopulation]
                    spawngenome_f[:, 1] = a2_irradiancesensitivity[parentid, currentsubpopulation]
                    spawngenome_f[:, 2] = a3_pred1sensitivity[parentid, currentsubpopulation]
                    spawngenome_f[:, 3] = a4_pred1reactivity[parentid, currentsubpopulation]
                    spawngenome_f[:, 4] = a5_energyallocation[parentid, currentsubpopulation]
                    spawngenome_f[:, 5] = a6_diapauseprobability[parentid, currentsubpopulation]
                    spawngenome_f[:, 6] = a7_diapauseentry[parentid, currentsubpopulation]
                    spawngenome_f[:, 7] = a8_diapauseexit[parentid, currentsubpopulation]

                    #this is the genome of newly spawned super individuals
                    spawngenome_n = np.repeat(0.00, nspawns * ndeattr).reshape(nspawns, ndeattr)

                    #crossover and mutation algorithms
                    for currentspawn in range(nspawns):

                        for currentdeattr in range(ndeattr):
                            
                            #this is the crossover probability per-gene (if this value is lower than crossover threshold, then crossover occurs)
                            cxprob = np.random.rand(1).squeeze()
                            #this is mutation probability per-gene (if this value is lower than the mutation threshold, then mutation occurs)
                            muprob = np.random.rand(1).squeeze()
                            #this is the blend value per-gene in BLX-alpha algorithm
                            #nb: see Tkahashi et al, (2001) 10.1109/CEC.2001.934452
                            cxval = np.random.rand(1).squeeze()

                            #crossover algorithm (BLX-alpha)
                            if cxprob < cxthreshold:
                            
                                #if the crossover threshold is met, then male and female genomes are blended with BLX-alpha crossover 
                                spawngenome_n[currentspawn, currentdeattr] = (cxval * spawngenome_f[currentspawn, currentdeattr]) + ((1.00 - cxval) * spawngenome_m[currentspawn, currentdeattr])
                            
                            else:

                                #otherwise, the female genome is inherited without blending
                                spawngenome_n[currentspawn, currentdeattr] = spawngenome_f[currentspawn, currentdeattr]

                            #end if

                            #mutation algorithm (random replacement)
                            #nb:the <else> condition is not written because, no mutation does not change the genome 
                            if muprob < muthreshold:
                            
                                spawngenome_n[currentspawn, currentdeattr] = np.random.rand(1).squeeze()
                            
                            #end if
                        
                        #end for
                    
                    #end for
                    
                    #initializing dynamic evolvable attributes
                    #dynamic evolvable attributes ('genes') are initialized based to their seeding/spawning configurations
                    a1_bodysize[spawningid, currentsubpopulation] = spawngenome_n[:, 0]
                    a2_irradiancesensitivity[spawningid, currentsubpopulation] = spawngenome_n[:, 1]
                    a3_pred1sensitivity[spawningid, currentsubpopulation] = spawngenome_n[:, 2]
                    a4_pred1reactivity[spawningid, currentsubpopulation] = spawngenome_n[:, 3]
                    a5_energyallocation[spawningid, currentsubpopulation] = spawngenome_n[:, 4]
                    a6_diapauseprobability[spawningid, currentsubpopulation] = spawngenome_n[:, 5]
                    a7_diapauseentry[spawningid, currentsubpopulation] = spawngenome_n[:, 6]
                    a8_diapauseexit[spawningid, currentsubpopulation] = spawngenome_n[:, 7]
                
                #end for

            else:

                #here, the available spaces in the population is lesser than the no. of seeds and spawns
                #seeding is ceased - spawning proceeds fully or partly
                nseeds = 0
                #if spawning happens fully, then no competition between inseminated females; if not, inseminated females compete for egg-placement via a fecundity-proportional selection process

                if nspaces >= nspawns:
                
                    #no competition for egg placement in the new generation - spawning proceeds without constraints
                    #these are the identities (index positions) of the empty spaces in the subpopulation
                    spacesid = np.array(np.where(lifestatus[:, currentsubpopulation] == 0)).squeeze()
                    #these are the identities (index positions) replaced by newly spawned super individuals
                    #nb:no seeding at this point!
                    replacementid = np.random.choice(a = spacesid, size = nspawns, replace = False)
                    
                    if nspawns > 0:
                        
                        #these are the identities (index positions) of the inseminated females that are producing eggs (unique and repeated - see comments below)
                        #obs:one female can produce several eggs, and thus its parent id must be replicated to reflect the no. of eggs produced
                        parentid_unique = np.array(np.where(potentialfecundity[:, currentsubpopulation] > 0)).squeeze()
                        parentid = np.repeat(parentid_unique, potentialfecundity[parentid_unique, currentsubpopulation])
                        
                    #end if
                
                else:

                    #competition for egg placement in the new generation - spawning proceeds with constraints
                    #fecundity-proportional selection (FPS)
                    #this writes the FPS output into the realized fecundity state variable
                    #nb:the np.floor() is taken instead of np.round() beacuse the latter bares the risk of the nspawns (i.e., sum(realizedfecundity)) becoming higher than the available empty spaces ('nspaces')
                    realizedfecundity[:, currentsubpopulation] = np.floor((potentialfecundity[:, currentsubpopulation] / nspawns) * nspaces).astype(np.int32)
                    #this the no. of spawns is redefined by the realized fecundity as:
                    nspawns = np.sum(realizedfecundity[:, currentsubpopulation])

                    #these are the identities (index positions) of the empty spaces in the subpopulation
                    spacesid = np.array(np.where(lifestatus[:, currentsubpopulation] == 0)).squeeze()
                    #these are the identities (index positions) replaced by newly spawned super individuals
                    #nb:no seeding at this point!
                    replacementid = np.random.choice(a = spacesid, size = nspawns, replace = False)
                    
                    if nspawns > 0:
                    
                        #these are the identities (index positions) of the inseminated females that are producing eggs (unique and repeated - see comments below)
                        #obs:one female can produce several eggs, and thus its parent id must be replicated to reflect the no. of eggs produced
                        parentid_unique = np.array(np.where(realizedfecundity[:, currentsubpopulation] > 0)).squeeze()
                        parentid = np.repeat(parentid_unique, realizedfecundity[parentid_unique, currentsubpopulation])
                        
                    #end if
                    
                #end if
                
                if nspawns > 0:
                
                    #spawning
                    #--------
                    #this is complicated than random seeding, because male and female genomes should interact (recombine) and shift (mutation)
                    #these are the index positions of super individual slots that will be filled-in during spawning
                    spawningid = replacementid[:]

                    #extracting the genome of spawning males ('gene' values stored during insemination)
                    #nb:this is a 2d numpy array (no hanging index on the 1st dimension: 0-1-2)
                    spawngenome_m = malegenome[parentid, currentsubpopulation, :]

                    #extracting the genome of spawning females ('gene' values are stored in their respective 'genes')
                    #the shape of the numpy nd array must be defined first before storing the female genome
                    spawngenome_f = np.repeat(0.00, nspawns * ndeattr).reshape(nspawns, ndeattr)

                    #filling in the extracted female 'gene' values in the numpy nd array
                    spawngenome_f[:, 0] = a1_bodysize[parentid, currentsubpopulation]
                    spawngenome_f[:, 1] = a2_irradiancesensitivity[parentid, currentsubpopulation]
                    spawngenome_f[:, 2] = a3_pred1sensitivity[parentid, currentsubpopulation]
                    spawngenome_f[:, 3] = a4_pred1reactivity[parentid, currentsubpopulation]
                    spawngenome_f[:, 4] = a5_energyallocation[parentid, currentsubpopulation]
                    spawngenome_f[:, 5] = a6_diapauseprobability[parentid, currentsubpopulation]
                    spawngenome_f[:, 6] = a7_diapauseentry[parentid, currentsubpopulation]
                    spawngenome_f[:, 7] = a8_diapauseexit[parentid, currentsubpopulation]

                    #this is the genome of newly spawned super individuals
                    spawngenome_n = np.repeat(0.00, nspawns * ndeattr).reshape(nspawns, ndeattr)

                    #crossover and mutation algorithms
                    for currentspawn in range(nspawns):

                        for currentdeattr in range(ndeattr):
                            
                            #this is the crossover probability per-gene (if this value is lower than crossover threshold, then crossover occurs)
                            cxprob = np.random.rand(1).squeeze()
                            #this is mutation probability per-gene (if this value is lower than the mutation threshold, then mutation occurs)
                            muprob = np.random.rand(1).squeeze()
                            #this is the blend value per-gene in BLX-alpha algorithm
                            #nb: see Tkahashi et al, (2001) 10.1109/CEC.2001.934452
                            cxval = np.random.rand(1).squeeze()

                            #crossover algorithm (BLX-alpha)
                            if cxprob < cxthreshold:
                            
                                #if the crossover threshold is met, then male and female genomes are blended with BLX-alpha crossover 
                                spawngenome_n[currentspawn, currentdeattr] = (cxval * spawngenome_f[currentspawn, currentdeattr]) + ((1.00 - cxval) * spawngenome_m[currentspawn, currentdeattr])
                            
                            else:

                                #otherwise, the female genome is inherited without blending
                                spawngenome_n[currentspawn, currentdeattr] = spawngenome_f[currentspawn, currentdeattr]

                            #end if

                            #mutation algorithm (random replacement)
                            #nb:the <else> condition is not written because, no mutation does not change the genome 
                            if muprob < muthreshold:
                            
                                spawngenome_n[currentspawn, currentdeattr] = np.random.rand(1).squeeze()
                            
                            #end if
                        
                        #end for
                    
                    #end for
                    
                    #initializing dynamic evolvable attributes
                    #dynamic evolvable attributes ('genes') are initialized based to their seeding/spawning configurations
                    a1_bodysize[spawningid, currentsubpopulation] = spawngenome_n[:, 0]
                    a2_irradiancesensitivity[spawningid, currentsubpopulation] = spawngenome_n[:, 1]
                    a3_pred1sensitivity[spawningid, currentsubpopulation] = spawngenome_n[:, 2]
                    a4_pred1reactivity[spawningid, currentsubpopulation] = spawngenome_n[:, 3]
                    a5_energyallocation[spawningid, currentsubpopulation] = spawngenome_n[:, 4]
                    a6_diapauseprobability[spawningid, currentsubpopulation] = spawngenome_n[:, 5]
                    a7_diapauseentry[spawningid, currentsubpopulation] = spawngenome_n[:, 6]
                    a8_diapauseexit[spawningid, currentsubpopulation] = spawngenome_n[:, 7]
                
                #end if

            #end if

            #initialization of state variables, loggers and trackers
            #nb:this is common for seeding and spawning (i.e., it uses the 'replacementid' for initialization)
            #nb:even when the replacement id is an empty array ('[]'), it has no impact on the replacements below - i.e., it doesnt initialize anything
            lifestatus[replacementid, currentsubpopulation] = 1
            developmentalstage[replacementid, currentsubpopulation] = 0
            thermalhistory[replacementid, currentsubpopulation] = 0.00
            nvindividuals[replacementid, currentsubpopulation] = nvindividualspersupindividual
            age[replacementid, currentsubpopulation] = 0
            #sex determination is random (50:50, M:F) and occurs when a super individual arrives at the adult stage (see below)
            #until then, the state variable is initialized as undefined "U" (vals: "U", "M", "F")
            sex[replacementid, currentsubpopulation] = "U"
            structuralmass[replacementid, currentsubpopulation] = eggmass
            maxstructuralmass[replacementid, currentsubpopulation] = eggmass
            reservemass[replacementid, currentsubpopulation] = 0.00
            timeofdiapauseentry[replacementid, currentsubpopulation] = 0
            developmentalstageatdiapauseentry[replacementid, currentsubpopulation] = 0
            timeofdiapauseexit[replacementid, currentsubpopulation] = 0
            structuralmassatdiapauseentry[replacementid, currentsubpopulation] = 0.00
            reservemassatdiapauseentry[replacementid, currentsubpopulation] = 0.00
            reservemassatdiapauseexit[replacementid, currentsubpopulation] = 0.00
            #diapause depth is selected randomly well below the permenant thermocline (of 700 m)
            #nb:this needs to change as the max depth varies on drift (coastal, offshore, shelfbreak)
            diapausedepth[replacementid, currentsubpopulation] = depthgrade[np.random.randint(low = 32, high = 36, size = nseeds + nspawns)]
            #diapause strategy determination is partially deterministic (partially random)
            #it is linked to the dynamic evolvable attribute 'a6_diapauseprobability' and occurs in place (see below)
            #until then, the state variable is initialized as -1 (vals: -1, 0, 1)
            diapausestrategy[replacementid, currentsubpopulation] = -1
            diapausestate[replacementid, currentsubpopulation] = "A"
            inseminationstate[replacementid, currentsubpopulation] = 0
            reproductiveallocation[replacementid, currentsubpopulation] =  0.00
            malegenome[replacementid, currentsubpopulation, :] = 0.00
            potentialfecundity[replacementid, currentsubpopulation] = 0
            realizedfecundity[replacementid, currentsubpopulation] = 0

            #for spawning, the position of offspring equals to the position of their mother
            #for seeding, it is random (or a fixed initiation point)
            #for starters, a fixed initiation point is given for all offspring irrespective of seeding/spawning
            xpos[replacementid, currentsubpopulation] = 2.0000
            ypos[replacementid, currentsubpopulation] = 68.0000
            zpos[replacementid, currentsubpopulation] = 1

            #resetting intermediate values
            nseeds = 0; nspawns = 0; nspaces = 0
            seedingid = None; spawningid = None; spacesid = None; replacementid = None; spawngenome_f = None; spawngenome_m = None; spawngenome_n = None

            #resetting state variables for egg production
            potentialfecundity[:, currentsubpopulation] = 0
            realizedfecundity[:, currentsubpopulation] = 0

        #end if

        #super individual level simulation
        #__________________________________
         
        #at each timepoint, growth & development, survival and reproduction of every super individual is simulated
        #the for loop skips if a super individual is dead or not seeded yet (given by the '0' lifestatus)
        #the no.of living super individuals and their index positions in the subpopulation needs to be inquired before running the loop
        #squeeze is mandatory to remove the hanging dimension of the numpy array
        livingsupindividuals = np.array(np.where(lifestatus[:, currentsubpopulation] == 1)).squeeze()
        nlivingsupindividuals = livingsupindividuals.size

        #to avoid iteration over zero-dim array errors, a condition check is performed at the beginning
        #the aim of this check is to not to proceed with the super individual loop until there are > 1 super individuals to process - otherwise, it returns nothing (skips the super-individual level iteration)
        # > 1 is used because of the male-female interaction, where at least one male and female should be present to continue the population (e.g., resembling a sort-of an effective population size)
        #also, running the super individual loop returns an error if it runs over 1 living super individual, because python loops do not run over scalars
        #this <else> condition is not written, as the super individual loop is processed within the <if> condition
        if nlivingsupindividuals > 1:
        
            #the super individual loop happens here (its an irregular index loop - because of the index positions of living super individuals are unpredictable)
            #if the living super individuals return an empty numpy array, the super individual loop returns nothing - and goes to the next time step 
            for currentsupindividual in livingsupindividuals:
                
                #nb:all super individuals that enters this loop are alive
                #nb:their index positions are given by "currentsupindividual" iterator
                
                #the x-y position tracking system operates here
                #outputs are currentxidx and currentyidx
                #to be coded for a basic functionality test and driven by opendrift in simulations
                #opening of state variable queries related to positional tracking (start position)
                currentxpos = xpos[currentsupindividual, currentsubpopulation]
                currentypos = ypos[currentsupindividual, currentsubpopulation]
                currentzpos = zpos[currentsupindividual, currentsubpopulation]
                
                #here goes the x-y position tracking system
                #<code here or use unidimensional fixation for testing>

                #the updated x-y positions are available at the end
                #x-y-z encoding (absolute value -> index) using the minimum absolute error (mae) approach for environmental variable array slicing
                currentxidx = np.argmin(abs(longitudegrade - currentxpos))
                currentyidx = np.argmin(abs(latitudegrade - currentypos))
                currentzidx = np.argmin(abs(depthgrade - currentzpos))

                #the growth & development, survival and reproductive simulation happens within this if() condition
                #no else() condition is written, as the loop skips if a super indivdual is dead or unseeded/uninitialized
                #extracts the developmental stage of the currently processed super individual
                currentdevelopmentalstage = developmentalstage[currentsupindividual, currentsubpopulation]

                #this structures the simulation into following developmental stage categories:
                #1. non-feeding egg, NI and NII stages (index: 0, 1, 2)
                #2. feeding but non-energy-storing NIII-NVI,CI-CIII stages (index: 3, 4, 5, 6, 7, 8, 9)
                #3. feeding and energy-storing CIV and CV (diapausing) stages (index: 10, 11)
                #4. adult females (index: 12)
                #5. adult males (index: 13)
                #nb:these stage groupings are for C. finmarchicus only - for C.glacialis and C.hyperboreus, stage compositions of some categories may vary

                #simulation of de-growth, development and survival of non-feeding stages (egg, NI, NII): dsc-I
                #______________________________________________________________________________________________

                if currentdevelopmentalstage <= 2:
                    
                    #this is developmental stage category - I (dsc-I)
                    #these developmental stages do not feed, so a general de-growth takes place over time
                    #their development is temperature dependent but food-independent

                    #opening state variable queries: extraction of 'current' internal states (from apropriate state variables) relevant to the stage category
                    #nb:not all state variables are inquired because some are not relevant to certain stage categories (e.g., state variables related to energy storage & diapause is not relevant until dsc-III)
                    currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                    currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                    currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                    currentage = age[currentsupindividual, currentsubpopulation]
                    currentthermalhistory = thermalhistory[currentsupindividual, currentsubpopulation]

                    #dsc-I :: diel and seasonal vertical migration submodel
                    #------------------------------------------------------
                    #this uses a module-driven function to estimate the relative (index) and absolute vertical positions of the super individual based on the developmental stage and environmental variables
                    #for non-feeding stages, the model assumes no individual vertical swimming capability, e.g., see: https://doi.org/10.1016/1054-3139(95)80062-X 
                    #their vertical position is thus assumed to be vary uniformly randomly within the surface mixed layer (but with some modification, see the "vm" module)
                    #this extracts the current surface mixed layer depth from the 3D array (<time, lon, lat>)
                    #nb:the mixed layer depth data are in np.float32 type, which needs to be converted to integers before proceeding further
                    currentsmld = int(smld[currenttime, currentxidx, currentyidx])
                    #this trims the endpoint of the current mixed layer depth to maximum depth to prevent issues
                    if currentsmld > ndepth:
                    
                        currentsmld = ndepth
                    
                    #end if
                    
                    #calls the vertical migration estimator function of the developmental stage category 1 (dsc1: egg, NI, NII)
                    currentzpos, currentzidx = vm.verticalmigration_dsc1(smld = currentsmld)

                    #dsc-I:growth and development submodel
                    #-------------------------------------
                    #extraction of apropriate environmental variables based on the current xidx, yidx and zidx
                    #data are extracted from a 4D numpy array (<time, depth, lon, lat>)
                    currenttemperature = temperature[currenttime, currentzidx, currentxidx, currentyidx]
                    #update the current thermal history (this is an arithmetic mean)
                    currentthermalhistory = (currentthermalhistory + currenttemperature) / 2.00

                    #this is the parameter "a" in Belehrádek’s (1935) temperature function, adopted from Campbel et al. (2001), see: https://doi.org/10.3354/meps221161
                    currentdevelopmentalcoefficient = developmentalcoefficient[currentdevelopmentalstage]
                    #this uses a module-driven function to estimate the growth and development rate of the super individual based on internal states and environmental variables 
                    #at egg, nauplius I & II only degrowth occurs (due to no feeding)
                    #the developmental and growth rates are solely dependent on ambient temperatures
                    #call the growth and development function for the dsc#1, which returns two outputs
                    #output units: 6h pings for developmental time; 6h estimate for growth rate - but it is negative, signifying degrowth
                    #nb:this degrowth rate is reduced the basal metabolic rate (= total metabolic rate at dsc-I)
                    currentdevelopmentaltime, currentgrowthrate = gd.growthanddevelopment_dsc1(temperature = currenttemperature, 
                                                                                            devcoef = currentdevelopmentalcoefficient,
                                                                                            thist = currentthermalhistory, 
                                                                                            strmass = currentstructuralmass, modelres = 6)

                    #this updates the structural mass of the super individual by adding the growth accumulated at 'currenttime'
                    #nb:despite the addition, this is effectively a substraction because of the negative growth output by the dsc-I growth function above
                    #nb:the "maxstructuralmass" does not need an update, because of structural degrowth during egg, and nauplii I & II
                    currentstructuralmass = currentstructuralmass + currentgrowthrate
                    
                    #dsc-I: age and developmental stage advancement
                    #----------------------------------------------
                    #this updates the age of the super individual
                    #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
                    currentage += 1

                    #if the currentage is greater than or equals to the mean developmental time sustained across the lifespan of the super individual, then the developmental stage advances
                    if currentage >= currentmeandevelopmentaltime:
                            
                        currentdevelopmentalstage += 1

                        #update state variable in-place (due to conditional state change)
                        developmentalstage[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage
                            
                    #end if

                    #dsc-I: survival submodel
                    #------------------------
                    #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                    currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                    #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                    currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                    #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                    currentmortalityrisk = sv.mortalityrisk_dsc1(strmass = currentstructuralmass, 
                                                                maxstrmass = currentmaxstructuralmass, 
                                                                devstage = currentdevelopmentalstage, 
                                                                p1dens = currentpred1dens,
                                                                p1lightdp = currentpred1lightdep, 
                                                                p2risk = pred2risk, 
                                                                bgmrisk = bgmortalityrisk)
                    #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                    #when all virtual individuals contained in a super individual dies, then the super individual also dies
                    #this death is simulated after the stage-specific processes
                    currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                    #update the non-conditional state variables (conditionally state changed state variables are update in-place, e.g., developmental stage - see above)
                    structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                    maxstructuralmass[currentsupindividual, currentsubpopulation] = currentmaxstructuralmass
                    nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                    age[currentsupindividual, currentsubpopulation] = currentage
                    thermalhistory[currentsupindividual, currentsubpopulation] = currentthermalhistory

                    xpos[currentsupindividual, currentsubpopulation] = currentxpos
                    ypos[currentsupindividual, currentsubpopulation] = currentypos
                    zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    #data logging
                    #------------
                    #nb:happens only if the 'datalogger' is enabled (val = 1)
                    if datalogger == 1:

                        #logging population size by sequentual addition
                        #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                        spatialdistribution_ps[currentdevelopmentalstage, currentxidx, currentyidx, currentzidx, currenttime] += currentnvindividuals
                        #logging biomass (gC) by sequential addition
                        #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                        currenttotalmass = (currentstructuralmass * currentnvindividuals) / 1e6
                        spatialdistribution_bm[currentdevelopmentalstage, currentxidx, currentyidx, currentzidx, currenttime] += currenttotalmass
                    
                    #end if

                #simulation of growth, development and survival of feeding but non-energy-storing stages (NIII, CIII): dsc-II
                #____________________________________________________________________________________________________________

                elif currentdevelopmentalstage >= 3 and currentdevelopmentalstage < 10:
                    
                    #these developmental stages feed, but do not channel the assimilated energy into an energy reserve
                    #their development is temperature and food dependent
                    #when they feed, it results in a propotional reduction in the food concentration (individual-environment feedbacks) :: <to be built>
                    #opening state variable queries: extraction of 'current' internal states (from apropriate state variables) relevant to the stage category
                    #nb:the cumulative development time state variable is discontinued from this stage category onwards (because henceforth, development is taken as a function of somatic growth)
                    currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                    currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                    currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                    currentage = age[currentsupindividual, currentsubpopulation]

                    #dsc-II: diel and seasonal vertical migration submodel
                    #-----------------------------------------------------
                    #this uses a modular function to estimate the relative (index) and absolute vertical positions of the super individual based on the developmental stage and environmental variables
                    #for feeding but non-energy-storing stages (dsc-III: NIII-CIII), the vertical position is estimated as a function of environmental variables (resource & risks) and super-individual-specific attribute values ('genes')

                    #this extracts relevant attribute ('gene') values:
                    current_a2_irradiancesensitivity = a2_irradiancesensitivity[currentsupindividual, currentsubpopulation]
                    current_a3_pred1sensitivity = a3_pred1sensitivity[currentsupindividual, currentsubpopulation]
                    current_a4_pred1reactivity = a4_pred1reactivity[currentsupindividual, currentsubpopulation]
                    
                    #this extracts the relevant time- and space-specific environmental variable ranges (<time>, <depth>, <longitude>, <latitude>)
                    #ranges are such that it includes data across entire depth range (nb: irregular depth intervals; see the "depthgrade" variable)
                    #indices used to slice these are encoded from absolute time, depth, longitude and latitude values in the spatial tracking submodel
                    currenttemperature_range = temperature[currenttime, :, currentxidx, currentyidx]
                    currentfood1concentration_range = food1concentration[currenttime, :, currentxidx, currentyidx]
                    currentirradiance_range = irradiance[currenttime, :, currentxidx, currentyidx]
                    currentpred1dens_range = pred1dens[currenttime, :, currentxidx, currentyidx]

                    #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
                    #calling the vertical position estimation function from the module
                    #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
                    currentzpos, currentzidx, currentmaxzdistance, currentactualzdistance = vm.verticalmigration_dsc2(temprange = currenttemperature_range, 
                                                                                                                    f1conrange = currentfood1concentration_range, 
                                                                                                                    iradrange = currentirradiance_range, 
                                                                                                                    maxirad = maxirradiance, 
                                                                                                                    p1dnsrange= currentpred1dens_range, 
                                                                                                                    a2 = current_a2_irradiancesensitivity, 
                                                                                                                    a3 = current_a3_pred1sensitivity, 
                                                                                                                    a4 = current_a4_pred1reactivity, 
                                                                                                                    pvp = currentzpos, 
                                                                                                                    pvi = currentzidx, 
                                                                                                                    strmass = currentstructuralmass, 
                                                                                                                    modelres = 6)

                    #dsc-II: growth and development submodel
                    #---------------------------------------
                    #extraction of apropriate environmental variables based on the current relative vertical position ("currrentzidx")
                    #no need for the data to be extracted from a 4D numpy array (<time, depth, lon, lat>), as the whole depth array is already sliced for vertical migration submodel input
                    currenttemperature = currenttemperature_range[currentzidx]
                    currentfood1concentration = currentfood1concentration_range[currentzidx]

                    #this function estimates the somatic growth rate, which is used in the calculation of development rate (= 1 / developmental time)
                    #only somatic growth (structural growth) occurs at this stage, no energy reserves are maintained
                    #the function takes ambient temperature and food concentration as environmental inputs and current structural mass and developmental stage as internal state inputs
                    currentgrowthrate = gd.growthanddevelopment_dsc2(temperature = currenttemperature, 
                                                                    f1con = currentfood1concentration, 
                                                                    strmass = currentstructuralmass, 
                                                                    maxzd = currentmaxzdistance, 
                                                                    actzd = currentactualzdistance, 
                                                                    modelres = 6)

                    #this updates the structural mass after growth no growth or degrowth
                    #despite the "+" operator, negative growth results in subtraction (signifying degrowth)
                    currentstructuralmass = currentstructuralmass + currentgrowthrate

                    #this updates the maximum lifetime structural mass (for starvation risk estimation) 
                    if currentstructuralmass > currentmaxstructuralmass:

                        currentmaxstructuralmass = currentstructuralmass
                        
                    #end if

                    #dsc-II: age and developmental stage advancement
                    #-----------------------------------------------
                    #this updates the age of the super individual
                    #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
                    currentage += 1

                    #the development rate or development time is not estimated by the above growth and development function
                    #however, it is calculated based on the somatic growth rate
                    #to estimate the molting state (i.e., whether a super individual is ready to molt to the next stage or not) of a super individual, the stage- and super-individual-specific critical molting mass is required
                    #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
                    #this extracts the 'gene' a1 (body size):
                    current_a1_bodysize = a1_bodysize[currentsupindividual, currentsubpopulation]
                    #this estimates the stage-specific critical molting mass based on the 'gene' a1:
                    currentcmm = cmm_lower[currentdevelopmentalstage] + (cmm_upper[currentdevelopmentalstage] - cmm_lower[currentdevelopmentalstage]) * current_a1_bodysize

                    #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
                    #the else condition is not mentioned here, as no stage increment occurs if the if() condition is invalid
                    if currentstructuralmass >= currentcmm:

                        #molting occurs and stage is updated in-place(due to conditional state change)
                        currentdevelopmentalstage += 1
                        developmentalstage[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage

                    #end if

                    #dsc-II: survival submodel
                    #-------------------------
                    #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                    currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                    #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                    currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                    
                    #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                    currentmortalityrisk = sv.mortalityrisk_dsc2(strmass = currentstructuralmass, maxstrmass = currentmaxstructuralmass, p1dens = currentpred1dens, p1lightdp = currentpred1lightdep, p2risk = pred2risk, bgmrisk = bgmortalityrisk)

                    #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                    #when all virtual individuals contained in a super individual dies, then the super individual also dies
                    currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                    #update the non-conditional state variables (conditionally state changed state variables are update in-place, e.g., developmental stage - see above)
                    structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                    maxstructuralmass[currentsupindividual, currentsubpopulation] = currentmaxstructuralmass
                    nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                    age[currentsupindividual, currentsubpopulation] = currentage

                    xpos[currentsupindividual, currentsubpopulation] = currentxpos
                    ypos[currentsupindividual, currentsubpopulation] = currentypos
                    zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    #data logging
                    #------------
                    #nb:happens only if the 'datalogger' is enabled (val = 1)
                    if datalogger == 1:

                        #logging population size by sequentual addition
                        #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                        spatialdistribution_ps[currentdevelopmentalstage, currentxidx, currentyidx, currentzidx, currenttime] += 1
                        #logging biomass (gC) by sequential addition
                        #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                        currenttotalmass = (currentstructuralmass * currentnvindividuals) / 1e6
                        spatialdistribution_bm[currentdevelopmentalstage, currentxidx, currentyidx, currentzidx, currenttime] += currenttotalmass
                    
                    #end if

                #simulation of growth, development and survival of feeding and energy-storing stages (CIV, CV): dsc-III
                #______________________________________________________________________________________________________

                elif currentdevelopmentalstage == 10 or currentdevelopmentalstage == 11:

                    #this is developmental stage category-IIII (dsc-III that includes energy-storing civ and cv stages)
                    #these are feeding stages that actively maintains an energy reserve
                    #the energy reserve is used for countering starvation mortality risk and for spending the unproductive part of the year in a diapause state
                    #depending on the state of diapause, this stage is split into 5 subcategories
                    #dsc-IIIA: active pre-diapause state; dsc-IIIE: active diapause entry state; dsc-IIID: diapause state; dsc-IIIX: active exit state; dscIII-P: active post-diapause state
                    #these subcategories have to be coded separately due to their intricate changes of physiology and behaviour 

                    #this opens the state variable query for diapause state
                    currentdiapausestate = diapausestate[currentsupindividual, currentsubpopulation]
                    #this opens the state variable query for diapause strategy
                    currentdiapausestrategy = diapausestrategy[currentsupindividual, currentsubpopulation]

                    #if the diapause strategy is undefined (-1: typical for newly seeded/spawned super individual arriving at civ/cv for the first time), define the diapause strategy (0, 1)
                    #nb:the diapause strategy is linked to the diapause probability 'gene'
                    if currentdiapausestrategy == -1:

                        #this inquires the diapause probability 'gene' for the super individual
                        current_a6_diapauseprobability = a6_diapauseprobability[currentsupindividual, currentsubpopulation]
                        #random number for diapause strategy determination
                        dsdet = np.random.rand(1).squeeze()
                        #falls into 0 (direct development, no diapause) or 1 (diapause) depending on the diapause probability 'gene' value
                        currentdiapausestrategy = 1 if dsdet < current_a6_diapauseprobability else 0
                        #update the state variable in-place
                        diapausestrategy[currentsupindividual, currentsubpopulation] = currentdiapausestrategy
                    
                    #end if

                    #this splits the dsc-III into subcategory-specific processing drives
                    if currentdiapausestate == "A":
                    
                        #this is the active pre-diapausing super individuals belonging to civ and cv stages
                        #they actively feed, grow and develop (civ-cv-adult) with (1-year life cycle)or without diapause (< 1 year life cycle)
                        #actively maintain an energy reserve for starvation compensation and diapause (if they diapause)
                        #they are subjected to two end pathways: (i)direct development into civ->cv->adult and onwards or (ii)become diapause entry stage (dsc-IIIE) and onwards

                        #this opens the state variable queries:
                        currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                        currentreservemass = reservemass[currentsupindividual, currentsubpopulation]
                        currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                        currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                        currentage = age[currentsupindividual, currentsubpopulation]

                        #dsc-IIIA: diel and seasonal vertical migration submodel
                        #-------------------------------------------------------
                        #this uses a modular function to estimate the absolute and relative vertical positions ("currentzpos", "currentzidx") of the super individual based on the internal states and environmental variables
                        #for feeding and energy-storing stages (CIV-CV:dsc-IIIA), the vertical position is estimated as a function of environmental variables (resource & risks) and super-individual-specific attribute values ('genes')

                        #this extracts relevant evolvable dynamic attribute ('gene') values:
                        current_a2_irradiancesensitivity = a2_irradiancesensitivity[currentsupindividual, currentsubpopulation]
                        current_a3_pred1sensitivity = a3_pred1sensitivity[currentsupindividual, currentsubpopulation]
                        current_a4_pred1reactivity = a4_pred1reactivity[currentsupindividual, currentsubpopulation]
                    
                        #this extracts the relevant time- and space-specific environmental variable ranges (<time>, <depth>, <longitude>, <latitude>)
                        #ranges are such that it includes data across entire depth range (nb: irregular intervals)
                        #indices used to slice these are encoded from absolute time, depth, longitude and latitude values in the spatial tracking submodel
                        currenttemperature_range = temperature[currenttime, :, currentxidx, currentyidx]
                        currentfood1concentration_range = food1concentration[currenttime, :, currentxidx, currentyidx]
                        currentirradiance_range = irradiance[currenttime, :, currentxidx, currentyidx]
                        currentpred1dens_range = pred1dens[currenttime, :, currentxidx, currentyidx]

                        #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
                        #calling the vertical position estimation function from the module
                        #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
                        currentzpos, currentzidx, currentmaxzdistance, currentactualzdistance = vm.verticalmigration_dsc3a(temprange = currenttemperature_range, 
                                                                                                                        f1conrange = currentfood1concentration_range, 
                                                                                                                        iradrange = currentirradiance_range, 
                                                                                                                        maxirad = maxirradiance, 
                                                                                                                        p1dnsrange = currentpred1dens_range, 
                                                                                                                        a2 = current_a2_irradiancesensitivity, 
                                                                                                                        a3 = current_a3_pred1sensitivity, 
                                                                                                                        a4 = current_a4_pred1reactivity, 
                                                                                                                        pvp = currentzpos, 
                                                                                                                        pvi = currentzidx, 
                                                                                                                        strmass = currentstructuralmass, 
                                                                                                                        resmass = currentreservemass, 
                                                                                                                        modelres = 6)

                        #dsc-IIIA: growth and development submodel
                        #-----------------------------------------
                        #this extracts the growth allocation evolvable dynamic attribute ('gene')
                        current_a5_energyallocation = a5_energyallocation[currentsupindividual, currentsubpopulation]

                        #extraction of apropriate environmental variables based on the current zidx
                        #no need for the data to be extracted from a 4D numpy array (<time, depth, lon, lat>), as the whole depth array is already sliced for vertical migration submodel input
                        currenttemperature = currenttemperature_range[currentzidx]
                        currentfood1concentration = currentfood1concentration_range[currentzidx]

                        #this function estimates the somatic growth rate, which is used in calculating the development rate (= 1 / development time)
                        #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
                        currentgrowthrate = gd.growthanddevelopment_dsc3a(temperature = currenttemperature, 
                                                                        f1con = currentfood1concentration, 
                                                                        strmass = currentstructuralmass, 
                                                                        resmass = currentreservemass,
                                                                        maxzd = currentmaxzdistance, 
                                                                        actzd = currentactualzdistance, 
                                                                        modelres = 6)
                        
                        #whether the super individual enters diapause or develop directly towards adulthood without diapause is governed by the 'gene' "a6_diapauseprobability"
                        #and this binary state is defined in the seeding and/or spawning submodel and updated in the state variable "diapausestrategy", which is estimated above at stage civ/cv entry
                        #this inquires the body size attribute ('gene') of the super individual
                        current_a1_bodysize = a1_bodysize[currentsupindividual, currentsubpopulation]
                        
                        #this estimates the maximum adult size reachable by the super individual (for establishing a structuralmass ceiling for diapausing individuals until energy reserves are sufficiently accumulated for diapause)
                        #to estimate the molting state of a super individual (i.e., whether a super individual is ready to molt to the next stage or not), the stage- and super-individual-specific critical molting mass is required
                        #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
                        #the index position 11 is the CV->CVI(M/F) molting mass, which is the maximum structural mass reachaable by a super individual in a given environment and given 'gene' value of "a1_bodysize"
                        currentadultsize = cmm_lower[11] + (cmm_upper[11] - cmm_lower[11]) * current_a1_bodysize

                        #based on the growth rate, this updates the structuralmass and energy reserve mass
                        #however, the surplus assimilation allocation patterns for somatic growth & reserve buildup are markedly different for directly developing super individuals and diapausing individuals
                        #nb:this <if> condition is not written for currentgrowthrate == 0.00 because it doesnt update neither the structural nor reserve masses
                        if currentgrowthrate > 0:

                            if currentdiapausestrategy == 0:

                                #this is the surplus assimilation allocation for non-diapausing super individuals
                                #all surplus assimilation is channeled to structural growth (somatic growth); no reserves are maintained (this is slightly differnt from pascal v.3.1)
                                #in other words, the "a5_energyallocation" 'gene' is effectively supressed (thats why these 'genes' are called dynamic evolvable attributes)
                                #therefore, no update in the "currentreservemass"
                                #since there is no reserve allocation, no control is needed to regulate the reserve:structure ratio (max = 1.00)
                                currentstructuralmass = currentstructuralmass + currentgrowthrate

                            else:
                                
                                #this is the surplus assimilation allocation for diapausing super individuals for which, the 'gene' "a5_energyallocation" is not supressed
                                #however, the surplus assimilation allocation depends on the structural mass of the super individual with respect to the maximum reachable body mass ("currentadultsize")
                                #the structural mass of the super individual must be held static, if it is at the "currentadultsize" until the reserve/structure ratio is satisfied for diapause entry
                                if currentstructuralmass >= currentadultsize:

                                    #this blocks further allocation of surplus assimilation to structural (or somatic) growth
                                    #and all assimilation is channeld to reserve build up
                                    #as a result, no update in the structural mass
                                    currentreservemass = currentreservemass + currentgrowthrate

                                else:

                                    #this allocates the surplus assimilation to both structural (somatic) growth and reserve build up based on proportions defined by the 'gene' "a5_energyallocation"
                                    #here, a fraction defined by the evolvable dynamic attribute "a5_energyallocation" is channeled to structural growth
                                    currentstructuralmass = currentstructuralmass + currentgrowthrate * (1.00 - current_a5_energyallocation)
                                    #the rest is channel to reserve build up
                                    #nb:here, there is no explicit reserve mass limitation applied, but self-limitation occur at the diapause entry condition, which has a reservemass/structuralmass ceiling of 1.00
                                    currentreservemass = currentreservemass + currentgrowthrate * current_a5_energyallocation

                                #end if
                            
                            #end if
                        
                        else:

                            #in case of negative growth, energy reserves (if any: means, among other things, diapausing or non-diapausing super individuals alike) are mobilized to balance the potential degrowth
                            #this reserve mobilization occurs only if the super individual has sufficent reserves for this balancing
                            if currentreservemass >= abs(currentgrowthrate):
                                
                                #if there are sufficient energy reserves to balance the negative growth (i.e., excess metabolic demands)
                                #the reserve mass is updated by subtracting the current growth rate (nb:since the "currentgrowthrate" is a negative entity, the "+" operator results in a subtraction)
                                #there is no update in the current structural mass, as reserve mobilization "saves" the super individual from structural catabolization
                                currentreservemass = currentreservemass + currentgrowthrate
                            
                            else:

                                #if there are no sufficient reserves to balance the negative growth (i.e., excess metabolic demands)
                                #no update to the reserve mass but structural mass is updated by subtracting the current growth rate (nb:since the "currentgrowthrate" is a negative entity, the "+" operator results in a subtraction)
                                #this is structural catabolization for compensating the excess metabolic demands
                                currentstructuralmass = currentstructuralmass + currentgrowthrate

                            #end if

                        #end if

                        #this conditionally updates the maximum lifetime structural mass (for starvation risk estimation)
                        #only potentially valid for positive structural growth
                        if currentstructuralmass > currentmaxstructuralmass:
                        
                            currentmaxstructuralmass = currentstructuralmass
                        
                        #end if
                        
                        #dsc-IIIA:age, developmental stage and/or diapause state advancement
                        #-------------------------------------------------------------------
                        #this extracts the diapause entry evolvable dynamic attribute for the super individual
                        current_a7_diapauseentry = a7_diapauseentry[currentsupindividual, currentsubpopulation]

                        #this updates the age of the super individual
                        #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
                        currentage += 1

                        #the development rate or development time is a function of growth rate calculated by the modular function above
                        #to estimate the molting state (i.e., whether a super individual is ready to molt to the next stage or not) of a super individual, the stage- and super-individual-specific critical molting mass is required
                        #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
                        currentcmm = cmm_lower[currentdevelopmentalstage] + (cmm_upper[currentdevelopmentalstage] - cmm_lower[currentdevelopmentalstage]) * current_a1_bodysize
                        #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
                        
                        if currentdiapausestrategy == 0:

                            #for super individuals that do not undergo diapause, an attempt is made to develop directly from civ - cv - adult
                            #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
                            #the else condition is not mentioned here, as no stage increment occurs if the if() condition is invalid
                            if currentstructuralmass >= currentcmm:

                                #molting occurs and stage is updated
                                #the diapause state does not change and remains at "A" (active)
                                currentdevelopmentalstage += 1

                                #update the state variable in-place (due to conditional state change)
                                developmentalstage[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage
                                
                                #data logging
                                #------------
                                if datalogger == 1:
                                    
                                    #log no. of direct-developments and their structural and reserve masses
                                    #obs:there is a time-lag here in terms of 'nvindividuals' because, it is updated in the survival submodel later down the stage
                                    if currentdevelopmentalstage == 12:
                                        
                                        #log spatial-integrated data (2D) for easy summarizing
                                        #log the no. of direct-developments
                                        lcstrategies_i[currenttime, 0] += currentnvindividuals
                                        #log the total structural and reserve masses (gC) 
                                        lcstrategies_f[currenttime, 0] += (currentstructuralmass * currentnvindividuals) / 1e6
                                        lcstrategies_f[currenttime, 1] += (currentreservemass * currentnvindividuals) / 1e6
                                    
                                    #end if
                                
                                #end if
                            
                            #end if

                        else:

                            #for super individuals that undergo diapause, development from civ-cv is allowed (not obligatory) but that from cv-adult is blocked
                            if currentdevelopmentalstage == 10:

                                #these are civ stages, and they are allowed to develop into cv stages if the diapause entry condition (structural to reserve mass ratio) allows
                                #first, the molting condition is checked:
                                #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
                                #the else condition is not mentioned here, as no stage increment occurs if the if() condition is invalid
                                if currentstructuralmass >= currentcmm:

                                    #first, the molting condition is checked
                                    #molting occurs and stage is updated
                                    #no change occur in the diapause state ("A":active)
                                    currentdevelopmentalstage += 1

                                    #update the state variable in-place (due to conditional state change)
                                    developmentalstage[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage
                                
                                #second, the diapause entry condition is checked:
                                #nb:the final else condition is not written as it doesnt update the diapause state or developmental stage
                                elif currentreservemass / currentstructuralmass >= current_a7_diapauseentry:

                                    #if this condition is satisfied, the super individual is in the "diapause entry" state ("E")
                                    #no update to the developmental stage
                                    currentdiapausestate = "E"

                                    #update state variables in-place
                                    diapausestate[currentsupindividual, currentsubpopulation] = currentdiapausestate
                                    timeofdiapauseentry[currentsupindividual, currentsubpopulation] = currenttime
                                    developmentalstageatdiapauseentry[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage
                                    structuralmassatdiapauseentry[currentsupindividual, currentsubpopulation] = currentstructuralmass
                                    reservemassatdiapauseentry[currentsupindividual, currentsubpopulation] = currentreservemass

                                    #data logging
                                    #------------
                                    if datalogger == 1:
                                        
                                        #log no. of direct-developments and their structural and reserve masses
                                        #obs:there is a time-lag here in terms of 'nvindividuals' because, it is updated in the survival submodel later down the stage
                                        #log spatial-integrated data (2D) for easy summarizing
                                        #log the no. of direct-developments (only civs arrive here - see above)
                                        lcstrategies_i[currenttime, 1] += currentnvindividuals
                                        #log the total structural and reserve masses (gC)
                                        lcstrategies_f[currenttime, 2] += (currentstructuralmass * currentnvindividuals) / 1e6
                                        lcstrategies_f[currenttime, 3] += (currentreservemass * currentnvindividuals) / 1e6
                                    
                                    #end if

                                #end if

                            else:
                            
                                #these are cv stages, and they are not allowed to develop into adults before entering diapause
                                #therefore, they are not checked for the developmental stage condition; also, in the allocation section above, a structural mass ceiling is established
                                #this holds their structural mass maximally at right below the adult size without development to adult
                                #this checks for the diapause entry condition:
                                #nb:no else condition is written as it does not update the diapause state
                                if currentreservemass / currentstructuralmass >= current_a7_diapauseentry:

                                    #if this condition is satisfied, the super individual is in the "diapause entry mode" ("E")
                                    #no update to the developmental stage
                                    #nb:<log data here!>
                                    currentdiapausestate = "E"

                                    #update state variable in-place
                                    diapausestate[currentsupindividual, currentsubpopulation] = currentdiapausestate
                                    timeofdiapauseentry[currentsupindividual, currentsubpopulation] = currenttime
                                    developmentalstageatdiapauseentry[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage
                                    structuralmassatdiapauseentry[currentsupindividual, currentsubpopulation] = currentstructuralmass
                                    reservemassatdiapauseentry[currentsupindividual, currentsubpopulation] = currentreservemass

                                    #data logging
                                    #------------
                                    if datalogger == 1:
                                        
                                        #log no. of direct-developments and their structural and reserve masses
                                        #obs:there is a time-lag here in terms of 'nvindividuals' because, it is updated in the survival submodel later down the stage
                                        #log spatial-integrated data (2D) for easy summarizing
                                        #log the no. of direct-developments (only cvs arrive here - see above)
                                        lcstrategies_i[currenttime, 2] += currentnvindividuals
                                        #log the total structural and reserve masses (gC)
                                        lcstrategies_f[currenttime, 4] += (currentstructuralmass * currentnvindividuals) / 1e6
                                        lcstrategies_f[currenttime, 5] += (currentreservemass * currentnvindividuals) / 1e6
                                    
                                    #end if

                                #end if

                            #end if

                        #end if
                        
                        #dsc-IIIA:survival submodel
                        #--------------------------
                        #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                        currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                        #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                        currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        currentmortalityrisk = sv.mortalityrisk_dsc3a(strmass = currentstructuralmass, 
                                                                    maxstrmass = currentmaxstructuralmass, 
                                                                    resmass = currentreservemass, 
                                                                    p1dens = currentpred1dens, 
                                                                    p1lightdp = currentpred1lightdep, 
                                                                    p2risk = pred2risk, 
                                                                    bgmrisk = bgmortalityrisk)

                        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                        #when all virtual individuals contained in a super individual dies, then the super individual also dies
                        #this death is simulated after stage-specific processes
                        currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                        #update the non-conditional state variables (conditionally state changed state variables are update in-place, e.g., developmental stage - see above)
                        structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                        reservemass[currentsupindividual, currentsubpopulation] = currentreservemass
                        maxstructuralmass[currentsupindividual, currentsubpopulation] = currentmaxstructuralmass
                        nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                        age[currentsupindividual, currentsubpopulation] = currentage

                        xpos[currentsupindividual, currentsubpopulation] = currentxpos
                        ypos[currentsupindividual, currentsubpopulation] = currentypos
                        zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    elif currentdiapausestate == "E":

                        #this developmental stage group subcategory includes super individuals that are at the diapause entry
                        #at this point, they do not feed but seek a preferred diapause depth
                        #despite not feeding, their metabolism occurrs at the regular rate (basal and active metabolic rates)
                        #for this, they are not in diapause yet - until they find the preferred diapause depth and 'settle' therein at a diapause state ("D":diapause)
                        
                        #this opens a state variable inquiry:
                        currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                        currentreservemass = reservemass[currentsupindividual, currentsubpopulation]
                        currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                        currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                        currentage = age[currentsupindividual, currentsubpopulation]
                        currentdiapausedepth = diapausedepth[currentsupindividual, currentsubpopulation]

                        #this uses a modular function to estimate the vertical position (i.e., seasonal vertical migration to diapause depths)
                        currentzpos, currentzidx, currentmaxzdistance, currentactualzdistance = vm.verticalmigration_dsc3e(diapdepth = currentdiapausedepth, 
                                                                                                                        strmass = currentstructuralmass, 
                                                                                                                        pvp = currentzpos, 
                                                                                                                        modelres = 6)

                        #dsc-IIIE: growth and development submodel
                        #-----------------------------------------
                        #nb:somatic growth do not occur at dsc-IIIE because super individuals stop feeding and begins to use the energy reserve for survival
                        #however, until reaching the diapause depth, the metabolic rate occurs at a regular rate (i.e., reserves are burnt faster than at diapause)
                        #the potential degrowth and/or reserve utilization is therefore, depndent on the ambient temperature and the total bodymass of the super individual
                        
                        #this extracts the ambient temperature at the current depth from a 4D numpy array (<time, depth, lon, lat>)
                        currenttemperature = temperature[currenttime, currentzidx, currentxidx, currentyidx]

                        #this uses a modular function to estimate the potential degrowth and/or reserve utilization of super individuals
                        currentgrowthrate = gd.growthanddevelopment_dsc3e(temperature = currenttemperature, 
                                                                        strmass = currentstructuralmass, 
                                                                        resmass = currentreservemass, 
                                                                        actzd = currentactualzdistance, 
                                                                        maxzd = currentmaxzdistance, 
                                                                        modelres = 6)

                        #super individuals with limited energy reserves can also enter the diapause entry stage (dsc-IIIE: cf. lower values of the a7_diapauseentry 'gene')
                        #therefore, the reserve exhaustion and/or structural catabolization is done as follows:
                        if currentreservemass >= abs(currentgrowthrate):

                            #if the reserve mass is sufficient to balance the degrowth (i.e., metabolic demands of diapause entry):
                            #reserves are proportionally mobilized and no change occurs in the structural mass (despite the "+" operator, it is effectively a subtraction as growt rate is negative)
                            currentreservemass = currentreservemass + currentgrowthrate
                        
                        else:

                            #if reserves are not sufficient to balance the degrowth:
                            #the structural mass is proportionally catabolized (despite the + operator, it is effectively a subtraction as growt hrate is negative)
                            #no change to the reserve mass
                            currentstructuralmass = currentstructuralmass + currentgrowthrate
                        
                        #end if

                        #dsc-IIIE: age and diapause state advancement
                        #-----------------------------------------------
                        #nb:no developmental stage advancement at this level (due to no feeding & growth)
                        #this advances the current age by 1 ping (= 6 hours)
                        currentage += 1

                        #this advances the diapause state (to "D":diapause) if the super individual had reached the diapause depth
                        if currentzpos == currentdiapausedepth:
                            
                            currentdiapausestate = "D"

                            #update state variable for diapause state in-place (due to conditional state change)
                            diapausestate[currentsupindividual, currentsubpopulation] = currentdiapausestate

                        #end if

                        #dsc-IIIE: survival submodel
                        #---------------------------
                        #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                        currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                        #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                        currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        currentmortalityrisk = sv.mortalityrisk_dsc3e(strmass = currentstructuralmass, 
                                                                    maxstrmass = currentmaxstructuralmass, 
                                                                    resmass = currentreservemass, 
                                                                    p1dens = currentpred1dens, 
                                                                    p1lightdp = currentpred1lightdep, 
                                                                    p2risk = pred2risk, 
                                                                    bgmrisk = bgmortalityrisk)

                        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                        #when all virtual individuals contained in a super individual dies, then the super individual also dies
                        currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                        #nb:some state variables are updated in-place (see above)
                        #nb:"maxstructuralmass" is not updated due to no feeding and growth - although the query is open, the state variable values are not updated in this subcategory
                        structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                        reservemass[currentsupindividual, currentsubpopulation] = currentreservemass
                        nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                        age[currentsupindividual, currentsubpopulation] = currentage

                        xpos[currentsupindividual, currentsubpopulation] = currentxpos
                        ypos[currentsupindividual, currentsubpopulation] = currentypos
                        zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    elif currentdiapausestate == "D":

                        #this developmental stage subcategory refers to diapausing individuals at their preferred diapause depths
                        #they dont feed, dont grow or develop
                        #their basal metabolic rate occurrs at 25% of the regular basal metabolic rate, see: Maps et al.(2012): https://doi.org/10.1093/plankt/fbt100 
                        #there is no active metabolic costs during diapause, which means no active movements - only passive drifts brought about by water currents
                        #therefore, the diel and seasonal vertical migration submodel is not called
                        
                        #this opens a state variable inquiry:
                        currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                        currentreservemass = reservemass[currentsupindividual, currentsubpopulation]
                        currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                        currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                        currentage = age[currentsupindividual, currentsubpopulation]

                        #dsc-IIID: growth and development submodel
                        #-----------------------------------------
                        #nb:no somatic growth occurs at dsc-IIID because super individuals are at diapause
                        #the metabolic rate occurs at a reduced rate (i.e., 25% of the regular metabolic rate)
                        #the potential degrowth and/or reserve utilization are depndent on the ambient temperature and the total bodymass of the super individual
                        
                        #this extracts the ambient temperature at the current depth from a 4D numpy array (<time, depth, lon, lat>)
                        #nb:here, zposition and zidx are extracted from their respective state variables before the stage segregation (i.e., in the 3d position tracking)
                        currenttemperature = temperature[currenttime, currentzidx, currentxidx, currentyidx]

                        #this uses a modular function to estimate the potential degrowth and/or reserve utilization of super individuals
                        currentgrowthrate = gd.growthanddevelopment_dsc3d(temperature = currenttemperature, 
                                                                        strmass = currentstructuralmass, 
                                                                        resmass = currentreservemass, 
                                                                        modelres = 6)
                        
                        #super individuals with limited energy reserves can also enter diapause (dsc-IIID: cf. lower values of the "a7_diapauseentry" 'gene')
                        #nb:however, their diapause state changes from "D" to "X" if the reserves run out (cf. "a8_diapauseexit" 'gene')
                        #therefore, the reserve exhaustion and/or structural catabolization is done as follows:
                        if currentreservemass >= abs(currentgrowthrate):

                            #if the reserve mass is sufficient to balance the degrowth (i.e., metabolic demands of diapause entry):
                            #reserves are proportionally mobilized and no change occurs in the structural mass (despite the + operator, it is effectively a subtraction as growt hrate is negative)
                            currentreservemass = currentreservemass + currentgrowthrate
                        
                        else:

                            #if reserves are not sufficient to balance the degrowth:
                            #the structural mass is proportionally catabolized (despite the + operator, it is effectively a subtraction as growt hrate is negative)
                            #no change to the reserve mass
                            currentstructuralmass = currentstructuralmass + currentgrowthrate
                        
                        #end if
                        
                        #dsc-IIID:age and diapause state advancement
                        #-------------------------------------------
                        #nb:no developmental stage advancement at this level (due to no feeding & growth)
                        #this advances the current age by 1 ping (= 6 hours)
                        currentage += 1

                        #this inquires the diapause exit 'gene' value of the super individual
                        current_a8_diapauseexit = a8_diapauseexit[currentsupindividual, currentsubpopulation]

                        #super individual exits from diapause if a fraction of energy reserves defined by the diapause exit 'gene' has been exhausted
                        #this requires the energy reserve size at diapause entry and it is inquired as:
                        currentreservemassatdiapauseentry = reservemassatdiapauseentry[currentsupindividual, currentsubpopulation]

                        #if the diapause entry reserve mass is zero (due to diapause entry 'gene' value being 0.00), a separate condition is used to evaluate the exit-state
                        #this is to avoid errors emerged from dividing by zero
                        if currentreservemassatdiapauseentry <= 0:
                        
                            #the diapsue state changes to diapause exit or "X"
                            currentdiapausestate = "X"
                            
                            #update state variables in-place
                            diapausestate[currentsupindividual, currentsubpopulation] = currentdiapausestate
                            reservemassatdiapauseexit[currentsupindividual, currentsubpopulation] = currentreservemass
                            timeofdiapauseexit[currentsupindividual, currentsubpopulation] = currenttime
                        
                        else:
                            
                            if 1.00 - currentreservemass / currentreservemassatdiapauseentry >= current_a8_diapauseexit:

                                #the diapsue state changes to diapause exit or "X"
                                currentdiapausestate = "X"
                                
                                #update state variables in-place
                                diapausestate[currentsupindividual, currentsubpopulation] = currentdiapausestate
                                reservemassatdiapauseexit[currentsupindividual, currentsubpopulation] = currentreservemass
                                timeofdiapauseexit[currentsupindividual, currentsubpopulation] = currenttime

                            #end if
                                                    
                        #end if
                        
                        #dsc-IIID: survival submodel
                        #---------------------------
                        #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                        currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                        #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                        currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        currentmortalityrisk = sv.mortalityrisk_dsc3d(strmass = currentstructuralmass, 
                                                                    maxstrmass = currentmaxstructuralmass, 
                                                                    resmass = currentreservemass, 
                                                                    p1dens = currentpred1dens, 
                                                                    p1lightdp = currentpred1lightdep, 
                                                                    p2risk = pred2risk, 
                                                                    bgmrisk = bgmortalityrisk)

                        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                        #when all virtual individuals contained in a super individual dies, then the super individual also dies
                        #this death is simulated after stage-specific processes
                        currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                        #nb:"maxstructuralmass" is not updated due to no feeding and growth - although the query is open, the state variable values are not updated in this subcategory
                        #nb:some state variables are conditionally updated in-place depending on whether the diapsuse state had changed (see above)
                        structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                        reservemass[currentsupindividual, currentsubpopulation] = currentreservemass
                        nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                        age[currentsupindividual, currentsubpopulation] = currentage

                        xpos[currentsupindividual, currentsubpopulation] = currentxpos
                        ypos[currentsupindividual, currentsubpopulation] = currentypos
                        zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    elif currentdiapausestate == "X":

                        #these include the super individuals belonging to the developmental stage subcategory that are exitting diapause
                        #they seek to move out of the diapause habitat (depth) and ascend to the surface (a random photic zone depth zidx = 0:25)
                        #their basal and active metabolic rates are back to regular levels
                        #however, they do not feed yet - do so only after completing the seasonal ascent, i.e., at diapause state "P" (dsc-IIIP)

                        #this opens a state variable inquiry:
                        currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                        currentreservemass = reservemass[currentsupindividual, currentsubpopulation]
                        currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                        currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                        currentage = age[currentsupindividual, currentsubpopulation]

                        #this uses a modular function to estimate the vertical position (i.e., seasonal vertical migration out of diapause depths)
                        currentzpos, currentzidx, currentmaxzdistance, currentactualzdistance = vm.verticalmigration_dsc3x(pvp = currentzpos, 
                                                                                                                        pvi = currentzidx, 
                                                                                                                        strmass = currentstructuralmass, 
                                                                                                                        modelres = 6)

                        #dsc-IIIX: growth and development submodel
                        #-----------------------------------------
                        #nb:no somatic growth occurs at dsc-IIIX because super individuals stop feeding and still continue to use the energy reserve for survival
                        #nb:feeding, growth and development resumes only at the dsc-IIIP stage subcategory (see below)
                        #however, the metabolic rate occurs at a regular rate (i.e., reserves are burnt faster than at diapause)
                        #the potential degrowth and/or reserve utilization is therefore, depndent on the ambient temperature and the total bodymass of the super individual
                        
                        #this extracts the ambient temperature at the current depth from a 4D numpy array (<time, depth, lon, lat>)
                        currenttemperature = temperature[currenttime, currentzidx, currentxidx, currentyidx]

                        #this uses a modular function to estimate the potential degrowth and/or reserve utilization of super individuals
                        currentgrowthrate = gd.growthanddevelopment_dsc3x(temperature = currenttemperature, 
                                                                        strmass = currentstructuralmass, 
                                                                        resmass = currentreservemass, 
                                                                        actzd = currentactualzdistance, 
                                                                        maxzd = currentmaxzdistance, 
                                                                        modelres = 6)
                        
                        #super individuals with limited energy reserves can also enter the diapause entry stage (dsc-IIIE: cf. lower values of the a7_diapauseentry 'gene')
                        #therefore, the reserve exhaustion and/or structural catabolization is done as follows:
                        if currentreservemass >= abs(currentgrowthrate):

                            #if the reserve mass is sufficient to balance the degrowth (i.e., metabolic demands of diapause entry):
                            #reserves are proportionally mobilized and no change occurs in the structural mass (despite the + operator, it is effectively a subtraction as growt hrate is negative)
                            currentreservemass = currentreservemass + currentgrowthrate
                        
                        else:

                            #if reserves are not sufficient to balance the degrowth:
                            #the structural mass is proportionally catabolized (despite the + operator, it is effectively a subtraction as growt hrate is negative)
                            #no change to the reserve mass
                            currentstructuralmass = currentstructuralmass + currentgrowthrate
                        
                        #end if
                        
                        #dsc-IIIX: age and diapause state advancement
                        #--------------------------------------------
                        #nb:no developmental stage advancement at this level (due to no feeding & growth)
                        #this advances the current age by 1 ping (= 6 hours)
                        currentage += 1

                        #this advances the diapause state (to "P":post-diapause) if the super individual had reached the upper pelagial (<= 100 m)
                        if currentzpos <= 100:
                            
                            currentdiapausestate = "P"
                            #update state variable for diapause state in-place
                            diapausestate[currentsupindividual, currentsubpopulation] = currentdiapausestate

                            #data logging
                            #------------
                            if datalogger == 1:
                                
                                #log no. of diapause-exits and their reserve masses
                                #nb:structural mass is not logged in here (as diapause is based on energy reserves)
                                #obs:there is a time-lag here in terms of 'nvindividuals' because, it is updated in the survival submodel later down the stage
                                #nb: these are spatial-integrated data (2D) for easy summarizing
                                
                                if currentdevelopmentalstage == 10:
                                    
                                    #for civ diapause-exits
                                    lcstrategies_i[currenttime, 3] += currentnvindividuals
                                    lcstrategies_f[currenttime, 6] += (currentreservemass * currentnvindividuals) / 1e6

                                elif currentdevelopmentalstage == 11:

                                    #for cv diapause-exits
                                    lcstrategies_i[currenttime, 4] += currentnvindividuals
                                    lcstrategies_f[currenttime, 7] += (currentreservemass * currentnvindividuals) / 1e6

                                #end if

                            #end if

                        #end if

                        #dsc-IIIX: survival submodel
                        #---------------------------
                        #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                        currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                        #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                        currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        currentmortalityrisk = sv.mortalityrisk_dsc3x(strmass = currentstructuralmass, 
                                                                    maxstrmass = currentmaxstructuralmass, 
                                                                    resmass = currentreservemass, 
                                                                    p1dens = currentpred1dens, 
                                                                    p1lightdp = currentpred1lightdep, 
                                                                    p2risk = pred2risk, 
                                                                    bgmrisk = bgmortalityrisk)

                        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                        #when all virtual individuals contained in a super individual dies, then the super individual also dies
                        currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                        #nb:"maxstructuralmass" is not updated due to no feeding and growth - although the query is open, the state variable values are not updated in this subcategory
                        #nb:some state variables, such as the "diapausestate" is updated in-place
                        structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                        reservemass[currentsupindividual, currentsubpopulation] = currentreservemass
                        nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                        age[currentsupindividual, currentsubpopulation] = currentage

                        xpos[currentsupindividual, currentsubpopulation] = currentxpos
                        ypos[currentsupindividual, currentsubpopulation] = currentypos
                        zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    elif currentdiapausestate == "P":
                        
                        #these include super individuals belonging to the post diapause stages ("P") at the developmental stage category III (dsc-IIIP)
                        #they feed, grow and develop as usual (this is the first state after which regular metabolism remains after diapause exit)
                        #they do not actively maintain an energy reserve - but may use the remaining reserves for managing starvation risk and channeling into structural growth during food shortage
                        #they develop directly to adults
                        #nb:no else condition is written due to the presence of "U" undefined diapause state, which is not processed

                        currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                        currentreservemass = reservemass[currentsupindividual, currentsubpopulation]
                        currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                        currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                        currentage = age[currentsupindividual, currentsubpopulation]

                        #dsc-IIIP: diel and seasonal vertical migration submodel
                        #-----------------------------------------------------
                        #this uses a module-driven function to estimate the vertical position & index ("currentzpos", "currentzidx") of the super individual based on the developmental stage and environmental variables
                        #for feeding and energy-storing stages (CIV-CV:subcategory-P), the vertical position is estimated as a function of environmental variables (resource & risks) and super-individual-specific attribute values ('genes')

                        #this extracts relevant attribute ('gene') values:
                        current_a2_irradiancesensitivity = a2_irradiancesensitivity[currentsupindividual, currentsubpopulation]
                        current_a3_pred1sensitivity = a3_pred1sensitivity[currentsupindividual, currentsubpopulation]
                        current_a4_pred1reactivity = a4_pred1reactivity[currentsupindividual, currentsubpopulation]
                    
                        #this extracts the relevant time- and space-specific environmental variable ranges (<time>, <depth>, <longitude>, <latitude>)
                        #ranges are such that it includes data across entire depth range (nb: irregular intervals)
                        #indices used to slice these are encoded from absolute time, depth, longitude and latitude values in the spatial tracking submodel
                        currenttemperature_range = temperature[currenttime, :, currentxidx, currentyidx]
                        currentfood1concentration_range = food1concentration[currenttime, :, currentxidx, currentyidx]
                        currentirradiance_range = irradiance[currenttime, :, currentxidx, currentyidx]
                        currentpred1dens_range = pred1dens[currenttime, :, currentxidx, currentyidx]

                        #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
                        #calling the vertical position estimation function from the module
                        #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
                        currentzpos, currentzidx, currentmaxzdistance, currentactualzdistance = vm.verticalmigration_dsc3p(temprange = currenttemperature_range, 
                                                                                                                        f1conrange = currentfood1concentration_range, 
                                                                                                                        iradrange = currentirradiance_range, 
                                                                                                                        maxirad = maxirradiance,
                                                                                                                        p1dnsrange= currentpred1dens_range, 
                                                                                                                        a2 = current_a2_irradiancesensitivity, 
                                                                                                                        a3 = current_a3_pred1sensitivity, 
                                                                                                                        a4 = current_a4_pred1reactivity,
                                                                                                                        pvp = currentzpos, 
                                                                                                                        pvi = currentzidx, 
                                                                                                                        strmass = currentstructuralmass, 
                                                                                                                        resmass = currentreservemass, 
                                                                                                                        modelres = 6)

                        #dsc-IIIP: growth and development submodel
                        #-----------------------------------------
                        #extraction of apropriate environmental variables based on the current zidx
                        #no need for the data to be extracted from a 4D numpy array (<time, depth, lon, lat>), as the whole depth array is already sliced for vertical migration submodel input
                        currenttemperature = currenttemperature_range[currentzidx]
                        currentfood1concentration = currentfood1concentration_range[currentzidx]

                        #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
                        #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
                        currentgrowthrate = gd.growthanddevelopment_dsc3p(temperature = currenttemperature, 
                                                                        f1con = currentfood1concentration, 
                                                                        strmass = currentstructuralmass, 
                                                                        resmass = currentreservemass,
                                                                        maxzd = currentmaxzdistance, 
                                                                        actzd = currentactualzdistance, 
                                                                        modelres = 6)
                        
                        #all surplus assimilations are channeled to structural growth: no energy reserves are maintained or replinshed
                        #however, reserves may be used for balancing degrowth and starvation risk therein
                        #nb:the no growth condition (currentgrowthrate == 0) is not written as it does not affect neither structural mass nor reserve mass
                        if currentgrowthrate > 0:
                        
                            #if the net growth rate is positive, all the surplus assimilation is channeled to somatic growth
                            #no changes in the reserve mass
                            currentstructuralmass = currentstructuralmass + currentgrowthrate

                            #this updates the maximum structural mass if necessary:
                            if currentstructuralmass >= currentmaxstructuralmass:

                                currentmaxstructuralmass = currentstructuralmass

                            #end if
                        
                        else:

                            #if the current net growth rate is negative (metabolic demands > assimilation)
                            if currentreservemass >= abs(currentgrowthrate):

                                #the excess metabolic demands can be balanced by the energy reserves
                                #nb:the + operator is effectively a subtraction because the net growth rate is negative
                                #nb:no update in the structural mass
                                currentreservemass = currentreservemass + currentgrowthrate
                            
                            else:

                                #the excess metabolic demands are balanced by catabolizing structural growth (degrowth)
                                #nb:the + operator is effectively a subtraction because the net growth rate is negative
                                #nb:no changes to the reserve mass
                                currentstructuralmass = currentstructuralmass + currentgrowthrate
                            
                            #end if

                        #end if

                        #dsc-IIIP:age and developmental stage advancement
                        #------------------------------------------------
                        #this updates the age of the super individual
                        #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
                        currentage += 1

                        #the developmental stage advancement can be from civ-cv and cv-cvi(F/M) depending on the diapause stage
                        #to estimate the molting state (i.e., whether a super individual is ready to molt to the next stage or not) of a super individual, the stage- and super-individual-specific critical molting mass is required
                        #this extracts the body size dynamic evolvable attribute ("a1_bodysize")
                        current_a1_bodysize = a1_bodysize[currentsupindividual, currentsubpopulation]
                        #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
                        currentcmm = cmm_lower[currentdevelopmentalstage] + (cmm_upper[currentdevelopmentalstage] - cmm_lower[currentdevelopmentalstage]) * current_a1_bodysize

                        #molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass
                        #the else condition is not mentioned here, as no stage increment occurs if the if() condition is invalid
                        if currentstructuralmass >= currentcmm:

                            #molting occurs and stage is updated
                            #the diapause state does not change and remains at "A" (active)
                            currentdevelopmentalstage += 1
                            #update the state variable in-place (due to conditional state change)
                            developmentalstage[currentsupindividual, currentsubpopulation] = currentdevelopmentalstage
                            
                        #end if

                        #dsc-IIIP:survival submodel
                        #--------------------------
                        #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                        currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                        #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                        currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        
                        #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                        currentmortalityrisk = sv.mortalityrisk_dsc3p(strmass = currentstructuralmass, 
                                                                    maxstrmass = currentmaxstructuralmass, 
                                                                    resmass = currentreservemass, 
                                                                    p1dens = currentpred1dens, 
                                                                    p1lightdp = currentpred1lightdep, 
                                                                    p2risk = pred2risk, 
                                                                    bgmrisk = bgmortalityrisk)

                        #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                        #when all virtual individuals contained in a super individual dies, then the super individual also dies
                        #this death is simulated after stage-specific processes
                        currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                        #nb:when a state variable query is opened, that must be closed before quitting the stage grouping!
                        #nb:some state variables are conditionally updated in-place
                        #nb:since the super individual is not dead, the "lifestatus" state variable does not change from the initial value of 1
                        structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                        reservemass[currentsupindividual, currentsubpopulation] = currentreservemass
                        maxstructuralmass[currentsupindividual, currentsubpopulation] = currentmaxstructuralmass
                        nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                        age[currentsupindividual, currentsubpopulation] = currentage

                        xpos[currentsupindividual, currentsubpopulation] = currentxpos
                        ypos[currentsupindividual, currentsubpopulation] = currentypos
                        zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    #end if

                    #data logging
                    #------------
                    #happens only if the 'datalogger' is enabled (val = 1)
                    if datalogger == 1:

                        #logging population size by sequentual addition
                        #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                        spatialdistribution_ps[currentdevelopmentalstage, currentxidx, currentyidx, currentzidx, currenttime] += 1
                        #logging biomass (gC) by sequential addition
                        #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                        #nb:reserve-driven biomass can be logged separately if need be
                        currenttotalmass = ((currentstructuralmass + currentreservemass) * currentnvindividuals) / 1e6
                        spatialdistribution_bm[currentdevelopmentalstage, currentxidx, currentyidx, currentzidx, currenttime] += currenttotalmass
                    
                    #end if

                #simulation of growth, development and survival of adult male and female stages (CVI-F, CVI-M): dsc-IV
                #_____________________________________________________________________________________________________

                else:

                    #these are adult stages (dsc-IV) that can be males or females
                    #the sex should be determined at the first entry to this stage (not initialized in the seeding/spawning)
                    currentsex = sex[currentsupindividual, currentsubpopulation]
                    
                    #sex is determined randomly at ca. 0.5:0.5 M:F probability
                    if currentsex == "U":
                        
                        #drawing a uniform random number to compare with the threshold of 0.50
                        sexdet = np.random.rand(1).squeeze()
                        currentsex = "M" if sexdet < 0.5 else "F"
                        #update state variable in-place
                        sex[currentsupindividual, currentsubpopulation] = currentsex

                    #end if
                    
                    #they feed, grow but do not maintain an active energy reserve - instead they use stored energy for fulfilling metabolic demands
                    #reproduction and spawning happens during the adult stage, where reproduction includes mate finding, insemination, recombination and mutation - and eventually, spawning (egg production)
                    #adult males do not feed - but use whatever the energy reserves they possess for survival (hence, they are short-lived and maintians 'genetic' diversity within the subpopulation)
                    #both sexes engage in shorter term vertical behavior because it improves the chances of male-female encounter
                    #nb:the "sex" and "maxstructuralmass" are inquired but not updated within the stage - so there is no updates therein at the closure (see below)
                    #the "malegenome" is not inquired but updated in-place if a female is inseminated (see below)
                    currentstructuralmass = structuralmass[currentsupindividual, currentsubpopulation]
                    currentreservemass = reservemass[currentsupindividual, currentsubpopulation]
                    currentmaxstructuralmass = maxstructuralmass[currentsupindividual, currentsubpopulation]
                    currentnvindividuals = nvindividuals[currentsupindividual, currentsubpopulation]
                    currentage = age[currentsupindividual, currentsubpopulation]
                    currentinseminationstate = inseminationstate[currentsupindividual, currentsubpopulation]
                    currentreproductiveallocation = reproductiveallocation[currentsupindividual, currentsubpopulation]
                    currenttotalfecundity = totalfecundity[currentsupindividual, currentsubpopulation]

                    #dsc-IV: diel and seasonal vertical migration submodel
                    #-----------------------------------------------------
                    #this uses a module-driven function to estimate the vertical position & index ("currentzpos", "currentzidx") of the super individual based on the developmental stage and environmental variables
                    #for adult male and female stages (CVI:subcategories M & F), the vertical position is estimated as a function of environmental variables (resource & risks) and super-individual-specific attribute values ('genes')

                    #this extracts relevant attribute ('gene') values:
                    current_a2_irradiancesensitivity = a2_irradiancesensitivity[currentsupindividual, currentsubpopulation]
                    current_a3_pred1sensitivity = a3_pred1sensitivity[currentsupindividual, currentsubpopulation]
                    current_a4_pred1reactivity = a4_pred1reactivity[currentsupindividual, currentsubpopulation]
                
                    #this extracts the relevant time- and space-specific environmental variable ranges (<time>, <depth>, <longitude>, <latitude>)
                    #ranges are such that it includes data across entire depth range (nb: irregular intervals)
                    #indices used to slice these are encoded from absolute time, depth, longitude and latitude values in the spatial tracking submodel
                    currenttemperature_range = temperature[currenttime, :, currentxidx, currentyidx]
                    currentfood1concentration_range = food1concentration[currenttime, :, currentxidx, currentyidx]
                    currentirradiance_range = irradiance[currenttime, :, currentxidx, currentyidx]
                    currentpred1dens_range = pred1dens[currenttime, :, currentxidx, currentyidx]

                    #evolvable attribute ('gene') values and the above environmental data ranges are inputs to the modular function for vertical position estimation
                    #calling the vertical position estimation function from the module
                    #nb:this outputs four integers: (i) absolute vertical position and (ii) relative vertical position (index), (iii) maximum vertical search distance and (iv) actual vertical search distance
                    currentzpos, currentzidx, currentmaxzdistance, currentactualzdistance = vm.verticalmigration_dsc4(temprange = currenttemperature_range, 
                                                                                                                    f1conrange = currentfood1concentration_range, 
                                                                                                                    iradrange = currentirradiance_range, 
                                                                                                                    maxirad = maxirradiance,
                                                                                                                    p1dnsrange= currentpred1dens_range, 
                                                                                                                    a2 = current_a2_irradiancesensitivity, 
                                                                                                                    a3 = current_a3_pred1sensitivity, 
                                                                                                                    a4 = current_a4_pred1reactivity,
                                                                                                                    pvp = currentzpos, 
                                                                                                                    pvi = currentzidx, 
                                                                                                                    strmass = currentstructuralmass, 
                                                                                                                    resmass = currentreservemass, 
                                                                                                                    modelres = 6)

                    #dsc-IV: growth and development submodel
                    #---------------------------------------
                    #this is male and female specific because adult males do not feed
                    #extraction of apropriate environmental variables based on the current zidx
                    #no need for the data to be extracted from a 4D numpy array (<time, depth, lon, lat>), as the whole depth array is already sliced for vertical migration submodel input
                    currenttemperature = currenttemperature_range[currentzidx]
                    currentfood1concentration = currentfood1concentration_range[currentzidx]

                    #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
                    #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
                    #obligatory negative growth (degrowth; structural or energy reserve) for adult male and growth/degrowth for adult female
                    if currentsex == "F":
                        #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
                        #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
                        #this is female-specific (growth/degrowth both possible)
                        currentgrowthrate = gd.growthanddevelopment_dsc4f(temperature = currenttemperature, 
                                                                        f1con = currentfood1concentration, 
                                                                        strmass = currentstructuralmass, 
                                                                        resmass = currentreservemass,
                                                                        maxzd = currentmaxzdistance, 
                                                                        actzd = currentactualzdistance, 
                                                                        modelres = 6)
                    
                    else:

                        #this function estimates the somatic growth rate and developmental rates (development is a function of growth - stage progression is coded below)
                        #the function takes ambient temperature and food concentration as environmental inputs and current structural & reserve masses as internal state inputs
                        #this is male-specific (degrowth is only possible)
                        currentgrowthrate = gd.growthanddevelopment_dsc4m(temperature = currenttemperature, 
                                                                        f1con = currentfood1concentration, 
                                                                        strmass = currentstructuralmass, 
                                                                        resmass = currentreservemass,
                                                                        maxzd = currentmaxzdistance, 
                                                                        actzd = currentactualzdistance, 
                                                                        modelres = 6)
                    
                    #end if
                    
                    #dsc-IV:age advancement
                    #------------------------
                    #this updates the age of the super individual
                    #nb:+=1 means it adds bins of 6 hrs (1 time ping in the model clock = 6 hrs in real clock)
                    #there is no developmental stage advancement after reaching the adulthood (CVI-F/M)
                    #the structural mass also reaches a maximum after reaching adulthood (check with v3.1 - check, ok!)
                    currentage += 1

                    #dsc-IV:reproduction and spawning submodel - part-A: energy allocation
                    #---------------------------------------------------------------------
                    #this submodel is sex-specific
                    #nearby males and females mate, and a given female gets inseminated by one (1) male only
                    #a given male can mate with many females during its short lifespan
                    #the mate choice is assumed to be random

                    if currentsex == "M":
                        
                        #for the male, only the structural and reserve masses are updated
                        #nb:only degrowth is possible because males do not feed
                        if currentreservemass >= abs(currentgrowthrate):
                        
                            #if reserves can balance the degrowth, update the reservemass with no change to structural mass
                            #despite the "+" operator, the operation is effectively a subtraction because the growth rate is negative
                            currentreservemass = currentreservemass + currentgrowthrate
                        
                        else:

                            #if the reserves are not sufficient to balance the degrowth, then structural mass is catabolized to meet the degrowth (metabolic demands)
                            #only update the structural mass, as no change occur to reserve mass
                            #despite the "+" operator, the operation is effectively a subtraction because the growth rate is negative
                            currentstructuralmass = currentstructuralmass + currentgrowthrate
                        
                        #end if

                    else:

                        #for the female, growth can be positive or negative (depending on food conditions and vertical behaviour)
                        #in phases of positive growth, the surplus energy is allocated to egg production and/or structural growth (only if a female sustained degrowth in the past)
                        #in case of past degrowth, the female channels 50% of the surplus energy for structural growth and 50% to egg production (a sensible estimate)
                        
                        #this extracts the body size dynamic evolvable attribute ("a1_bodysize")
                        current_a1_bodysize = a1_bodysize[currentsupindividual, currentsubpopulation]
                        #this is defined by the environment (lower and upper bounds of critical molting masses) and the trajectory is defined by the body size attribute (a1)
                        currentadultsize = cmm_lower[12] + (cmm_upper[12] - cmm_lower[12]) * current_a1_bodysize

                        if currentgrowthrate >= 0.00:
                        
                            #here, the growth rate is positive (i.e., there is surplus assimilation)
                            #surplus assimilation is fully or partly channeled to reproductive output
                            if currentstructuralmass >= currentadultsize:
                            
                                #in this case, the female is healthy (i.e., havent sustained structural degrowth)
                                #surplus assimilation is fully allocated to reproductive output (this can be fitted with a conversion factor, which is ca. 80% in some models)
                                #but only if the female is inseminated
                                #if so or else, no change to structural mass; no change to reserve mass - the surplus assimilation is discounted for non-inseminated females at structural mass ceiling
                                
                                if currentinseminationstate == 1:
                                    #reproductive allocation only if the female is inseminated - otherwise, surplus assimilation is discounted (no else condition is written)
                                    currentreproductiveallocation = currentreproductiveallocation + currentgrowthrate
                                #end if
                                
                            else:

                                #in this case, the female has sustained structural degrowth and needs recovery
                                #surplus assimilation is only partly allocated to reproductive output (50%) if the female is inseminated: no change to energy reserve mass
                                #if the female is not inseminated, all the surplus assimilation is channeled to structural growth
                                #no need to cap the structural mass, as it does not grow beyond (apart from a very small amount) the limit ("currentadultsize") because of the <if> condition above
                                #this means, as the female grows beyond the maximum structural mass the if condition above is activated and it does not have a structural growth allocation routine

                                if currentinseminationstate == 1:

                                    #surplus assimilation is equally channeled between reproduction and structural growth for inseminated females
                                    currentreproductiveallocation = currentreproductiveallocation + 0.50 * currentgrowthrate
                                    currentstructuralmass = currentstructuralmass + 0.50 * currentgrowthrate
                                
                                else:

                                    #for non-inseminated females, all surplus assimilation is allocated to structural growth
                                    currentstructuralmass = currentstructuralmass + currentgrowthrate

                                #end if
                            #end if
                        
                        else:

                            #here, the growth rate is negative, and hence no eggs are produced
                            #the metabolic demands are balanced by the reserves (if any) or by structural degrowth
                            if currentreservemass >= abs(currentgrowthrate):
                            
                                #if reserves can balance the degrowth, update the reservemass with no change to structural mass
                                #despite the "+" operator, the operation is effectively a subtraction because the growth rate is negative
                                currentreservemass = currentreservemass + currentgrowthrate
                            
                            else:

                                #if the reserves are not sufficient to balance the degrowth, then structural mass is catabolized to meet the degrowth (metabolic demands)
                                #only update the structural mass, as no change occur to reserve mass
                                #despite the "+" operator, the operation is effectively a subtraction because the growth rate is negative
                                currentstructuralmass = currentstructuralmass + currentgrowthrate
                            
                            #end if

                        #end if
                    
                    #end if
                    
                    #dsc-IV:reproduction and spawning submodel - part-B: mate selection and spawning
                    #-------------------------------------------------------------------------------
                    #this applies to the adult female only
                    #if the female is non-inseminated, it finds a male (randomly) in the proximity and mates
                    #during mating the male 'genome' is copied to the female into a state variable 
                    #if the female is inseminated, it produces eggs using the reproductive allocation upadted above
                    #the else condition ('currentsex == male') is not written as it returns nothing
                    
                    if currentsex == "F":

                        if currentinseminationstate == 0:

                            #these are non-inseminated females that needs to find a male to mate
                            #this routine finds a mate for the female, but the spawning does not happen until the next timepoint
                            #this returns the identities (index positions of males in the subpopulation)
                            #nb:no inter-subpopulation geneflow is coded yet (this should be done in a future version)
                            #nb:mate-finding is also simple: a proximity-driven search should be coded in a future version
                            maleids = np.array(np.where(sex[:, currentsubpopulation] == "M")).squeeze()
                            nmales = maleids.size

                            #the mate finding is performed only if there are males in the subpopulation
                            #the else condition ('nmales == 0') is not written as it returns nothing
                            if nmales > 0:
                                
                                #this uses a module-driven function to randomly select a male from a non-proximity-drive (npd) routine
                                #npd routine is the simplest form; an advanced routine based on proximity to be designed at a later stage
                                selectedmale = rp.mateselection_npd(malelist = maleids)
                                
                                #this updates the insemination state and the state variable therein
                                currentinseminationstate = 1
                                inseminationstate[currentsupindividual, currentsubpopulation] = currentinseminationstate

                                #this simulates insemination, where the male genome is copied to the female (into the state variable 'malegenome')
                                #these genes are used in recombination process in the seeding and spawning submodel
                                #updating the state variable 'malegenome' in-place
                                malegenome[currentsupindividual, currentsubpopulation, 0] = a1_bodysize[selectedmale, currentsubpopulation]
                                malegenome[currentsupindividual, currentsubpopulation, 1] = a2_irradiancesensitivity[selectedmale, currentsubpopulation]
                                malegenome[currentsupindividual, currentsubpopulation, 2] = a3_pred1sensitivity[selectedmale, currentsubpopulation]
                                malegenome[currentsupindividual, currentsubpopulation, 3] = a4_pred1reactivity[selectedmale, currentsubpopulation]
                                malegenome[currentsupindividual, currentsubpopulation, 4] = a5_energyallocation[selectedmale, currentsubpopulation]
                                malegenome[currentsupindividual, currentsubpopulation, 5] = a6_diapauseprobability[selectedmale, currentsubpopulation]
                                malegenome[currentsupindividual, currentsubpopulation, 6] = a7_diapauseentry[selectedmale, currentsubpopulation]
                                malegenome[currentsupindividual, currentsubpopulation, 7] = a8_diapauseexit[selectedmale, currentsubpopulation]

                            #end if

                        else:

                            #for inseminated females, egg production may occur depending on the reproductive allocation
                            #the else condition ('currentreproductiveallocation < eggmass') is not written as it returns nothing 
                            
                            if currentreproductiveallocation >= eggmass:
                            
                                #spawning can occur since there is a surplus allocation for this purpose
                                #nb:int does not round up or down - it takes the significant digit: therefore, a d0 rounded value is converted to int
                                #assimilated energy cannot vanish: so, the reproductive allocation is re-updated with the remainder
                                currentpotentialfecundity = currentreproductiveallocation / eggmass
                                currentreproductiveallocation = currentreproductiveallocation - (eggmass * currentpotentialfecundity)
                                currentpotentialfecundity = int(np.round(currentpotentialfecundity, decimals = 0))
                                currenttotalfecundity = currenttotalfecundity + currentpotentialfecundity

                                #this updates the state variable 'potentialfecundity' in-place
                                #reproductive allocation is updated below, as it is updated with or without egg production (this is sort of a minor flaw in the model - i.e., a minor energy balancing issue)
                                potentialfecundity[currentsupindividual, currentsubpopulation] = currentpotentialfecundity
                                #this updates the state variable 'total fecundity'
                                totalfecundity[currentsupindividual, currentsubpopulation] = currenttotalfecundity
                            
                            #end if

                        #end if

                    #end if
                    
                    #dsc-IV:survival submodel
                    #------------------------
                    #this estimates the visual predator density ("pred1dens") as a probability of death sliced from the 4d array (<time, depth, lon, lat>)
                    currentpred1dens = pred1dens[currenttime, currentzidx, currentxidx, currentyidx]
                    #this estimates the normalized and range-scaled (0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
                    currentpred1lightdep = pred1lightdep[currenttime, currentzidx, currentxidx, currentyidx]
                    #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                    
                    #this uses a modular function to estimate the total mortality risk faced by the super individual (as a probability of death)
                    currentmortalityrisk = sv.mortalityrisk_dsc4(strmass = currentstructuralmass, 
                                                                maxstrmass = currentmaxstructuralmass, 
                                                                resmass = currentreservemass, 
                                                                p1dens = currentpred1dens, 
                                                                p1lightdp = currentpred1lightdep, 
                                                                p2risk = pred2risk, 
                                                                bgmrisk = bgmortalityrisk)

                    #the total mortality risk translates to the death of virtual individuals contained in a given super individual
                    #when all virtual individuals contained in a super individual dies, then the super individual also dies
                    #this death is simulated after stage-specific processes 
                    currentnvindividuals = currentnvindividuals - int(currentnvindividuals * currentmortalityrisk)

                    #nb:when a state variable query is opened, that must be closed before quitting the stage grouping!
                    #nb:some state variables are conditionally updated in-place
                    #nb:since the super individual is not dead, the "lifestatus" state variable does not change from the initial value of 1
                    #nb:the "maxstrucuralmass" and "sex" state variables are not updated here as they are not altered in the dsc-IV ('sex' is updated conditionally if defined at stage category entry)
                    structuralmass[currentsupindividual, currentsubpopulation] = currentstructuralmass
                    reservemass[currentsupindividual, currentsubpopulation] = currentreservemass
                    nvindividuals[currentsupindividual, currentsubpopulation] = currentnvindividuals
                    age[currentsupindividual, currentsubpopulation] = currentage

                    xpos[currentsupindividual, currentsubpopulation] = currentxpos
                    ypos[currentsupindividual, currentsubpopulation] = currentypos
                    zpos[currentsupindividual, currentsubpopulation] = currentzpos

                    #data logging
                    #happens only if the 'datalogger' is enabled (val = 1)
                    if datalogger == 1:

                        #nb:at adult stage ('developmentalstage = 12'), indexing code for datalogger for female ("F") is 12 and male ("M") is 13
                        if currentsex == "F":
                            
                            #female developmental stage index no. is 12
                            #logging population size by sequentual addition
                            #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                            spatialdistribution_ps[12, currentxidx, currentyidx, currentzidx, currenttime] += 1

                            #logging biomass (gC) by sequential addition
                            #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                            #nb:reserve-driven biomass can be logged separately if need be
                            currenttotalmass = ((currentstructuralmass + currentreservemass) * currentnvindividuals) / 1e6
                            spatialdistribution_bm[12, currentxidx, currentyidx, currentzidx, currenttime] += currenttotalmass

                        else:

                            #male developmental stage index no. is 13
                            #logging population size by sequentual addition
                            #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                            spatialdistribution_ps[13, currentxidx, currentyidx, currentzidx, currenttime] += 1

                            #logging biomass (gC) by sequential addition
                            #nb:indexing done as: <stage.s> <longitude.x> <latitude.y> <depth.z> <time.t>
                            #nb:reserve-driven biomass can be logged separately if need be
                            currenttotalmass = ((currentstructuralmass + currentreservemass) * currentnvindividuals) / 1e6
                            spatialdistribution_bm[13, currentxidx, currentyidx, currentzidx, currenttime] += currenttotalmass

                        #end if
                    
                    #end if

                #end if
                
                #post-developmental-stage processing
                #___________________________________

                #simulation of death of super individuals (i.e., when all virtual individual dies, a super individual also dies)
                #all state variables, gene values, loggers etc. are reset for a new super individual to take its place (these do not need to be re-initialized at seeding/spawning; only the 'gene' values do)
                #nb:total fecundity has to be re-called here because it is a stage-specific condition (dsc4) and if not, the value from a previous super individual will erroneously be used in the condition below
                currenttotalfecundity = totalfecundity[currentsupindividual, currentsubpopulation]

                #this simulates death of a super individual in three potential ways
                #death by natural causes (predation, starvation) or death imposed by an age limit of 1.5 years or death imposed by a fecundity limit of 1000 eggs per lifespan (only for dsc4 female)
                if currentnvindividuals <= 0 or currentage >= ageceiling or currenttotalfecundity >= fecundityceiling:

                    #here, the super individual die
                    currentlifestatus = 0

                    #all state variables are reset to their respective default values
                    lifestatus[currentsupindividual, currentsubpopulation] = currentlifestatus
                    developmentalstage[currentsupindividual, currentsubpopulation] = 0
                    thermalhistory[currentsupindividual, currentsubpopulation] = 0.00
                    nvindividuals[currentsupindividual, currentsubpopulation] = 0
                    age[currentsupindividual, currentsubpopulation] = 0
                    sex[currentsupindividual, currentsubpopulation] = "U"
                    structuralmass[currentsupindividual, currentsubpopulation] = 0.00
                    maxstructuralmass[currentsupindividual, currentsubpopulation] = 0.00
                    reservemass[currentsupindividual, currentsubpopulation] = 0.00
                    timeofdiapauseentry[currentsupindividual, currentsubpopulation] = 0
                    developmentalstageatdiapauseentry[currentsupindividual, currentsubpopulation] = 0
                    timeofdiapauseexit[currentsupindividual, currentsubpopulation] = 0
                    structuralmassatdiapauseentry[currentsupindividual, currentsubpopulation] = 0.00
                    reservemassatdiapauseentry[currentsupindividual, currentsubpopulation] = 0.00
                    reservemassatdiapauseexit[currentsupindividual, currentsubpopulation] = 0.00
                    diapausedepth[currentsupindividual, currentsubpopulation] = 0
                    diapausestate[currentsupindividual, currentsubpopulation] = "U"
                    diapausestrategy[currentsupindividual, currentsubpopulation] = -1
                    inseminationstate[currentsupindividual, currentsubpopulation] = 0
                    reproductiveallocation[currentsupindividual, currentsubpopulation] = 0.00
                    malegenome[currentsupindividual, currentsubpopulation, :] = 0.00
                    potentialfecundity[currentsupindividual, currentsubpopulation] = 0
                    realizedfecundity[currentsupindividual, currentsubpopulation] = 0
                    xpos[currentsupindividual, currentsubpopulation] = 0.00
                    ypos[currentsupindividual, currentsubpopulation] = 0.00
                    zpos[currentsupindividual, currentsubpopulation] = 0

                    #all dynamic evolvable attributes are reset to their default states
                    a1_bodysize[currentsupindividual, currentsubpopulation] = 0.00
                    a2_irradiancesensitivity[currentsupindividual, currentsubpopulation] = 0.00
                    a3_pred1sensitivity[currentsupindividual, currentsubpopulation] = 0.00
                    a4_pred1reactivity[currentsupindividual, currentsubpopulation] = 0.00
                    a5_energyallocation[currentsupindividual, currentsubpopulation] = 0.00
                    a6_diapauseprobability[currentsupindividual, currentsubpopulation] = 0.00
                    a7_diapauseentry[currentsupindividual, currentsubpopulation] = 0.00
                    a8_diapauseexit[currentsupindividual, currentsubpopulation] = 0.00
                
                #end if
            
            #end for
        
        #end if

        #breaking condition, processing and writing log files to disk
        #_____________________________________________________________

        if currenttime == ntime - 1 and currentyear == nyears:
            
            #initialize the breaking condition for the time loop
            timecondition = False
            
            #file processing and writing occurs only if the datalogging is enabled (it is usually enabled in the final calendar year)
            if datalogger == 1:
                
                #file-writing status print
                termcolor.cprint(text = "[WRITING FILES TO DISK - THIS MAY TAKE A WHILE]", color = "light_red")
                
                #this makes a folder by using the unique execution id input by the user as a folder name to write output
                os.mkdir(outputpath / outputfolder)

                #file1: space-, time-, and tage-specific population size (datatype = np.int32)
                #-----------------------------------------------------------------------------
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
                populationsize_ds.warning = "evaluation output - do not use for analyses"

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
                datavar.longname = "estimated stage-, time- and space-specific population size of Calanus finmarchicus"

                stagevar[:] = np.arange(0, 14, 1)
                lonvar[:] = longitudegrade
                latvar[:] = latitudegrade
                depthvar[:] = depthgrade
                timevar[:] = np.arange(0, 1460, 1)
                datavar[:] = spatialdistribution_ps

                #file2: space-, time-, and tage-specific biomass (datatype = np.float32)
                #-------------------------------------------------------------------------------
                #nb: dimensions: <stage> <longitude> <latitude> <depth> <time>
                #datafile creation
                ncfilename = "biomass_" + "sbp_" + str(currentsubpopulation) + ".nc"
                outputfile = outputpath / outputfolder / ncfilename
                biomass_ds = nc.Dataset(outputfile, "w", format = "NETCDF4_CLASSIC")
                
                #writing datafile attributes (add as needed)
                biomass_ds.title = "PASCALv4 output datafile: integrated biomass"
                biomass_ds.subtitle = "subpopulation ID: " + str(currentsubpopulation)
                biomass_ds.project = "NFR Migratory Crossroads"
                biomass_ds.author = "Kanchana Bandra"
                biomass_ds.warning = "evaluation output - do not use for analyses"

                #creating dataset dimensions
                stagedim = biomass_ds.createDimension("devstage", ndevelopmentalstage)
                londim = biomass_ds.createDimension("lon", longitudegrade.size)
                latdim = biomass_ds.createDimension("lat", latitudegrade.size)
                depthdim = biomass_ds.createDimension("depth", depthgrade.size)
                timedim = biomass_ds.createDimension("time", ntime)

                #creating dimensionality variabels & data variables
                stagevar = biomass_ds.createVariable("devstage", np.int32, ("devstage", ))
                stagevar.units = "dim.less"
                stagevar.longname = "developmental stage"

                lonvar = biomass_ds.createVariable("lon", np.float32, ("lon", ))
                lonvar.units = "degrees east"
                lonvar.longname = "longitude"

                latvar = biomass_ds.createVariable("lat", np.float32, ("lat", ))
                latvar.units = "degrees north"
                latvar.longname = "latitude"

                depthvar = biomass_ds.createVariable("depth", np.int32, ("depth", ))
                depthvar.units = "m"
                depthvar.longname = "depth levels"

                timevar = biomass_ds.createVariable("time", np.int32, ("time", ))
                timevar.units = "6 h"
                timevar.longname = "time of year in 6h intervals"

                datavar = biomass_ds.createVariable("biomass", np.float32, ("devstage", "lon", "lat", "depth", "time", ))
                datavar.units = "gC"
                datavar.longname = "estimated stage-, time- and space-specific biomass of Calanus finmarchicus"

                stagevar[:] = np.arange(0, 14, 1)
                lonvar[:] = longitudegrade
                latvar[:] = latitudegrade
                depthvar[:] = depthgrade
                timevar[:] = np.arange(0, 1460, 1)
                datavar[:] = spatialdistribution_bm

                #file3: spatially-integrated time-specific life strategies (datatype = np.int32)
                #-------------------------------------------------------------------------------
                #horizontal stacking and conversion to a pandas dataframe for easy processing
                lcstrategies = np.hstack(tup = (lcstrategies_i, lcstrategies_f), dtype = np.float32)
                lcstrategies_pd = pd.DataFrame(data = lcstrategies, 
                                            columns = ["nddev", "nden_c4", "nden_c5", "ndex_c4", "ndex_c5", "strm_ddev", "stom_ddev", "strm_den_c4", "stom_den_c4", "strm_den_c5", "stom_den_c5", "stom_dex_civ", "stom_dex_cv"])
                
                #auto-generated path and filename
                txtfilename = "lifestrategies_" + "sbp_" + str(currentsubpopulation) + ".csv"
                outputfile = outputpath / outputfolder / txtfilename

                #writing csv
                lcstrategies_pd.to_csv(path_or_buf = outputfile, index = True, header = True)

                #file-write status print
                termcolor.cprint(text = "[FILE WRITING COMPLETED]", color = "light_red")
            
            #end if
            
        #end if
        
    #end while

#end for

#simulation termination: time-tracking and status print
#_______________________________________________________
execendtime_rec = datetime.now()
execendtime_prt = strftime("%Y-%m-%d %H:%M:%S", gmtime())

termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")
termcolor.cprint(text = f"\nexecution terminated at  : {execendtime_prt} GMT", color = "light_blue")
termcolor.cprint(text = f"time taken for simulation: {execendtime_rec - execstarttime_rec} HH:MM:SS.SSSS", color = "light_blue")
termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")
termcolor.cprint(text = f"\n[SIMULATION COMPLETED]", color = "light_red")
termcolor.cprint(text = "____________________________________________________________________________________", color = "light_blue")