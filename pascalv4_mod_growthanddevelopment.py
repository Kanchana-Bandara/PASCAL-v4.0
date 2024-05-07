########################################################################
#pan arctic behavioural and life-history simulator for calanus (pascal)#
########################################################################

#version 4.00 :: python development :: temporary macmini edition :: merge later
#super-individual-based model for simulating behavioural and life-history strategies of the north atlantic copepod, calanus finmarchicus

#*** module file ***

#module for computing the growth and developmental rates of super individuals in the simulation based on inputdata
#returns two floats: development time (= 1 / development rate) and growth rate
#nb:developmental times are not estimated per-se for stgc 2 and above; because their development rate is inferred from the growth rates
#growth and developmental rate calculation functions are structured by stage categories

def growthanddevelopment_dsc1(temperature: float, devcoef: float, thist: float, strmass: float, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate and developmental time (= 1 / developmental rate) of a super individual in the developmental stage category 1 (dsc1: eggs, nauplius I and II)
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    devcoef                  : the developmental coefficient of the current developmental stage (s, d)
    thist                    : the mean temperature encountered during the early lifespan (s, x, y, z, t)
    strmass                  : the somatic body mass (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)
    devtime                  : estimated current developmental time (6hr)

    """
    
    import math

    #this estimates the developmental time of the super individual using a Belehrádek’s function
    #parameterization based on: Campbell et al. (2001): doi:10.3354/meps221161
    #for source, see: Belehradek, J. (1935). Temperature and living matter (Vol. 8). Berlin: Borntraeger
    devtime = 4.00 * devcoef * (thist + 9.11) ** -2.05

    #this estimates the degrowth rate of the super individual using the formulations and parameterization of:
    #Bandara et al. (2019): https://doi.org/10.1016/j.pocean.2019.02.006 and
    #Maps et al. (2012): https://doi.org/10.1093/icesjms/fsr182
    #nb:degrowth because these early developmental stages do not feed
    #nb:metadj is the adjustment of the metabolic rate to represent the natural reduction therein at earlier developmental stages (see Maps et al. 2012 above)
    #nb:-1.00 multiplier is because this growth rate is indeed negative (essentially, the body degrows at a rate similar to reduced basal metabolic rate) 
    mass_metcoef = 0.0008487
    mass_metexpo = 0.7502
    temp_metcoef = 1.2956
    temp_metexpo = 0.1170
    metadj = 0.50

    growthrate_mass = mass_metcoef * strmass ** mass_metexpo
    growthrate_temp = growthrate_mass * temp_metcoef * math.exp(temp_metexpo * temperature)
    growthrate = growthrate_temp * modelres * metadj * -1.00

    return devtime, growthrate

#end def

def growthanddevelopment_dsc2(temperature: float, maxtemperature: float, f1con: float, strmass: float, maxzd: int, actzd: int, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate and basal metabolic rate of a super individual in the developmental stage category 2 (dsc1: NIII-CIII)
    this does not explicitly return the development rate, but it is calculated in the main program using the growth rate

    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    maxtemperature           : maximum ambient temperature
    f1con                    : ambient category#1 food concentration (i.e., phytoplankton)
    strmass                  : the somatic body mass (s, t)
    maxzd                    : the maximum vertical distance searhable by the super individual (s, t)
    actzd                    : the actual vertical distance travelled (actively) by the super individual (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """

    import math
    
    #this estimates the growth rate of the super individual using the formulations and parameterization of:
    #Bandara et al. (2019): https://doi.org/10.1016/j.pocean.2019.02.006 and
    #Maps et al. (2012): https://doi.org/10.1093/icesjms/fsr182
    #nb:depending on the environmental conditions (e.g., food availability) the somatic growth rate can be positive or negative
    mass_ingestioncoef = 0.009283
    temp_ingestioncoef = 1.2392
    mass_ingestionexpo = 0.7524
    temp_ingestionexpo = 0.0966
    chltocarbon = 30
    assimilationcoef = 0.60

    #this estimates the mass aspect of the ingestion rate at a reference temperature of -2.00 C as a power function
    ingestionrate_mass = mass_ingestioncoef * (strmass ** mass_ingestionexpo)
    #this is the temperature scaling of the mass aspect of ingestion rate
    ingestionrate_temp = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * temperature)
    #this is the satiation food concentration per a given body mass
    food1_ingestioncoef = 0.30 * (strmass ** -0.138)
    #this is the food concentration estimated based on chlorophyll-to-carbon conversion factor defined above (this can be a dynamic attribute that reflect the food quality in a future development)
    food1con_carbonunits = f1con * chltocarbon
    #this scales the temperature aspect of the ingestion rate to a range of 0-1 depending on the food availability and satiation state
    ingestionrate_food = ingestionrate_temp * ((food1_ingestioncoef * food1con_carbonunits) / (1.00 + food1_ingestioncoef * food1con_carbonunits))
    #not all ingested food is assimilated; only ca. 60%, see: Huntley and Boyd (1984): https://doi.org/10.1086/284288
    #in theory, this quantity is the gross growth rate (ugC/sind/hr)
    assimilationrate = ingestionrate_food * assimilationcoef

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    ambmscalar = 1.50
    sdascalar = 1.60

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (strmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)
    #this computes the active metabolic rate by considering the distance travelled by the super individual between time 't' and 't + 1'
    #the differential distances are computed in the vertical migration submodel (see vm module)
    #no need of multiplying the active metabolic rate with "modelres" as it works atop the resolution-adjusted basal metabolic rate
    activemetabolicrate = ambmscalar * basalmetabolicrate * (actzd / maxzd)
    #this estimates the specific dynamic action (sda) - which scales with the ingestion rate
    #the max. value for the scalar is 1.60 (based on Thor, 2002: https://doi.org/10.1016/S0022-0981(02)00131-4) 
    #this is scaled based on the max. ingestion rate (at highest temperature, no food limitation) to current ingestion rate ratio
    ingestionrate_max = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * maxtemperature)
    sda = basalmetabolicrate * sdascalar * (ingestionrate_food / ingestionrate_max)

    #the net growth rate is the difference between the assimilation rate and the total metabolic rate
    growthrate = (modelres * assimilationrate) - (basalmetabolicrate + activemetabolicrate + sda)
    
    return growthrate

#end def

def growthanddevelopment_dsc3a(temperature: float, maxtemperature: float, f1con: float, strmass: float, resmass: float, maxzd: int, actzd: int, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate and basal metabolic rate of a super individual in the developmental stage category 2 (dsc1: NIII-CIII)
    this does not explicitly return the development rate, but it is calculated in the main program using the growth rate

    note:
    the only difference between the dsc2 and dsc3a growth and development functions is that in dsc3a, the metabolic rate is estimated as function of total body mass (strucural mass + reserve mass)
    for super individuals in dsc2, no energy reserves exist

    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    maxtemperature           : maximum ambient temperature
    f1con                    : ambient category#1 food concentration (i.e., phytoplankton)
    strmass                  : the somatic body mass (s, t)
    resmass                  : the mass of the energy reserve (s, t)
    maxzd                    : the maximum vertical distance searhable by the super individual (s, t)
    actzd                    : the actual vertical distance travelled (actively) by the super individual (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """

    import math
    
    #this estimates the growth rate of the super individual using the formulations and parameterization of:
    #Bandara et al. (2019): https://doi.org/10.1016/j.pocean.2019.02.006 and
    #Maps et al. (2012): https://doi.org/10.1093/icesjms/fsr182
    #nb:depending on the environmental conditions (e.g., food availability) the somatic growth rate can be positive or negative
    mass_ingestioncoef = 0.009283
    temp_ingestioncoef = 1.2392
    mass_ingestionexpo = 0.7524
    temp_ingestionexpo = 0.0966
    chltocarbon = 30
    assimilationcoef = 0.60


    #this estimates the mass aspect of the ingestion rate at a reference temperature of -2.00 C as a power function
    ingestionrate_mass = mass_ingestioncoef * (strmass ** mass_ingestionexpo)
    #this is the temperature scaling of the mass aspect of ingestion rate
    ingestionrate_temp = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * temperature)
    #this is the satiation food concentration per a given body mass
    food1_ingestioncoef = 0.30 * (strmass ** -0.138)
    #this is the food concentration estimated based on chlorophyll-to-carbon conversion factor defined above (this can be a dynamic attribute that reflect the food quality in a future development)
    food1con_carbonunits = f1con * chltocarbon
    #this scales the temperature aspect of the ingestion rate to a range of 0-1 depending on the food availability and satiation state
    ingestionrate_food = ingestionrate_temp * ((food1_ingestioncoef * food1con_carbonunits) / (1.00 + food1_ingestioncoef * food1con_carbonunits))
    #not all ingested food is assimilated; only ca. 60%, see: Huntley and Boyd (1984): https://doi.org/10.1086/284288
    #in theory, this quantity is the gross growth rate (ugC/sind/hr)
    assimilationrate = ingestionrate_food * assimilationcoef

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    ambmscalar = 1.50
    sdascalar = 1.60

    #this estimates the total body mass of the super individual
    totalmass = strmass + resmass

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (totalmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)
    #this computes the active metabolic rate by considering the distance travelled by the super individual between time 't' and 't + 1'
    #the differential distances are computed in the vertical migration submodel (see vm module)
    #no need of multiplying the active metabolic rate with "modelres" as it works atop the resolution-adjusted basal metabolic rate
    activemetabolicrate = ambmscalar * basalmetabolicrate * (actzd / maxzd)
    #this estimates the specific dynamic action (sda) - which scales with the ingestion rate
    #the max. value for the scalar is 1.60 (based on Thor, 2002: https://doi.org/10.1016/S0022-0981(02)00131-4) 
    #this is scaled based on the max. ingestion rate (at highest temperature, no food limitation) to current ingestion rate ratio
    ingestionrate_max = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * maxtemperature)
    sda = basalmetabolicrate * sdascalar * (ingestionrate_food / ingestionrate_max)

    #the net growth rate is the difference between the assimilation rate and the total metabolic rate
    growthrate = (modelres * assimilationrate) - (basalmetabolicrate + activemetabolicrate + sda)
    
    return growthrate

#end def

def growthanddevelopment_dsc3e(temperature: float, strmass: float, resmass: float, actzd: int, maxzd: int, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate of a super individual in the developmental stage category IIIE(civ and cv at diapause entry)
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxzd                    : the maximum vertical distance searhable by the super individual (s, t)
    actzd                    : the actual vertical distance travelled (actively) by the super individual (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """
    
    import math

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    ambmscalar = 1.50

    #this estimates the total body mass of the super individual
    totalmass = strmass + resmass

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (totalmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)
    #this computes the active metabolic rate by considering the distance travelled by the super individual between time 't' and 't + 1'
    #the differential distances are computed in the vertical migration submodel (see vm module)
    activemetabolicrate = ambmscalar * basalmetabolicrate * (actzd / maxzd)

    #the -1.00 scalar because of a potential degrowth
    growthrate = -1.00 * (basalmetabolicrate + activemetabolicrate)

    return growthrate

#end def

def growthanddevelopment_dsc3d(temperature: float, strmass: float, resmass: float, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate of a super individual in the developmental stage category IIID(civ and cv at diapause)
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """
    
    import math

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    metadj = 0.25

    #this estimates the total body mass of the super individual
    totalmass = strmass + resmass

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (totalmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #nb:there is no active metabolic rate at diapause stage due to physical inactivity, see: Hirche, (1996) https://doi.org/10.1080/00785326.1995.10429843
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)

    #the -1.00 scalar because of a potential degrowth
    #"metadj" is the adjustment of metabolic rate at diapause
    growthrate = -1.00 * basalmetabolicrate * metadj

    return growthrate

#end def

def growthanddevelopment_dsc3x(temperature: float, strmass: float, resmass: float, actzd: int, maxzd: int, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate of a super individual in the developmental stage category IIIE(civ and cv at diapause entry)
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxzd                    : the maximum vertical distance searhable by the super individual (s, t)
    actzd                    : the actual vertical distance travelled (actively) by the super individual (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """
    
    import math

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    ambmscalar = 1.50

    #this estimates the total body mass of the super individual
    totalmass = strmass + resmass

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (totalmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)
    #this computes the active metabolic rate by considering the distance travelled by the super individual between time 't' and 't + 1'
    #the differential distances are computed in the vertical migration submodel (see vm module)
    activemetabolicrate = ambmscalar * basalmetabolicrate * (actzd / maxzd)

    #the -1.00 scalar because of a potential degrowth
    growthrate = -1.00 * (basalmetabolicrate + activemetabolicrate)

    return growthrate

#end def

def growthanddevelopment_dsc3p(temperature: float, maxtemperature: float, f1con: float, strmass: float, resmass: float, maxzd: int, actzd: int, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate and basal metabolic rate of a super individual in the developmental stage category 2 (dsc1: NIII-CIII)
    this does not explicitly return the development rate, but it is calculated in the main program using the growth rate

    note:
    the only difference between the dsc2 and dsc3a growth and development functions is that in dsc3a, the metabolic rate is estimated as function of total body mass (strucural mass + reserve mass)
    for super individuals in dsc2, no energy reserves exist

    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    maxtemperature           : maximum ambient temperature
    f1con                    : ambient category#1 food concentration (i.e., phytoplankton)
    strmass                  : the somatic body mass (s, t)
    resmass                  : the mass of the energy reserve (s, t)
    maxzd                    : the maximum vertical distance searhable by the super individual (s, t)
    actzd                    : the actual vertical distance travelled (actively) by the super individual (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """

    import math
    
    #this estimates the growth rate of the super individual using the formulations and parameterization of:
    #Bandara et al. (2019): https://doi.org/10.1016/j.pocean.2019.02.006 and
    #Maps et al. (2012): https://doi.org/10.1093/icesjms/fsr182
    #nb:depending on the environmental conditions (e.g., food availability) the somatic growth rate can be positive or negative
    mass_ingestioncoef = 0.009283
    temp_ingestioncoef = 1.2392
    mass_ingestionexpo = 0.7524
    temp_ingestionexpo = 0.0966
    chltocarbon = 30
    assimilationcoef = 0.60


    #this estimates the mass aspect of the ingestion rate at a reference temperature of -2.00 C as a power function
    ingestionrate_mass = mass_ingestioncoef * (strmass ** mass_ingestionexpo)
    #this is the temperature scaling of the mass aspect of ingestion rate
    ingestionrate_temp = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * temperature)
    #this is the satiation food concentration per a given body mass
    food1_ingestioncoef = 0.30 * (strmass ** -0.138)
    #this is the food concentration estimated based on chlorophyll-to-carbon conversion factor defined above (this can be a dynamic attribute that reflect the food quality in a future development)
    food1con_carbonunits = f1con * chltocarbon
    #this scales the temperature aspect of the ingestion rate to a range of 0-1 depending on the food availability and satiation state
    ingestionrate_food = ingestionrate_temp * ((food1_ingestioncoef * food1con_carbonunits) / (1.00 + food1_ingestioncoef * food1con_carbonunits))
    #not all ingested food is assimilated; only ca. 60%, see: Huntley and Boyd (1984): https://doi.org/10.1086/284288
    #in theory, this quantity is the gross growth rate (ugC/sind/hr)
    assimilationrate = ingestionrate_food * assimilationcoef

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    ambmscalar = 1.50
    sdascalar = 1.60

    #this estimates the total body mass of the super individual
    totalmass = strmass + resmass

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (totalmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)
    #this computes the active metabolic rate by considering the distance travelled by the super individual between time 't' and 't + 1'
    #the differential distances are computed in the vertical migration submodel (see vm module)
    #no need of multiplying the active metabolic rate with "modelres" as it works atop the resolution-adjusted basal metabolic rate
    activemetabolicrate = ambmscalar * basalmetabolicrate * (actzd / maxzd)
    #this estimates the specific dynamic action (sda) - which scales with the ingestion rate
    #the max. value for the scalar is 1.60 (based on Thor, 2002: https://doi.org/10.1016/S0022-0981(02)00131-4) 
    #this is scaled based on the max. ingestion rate (at highest temperature, no food limitation) to current ingestion rate ratio
    ingestionrate_max = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * maxtemperature)
    sda = basalmetabolicrate * sdascalar * (ingestionrate_food / ingestionrate_max)

    #the net growth rate is the difference between the assimilation rate and the total metabolic rate
    growthrate = (modelres * assimilationrate) - (basalmetabolicrate + activemetabolicrate + sda)
    
    return growthrate

#end def

def growthanddevelopment_dsc4f(temperature: float, maxtemperature: float, f1con: float, strmass: float, resmass: float, maxzd: int, actzd: int, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate and basal metabolic rate of a super individual in the developmental stage category 4f (dsc-IV: adult female)
    this does not explicitly return the development rate, but it is calculated in the main program using the growth rate

    note:
    the only difference between the dsc2 and dsc3a growth and development functions is that in dsc3a, the metabolic rate is estimated as function of total body mass (strucural mass + reserve mass)
    for super individuals in dsc2, no energy reserves exist

    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    maxtemperature           : maximum ambient temperature
    f1con                    : ambient category#1 food concentration (i.e., phytoplankton)
    strmass                  : the somatic body mass (s, t)
    resmass                  : the mass of the energy reserve (s, t)
    maxzd                    : the maximum vertical distance searhable by the super individual (s, t)
    actzd                    : the actual vertical distance travelled (actively) by the super individual (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """

    import math
    
    #this estimates the growth rate of the super individual using the formulations and parameterization of:
    #Bandara et al. (2019): https://doi.org/10.1016/j.pocean.2019.02.006 and
    #Maps et al. (2012): https://doi.org/10.1093/icesjms/fsr182
    #nb:depending on the environmental conditions (e.g., food availability) the somatic growth rate can be positive or negative
    mass_ingestioncoef = 0.009283
    temp_ingestioncoef = 1.2392
    mass_ingestionexpo = 0.7524
    temp_ingestionexpo = 0.0966
    chltocarbon = 30
    assimilationcoef = 0.60


    #this estimates the mass aspect of the ingestion rate at a reference temperature of -2.00 C as a power function
    ingestionrate_mass = mass_ingestioncoef * (strmass ** mass_ingestionexpo)
    #this is the temperature scaling of the mass aspect of ingestion rate
    ingestionrate_temp = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * temperature)
    #this is the satiation food concentration per a given body mass
    food1_ingestioncoef = 0.30 * (strmass ** -0.138)
    #this is the food concentration estimated based on chlorophyll-to-carbon conversion factor defined above (this can be a dynamic attribute that reflect the food quality in a future development)
    food1con_carbonunits = f1con * chltocarbon
    #this scales the temperature aspect of the ingestion rate to a range of 0-1 depending on the food availability and satiation state
    ingestionrate_food = ingestionrate_temp * ((food1_ingestioncoef * food1con_carbonunits) / (1.00 + food1_ingestioncoef * food1con_carbonunits))
    #not all ingested food is assimilated; only ca. 60%, see: Huntley and Boyd (1984): https://doi.org/10.1086/284288
    #in theory, this quantity is the gross growth rate (ugC/sind/hr)
    assimilationrate = ingestionrate_food * assimilationcoef

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    ambmscalar = 1.50
    sdascalar = 1.60

    #this estimates the total body mass of the super individual
    totalmass = strmass + resmass

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (totalmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)
    #this computes the active metabolic rate by considering the distance travelled by the super individual between time 't' and 't + 1'
    #the differential distances are computed in the vertical migration submodel (see vm module)
    #no need of multiplying the active metabolic rate with "modelres" as it works atop the resolution-adjusted basal metabolic rate
    activemetabolicrate = ambmscalar * basalmetabolicrate * (actzd / maxzd)
    #this estimates the specific dynamic action (sda) - which scales with the ingestion rate
    #the max. value for the scalar is 1.60 (based on Thor, 2002: https://doi.org/10.1016/S0022-0981(02)00131-4) 
    #this is scaled based on the max. ingestion rate (at highest temperature, no food limitation) to current ingestion rate ratio
    ingestionrate_max = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * maxtemperature)
    sda = basalmetabolicrate * sdascalar * (ingestionrate_food / ingestionrate_max)

    #the net growth rate is the difference between the assimilation rate and the total metabolic rate
    growthrate = (modelres * assimilationrate) - (basalmetabolicrate + activemetabolicrate + sda)
    
    return growthrate

#end def

def growthanddevelopment_dsc4m(temperature: float, f1con: float, strmass: float, resmass: float, maxzd: int, actzd: int, modelres: int) -> float:

    """
    functionality:
    this returns the estimated growth rate and basal metabolic rate of a super individual in the developmental stage category 4m (dsc-IV: adult male)
    this does not explicitly return the development rate, but it is calculated in the main program using the growth rate

    note:
    the only difference between the dsc2 and dsc3a growth and development functions is that in dsc3a, the metabolic rate is estimated as function of total body mass (strucural mass + reserve mass)
    for super individuals in dsc2, no energy reserves exist

    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    temperature              : ambient temperature (4D: t, z, x, y)
    f1con                    : ambient category#1 food concentration (i.e., phytoplankton)
    strmass                  : the somatic body mass (s, t)
    resmass                  : the mass of the energy reserve (s, t)
    maxzd                    : the maximum vertical distance searhable by the super individual (s, t)
    actzd                    : the actual vertical distance travelled (actively) by the super individual (s, t)
    modelres                 : the temporal resolution of the model
    
    return:
    growthrate               : estimated current somatic growth rate (ugC/sup.ind/6-hr)

    """

    import math
    
    #this estimates the growth rate of the super individual using the formulations and parameterization of:
    #Bandara et al. (2019): https://doi.org/10.1016/j.pocean.2019.02.006 and
    #Maps et al. (2012): https://doi.org/10.1093/icesjms/fsr182
    #nb:depending on the environmental conditions (e.g., food availability) the somatic growth rate can be positive or negative
    # mass_ingestioncoef = 0.009283
    # temp_ingestioncoef = 1.2392
    # mass_ingestionexpo = 0.7524
    # temp_ingestionexpo = 0.0966
    # chltocarbon = 30
    # assimilationcoef = 0.60


    # #this estimates the mass aspect of the ingestion rate at a reference temperature of -2.00 C as a power function
    # ingestionrate_mass = mass_ingestioncoef * (strmass ** mass_ingestionexpo)
    # #this is the temperature scaling of the mass aspect of ingestion rate
    # ingestionrate_temp = ingestionrate_mass * temp_ingestioncoef * math.exp(temp_ingestionexpo * temperature)
    # #this is the satiation food concentration per a given body mass
    # food1_ingestioncoef = 0.30 * (strmass ** -0.138)
    # #this is the food concentration estimated based on chlorophyll-to-carbon conversion factor defined above (this can be a dynamic attribute that reflect the food quality in a future development)
    # food1con_carbonunits = f1con * chltocarbon
    # #this scales the temperature aspect of the ingestion rate to a range of 0-1 depending on the food availability and satiation state
    # ingestionrate_food = ingestionrate_temp * ((food1_ingestioncoef * food1con_carbonunits) / (1.00 + food1_ingestioncoef * food1con_carbonunits))
    # #not all ingested food is assimilated; only ca. 60%, see: Huntley and Boyd (1984): https://doi.org/10.1086/284288
    # #in theory, this quantity is the gross growth rate (ugC/sind/hr)
    # assimilationrate = ingestionrate_food * assimilationcoef

    #to estimate the net somatic growth rate, an estimation of the total metabolic rate is needed
    #here, the total metabolic rate is the sum of basal metabolic rate and the active metabolic rate
    #the active metabolic rate is calculated depending on the ratio between actual vertical distance travelled and the theoreitical maximum distance that can be travelled by a super individual
    #this can be adjusted later for upwelling and downwelling assist (cf. prospective DEEP project)
    mass_metaboliccoef = 0.0008487
    temp_metaboliccoef = 1.2956
    mass_metabolicexpo = 0.7502
    temp_metabolicexpo = 0.1170
    ambmscalar = 1.50

    #this estimates the total body mass of the super individual
    totalmass = strmass + resmass

    #this is the mass aspect of basal metabolic rate at a reference temperature of -2.00 C
    basalmetabolicrate_mass = mass_metaboliccoef * (totalmass ** mass_metabolicexpo)
    #this is the temperature scaling of the mass aspect of the basal metabolic rate
    #the scalar 6.00 adjusts the estimates to the 6 hr model resolution
    basalmetabolicrate = modelres * basalmetabolicrate_mass * temp_metaboliccoef * math.exp(temp_metabolicexpo * temperature)
    #this computes the active metabolic rate by considering the distance travelled by the super individual between time 't' and 't + 1'
    #the differential distances are computed in the vertical migration submodel (see vm module)
    #no need of multiplying the active metabolic rate with "modelres" as it works atop the resolution-adjusted basal metabolic rate
    activemetabolicrate = ambmscalar * basalmetabolicrate * (actzd / maxzd)

    #the net growth rate is the difference between the assimilation rate and the total metabolic rate
    growthrate = -1.00 * (basalmetabolicrate + activemetabolicrate)
    
    return growthrate

#end def