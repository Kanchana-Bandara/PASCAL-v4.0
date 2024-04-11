########################################################################
#pan arctic behavioural and life-history simulator for calanus (pascal)#
########################################################################

#version 4.00 :: python development :: temporary macmini edition :: merge later
#super-individual-based model for simulating behavioural and life-history strategies of the north atlantic copepod, calanus finmarchicus

#*** module file ***

#module for computing the vertical position of super individuals in the simulation based on inputdata
#returns an integer, which is the absolute vertical position of a super individual (nb: this does not match the depth grading of the model environment, should be done in main.py)
#vertical migration functions are structured by stage categories

def verticalmigration_dsc1(smld: int) -> int:

    """
    functionality:
    this returns the absolute and relative (index) vertical positions of a super individual in the developmental stage category I (dsc1: eggs, nauplius I and II)
    this is not diel or seasonal vertical migrations - at these stages, super individuals are too small and do not have resources to perform notable active vertical movements
    
    args:
    smld                        : current surface mixed layer depth
    
    return:
    zpos                        : estimated current vertical position of the super individual

    """
    
    import numpy as np
    
    #these are the standard depths of the model environment (similar to variable "depthgrade" in the main program)
    depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    maxdepth = np.max(depthrange)

    #this estimates the absolute and relative vertical positions of the super individual as a function of mixed layer depth
    #nb:thsis routine needs an update (assumptions are too much and formulation is too simple)
    if smld > maxdepth:
        
        ztrim = maxdepth
        depthrangeidx = np.arange(0, 37, 1)
        zidx = np.random.choice(a = depthrangeidx, size = 1).squeeze()
        zpos = depthrange[zidx]
    
    elif smld == 0:

        zidx = 0
        zpos = depthrange[zidx]

    else:

        ztrim = np.argmin(abs(depthrange - smld))
        depthrangeidx = np.arange(0, ztrim, 1)
        zidx = np.random.choice(a = depthrangeidx, size = 1).squeeze()
        zpos = depthrange[zidx]

    #end if

    return zpos, zidx

#end def

def verticalmigration_dsc2(temprange: float, f1conrange: float, iradrange: float, maxirad: float, p1dnsrange: float, a2: float, a3: float, a4: float, pvp: int, pvi: int, strmass: float, modelres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category II (dsc1: NIII-CIII)
    
    args:
    temprange                   : range of temperatures across all depth layers at a given longitude, latitude and time
    f1conrange                  : range of food concentration category 1 (= phytoplankton) at a given longitude, latitude and time
    iradrange                   : range of shortwave solar irradiance at a given longitude, latitude and time
    p1dnsrange                  : range of category 1 predator density at a given longitude, latitude and time
    a2                          : evolvable attribute for irradiance sensitivity of a given super individual
    a3                          : evolvable attribute for predator sensitivity of a given super individual
    a4                          : evolvable attribute for predator reactivity of a given super individual
    pvp                         : absolute vertical position at the previous timepoint (the function calculates the current vertical position)
    pvi                         : relative vertical position (index) at the previous timepoint
    strmass                     : structural mass of a given super individual
    modelres                    : the temporal resolution of the model
    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated current vertical position index (relative vertical position)
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """

    import numpy as np
    import math
    
    #these are the standard depths of the model environment (similar to variable "depthgrade" in the main program)
    depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    #this estimates the theoreitical maximum distance that be vertically searched by a given super individual as a power function of somatic body mass
    #nb:the estimate is adjusted to the temporal resolution of the model
    #the function is based on Bandara et al. (2021), which follows a literature review of copepod swimming speeds in natural habitats, see: https://doi.org/10.1016/j.ecolmodel.2021.109739
    maxdistance = int(modelres * 5.2287 * strmass ** 0.4862)
    #this estimates the minimum and maximum points of the 1D water column that the super individual can search
    searchceiling = pvp - maxdistance
    searchfloor = pvp + maxdistance
    #these estimates surface and bottom depths of the model environment
    mindepth = np.min(depthrange)
    maxdepth = np.max(depthrange)

    #both above estimates must be trimmed so that they does not exceed minimum and maximum depths (else, in theory, animals get burried in the sediment like worms or shoot out into the air like goddamn flying fish!!!)
    if searchceiling < mindepth:

        searchceiling = mindepth
    
    #end if
    
    if searchfloor > maxdepth:

        searchfloor = maxdepth
    
    #end if
    
    #these are index positions related to the minimum and maximum points of the water column searchable by a given super individual
    searchceiling_idx = np.argmin(abs(depthrange - searchceiling))
    searchfloor_idx = np.argmin(abs(depthrange - searchfloor))

    #if the vertical search distance is not sufficient to move between model-resolvable adjacent depth layers, then the model assums no movement of the super individual
    if searchceiling_idx == searchfloor_idx:
        
        zpos = pvp
        zidx = pvi
    
    #if not (i.e., search range spans across 2 or more model-resolved depth gradings), the vertical behavioural submodel applies
    else:
        
        #this is the dynamic evolvable attribute scaling function defined by Edirisinghe (2022), where vertical behavioural reactions of super individuals are modified by percieved visual predation risk
        #where behavioural reaction is by default elicited via ambient irradiance & it is altered by predator presence and copepod detection & reaction capabilities therein
        #the asymptotic exponantial scalar expresses the size-dependence of irradiance sensitivity by copepods (where, smaller ones tend to stay closer to the surface and vice versa)
        if p1dnsrange[pvi] >= a3:
        
            thresholdirradiance = a2 * a4 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00))))
        
        else:

            thresholdirradiance = a2 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00))))
        
        #end if
        
        #this is the vertical search range of the super individual given as a numpy array
        searchrange = np.arange(searchceiling_idx, searchfloor_idx)
        #this is the range of irradiance values corresponding to the search range of the super individual    
        iradrange_sr = iradrange[searchrange]
        #these are environmental variables corresponding to the search range of the super individual
        #nb:the celsius is converted to kelvin for growth potential estimation; see bellow
        temprange_sr = temprange[searchrange] + 273.15
        f1conrange_sr = f1conrange[searchrange]

        #the preferred depth by the super individual is the depth that meets or closest to meeting the irraiance threshold with maximum growth potential
        #the growth potential is estimated as a product of temperature and foodconcentration as described in Edirisinghe (2022)
        growthpotential_sr = temprange_sr * f1conrange_sr

        #if at least one depth bin offers a refuge from light and light-dependent predation
        if any(iradrange_sr <= thresholdirradiance):
            
            potentialzpos = np.array(np.where(iradrange_sr <= thresholdirradiance)).squeeze()
            
            if potentialzpos.size == 1:
            
                zidx = searchrange[potentialzpos]
                zpos = depthrange[zidx]
            
            else:
                
                zidx = searchrange[potentialzpos[np.argmax(growthpotential_sr[potentialzpos])]]
                zpos = depthrange[zidx]
            
            #end if
            
        #if none of the searchable depths offer a refuge from light and light-dependent predation
        else:
            
            #super individuals select the deepest depth (as downwelling light is vertically extinct in the water column)
            zidx = searchrange[-1]
            zpos = depthrange[zidx]

        #end if
            
    #end if

    traveldistance = abs(pvp - zpos)

    return zpos, zidx, maxdistance, traveldistance

#end def

def verticalmigration_dsc3a(temprange: float, f1conrange: float, iradrange: float, maxirad: float, p1dnsrange: float, a2: float, a3: float, a4: float, pvp: int, pvi: int, strmass: float, resmass: float, modelres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category III-A (dsc3A: CIV and CV with diapause status "A")

    note:
    the only difference between this dsc3a and the dsc2 functions is that the allometric scaling of irradiance sensitivity operates on the total body mass of the super individual (structural + reserve masses)
    for super individuals at dsc1 and dsc2, they do not possess an energy reserve and the allometric scaling of irradiance sensitivity operates on the structural mass only
    
    args:
    temprange                   : range of temperatures across all depth layers at a given longitude, latitude and time
    f1conrange                  : range of food concentration category 1 (= phytoplankton) at a given longitude, latitude and time
    iradrange                   : range of shortwave solar irradiance at a given longitude, latitude and time
    p1dnsrange                  : range of category 1 predator density at a given longitude, latitude and time
    a2                          : evolvable attribute for irradiance sensitivity of a given super individual
    a3                          : evolvable attribute for predator sensitivity of a given super individual
    a4                          : evolvable attribute for predator reactivity of a given super individual
    pvp                         : absolute vertical position at the previous timepoint (the function calculates the current vertical position)
    pvi                         : relative vertical position (index) at the previous timepoint
    strmass                     : structural mass of a given super individual
    resmass                     : reserve mass of a given super individual
    modelres                    : the temporal resolution of the model
    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated current vertical position index (relative vertical position)
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """

    import numpy as np
    import math
    
    #this is the total bodymass of the super individual
    totalmass = strmass + resmass
    #these are the standard depths of the model environment (similar to variable "depthgrade" in the main program)
    depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    #this estimates the theoreitical maximum distance that be vertically searched by a given super individual as a power function of somatic body mass
    #nb:the estimate is adjusted to the temporal resolution of the model
    #the function is based on Bandara et al. (2021), which follows a literature review of copepod swimming speeds in natural habitats, see: https://doi.org/10.1016/j.ecolmodel.2021.109739
    maxdistance = int(modelres * 5.2287 * strmass ** 0.4862)
    #this estimates the minimum and maximum points of the 1D water column that the super individual can search
    searchceiling = pvp - maxdistance
    searchfloor = pvp + maxdistance
    #these estimates surface and bottom depths of the model environment
    mindepth = np.min(depthrange)
    maxdepth = np.max(depthrange)

    #both above estimates must be trimmed so that they does not exceed minimum and maximum depths (else, in theory, animals get burried in the sediment like worms or shoot out into the air like goddamn flying fish!!!)
    if searchceiling < mindepth:

        searchceiling = mindepth
    
    #end if
    
    if searchfloor > maxdepth:

        searchfloor = maxdepth
    
    #end if
    
    #these are index positions related to the minimum and maximum points of the water column searchable by a given super individual
    searchceiling_idx = np.argmin(abs(depthrange - searchceiling))
    searchfloor_idx = np.argmin(abs(depthrange - searchfloor))

    #if the vertical search distance is not sufficient to move between model-resolvable adjacent depth layers, then the model assums no movement of the super individual
    if searchceiling_idx == searchfloor_idx:
        
        zpos = pvp
        zidx = pvi
    
    #if not (i.e., search range spans across 2 or more model-resolved depth gradings), the vertical behavioural submodel applies
    else:
        
        #this is the dynamic evolvable attribute scaling function defined by Edirisinghe (2022), where vertical behavioural reactions of super individuals are modified by percieved visual predation risk
        #where behavioural reaction is by default elicited via ambient irradiance & it is altered by predator presence and copepod detection & reaction capabilities therein
        #the asymptotic exponantial scalar expresses the size-dependence of irradiance sensitivity by copepods (where, smaller ones tend to stay closer to the surface and vice versa)
        if p1dnsrange[pvi] >= a3:
        
            thresholdirradiance = a2 * a4 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        else:

            thresholdirradiance = a2 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        #end if
        
        #this is the vertical search range of the super individual given as a numpy array
        searchrange = np.arange(searchceiling_idx, searchfloor_idx)
        #this is the range of irradiance values corresponding to the search range of the super individual    
        iradrange_sr = iradrange[searchrange]
        #these are environmental variables corresponding to the search range of the super individual
        #nb:the celsius is converted to kelvin for growth potential estimation; see bellow
        temprange_sr = temprange[searchrange] + 273.15
        f1conrange_sr = f1conrange[searchrange]

        #the preferred depth by the super individual is the depth that meets or closest to meeting the irraiance threshold with maximum growth potential
        #the growth potential is estimated as a product of temperature and foodconcentration as described in Edirisinghe (2022)
        growthpotential_sr = temprange_sr * f1conrange_sr

        #if at least one depth bin offers a refuge from light and light-dependent predation
        if any(iradrange_sr <= thresholdirradiance):
            
            potentialzpos = np.array(np.where(iradrange_sr <= thresholdirradiance)).squeeze()
            
            if potentialzpos.size == 1:
            
                zidx = searchrange[potentialzpos]
                zpos = depthrange[zidx]
            
            else:
                
                zidx = searchrange[potentialzpos[np.argmax(growthpotential_sr[potentialzpos])]]
                zpos = depthrange[zidx]
            
            #end if

        #if none of the searchable depths offer a refuge from light and light-dependent predation
        else:
            
            #super individuals select the deepest depth (as downwelling light is vertically extinct in the water column)
            zidx = searchrange[-1]
            zpos = depthrange[zidx]

        #end if
            
    #end if

    traveldistance = abs(pvp - zpos)

    return zpos, zidx, maxdistance, traveldistance

#end def

def verticalmigration_dsc3e(diapdepth: int, strmass: float, pvp: int, modelres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category IIII and subcategory "E" (civ & cv)
    this is seasonal vertical migration 
    
    args:
    diapdepth                   : preferred diapause depth of the super individual
    pvp                         : absolute vertical position at the previous timepoint (the function calculates the current vertical position)
    strmass                     : structural mass of a given super individual
    modelres                    : the temporal resolution of the model

    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated relative vertical position (index) of the super individual
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """
    
    import numpy as np

    depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    maxdistance = int(modelres * 5.2287 * strmass ** 0.4862)

    #this checks whether the super individual needs an ascent or a descent to reach the desired diapause depth
    #nb:an ascent is unlikely, but coding for that covers for all posibilities

    if pvp == diapdepth:

        #no additional vertical movement is needed; as the super individual is already at the preferred diapause depth
        zpos = diapdepth
        zidx = np.array(np.where(depthrange == zpos)).squeeze()
        traveldistance = 0
    
    elif pvp < diapdepth:

        #a descent is required to reach the desired diapause depth
        #this checks if the maximum vertical search distance is sufficient to reach the diapause depth
        if pvp + maxdistance >= diapdepth:

            #reaches the diapause depth
            zpos = diapdepth
            zidx = np.array(np.where(depthrange == zpos)).squeeze()
            traveldistance = abs(zpos - pvp)
        
        else:

            #reaches the maximum possible depth (and continue the downward voyage at the next time point)
            zpos = pvp + maxdistance
            zidx = np.argmin(abs(depthrange - zpos))
            zpos = depthrange[zidx]
            traveldistance = abs(zpos - pvp)

        #end if

    else:

        #in an unlikely case, an ascent is required to reach the preferred diapause depth
        if pvp - maxdistance <= diapdepth:

            #reaches the diapause depth
            zpos = diapdepth
            zidx = np.array(np.where(depthrange == zpos)).squeeze()
            traveldistance = abs(pvp - zpos)
        
        else:

            #reaches the minimum possible depth (and continue the unlikely upward voyage at the next time point)
            zpos = pvp - maxdistance
            zidx = np.argmin(abs(depthrange - zpos))
            zpos = depthrange[zidx]
            traveldistance = abs(pvp - zpos)

        #end if
        
    #end if

    return zpos, zidx, maxdistance, traveldistance

#end def

def verticalmigration_dsc3x(pvp: int, pvi: int, strmass: float, modelres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category IIII and subcategory "X" (civ & cv)
    this is seasonal vertical migration 
    
    args:
    diapdepth                   : preferred diapause depth of the super individual
    strmass                     : structural mass of a given super individual
    modelres                    : the temporal resolution of the model

    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated relative vertical position (index) of the super individual
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """
    
    import numpy as np

    depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    depthrange_upperpelagial = depthrange[0:21]
    
    maxdistance = int(modelres * 5.2287 * strmass ** 0.4862)

    #the target depth is arbitary < 100 m - index positions 0-to-21 upper pelagial depth
    targetdepth = np.random.choice(a = depthrange_upperpelagial, replace = False, size = 1).squeeze()
    targetidx = np.array(np.where(depthrange == targetdepth)).squeeze()
    
    #this checks whether the super individual needs an ascent or a no ascent to reach/stay at the desired target depth
    #nb:a non-ascent is unlikely, but coding for that covers for all posibilities
    if pvp <= targetdepth:

        #if the super individual is above the target depth, the super individual stays at where it is
        zpos = pvp
        zidx = pvi
        traveldistance = 0
    
    else:

        #if the super individual is below the target depth
        if pvp - maxdistance <= targetdepth:

            #reaches the diapause depth
            zpos = targetdepth
            zidx = targetidx
            
            traveldistance = abs(pvp - zpos)
        
        else:

            #reaches the minimum possible depth (and continue the unlikely upward voyage at the next time point)
            #no need of bound corrections, as this is always higher than the target depth, which is non-negative & does not exceed maxdepth (due to 'pvp')
            zpos = pvp - maxdistance
            
            #correction of zpos based on the depth levels
            zidx = np.argmin(abs(depthrange - zpos))
            zpos = depthrange[zidx]
            
            traveldistance = abs(pvp - zpos)

        #end if

    #end if

    return zpos, zidx, maxdistance, traveldistance

#end def

def verticalmigration_dsc3p(temprange: float, f1conrange: float, iradrange: float, maxirad: float, p1dnsrange: float, a2: float, a3: float, a4: float, pvp: int, pvi: int, strmass: float, resmass: float, modelres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category III-P (dsc3P: post-diapause CIV and CV stages)
    this is diel vertical migration

    note:
    the only difference between this dsc3a and the dsc2 functions is that the allometric scaling of irradiance sensitivity operates on the total body mass of the super individual (structural + reserve masses)
    for super individuals at dsc1 and dsc2, they do not possess an energy reserve and the allometric scaling of irradiance sensitivity operates on the structural mass only
    
    args:
    temprange                   : range of temperatures across all depth layers at a given longitude, latitude and time
    f1conrange                  : range of food concentration category 1 (= phytoplankton) at a given longitude, latitude and time
    iradrange                   : range of shortwave solar irradiance at a given longitude, latitude and time
    p1dnsrange                  : range of category 1 predator density at a given longitude, latitude and time
    a2                          : evolvable attribute for irradiance sensitivity of a given super individual
    a3                          : evolvable attribute for predator sensitivity of a given super individual
    a4                          : evolvable attribute for predator reactivity of a given super individual
    pvp                         : absolute vertical position at the previous timepoint (the function calculates the current vertical position)
    pvi                         : relative vertical position (index) at the previous timepoint
    strmass                     : structural mass of a given super individual
    resmass                     : reserve mass of a given super individual
    modelres                    : the temporal resolution of the model
    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated current vertical position index (relative vertical position)
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """

    import numpy as np
    import math
    
    #this is the total bodymass of the super individual
    totalmass = strmass + resmass
    #these are the standard depths of the model environment (similar to variable "depthgrade" in the main program)
    depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    #this estimates the theoreitical maximum distance that be vertically searched by a given super individual as a power function of somatic body mass
    #nb:the estimate is adjusted to the temporal resolution of the model
    #the function is based on Bandara et al. (2021), which follows a literature review of copepod swimming speeds in natural habitats, see: https://doi.org/10.1016/j.ecolmodel.2021.109739
    maxdistance = int(modelres * 5.2287 * strmass ** 0.4862)
    #this estimates the minimum and maximum points of the 1D water column that the super individual can search
    searchceiling = pvp - maxdistance
    searchfloor = pvp + maxdistance
    #these estimates surface and bottom depths of the model environment
    mindepth = np.min(depthrange)
    maxdepth = np.max(depthrange)

    #both above estimates must be trimmed so that they does not exceed minimum and maximum depths (else, in theory, animals get burried in the sediment like worms or shoot out into the air like goddamn flying fish!!!)
    if searchceiling < mindepth:

        searchceiling = mindepth
    
    #end if
    
    if searchfloor > maxdepth:

        searchfloor = maxdepth
    
    #end if
    
    #these are index positions related to the minimum and maximum points of the water column searchable by a given super individual
    searchceiling_idx = np.argmin(abs(depthrange - searchceiling))
    searchfloor_idx = np.argmin(abs(depthrange - searchfloor))

    #if the vertical search distance is not sufficient to move between model-resolvable adjacent depth layers, then the model assums no movement of the super individual
    if searchceiling_idx == searchfloor_idx:
        
        zpos = pvp
        zidx = pvi
    
    #if not (i.e., search range spans across 2 or more model-resolved depth gradings), the vertical behavioural submodel applies
    else:
        
        #this is the dynamic evolvable attribute scaling function defined by Edirisinghe (2022), where vertical behavioural reactions of super individuals are modified by percieved visual predation risk
        #where behavioural reaction is by default elicited via ambient irradiance & it is altered by predator presence and copepod detection & reaction capabilities therein
        #the asymptotic exponantial scalar expresses the size-dependence of irradiance sensitivity by copepods (where, smaller ones tend to stay closer to the surface and vice versa)
        if p1dnsrange[pvi] >= a3:
        
            thresholdirradiance = a2 * a4 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        else:

            thresholdirradiance = a2 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        #end if
        
        #this is the vertical search range of the super individual given as a numpy array
        searchrange = np.arange(searchceiling_idx, searchfloor_idx)
        #this is the range of irradiance values corresponding to the search range of the super individual    
        iradrange_sr = iradrange[searchrange]
        #these are environmental variables corresponding to the search range of the super individual
        #nb:the celsius is converted to kelvin for growth potential estimation; see bellow
        temprange_sr = temprange[searchrange] + 273.15
        f1conrange_sr = f1conrange[searchrange]

        #the preferred depth by the super individual is the depth that meets or closest to meeting the irraiance threshold with maximum growth potential
        #the growth potential is estimated as a product of temperature and foodconcentration as described in Edirisinghe (2022)
        growthpotential_sr = temprange_sr * f1conrange_sr

        #if at least one depth bin offers a refuge from light and light-dependent predation
        if any(iradrange_sr <= thresholdirradiance):
            
            potentialzpos = np.array(np.where(iradrange_sr <= thresholdirradiance)).squeeze()
            
            if potentialzpos.size == 1:
            
                zidx = searchrange[potentialzpos]
                zpos = depthrange[zidx]
            
            else:
                
                zidx = searchrange[potentialzpos[np.argmax(growthpotential_sr[potentialzpos])]]
                zpos = depthrange[zidx]
            
            #end if

        #if none of the searchable depths offer a refuge from light and light-dependent predation
        else:
            
            #super individuals select the deepest depth (as downwelling light is vertically extinct in the water column)
            zidx = searchrange[-1]
            zpos = depthrange[zidx]

        #end if
            
    #end if

    traveldistance = abs(pvp - zpos)

    return zpos, zidx, maxdistance, traveldistance

#end def

def verticalmigration_dsc4(temprange: float, f1conrange: float, iradrange: float, maxirad: float, p1dnsrange: float, a2: float, a3: float, a4: float, pvp: int, pvi: int, strmass: float, resmass: float, modelres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category IV (dsc4: adult male and female stages)
    this is diel vertical migration

    note:
    the only difference between this dsc3a and the dsc2 functions is that the allometric scaling of irradiance sensitivity operates on the total body mass of the super individual (structural + reserve masses)
    for super individuals at dsc1 and dsc2, they do not possess an energy reserve and the allometric scaling of irradiance sensitivity operates on the structural mass only
    
    args:
    temprange                   : range of temperatures across all depth layers at a given longitude, latitude and time
    f1conrange                  : range of food concentration category 1 (= phytoplankton) at a given longitude, latitude and time
    iradrange                   : range of shortwave solar irradiance at a given longitude, latitude and time
    p1dnsrange                  : range of category 1 predator density at a given longitude, latitude and time
    a2                          : evolvable attribute for irradiance sensitivity of a given super individual
    a3                          : evolvable attribute for predator sensitivity of a given super individual
    a4                          : evolvable attribute for predator reactivity of a given super individual
    pvp                         : absolute vertical position at the previous timepoint (the function calculates the current vertical position)
    pvi                         : relative vertical position (index) at the previous timepoint
    strmass                     : structural mass of a given super individual
    resmass                     : reserve mass of a given super individual
    modelres                    : the temporal resolution of the model
    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated current vertical position index (relative vertical position)
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """

    import numpy as np
    import math
    
    #this is the total bodymass of the super individual
    totalmass = strmass + resmass
    #these are the standard depths of the model environment (similar to variable "depthgrade" in the main program)
    depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    #this estimates the theoreitical maximum distance that be vertically searched by a given super individual as a power function of somatic body mass
    #nb:the estimate is adjusted to the temporal resolution of the model
    #the function is based on Bandara et al. (2021), which follows a literature review of copepod swimming speeds in natural habitats, see: https://doi.org/10.1016/j.ecolmodel.2021.109739
    maxdistance = int(modelres * 5.2287 * strmass ** 0.4862)
    #this estimates the minimum and maximum points of the 1D water column that the super individual can search
    searchceiling = pvp - maxdistance
    searchfloor = pvp + maxdistance
    #these estimates surface and bottom depths of the model environment
    mindepth = np.min(depthrange)
    maxdepth = np.max(depthrange)

    #both above estimates must be trimmed so that they does not exceed minimum and maximum depths (else, in theory, animals get burried in the sediment like worms or shoot out into the air like goddamn flying fish!!!)
    if searchceiling < mindepth:

        searchceiling = mindepth
    
    #end if
    
    if searchfloor > maxdepth:

        searchfloor = maxdepth
    
    #end if
    
    #these are index positions related to the minimum and maximum points of the water column searchable by a given super individual
    searchceiling_idx = np.argmin(abs(depthrange - searchceiling))
    searchfloor_idx = np.argmin(abs(depthrange - searchfloor))

    #if the vertical search distance is not sufficient to move between model-resolvable adjacent depth layers, then the model assums no movement of the super individual
    if searchceiling_idx == searchfloor_idx:
        
        zpos = pvp
        zidx = pvi
    
    #if not (i.e., search range spans across 2 or more model-resolved depth gradings), the vertical behavioural submodel applies
    else:
        
        #this is the dynamic evolvable attribute scaling function defined by Edirisinghe (2022), where vertical behavioural reactions of super individuals are modified by percieved visual predation risk
        #where behavioural reaction is by default elicited via ambient irradiance & it is altered by predator presence and copepod detection & reaction capabilities therein
        #the asymptotic exponantial scalar expresses the size-dependence of irradiance sensitivity by copepods (where, smaller ones tend to stay closer to the surface and vice versa)
        if p1dnsrange[pvi] >= a3:
        
            thresholdirradiance = a2 * a4 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        else:

            thresholdirradiance = a2 * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        #end if
        
        #this is the vertical search range of the super individual given as a numpy array
        searchrange = np.arange(searchceiling_idx, searchfloor_idx)
        #this is the range of irradiance values corresponding to the search range of the super individual    
        iradrange_sr = iradrange[searchrange]
        #these are environmental variables corresponding to the search range of the super individual
        #nb:the celsius is converted to kelvin for growth potential estimation; see bellow
        temprange_sr = temprange[searchrange] + 273.15
        f1conrange_sr = f1conrange[searchrange]

        #the preferred depth by the super individual is the depth that meets or closest to meeting the irraiance threshold with maximum growth potential
        #the growth potential is estimated as a product of temperature and foodconcentration as described in Edirisinghe (2022)
        growthpotential_sr = temprange_sr * f1conrange_sr

        #if at least one depth bin offers a refuge from light and light-dependent predation
        if any(iradrange_sr <= thresholdirradiance):
            
            potentialzpos = np.array(np.where(iradrange_sr <= thresholdirradiance)).squeeze()
            
            if potentialzpos.size == 1:
            
                zidx = searchrange[potentialzpos]
                zpos = depthrange[zidx]
            
            else:
                
                zidx = searchrange[potentialzpos[np.argmax(growthpotential_sr[potentialzpos])]]
                zpos = depthrange[zidx]
            
            #end if

        #if none of the searchable depths offer a refuge from light and light-dependent predation
        else:
            
            #super individuals select the deepest depth (as downwelling light is vertically extinct in the water column)
            zidx = searchrange[-1]
            zpos = depthrange[zidx]

        #end if
            
    #end if

    traveldistance = abs(pvp - zpos)

    return zpos, zidx, maxdistance, traveldistance

#end def
