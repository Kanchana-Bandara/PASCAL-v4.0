########################################################################
#pan arctic behavioural and life-history simulator for calanus (pascal)#
########################################################################

#version 4.00 :: python development :: temporary macmini edition :: merge later
#super-individual-based model for simulating behavioural and life-history strategies of the north atlantic copepod, calanus finmarchicus

#*** module file ***

#module for estimating various sources of mortality of super individuals in the simulation based on inputdata
#there are a number of sources of mortality in the functions nested witin this module, including those for the calculation of starvation risk, predator-driven mortality risk and background mortality risk

def mortalityrisk_dsc1(strmass: float, maxstrmass: float, devstage: int, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 1 (dsc1: eggs, nauplius I and II)
    nb:for starvation risk, eggs are skipped in this estimation using an <if> condition
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    devstage                 : the developmental stage (s, t)
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    strcat = (maxstrmass - strmass) / maxstrmass
    #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
    #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
    #nb:eggs (s = 0) are skipped from starvation risk estimation (i.e., it returns 0.00)
    
    if strcat <= 0.10 or devstage == 0:

        starvationrisk = 0.00
    
    else:

        starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
    #end if
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def


def mortalityrisk_dsc2(strmass: float, maxstrmass: float, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 2 (dsc2: NIII-CIII)
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    devstage                 : the developmental stage (s, t)
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if    
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def

def mortalityrisk_dsc3a(strmass: float, maxstrmass: float, resmass: float, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 3A (dsc-IIIA: active civ and cv)
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    devstage                 : the developmental stage (s, t)
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if    
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    
    totalmass = strmass + resmass
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def

def mortalityrisk_dsc3e(strmass: float, maxstrmass: float, resmass: float, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 3E (dsc-IIIE: civ & cv at diapause entry)
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if    
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    
    totalmass = strmass + resmass
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def

def mortalityrisk_dsc3d(strmass: float, maxstrmass: float, resmass: float, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 3E (dsc-IIIE: civ & cv at diapause entry)
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if    
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    
    totalmass = strmass + resmass
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def

def mortalityrisk_dsc3x(strmass: float, maxstrmass: float, resmass: float, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 3E (dsc-IIIE: civ & cv at diapause entry)
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if    
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    
    totalmass = strmass + resmass
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def

def mortalityrisk_dsc3p(strmass: float, maxstrmass: float, resmass: float, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 3A (dsc-IIIA: active civ and cv)
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    devstage                 : the developmental stage (s, t)
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if    
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    
    totalmass = strmass + resmass
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def

def mortalityrisk_dsc4(strmass: float, maxstrmass: float, resmass: float, p1dens: float, p1lightdp: float, p2risk: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 4 (dsc4: adult males and females)
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    resmass                  : the energy reserve mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    devstage                 : the developmental stage (s, t)
    p1dens                   : the visual predator (predator#1) density (x, y, t, z)
    p1lightdp                : the range scaled (0.1-0.9) shortwave irraidiance signifying the light dependency of visual predation risk
    p2risk                   : the tactile predation risk expressed as probability of death
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #1:estimation of starvation risk
    #_______________________________
    
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if    
    
    #2:estimation of light- and size-dependent predation risk
    #________________________________________________________
    
    totalmass = strmass + resmass
    pred1risk = p1dens * p1lightdp * (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00)))

    #3:estimation of total mortality risk 
    #_____________________________________
    #this is a function of starvation risk, light- and size-dependent visual predation risk (pred1risk), light- and size-independent tactile predation risk (pred2risk) and background mortality risk
    totalmortalityrisk = starvationrisk + pred1risk + p2risk + bgmrisk

    #this adjusts truncates the total mortality risk to 1.00 if it exceeds 1.00 (nb:total mortality risk is a probability of death)
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk

#end def