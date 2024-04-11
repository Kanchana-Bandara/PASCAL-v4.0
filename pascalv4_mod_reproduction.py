########################################################################
#pan arctic behavioural and life-history simulator for calanus (pascal)#
########################################################################

#version 4.00 :: python development :: temporary macmini edition :: merge later
#super-individual-based model for simulating behavioural and life-history strategies of the north atlantic copepod, calanus finmarchicus

#*** module file ***

#module for mating
#this module selects a male from a specific subpopulation (or from all subpopulations) depending or not depending on proximity

def mateselection_npd(malelist: int) -> int:

    """
    functionality:
    this returns the identity (index position) of a male following a non-proximity-driven (npd) random selection
    #this is the simplest form of mate selection, with no regards to the spatial positioning and encounter probabilities of males and females
    
    args:
    malelist                    : the list of index positions of the males in the subpopulation
    
    return:
    malesel                     : selected male from the list of males in the subpopulation

    """
    
    import numpy as np
    
    malesel = np.random.choice(a = malelist, size = 1, replace = False).squeeze()

    return malesel

#end def
