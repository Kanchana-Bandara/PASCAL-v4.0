# Pan-Arctic Behavioral and Life-history Simulator for Calanus (PASCAL v4.0)
## Overview
PASCAL is the 4th iteration of a very-high biological resolution behavioral and life-history simulation model designed for the copepods of genus _Calanus_ inhabiting the North Atlantic and the Arctic. It was initiated in 2016 under the VISTA PhD project of Kanchana Bandara (Akvaplan niva AS, Norway), titled "High-Resolution Modelling of Diel and Seasonal Vertical Migration of High-Latitude Zooplankton" supervised by Ketil Eiane (Nord University, Norway), Øystein Varpe (University of Bergen) and Rubao Ji (Woods Hole Oceanographic Institution, USA). Since its inception, PASCAL recieved 2 subsequent upgrades; PASCAL v2.0 in 2018 under the same VISTA project and and PASCAL v3.1 in 2021 under the NFR GLIDER Phase - I project with the oversight of Vigdis Tverberg (Nord University, Norway). PASCAL v4.0 is the latest iteration that sees an overhaul of the core model architecture and brings improved performance. A basic comparison of the four versions of PASCAL is listed below:

#### Table 1: A basic comparison of four versions of PASCAL
|Attribute|PASCAL v1.0|PASCAL v2.0|PASCAL v3.1|Pascal v4.0|
|-------|-------|-------|------|------|
|Model Architecture|strategy-oriented|strategy-oriented|individual-based|super-individual-based
|Model Dimensions|2D|2D|2D|4D|
|Temporal Resolution|1 h|1 h|6 h|6 h
|Spatial Resolution (x, y)|-|-|-|> 9 km
|Spatial Resolution (z)|1 m |1 m|1 m|> 1 m
|Simulated Population Size|1 x 10<sup>6</sup>|2.5 x 10<sup>6</sup>|1 x 10<sup>6</sup>|resource/risk-dependent ceiling
|Simulated Taxa|Generalized copepod|_Calanus_ spp. |_C.finmarchicus_|_C.finmarchicus_
|Programming Language|R|R|FORTRAN(95)|Python(3.x)
|Project Attribution|VISTA 6165|VISTA 6165|GLIDER Phase-I|Migratory Crossroads

