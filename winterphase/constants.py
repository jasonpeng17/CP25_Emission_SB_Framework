################################################################
# Define some useful fundamental constants, unit conversion factors, and physical parameters for calculations
################################################################

import numpy as np
from astropy import constants as const

##### Fundamental constants:
kB = 1.3806488e-16 # Boltzmann constant (erg/K)
mp = 1.6726219e-24 # Proton mass (g)

##### Unit conversion factors:
kms = 1e5 # km/s in cm/s
pc = const.pc.cgs.value # 1 pc in cm
kpc = const.kpc.cgs.value # 1 kpc in cm
Msun = const.M_sun.cgs.value # Solar mass in g
s = 1 # Second (s)
yr = 3.154e7 # Year in seconds
Myr = 3.154e13 # Million years in seconds
Gyr = 3.154e16 # Billion years in seconds

##### Physical parameters:
gamma = 5./3. # Adiabatic index
mu = 1 # Mean molecular weight
muH = 1 # Hydrogen mean molecular weight
Z_solar = 0.0134 # Solar metallicity
# FB22 model parameters
CoolingAreaChiPower         =  0.5 
ColdTurbulenceChiPower      = -0.5 
TurbulentVelocityChiPower   =  0.0 
Mdot_coefficient            = 2.0 # consistent with public version of FB22 (online github version: 1.0 / 3.0)
Cooling_Factor              = 1.0
drag_coeff                  = 0.5
f_turb0                     = 10**-1.0