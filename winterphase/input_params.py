################################################################
# The input file for running the Fielding & Bryan 2022 (FB22) models and computing the surface brightness of certain emission lines (see Chen & Peng et al. 2025 for details)
#
# - Save and Plot main parameters are:
#   - master_folder_name        (master folder name for a given set of runs)
#   - folder_name_base          (the base of each folder name for a particular run of a set of runs) 
#   - num_decimal           (the number of decimal points for each parameter value when saving)
#
# - The FB22 main parameters are:
#   - SFR_inputs                (the input star formation rates)
#   - etaE                      (the fixed thermalization efficiency factor value when vary_which_eta = 'etaM')
#   - etaM                      (the fixed initial hot phase mass loading factor value when vary_which_eta = 'etaE')
#   - vary_which_eta            (vary 'etaM' or 'etaE' values when running the FB22 or CP25 model)
#   - etaM_grid                 (the grid of initial hot phase mass loading factor value when vary_which_eta = 'etaM')
#   - etaE_grid                 (the grid of thermalization efficiency factor value when vary_which_eta = 'etaE')
#   - etaM_cold                 (initial cold phase mass loading)
#   - log_Mcl                   (initial cloud mass)
#   - v_cloud_init              (initial cloud velocity)
#   - r_sonic                   (sonic radius)
#   - Z_wind_init               (initial wind metallicity)
#   - Z_cloud_init              (initial cloud metallicity)
#   - v_circ0                   (circular velocity of external isothermal gravitational potential)
#   - half_opening_angle        (half opening angle of the spherical symmetric wind)
#   - Omwind                    (solid angle of the spherical symmetric wind, based on half_opening_angle)
#
# - The Ploeckinger & Schaye 2020 (PS20) main parameter is:
#   - redshift_ps20             (redshift of the PS20 cooling tables)
#
# - CHIMES (Richings et al. 2014a,b) main parameters are:
#   - redshift_chimes           (redshift of the CHIMES grid)
#   - eq_or_noneq               (whether to import the equilibrium or non-equilibrium solution)
#   - noneq_time                (if ``eq_or_noneq'' = 'noneq', choose the time snapshot of the non-equilibrium solution)

# - Surface Brightness (SB) calculations' main parameters are:
#   - which_lines               (to derive the SB for which lines, see examples below)
#   - R_eval_arr                (the radii where to derive the SB, in units of r_sonic)
#   - z_galaxy                  (the redshift for simulating the SB profile)
################################################################

import numpy as np

from constants import *

# Save and Plot parameters
master_folder_name = 'hayes_2016_setups'
folder_name_base = 'hayes_2016'
num_decimal = 2

# FB22 parameters
SFR_inputs = np.arange(5, 101, 5) * Msun / yr
etaE = 1.0           
etaM = 0.2          
vary_which_eta = 'etaM' # 'etaM' or 'etaE'
# grids for iteration when varying etaM or etaE
etaM_grid = np.arange(0.1, 1.01, 0.05)
etaE_grid = np.arange(0.1, 1.01, 0.05)
etaM_cold = 0.1     
log_Mcl = 5.0    
v_cloud_init = 10**(1.5) * kms # in cm / s
r_sonic = 750 * pc        
Z_wind_init = 2.0 * Z_solar   
Z_cloud_init = 0.25 * Z_solar 
v_circ0 = 67 * kms # in cm / s
half_opening_angle = np.pi / 2
Omwind = 4 * np.pi * (1.0 - np.cos(half_opening_angle))

# PS20 parameters 
redshift_ps20 = 0.2

# CHIMES parameters
redshift_chimes = 0.2
eq_or_noneq = 'noneq'
noneq_time = 10 # in Myr (currently available options, from 1 to 10 Myr with a timestep of 1 Myr)

# SB parameters
# currently available emission lines 
# [b'H  1      6562.81A', b'H  1      4861.33A', b'O  1      6300.30A', b'S  2      4068.60A', b'Si 3      1206.50A', 
#  b'O  2      3726.03A', b'O  2      3728.81A', b'Blnd      1397.00A', b'O  3      5006.84A', b'O  3      4958.91A', 
#  b'O  3      4363.21A', b'Blnd      1549.00A', b'O  6      1031.91A', b'O  6      1037.62A', b'N  2      6583.45A']
# (or run chen23 grids by yourself; 
#  see https://github.com/jasonpeng17/CP25_Emission_SB_Framework/tree/main?tab=readme-ov-file#generating-your-own-chen23-trml-fluxfraction-grids-optional)
which_lines = np.array([b'O  6      1031.91A', b'O  6      1037.62A']) 
R_eval_arr = np.arange(1.01, 100, 0.1) # in units of r_sonic
z_galaxy = 0.235 




