################################################################
# This script is based on the MultiphaseGalacticWind.py script in https://github.com/dfielding14/MultiphaseGalacticWind (Fielding & Bryan 2022)

# Overview:
# - First the code calculates the structure of a single phase galactic wind in the manner of Chevalier and Clegg (1985). 
# - Then the code calculates the structure of a multiphase galactic wind. 
################################################################

import numpy as np
import h5py 
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import solve_ivp
import cmasher as cmr
from matplotlib.lines import Line2D
from astropy import constants as const
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
import os
from matplotlib import rc

from IPython import embed
from constants import *
from utils import *
from input_params import *

## Plot Styling
rc('text', usetex=True)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = "round"
matplotlib.rcParams['lines.solid_capstyle'] = "round"
matplotlib.rcParams['legend.handletextpad'] = 0.4
matplotlib.rcParams['axes.linewidth'] = 0.6
matplotlib.rcParams['ytick.major.width'] = 0.6
matplotlib.rcParams['xtick.major.width'] = 0.6
matplotlib.rcParams['ytick.minor.width'] = 0.45
matplotlib.rcParams['xtick.minor.width'] = 0.45
matplotlib.rcParams['ytick.major.size'] = 2.75
matplotlib.rcParams['xtick.major.size'] = 2.75
matplotlib.rcParams['ytick.minor.size'] = 1.75
matplotlib.rcParams['xtick.minor.size'] = 1.75
matplotlib.rcParams['legend.handlelength'] = 2
matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams['font.family'] = "serif"


"""
MAIN function to derive the FB22 outputs
"""
def Multiphase_Wind_Evo(r, state):
    """
    Calculates the derivative of v_wind, rho_wind, Pressure, rhoZ_wind, M_cloud, v_cloud, and Z_cloud. 
    Used with solve_ivp to calculate steady state structure of multiphase wind. 
    """
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4]
    v_cloud    = state[5]
    Z_cloud    = state[6]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    Z_wind       = rhoZ_wind/rho_wind
    vc           = v_circ0 * np.where(r<r_sonic, r/r_sonic, 1.0)

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) 
    Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) 

    # cloud properties
    Ndot_cloud              = Ndot_cloud_init * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud    = Ndot_cloud/(Omwind * v_cloud * r**2)
    cs_cl_sq                = gamma * kB*T_cloud/(mu*mp)

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kB*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind              # density contrast
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kB * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kB, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * M_cloud * v_turb_cold / r_cloud 
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow - Mdot_loss, 0)

    # density
    drhodt       = (number_density_cloud * Mdot_cloud)
    drhodt_plus  = (number_density_cloud * Mdot_loss)
    drhodt_minus = (number_density_cloud * Mdot_grow) 

    # momentum
    p_dot_drag   = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)
    dpdt_drag    = (number_density_cloud * p_dot_drag)
    
    # energy
    e_dot_cool   = 0.0 if (Cooling_Factor==0) else (rho_wind/(muH*mp))**2 * Lambda_P_rho(Pressure,rho_wind,Z_wind/Z_solar)

    # metallicity
    drhoZdt      = -1.0 * (number_density_cloud * (Z_wind*Mdot_grow + Z_cloud*Mdot_loss))

    # wind gradients
    # velocity
    dv_dr       = 2/Mach_sq_wind
    dv_dr      += - (vc/v_wind)**2
    dv_dr      +=  drhodt_minus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    dv_dr      += (gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    dv_dr      += -(gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    dv_dr      += -dpdt_drag/(rho_wind*v_wind**2/r)
    dv_dr      *= (v_wind/r)/(1.0-(1.0/Mach_sq_wind))
    
    # density
    drho_dr       = -2
    drho_dr      += (vc/v_wind)**2
    drho_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    drho_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    drho_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    drho_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    drho_dr      *= (rho_wind/r)/(1.0-(1.0/Mach_sq_wind))

    # pressure
    dP_dr       = -2
    dP_dr      += (vc/v_wind)**2
    dP_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.* (v_rel**2 / cs_sq_wind)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/cs_sq_wind)
    dP_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    dP_dr      *= (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind))


    drhoZ_dr   = drho_dr*(rhoZ_wind/rho_wind) + (rhoZ_wind/r) * drhodt_plus/(rho_wind*v_wind/r) * (Z_cloud/Z_wind - 1)

    # cloud gradients
    dM_cloud_dr = Mdot_cloud/v_cloud

    dv_cloud_dr = (p_dot_drag + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)

    dZ_cloud_dr = (Z_wind-Z_cloud) * Mdot_grow / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)

    return np.r_[dv_dr, drho_dr, dP_dr, drhoZ_dr, dM_cloud_dr, dv_cloud_dr, dZ_cloud_dr]

def Single_Phase_Wind_Evo(r, state):
    """
    Calculates the derivative of v_wind, rho_wind, Pressure for a single phase wind. 
    Used with solve_ivp to calculate steady state structure of a single phase wind with no cooling and no gravity. 
    """
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(r<r_sonic, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(r<r_sonic, 1.0, 0.0)

    # density
    drhodt          = Mdot_SN
    
    # momentum
    dpdt            = 0 

    # energy
    dedt            = Edot_SN

    dv_dr    = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( 2.0/Mach_sq_wind - 1/(rho_wind*v_wind/r) * (drhodt*(gamma+1)/2. + (gamma-1)*dedt/v_wind**2)) 
    drho_dr  = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind)) 
    dP_dr    = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt + drhodt * (gamma-1)/2.*Mach_sq_wind + (gamma-1)*Mach_sq_wind*dedt/v_wind**2))

    return np.r_[dv_dr, drho_dr, dP_dr]

def cloud_ksi(r, state):
    """
    function to calculate the value of ksi = t_mix / t_cool
    """
    v_wind       = state[0]
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    M_cloud      = state[4]
    v_cloud      = state[5]
    Z_cloud      = state[6]
    rho_cloud    = Pressure * (mu*mp) / (kB*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kB * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kB, Z_mix/Z_solar)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    return ksi

def supersonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 + epsilon)
supersonic.terminal = True

def subsonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 - epsilon)
subsonic.terminal = True

def cold_wind(r,z):
    return np.sqrt(gamma*z[2]/z[1])/np.sqrt(gamma*kB*T_cloud/(mu*mp)) - (1.0 + epsilon)
cold_wind.terminal = True

def cloud_stop(r,z):
    return z[5] - 10e5
cloud_stop.terminal = True

if __name__ == "__main__":

    # choose the grid(s)
    if vary_which_eta == 'etaM':
        etaM_arr = etaM_grid
        # keep etaE fixed from input_params
        norm = matplotlib.colors.Normalize(vmin=etaM_arr.min(), vmax=etaM_arr.max())
    elif vary_which_eta == 'etaE':
        etaE_arr = etaE_grid
        # keep etaM fixed from input_params
        norm = matplotlib.colors.Normalize(vmin=etaE_arr.min(), vmax=etaE_arr.max())
    else:
        raise ValueError("vary_which_eta must be 'etaM' or 'etaE'.")
    
    cmap = cm.plasma
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for SFR in SFR_inputs:

        # define the folder name
        SFR_Msun_per_yr = SFR / (Msun/yr)
        folder_name = f'{folder_name_base}_etaMcl={etaM_cold:.{num_decimal}f}_logMcl={log_Mcl:.{num_decimal}f}_sfr={SFR_Msun_per_yr:.{num_decimal}f}_setup'

        fig, axes = plt.subplots(2, 5, figsize=(32.0,9.6))
        fig.subplots_adjust(wspace=0.4)

        # build iteration over (etaE, etaM)
        if vary_which_eta == 'etaM':
            iter_vals = [(etaE, em) for em in etaM_arr]
        elif vary_which_eta == 'etaE':
            iter_vals = [(eE, etaM) for eE in etaE_arr]

        for etaE, etaM in iter_vals:

            # feedback and SF props
            M_cloud_min = 1e-2*Msun     ## minimum mass of clouds
            Mdot        = etaM * SFR
            # E_SN        = 1e51          ## energy per SN in erg
            # mstar       = 100*Msun      ## mass of stars formed per SN
            # Edot        = etaE * (E_SN/mstar) * SFR
            Edot        = 3e41 * etaE * (SFR / (Msun/yr)) # erg / s (etaE * (E_SN/mstar) * SFR)

            # properties at r_sonic if no clouds + gravity
            epsilon     = 1e-5              ## define a small number to jump above and below sonic radius / mach = 1
            Mach0       = 1.0 + epsilon
            v0          = np.sqrt(Edot/Mdot)*(1/((gamma-1)*Mach0) + 1/2.)**(-1/2.) ## velocity at sonic radius
            rho0        = Mdot/(Omwind*r_sonic**2 * v0)                            ## density at sonic radius
            P0          = rho0*v0**2 / Mach0**2 / gamma                            ## pressure at sonic radius
            rhoZ0       = rho0 * Z_wind_init
            print( "v_wind = %.1e km/s  n_wind = %.1e cm^-3  P_wind = %.1e kB K cm^-3" %(v0/1e5, rho0/(mu*mp), P0/kB))

            Edot_per_Vol = Edot / (4/3. * np.pi * r_sonic**3) # source terms from SN
            Mdot_per_Vol = Mdot / (4/3. * np.pi * r_sonic**3) # source terms from SN

            ##########
            ## integrate the single phase only solution
            ##########

            r_init = 100*pc ### inner radius for hot solution
            # r_init = 10*pc

            ## calculate gradients right at sonic radius 
            dv_dr0, drho_dr0, dP_dr0 = Single_Phase_Wind_Evo(r_sonic, np.r_[v0, rho0, P0])

            ## interpolate to within the subsonic region
            dlogvdlogr   = dv_dr0 * r_sonic/v0
            dlogrhodlogr = drho_dr0 * r_sonic/rho0
            dlogPdlogr   = dP_dr0 * r_sonic/P0
            dlogr0       = 1e-8

            v0_sub   = 10**(np.log10(v0) - dlogvdlogr * dlogr0)
            rho0_sub = 10**(np.log10(rho0) - dlogrhodlogr * dlogr0)
            P0_sub   = 10**(np.log10(P0) - dlogPdlogr * dlogr0)

            ### integrate (single phase only) from sonic radius to r_init in the subsonic region.
            sol = solve_ivp(Single_Phase_Wind_Evo, [10**(np.log10(r_sonic)-dlogr0),r_init], np.r_[v0_sub, rho0_sub, P0_sub], 
                            events=[supersonic], 
                            dense_output=True,
                            rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kB])

            r_init    = sol.t[-1]
            v_init    = sol.y[0][-1]
            rho_init  = sol.y[1][-1]
            P_init    = sol.y[2][-1]
            rhoZ_init = rho_init * Z_wind_init

            ## interpolate to within the supersonic region
            v0_sup   = 10**(np.log10(v0) + dlogvdlogr * dlogr0)
            rho0_sup = 10**(np.log10(rho0) + dlogrhodlogr * dlogr0)
            P0_sup   = 10**(np.log10(P0) + dlogPdlogr * dlogr0)

            ### integrate (single phase only) from sonic radius to 100x sonic radius
            sol_sup = solve_ivp(Single_Phase_Wind_Evo, [10**(np.log10(r_sonic)+dlogr0),(10**2)*r_sonic], np.r_[v0_sup, rho0_sup, P0_sup], 
                events=[supersonic], 
                dense_output=True,
                rtol=1e-12, atol=[1e-3, 1e-7*mp, 1e-2*kB])

            r_hot_only         = np.append(sol.t[::-1], sol_sup.t)
            v_wind_hot_only    = np.append(sol.y[0][::-1], sol_sup.y[0])
            rho_wind_hot_only  = np.append(sol.y[1][::-1], sol_sup.y[1])
            P_wind_hot_only    = np.append(sol.y[2][::-1], sol_sup.y[2])

            Mdot_wind_hot_only   = Omwind*r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only/(Msun/yr)
            cs_wind_hot_only     = np.sqrt(gamma * P_wind_hot_only / rho_wind_hot_only)
            T_wind_hot_only      = P_wind_hot_only/kB / (rho_wind_hot_only/(mu*mp))
            K_wind_hot_only      = (P_wind_hot_only/kB) / (rho_wind_hot_only/(mu*mp))**gamma
            Pdot_wind_hot_only   = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only**2/(1e5*Msun/yr)
            Edot_wind_hot_only   = Omwind * r_hot_only**2 * rho_wind_hot_only * v_wind_hot_only * (0.5 * v_wind_hot_only**2 + 1.5 * cs_wind_hot_only**2)/(1e5**2*Msun/yr)

            ##########
            ## integrate the multiphase solution
            ##########

            # cold cloud initial properties
            T_cloud             = 1e4
            log_etaM_cold      = np.log10(etaM_cold)

            #########################
            ## cold clouds can either be introduced instantaneously at cold_cloud_injection_radial_extent 
            ## or distributed in space according to some power law slope (cold_cloud_injection_radial_power) between cloud_radial_offest and cold_cloud_injection_radial_extent

            ## Immediate injection of cold clouds
            # cold_cloud_injection_radial_power   = np.inf 
            # cold_cloud_injection_radial_extent  = 1.33*r_sonic
            # cloud_radial_offest                 = 2e-2 ### don't start integration exactly at r_sonic

            ## Distributed injection of cold clouds --- uncomment for distributed cloud injection
            cold_cloud_injection_radial_power   = 6
            cold_cloud_injection_radial_extent  = 1.33*r_sonic
            cloud_radial_offest                 = 2e-2

            ### where to start the integration
            irstart         = np.argmin(np.abs(r_hot_only-r_sonic*(1.0+cloud_radial_offest))) 
            r_init          = r_hot_only[irstart]
            v_init          = v_wind_hot_only[irstart]
            rho_init        = rho_wind_hot_only[irstart]
            P_init          = P_wind_hot_only[irstart]
            M_cloud_init    = 10**log_Mcl * Msun
            # Z_cloud_init    = 1 * Z_solar 
            # Z_cloud_init    = 0.25 * Z_solar 

            # cold cloud total properties
            # v_cloud_init    = 10**1.5 * km/s 
            # v_cloud_init    = 100 * km/s 
            Mdot_cold_init  = etaM_cold * SFR              ## mass flux in cold clouds
            Ndot_cloud_init = Mdot_cold_init / M_cloud_init ## number flux in cold clouds

            #### ICs 
            supersonic_initial_conditions = np.r_[v_init, rho_init, P_init, Z_wind_init*rho_init, M_cloud_init, v_cloud_init, Z_cloud_init]

            ### integrate!
            sol = solve_ivp(Multiphase_Wind_Evo, [r_init, 1e3*r_sonic], supersonic_initial_conditions, events=[supersonic,cloud_stop,cold_wind], dense_output=True,rtol=1e-10)
            print(sol.message)
            print(sol.t_events)

            ## gather solution and manipulate into useful form
            r           = sol.t
            v_wind      = sol.y[0]
            rho_wind    = sol.y[1]
            P_wind      = sol.y[2]
            rhoZ_wind   = sol.y[3]
            M_cloud     = sol.y[4]
            v_cloud     = sol.y[5]
            Z_cloud     = sol.y[6]
            rho_cloud = P_wind * (mu*mp) / (kB*T_cloud) # cloud in pressure equilibrium

            cloud_Mdots = (np.outer(Ndot_cloud_init, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud / (Msun/yr))
            Mdot_wind   = Omwind * r**2 * rho_wind * v_wind/(Msun/yr)
            cs_wind     = np.sqrt(gamma * P_wind / rho_wind)
            T_wind      = P_wind/kB/(rho_wind/(mu*mp))
            K_wind      = (P_wind/kB) / (rho_wind/(mu*mp))**gamma

            Pdot_wind   = Omwind * r**2 * rho_wind * v_wind**2/(1e5*Msun/yr)
            cloud_Pdots = (np.outer(Ndot_cloud_init, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * v_cloud / (1e5 * Msun/yr))

            Edot_wind   = Omwind * r**2 * rho_wind * v_wind * (0.5 * v_wind**2 + 1.5 * cs_wind**2) / (1e5**2*Msun/yr) # divide by (1e5**2*Msun/yr) to get FB22 conversion
            cloud_Edots = (np.outer(Ndot_cloud_init, np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)) * M_cloud * (0.5 * v_cloud**2 + 2.5 * kB * T_cloud/(mu*mp)) / (1e5**2*Msun/yr)) # divide by (1e5**2*Msun/yr) to get FB22 conversion

            v_rel       = (v_wind-v_cloud) # relative velocity 
            M_rel       = v_rel / cs_wind # relative mach number

            ##########
            ## Plot ##
            ##########
            color = m.to_rgba(etaM) if vary_which_eta == 'etaM' else m.to_rgba(etaE)

            ## 10 panels figure
            for i in range(10):
                if i < 5:
                    row = 0       
                    # axes[row, i].set_xlabel("r (kpc)", fontsize=15)
                    axes[row, i].set_xscale('log')
                    axes[row, i].tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, 
                                   right = True, labelright = True)
                    axes[row, i].tick_params(which='minor',right = True)
                    axes[row, i].tick_params(which='minor',top = True)
                    axes[row, i].tick_params(axis='x', labelsize=14, which = 'both')
                    axes[row, i].tick_params(axis='y', labelsize=14, which = 'both')
                if i >= 5 and i < 10:
                    row = 1
                    axes[row, i-5].set_xlabel("r (kpc)", fontsize=24)
                    axes[row, i-5].set_xscale('log')
                    axes[row, i-5].tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, 
                                   right = True, labelright = True)
                    axes[row, i-5].tick_params(which='minor',right = True)
                    axes[row, i-5].tick_params(which='minor',top = True)
                    axes[row, i-5].tick_params(axis='x', labelsize=14, which = 'both')
                    axes[row, i-5].tick_params(axis='y', labelsize=14, which = 'both')
                # velocity 
                if i == 0:
                    axes[row, i].plot(r/kpc, v_wind/1e8, color=color, lw = 1.5, linestyle='-') # hot phase
                    axes[row, i].plot(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, v_cloud)/1e8, color=color, lw = 1.5, ls = '--')
                    axes[row, i].set_ylabel(r'V (1000 $\rm{km \ s^{-1}}$)', fontsize=24)
                # density
                if i == 1:
                    axes[row, i].plot(r/kpc, np.log10(rho_wind/(mu*mp)), color=color, lw = 1.5, linestyle='-')
                    axes[row, i].plot(r/kpc, np.log10(np.ma.masked_where(M_cloud<M_cloud_min, rho_cloud)/(mu*mp)), color=color, lw = 1.5, linestyle='--')
                    axes[row, i].set_ylabel(r'$\mathrm{log_{10}}$[n ($\rm{cm^{-3}}$)]', fontsize=24)
                # temperature
                if i == 2:
                    axes[row, i].plot(r/kpc, T_wind, color=color, lw = 1.5, linestyle='-')
                    axes[row, i].plot(r/kpc, (T_wind*T_cloud)**0.5, color=color, lw = 1.5, linestyle='--')
                    axes[row, i].set_yscale('log')
                    axes[row, i].set_ylabel(r'T (K)', fontsize=24)
                # cloud mass
                if i == 3:
                    axes[row, i].plot(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, M_cloud)/Msun, color = color, ls = '--', lw = 1.5)
                    axes[row, i].set_yscale('log')
                    axes[row, i].set_ylabel(r'$M_{\rm{cl}} \; (M_\odot)$', fontsize=24)
                # metallicity
                if i == 4:
                    axes[row, i].plot(r/kpc, rhoZ_wind / rho_wind / Z_solar, color = color, ls = '-', lw = 1.5)
                    axes[row, i].plot(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, Z_cloud)/Z_solar, color = color, ls = '--', lw = 1.5)
                    axes[row, i].set_yscale('log')
                    axes[row, i].set_ylabel(r'$Z \; (Z_\odot)$', fontsize=24)
                    # Set y-axis to use float notation
                    formatter = ScalarFormatter()
                    formatter.set_scientific(False)
                    axes[row, i].yaxis.set_major_formatter(formatter)
                    axes[row, i].yaxis.set_minor_formatter(formatter)
                # ksi 
                if i == 5:
                    axes[row, i-5].plot(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_ksi(r, sol.y)), color = color, ls = '--', lw = 1.5)
                    axes[row, i-5].set_yscale('log')
                    axes[row, i-5].set_ylabel(r'$\xi = r_{\rm{cl}} / v_{\rm{turb}} t_{\rm{cool}}$', fontsize=24)
                # mass outflow rate
                if i == 6:
                    axes[row, i-5].plot(r/kpc, Mdot_wind / (SFR / (Msun / yr)), color = color, ls = '-', lw = 1.5)
                    axes[row, i-5].plot(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Mdots[0]) / (SFR / (Msun / yr)), color = color, ls = '--', lw = 1.5)
                    axes[row, i-5].set_yscale('log')
                    axes[row, i-5].set_ylabel(r'$\dot{M} / \rm{SFR}$', fontsize=24)
                # energy outflow rate
                if i == 7:
                    axes[row, i-5].plot(r/kpc, Edot_wind, color = color, ls = '-', lw = 1.5)
                    axes[row, i-5].plot(r/kpc, np.ma.masked_where(M_cloud<M_cloud_min, cloud_Edots[0]), color = color, ls = '--', lw = 1.5)
                    axes[row, i-5].set_yscale('log')
                    axes[row, i-5].set_ylabel(r'$\dot{E} \; ({\rm km}^2/{\rm s}^2 \; M_\odot/{\rm yr})$', fontsize=24)
                # relative mach number
                if i == 8:
                    axes[row, i-5].plot(r/kpc, M_rel, color = color, ls = '-', lw = 1.5)
                    axes[row, i-5].set_yscale('log')
                    axes[row, i-5].set_ylabel(r'$\mathcal{M}_{\rm{rel}}$', fontsize=24)
                # thermal pressure
                if i == 9:
                    axes[row, i-5].plot(r/kpc, P_wind / kB / 1e3, color = color, ls = '-', lw = 1.5)
                    axes[row, i-5].set_yscale('log')
                    axes[row, i-5].set_ylabel(r'$P_3$', fontsize=24)

            ##########
            ## Save ##
            ##########

            v_c_km_per_s = v_circ0 / kms # circular velocity
            v_star = v_cloud / ((Edot/Mdot)**(0.5)) # for clouds
            rho_star = rho_cloud / ((r_sonic**(-2))*(Edot**(-1/2))*(Mdot**(3/2))) # for clouds
            T_star = ((T_wind*T_cloud)**0.5) / ((Edot/Mdot)*(mp/kB)*mu) # for T_mix = (T_wind*T_cloud)**0.5
            P_star = P_wind / ((r_sonic**(-2))*(Edot**(1/2))*(Mdot**(1/2))) # for clouds or hot winds (in pressure equilibrium)
            Z_star = Z_cloud / Z_solar # for clouds

            v_w_star = v_wind / ((Edot/Mdot)**(0.5)) # for hot winds
            rho_w_star = rho_wind / ((r_sonic**(-2))*(Edot**(-1/2))*(Mdot**(3/2))) # for hot winds
            T_w_star = T_wind / ((Edot/Mdot)*(mp/kB)*mu) # for hot winds
            Z_w_star = rhoZ_wind / rho_wind / Z_solar # for hot winds

            P3_wind = P_wind / kB / 1e3 # for clouds or hot winds (in pressure equilibrium)
            M1d5_wind = v_wind / cs_wind / 1.5 # for hot winds
            chi_cl_w = T_wind / T_cloud # overdensity of clouds: rho_cl / rho_w = T_w / T_cl for pressure equilibrium

            # save outputs
            base_root = "../fb22_model_outputs"
            tag = make_tag(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Z_wind_init/Z_solar, num_decimal)

            cloud_items = [
                ("u_arr",      "fb22_u_arr",     v_star),
                ("rho_arr",    "fb22_rho_arr",   rho_star),
                ("T_arr",      "fb22_T_arr",     T_star),
                ("Z_arr",      "fb22_Z_arr",     Z_star),
                ("M_cl_arr",   "fb22_Mcl_arr",   M_cloud),
            ]

            wind_items = [
                ("u_w_arr",    "fb22_u_w_arr",   v_w_star),
                ("rho_w_arr",  "fb22_rho_w_arr", rho_w_star),
                ("T_w_arr",    "fb22_T_w_arr",   T_w_star),
                ("Z_w_arr",    "fb22_Z_w_arr",   Z_w_star),
            ]

            extra_items = [
                ("x_arr",      "fb22_x_arr",     (r / r_init)),
                ("P_arr",      "fb22_P_arr",     P_star),
                ("P3_arr",     "fb22_P3_arr",    P3_wind),
                ("M1d5_arr",   "fb22_M1d5_arr",  M1d5_wind),
                ("chi_arr",    "fb22_chi_arr",   chi_cl_w),
            ]

            save_items(base_root, master_folder_name, folder_name, tag, cloud_items)
            save_items(base_root, master_folder_name, folder_name, tag, wind_items)
            save_items(base_root, master_folder_name, folder_name, tag, extra_items)

        cax = fig.add_axes([0.93, 0.11, 0.02, 0.77]) # Position colorbar
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='proportional')
        if vary_which_eta == 'etaM':
            cb.set_label(r'$\eta_{\rm{M}}$', size = 26)
        if vary_which_eta == 'etaE':
            cb.set_label(r'$\eta_{\rm{E}}$', size = 26)
        cb.ax.tick_params(labelsize=14)  # Adjusting the tick label size

        fig_direct = f"../fb22_model_outputs/fb22_solution_plots/{master_folder_name}/"
        if not os.path.exists(fig_direct):
            os.makedirs(fig_direct)
        if vary_which_eta == 'etaM':
            fig.savefig(fig_direct + f'/fb22_solution_{folder_name}_etaE={etaE:.{num_decimal}f}.pdf', 
                        dpi=300, bbox_inches='tight')
        if vary_which_eta == 'etaE':
            fig.savefig(fig_direct + f'/fb22_solution_{folder_name}_etaM={etaM:.{num_decimal}f}.pdf', 
                        dpi=300, bbox_inches='tight')
        plt.clf()
