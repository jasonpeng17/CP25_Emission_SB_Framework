################################################################
# Based on the Chen et al. (2023) TRML model to solve the TRML phase structure 
# (https://github.com/ziruichen11/1.5D_mixing/blob/main/surface_brightness_calculation/6.%20emissivity%20and%20surface%20brightness.ipynb)
################################################################
import numpy as np
import re
import math
import argparse
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy import optimize
from matplotlib import rc
from matplotlib import cm
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys, os
import cmasher as cmr
from astropy import constants as const

from IPython import embed

rc('text', usetex=False)
matplotlib.rc('font', family='sans-serif', size=12)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = 'round'
matplotlib.rcParams['lines.solid_capstyle'] = 'round'
matplotlib.rcParams['figure.dpi'] = 200
cycle=plt.rcParams['axes.prop_cycle'].by_key()['color']

# Define a function to dynamically adjust the factor based on the behavior of find_final_gradient
def adjust_factor(current_factor, current_gradient, previous_gradient):
    # Adjust the factor based on the change in gradient
    if abs(current_gradient) > abs(previous_gradient) / 2:
        # If the change in gradient is not significant, make a larger adjustment
        return max(current_factor * 0.9, 0.1)  # Adjust these values as needed
    else:
        # If the change in gradient is significant, make a finer adjustment
        return max(current_factor * 0.99, 0.5)  # Adjust these values as needed

# define functions that return values of vx and its 1st & 2nd derivatives 
# version 1: taking the vx profile as a cosine
# def vx_cos(z, h, mach_rel):
#     if (z > h):
#         return 0
#     else:
#         return (mach_rel/2)*(np.cos(np.pi*z/h)+1)

# vx_cos = np.vectorize(vx_cos)    
    
# def dvx_dz_cos(z, h, mach_rel):
#     if (z > h):
#         return 0
#     else:
#         return -(mach_rel/2) * np.pi/h * np.sin(np.pi*z/h)

# dvx_dz_cos = np.vectorize(dvx_dz_cos)
    
# def d2vx_dz2_cos(z, h, mach_rel):
#     if (z > h):
#         return 0
#     else:
#         return -(mach_rel/2) * (np.pi/h)**2 * np.cos(np.pi*z/h)

# d2vx_dz2_cos = np.vectorize(d2vx_dz2_cos)

# version 2: taking the rho vx profile as a cosine
def rho_vx_cos(z, h, mach_rel):
    # rho_hot = P_hot / T_hot
    rho_hot = 1
    A = rho_hot * mach_rel/2.
    return A*(np.cos(np.pi*z/h)+1)
    
def drho_vx_dz_cos(z, h, mach_rel):
    # rho_hot = P_hot / T_hot
    rho_hot = 1
    A = rho_hot * mach_rel/2.
    return -A * np.pi/h * np.sin(np.pi*z/h)

def d2rho_vx_dz2_cos(z, h, mach_rel):
    # rho_hot = P_hot / T_hot
    rho_hot = 1
    A = rho_hot * mach_rel/2.
    return -A * (np.pi/h)**2 * np.cos(np.pi*z/h)


# define a function that makes a 4-panel plot for a solution
# the plot includes the T, P, vz, and vx profile, the phase distributions, and the terms decomposition in position and temperature space

def plot_solution(sol, mdot_crit, mdot_over_mdot_crit, h, mach_rel, f_nu, kappa_0, Prandtl, edot_cool_double_equilibrium, T_cold, T_hot,
                  name='random_profile', gamma = 5.0/3.0):
    mdot        = mdot_crit * mdot_over_mdot_crit
    z           = sol.t
    dT_dz       = sol.y[0]
    T           = sol.y[1]
    dvz_dz      = sol.y[2]
    vz          = sol.y[3]
    P           = T*(mdot/vz)
    rho         = (mdot/vz)
    # version 1: vx as cosine
    # vx          = vx_cos(z, h, mach_rel)
    # dvx_dz      = dvx_dz_cos(z, h, mach_rel)
    # d2vx_dz2    = d2vx_dz2_cos(z, h, mach_rel)

    # version 2: rho*vx as cosine
    vx          = np.vectorize(rho_vx_cos)(z, h, mach_rel)/rho
    dvx_dz      = np.gradient(vx, z)
    d2vx_dz2    = np.gradient(dvx_dz, z)
    
    dEdot_dlogT = (T/-dT_dz)*edot_cool_double_equilibrium(T)
    dM_dlogT    = (T/-dT_dz)*rho

    # version 1: vx as cosine
    # kappa        = (mdot/vz) * (f_nu * h**2 * np.abs(dvx_dz) + kappa_0 / (mdot/vz))
    # mu        = Prandtl * (mdot/vz) * (f_nu * h**2 * np.abs(dvx_dz) + kappa_0 / (mdot/vz))
    # dkappa_dz       = f_nu * h**2 * ((mdot/vz) * d2vx_dz2 - (mdot/vz**2) * np.abs(dvx_dz) * dvz_dz)
    # dmu_dz       = Prandtl * (f_nu * h**2 * ((mdot/vz) * d2vx_dz2 - (mdot/vz**2) * np.abs(dvx_dz) * dvz_dz))
    # d2T_dz2     = edot_cool_double_equilibrium(T)/kappa + mdot*vz/kappa * (T/vz**2 * dvz_dz + dT_dz/(gamma-1)/vz) - dT_dz * dkappa_dz/kappa - mu/kappa * (dvx_dz**2 + (4/3.)*dvz_dz**2)
    # d2vz_dz2    = (3./4.) * (mdot/mu) * (dT_dz/vz + dvz_dz*(1-T/vz**2)) - dvz_dz * dmu_dz/mu

    # version 2: rho*vx as cosine
    kappa = f_nu * h**2 * (mdot/vz) * np.abs(dvx_dz) + kappa_0
    mu = Prandtl * kappa   
    a = - ((Prandtl*f_nu * h**2) / (mu*vz**2)) * dvz_dz**2 * np.abs(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) + ((Prandtl*f_nu * h**2) / (mu*vz)) * dvz_dz * np.sign(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) * (vz*d2rho_vx_dz2_cos(z, h, mach_rel) + 2*dvz_dz*drho_vx_dz_cos(z, h, mach_rel))           
    d2vz_dz2 = ((3.0/4)*(mdot / mu) * (dT_dz/vz + dvz_dz*(1 - T/vz**2)) - a) / (1 + ((Prandtl*f_nu * h**2) / (mu*vz)) * np.sign(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) * rho_vx_cos(z, h, mach_rel) * dvz_dz)
    dkappa_dz = - (f_nu * h**2 / vz**2) * dvz_dz * np.abs(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) + (f_nu * h**2 / vz) * np.sign(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) * (vz*d2rho_vx_dz2_cos(z, h, mach_rel) + 2*dvz_dz*drho_vx_dz_cos(z, h, mach_rel) + rho_vx_cos(z, h, mach_rel) * d2vz_dz2)
    d2T_dz2 = (1/kappa) * ((1/(gamma-1))*mdot*dT_dz + (mdot*T / vz)*dvz_dz + edot_cool_double_equilibrium(T) - mu*((dvx_dz)**2 + (4.0/3)*dvz_dz**2) - dkappa_dz*dT_dz)

    dHvisc_dlogT = (T/-dT_dz)*mu*(dvx_dz**2 + (4/3.)*dvz_dz**2)
    dP_dz = (mdot/vz)*dT_dz - (mdot*T/vz**2)*dvz_dz
    dWork_dlogT = (T/-dT_dz)*vz*dP_dz
    
    adv_enthalpy = (gamma/(gamma-1))*mdot*dT_dz
    
    dkappa_dz_dT_dz = dkappa_dz * dT_dz
    conduction = dkappa_dz_dT_dz + kappa*d2T_dz2
    
    cooling = -edot_cool_double_equilibrium(T)
    
    x_visc_heating = mu*(dvx_dz)**2
    z_visc_heating = mu*(4.0/3)*dvz_dz**2
    
    work = mdot*(dT_dz - dvz_dz*T/vz)
    
    
    fig, ((ax1,ax3),(ax2,ax4)) = plt.subplots(2,2)

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })


    ax1.plot(z,  T, color='#2A3132', label=r'$T  $')
    ax1.plot(z,  vz, color='#FF9C44', label=r'$v_z $')
    ax1.plot(z,  vx, color='#EF476F', label=r'$v_x $')
    ax1.plot(z,  P, color='#88B04B', label=r'$P  $')

    ax1.set_ylabel('Profiles', fontsize=8)

    ax1.annotate(r'$\frac{T}{T_{\rm hot}}$', xy=(0.23, 0.7), color='#2A3132',fontsize=8,
                xytext=(0.29, 0.5), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top')

    ax1.annotate(r'$\frac{P}{P_{\rm hot}}$', xy=(0.5, 0.94), color='#88B04B',fontsize=8,
                xytext=(0.5, 0.94), textcoords='axes fraction',
                horizontalalignment='right', verticalalignment='top')

    ax1.annotate(r'$\frac{v_z}{c_{\rm s,hot}}$', xy=(0.1, 0.26), color='#FF9C44',fontsize=8,
                xytext=(0.075, 0.22), textcoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')

    ax1.annotate(r'$\frac{v_x}{c_{\rm s,hot}}$', xy=(0.8, 0.57), color='#EF476F',fontsize=8,
                xytext=(0.8, 0.2), textcoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')


    ax2.semilogy(z,  adv_enthalpy,   color='#2A3132',              label='adv. enth.')
    ax2.semilogy(z,  conduction,            color='#702A70', label='conduction')
    ax2.semilogy(z,  cooling,  color='#118AB2', label='cooling')
    ax2.semilogy(z,  x_visc_heating,                      color='#EF476F', label='x visc. heat.')
    ax2.semilogy(z,  z_visc_heating,               color='#FF9C44', label='z visc. heat.')
    ax2.semilogy(z,  work,               color='#88B04B', label='work')

    ax2.semilogy(z,  -adv_enthalpy,   color='#2A3132', ls='--')
    ax2.semilogy(z,  -conduction,         color='#702A70', ls='--')
    ax2.semilogy(z,  -cooling,   color='#118AB2', ls='--')
    ax2.semilogy(z,  -x_visc_heating,                     color='#EF476F', ls='--')
    ax2.semilogy(z,  -z_visc_heating,              color='#FF9C44', ls='--')
    ax2.semilogy(z,  -work,              color='#88B04B', ls='--')

    ax2.set_ylabel(r'$\left. \dot \varepsilon \right/ \dot \varepsilon_0$')


    ax3.loglog(T,  dEdot_dlogT, color='#118AB2', label=r'$\frac{d \dot{E}}{d \log T}$')
    ax3.loglog(T,  dM_dlogT, color='#743023', label=r'$\frac{d M}{d \log T}$')
    i_lo = np.argmin(np.abs(T-1.2*T_cold))
    i_hi = np.argmin(np.abs(T-0.8*T_hot))
    maximum = np.max([np.max(dEdot_dlogT[i_hi:i_lo]),np.max(dM_dlogT[i_hi:i_lo])])
    minimum = np.min([np.min(np.abs(dEdot_dlogT)[i_hi:i_lo]),np.min(np.abs(dM_dlogT)[i_hi:i_lo])])
    ax3.set_ylim((minimum,maximum))

    ax3.set_ylabel('Phase Distributions', fontsize=8)

    ax3.annotate(r'$\frac{1}{\dot{E}_0}\frac{d \dot E_{\rm cool} }{d logT}$', xy=(0.3, 0.25), color='#118AB2',fontsize=8,
                xytext=(0.3, 0.2), textcoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')

    ax3.annotate(r'$\frac{1}{M_0} \frac{d M}{d logT}$', xy=(0.5, 0.45), color='#743023',fontsize=8,
                xytext=(0.5, 0.45), textcoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')


    ax4.loglog(T,  adv_enthalpy,   color='#2A3132', label='adv. enth.')
    ax4.loglog(T,  conduction,            color='#702A70', label='conduction')
    ax4.loglog(T,  cooling,  color='#118AB2', label='cooling')
    ax4.loglog(T,  x_visc_heating,                      color='#EF476F', label='x visc. heat.')
    ax4.loglog(T,  z_visc_heating,               color='#FF9C44', label='z visc. heat.')
    ax4.loglog(T,  work,               color='#88B04B', label='work')

    ax4.loglog(T,  -adv_enthalpy,   color='#2A3132', ls='--')
    ax4.loglog(T,  -conduction,         color='#702A70', ls='--')
    ax4.loglog(T,  -cooling,   color='#118AB2', ls='--')
    ax4.loglog(T,  -x_visc_heating,                     color='#EF476F', ls='--')
    ax4.loglog(T,  -z_visc_heating,              color='#FF9C44', ls='--')
    ax4.loglog(T,  -work,              color='#88B04B', ls='--')

    ax4.set_ylabel(r'$\left. \dot \varepsilon \right/ \dot \varepsilon_0$')

    ax2.set_ylim((2e-4,3e1))
    ax4.set_ylim((2e-4,3e1))



    ax2.set_xlabel(r'$\left. z \right/ h$', fontsize=8)
    ax4.set_xlabel(r'$\left. T \right/ T_{\rm hot}$', fontsize=8)


    ax2.legend(loc='upper right', ncol = 2, fontsize=6, handlelength=1.3, labelspacing=0.25, columnspacing=0.7)
    ax4.legend(loc='lower center', ncol = 3, fontsize=6, handlelength=1.3, labelspacing=0.25, columnspacing=0.7)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax1.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax3.tick_params(axis='both', labelsize=8)
    ax4.tick_params(axis='both', labelsize=8)

    fig.set_size_inches(7.1, 7.1*(2/3))

    plt.subplots_adjust(hspace=0.02, wspace=0.25)
    
    if name:
        plt.savefig(name+'.pdf',dpi=200, bbox_inches='tight')
        plt.clf()
        # plt.show()
    else:
        # plt.show()
        plt.clf()
    return

def plot_solution_given_mdot(norm_cooling_curve, mach_rel = 0.75, h = 1, f_nu = 0.01, kappa_0 = 1e-3, Prandtl = 0.1, tau = 10**(-1), T_cold_in_K = 1e4, T_hot_in_K = 10**(5.5),
                             max_iter = 15):
    # set the relative shear velocity between the phases here, note that in our setup cs,hot=sqrt(gamma)=sqrt(5/3)
    # mach_rel = 1.5

    # set the width of the mixing layer here, the fiducial value is 1 
    # h  = 1

    # set the value of f_nu here, suggested value is between 10**(-2.5) and 10**(-1.5)
    # f_nu  = 0.01

    # set the value of kappa_0 here. kappa_0 is the constant term in the expression of kappa that prevents singularities in the solution
    # the fiducial value is 10**(-6)
    # kappa_0 = 1e-3

    # set the value of the Prandtl number here, suggested value is between 0.1 and 1
    # Prandtl = 0.1

    # set the value of tau here, suggested value is between 10**(-2) and 10**(-1)
    # tau = 10**(-1)

    # define the hot wind temperature at the starburst radius (defined in CC85)
    k_B = const.k_B.cgs.value # boltzmann constant in cgs units
    m_p = const.m_p.to('g').value # proton mass in cgs units
    M_sun_per_yr = const.M_sun.cgs.value / (3.154e7) 
    mu = 1
    # T_hot_in_K = 3e41 * (alpha / beta) * (m_p / k_B) * mu * (0.0338 / 0.113) / M_sun_per_yr # T_hot = 3e41 ergs/s (P* / rho*) * (m_p * k_B / mu) * (alpha / beta) * (E_dot / M_dot)
    # T_hot_in_K = 1e6

    def edot_cool_double_equilibrium_func(T_hot_in_K, tau):
        # # enter the desired value of the hot phase temperature (in Kelvin) here, the fiducial value is 1e6 K
        # T_hot_in_K = 1e7
        gamma = 5.0/3.0 

        # read in data for the real cooling curve.
        # here we are using the real cooling curve at a pressure of p/kb=1000. 
        # The real cooling curve is basically independent on pressure in the range of pressure we are interested in
        data = np.load(norm_cooling_curve)
        Ts  =   data['Ts']
        edot   =   data['edot']
        P   =   float(data['P'])
        e_dot_cool_norm = float(data['norm'])

        f_edot = interp1d(Ts, edot, bounds_error=False,fill_value=0)
        # interpolate between the real cooling curve to get a function that returns the value of edot_cool
        # take the value of T_hot we chose before and set that temperature to be at thermal equilibrium (Edot_cool=Edot_heat)
        # assuming that Edot_heat is a constant, this setup also gives us the cold equilibrium temperature
        epsilon_T = 0.05

        #### version 1: original version
        # def edot_cool_double_equilibrium(T):
        #     Edot_heat = f_edot(T_hot_in_K)
        #     Edot_cool = f_edot(T_hot_in_K*T)
        #     Edot_heat *= np.where(T<(1+epsilon_T), 1, (T/(1+epsilon_T))**(-5))
        #     return 1.5/tau*(Edot_cool-Edot_heat)

        #### version 2: ensures T_cold_in_K is 1e4;
        def Edot_heat(T):
            f_edot_at_Tcold = f_edot(T_cold_in_K)
            f_edot_at_Thot = f_edot(T_hot_in_K)
            
            power_law_slope = (np.log10(f_edot_at_Thot) - np.log10(f_edot_at_Tcold)) / (np.log10(T_hot_in_K) - np.log10(T_cold_in_K))
            
            log_return_value = np.log10(f_edot_at_Tcold) + power_law_slope * (np.log10(T*T_hot_in_K) - np.log10(T_cold_in_K))
            
            return_value = 10**log_return_value
            
            return return_value

            # T_mix_in_K = np.sqrt(T_cold_in_K * T_hot_in_K)

            # f_edot_at_Tcold = f_edot(T_cold_in_K)
            # f_edot_at_Thot = f_edot(T_hot_in_K)
            # f_edot_at_Tmix = f_edot(T_cold_in_K) / 10
            
            # power_law_slope_ctom = (np.log10(f_edot_at_Tmix) - np.log10(f_edot_at_Tcold)) / \
            #                        (np.log10(T_mix_in_K) - np.log10(T_cold_in_K))

            # power_law_slope_mtoh = (np.log10(f_edot_at_Thot) - np.log10(f_edot_at_Tmix)) / \
            #                        (np.log10(T_hot_in_K) - np.log10(T_mix_in_K))
            
            # if T >= (T_mix_in_K / T_hot_in_K):
            #     log_return_value = np.log10(f_edot_at_Tmix) + power_law_slope_mtoh * (np.log10(T*T_hot_in_K) - np.log10(T_mix_in_K))
            # else:
            #     log_return_value = np.log10(f_edot_at_Tcold) + power_law_slope_ctom * (np.log10(T*T_mix_in_K) - np.log10(T_cold_in_K))

            # return_value = 10**log_return_value
            
            # return return_value

        def edot_cool_double_equilibrium(T):
            Edot_cool = f_edot(T_hot_in_K*T)
            
            return 1.5/tau*(Edot_cool-Edot_heat(T))

        edot_cool_double_equilibrium_vec = np.vectorize(edot_cool_double_equilibrium)

        return e_dot_cool_norm, P, edot_cool_double_equilibrium_vec

    e_dot_cool_norm, P_over_kb, edot_cool_double_equilibrium = edot_cool_double_equilibrium_func(T_hot_in_K, tau)

    # define some constants that are used in the integration
    # note that we need initial T gradient to be small and non-zero to give the solution a "nudge" away from the equilibrium at the hot phase temperature
    dT_dz_initial       = -1e-6 # default -1e-6
    dvz_dz_initial      = 0

    beta_lo             = -2
    beta_hi             = 1
    # density_contrast    = 1/scipy.optimize.root_scalar(edot_cool_double_equilibrium, bracket = [1e-1,1e-4]).root
    density_contrast    = T_hot_in_K / T_cold_in_K
    T_peak_over_T_cold  = density_contrast**(1/3.)

    gamma               = 5.0/3.0 
    P_hot               = 1.0
    T_cold              = 1.0/density_contrast
    T_hot               = 1.0
    T_peak              = T_peak_over_T_cold * T_cold
    epsilon_T           = 0.005
    rho_hot             = P_hot / T_hot
    mdot_crit           = rho_hot * np.sqrt(T_hot)
    alpha_heat          = (np.log10(T_cold/T_peak)/np.log10(density_contrast) * (beta_lo - beta_hi)) - beta_hi
    heating_coefficient = (T_cold/T_peak)**((beta_hi-beta_lo)*(1.0 + (np.log10(T_cold/T_peak)/np.log10(density_contrast))))

    # we define two termination events that will later be taken as inputs to the solve_ivp function.
    # we want to terminate the integration if temperature drops below the cold phase or exceeds the hot phase.
    def dip(z,w,mdot): # terminate when T drops below T_cold
        T = w[1]
        return T/T_cold - 0.999

    dip.terminal = True

    def bump(z,w,mdot): # terminate when T exceeds T_hot
        T = w[1]
        return T/T_hot - 1.001

    bump.terminal = True

    def length(z,w,modit): # terminate when z > 1
        return z - 1.001

    length.terminal = True 

    # define an integrator that will later be taken as an input to scipy's solve_ivp function.
    # we are solving 4 coupled differential equations (1st & 2nd derivatives of T and vz), so we need expressions of those as outputs of the integrator function
    def integrator(z, w, mdot): 
        dT_dz, T, dvz_dz, vz = w
        if (z <= 0):
            dvx_dz = 0
        elif (z > h):
            dvx_dz = 0
        else:
            dvx_dz = (1/mdot)*(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel))
    
        kappa = f_nu * h**2 * (mdot/vz) * np.abs(dvx_dz) + kappa_0
        mu = Prandtl * kappa   
        a = - ((Prandtl*f_nu * h**2) / (mu*vz**2)) * dvz_dz**2 * np.abs(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) + ((Prandtl*f_nu * h**2) / (mu*vz)) * dvz_dz * np.sign(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) * (vz*d2rho_vx_dz2_cos(z, h, mach_rel) + 2*dvz_dz*drho_vx_dz_cos(z, h, mach_rel))           
        d2vz_dz2 = ((3.0/4)*(mdot / mu) * (dT_dz/vz + dvz_dz*(1 - T/vz**2)) - a) / (1 + ((Prandtl*f_nu * h**2) / (mu*vz)) * np.sign(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) * rho_vx_cos(z, h, mach_rel) * dvz_dz)
        dkappa_dz = - (f_nu * h**2 / vz**2) * dvz_dz * np.abs(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) + (f_nu * h**2 / vz) * np.sign(rho_vx_cos(z, h, mach_rel)*dvz_dz + vz*drho_vx_dz_cos(z, h, mach_rel)) * (vz*d2rho_vx_dz2_cos(z, h, mach_rel) + 2*dvz_dz*drho_vx_dz_cos(z, h, mach_rel) + rho_vx_cos(z, h, mach_rel) * d2vz_dz2)
        d2T_dz2 = (1/kappa) * ((1/(gamma-1))*mdot*dT_dz + (mdot*T / vz)*dvz_dz + edot_cool_double_equilibrium(T) - mu*((dvx_dz)**2 + (4.0/3)*dvz_dz**2) - dkappa_dz*dT_dz)
        
        return np.array([d2T_dz2, dT_dz, d2vz_dz2, dvz_dz])

    # def integrator(z, w, mdot): 
    #     dT_dz, T, dvz_dz, vz = w
    #     if (z <= 0):
    #         dvx_dz = 0
    #         d2vx_dz2 =0 
    #     elif (z > h):
    #         dvx_dz = 0
    #         d2vx_dz2 =0 
    #     else:
    #         dvx_dz = -(mach_rel/2) * np.pi/h * np.sin(np.pi*z/h)
    #         d2vx_dz2 = -(mach_rel/2) * (np.pi/h)**2 * np.cos(np.pi*z/h)

    #     # the derivation of these equations can be found in the methods section of the paper
    #     kappa        = (mdot/vz) * (f_nu * h**2 * np.abs(dvx_dz) + kappa_0 / (mdot/vz))
    #     mu        = Prandtl * (mdot/vz) * (f_nu * h**2 * np.abs(dvx_dz) + kappa_0 / (mdot/vz))
    #     dkappa_dz       = f_nu * h**2 * ((mdot/vz) * d2vx_dz2 - (mdot/vz**2) * np.abs(dvx_dz) * dvz_dz)
    #     dmu_dz       = Prandtl * (f_nu * h**2 * ((mdot/vz) * d2vx_dz2 - (mdot/vz**2) * np.abs(dvx_dz) * dvz_dz))
    #     d2T_dz2     = edot_cool_double_equilibrium(T)/kappa + mdot*vz/kappa * (T/vz**2 * dvz_dz + dT_dz/(gamma-1)/vz) - dT_dz * dkappa_dz/kappa - mu/kappa * (dvx_dz**2 + (4/3.)*dvz_dz**2)
    #     d2vz_dz2    = (3./4.) * (mdot/mu) * (dT_dz/vz + dvz_dz*(1-T/vz**2)) - dvz_dz * dmu_dz/mu
    #     return np.array([d2T_dz2, dT_dz, d2vz_dz2, dvz_dz])

    # use the integrator and the termination events to integrate a solution for a guess of mdot
    # this function then returns the final T gradient of the solution, which is a crucial criterion that informs our bisection process
    def find_final_gradient(mdot_over_mdot_crit):
        mdot                = mdot_crit * mdot_over_mdot_crit
        T_initial           = T_hot
        vz_initial          = mdot/rho_hot
        dT_dz_initial       = -1e-6 # default -1e-6
        dvz_dz_initial      = 1e-6
        initial_conditions  = [dT_dz_initial, T_initial, dvz_dz_initial, vz_initial]
        stop_distance       = 10**4
        sol = solve_ivp(integrator, [0, stop_distance], initial_conditions, 
            dense_output=True, 
            events=[dip, bump, length],
            rtol=3e-14, atol=[1e-9,1e-11,1e-9,1e-11],
            # rtol=3e-10, atol=[1e-5,1e-7,1e-5,1e-7],
            args=[mdot], 
            method='Radau') # default method is 'RK45'
        return sol.y[0][-1]

    def find_gradient(mdot_over_mdot_crit):
        mdot                = mdot_crit * mdot_over_mdot_crit
        T_initial           = T_hot
        vz_initial          = mdot/rho_hot
        dT_dz_initial       = -1e-6 # default -1e-6
        dvz_dz_initial      = 1e-6
        initial_conditions  = [dT_dz_initial, T_initial, dvz_dz_initial, vz_initial]
        stop_distance       = 10**4
        sol = solve_ivp(integrator, [0, stop_distance], initial_conditions, 
            dense_output=True, 
            events=[dip, bump, length],
            rtol=3e-14, atol=[1e-9,1e-11,1e-9,1e-11],
            # rtol=3e-10, atol=[1e-5,1e-7,1e-5,1e-7],
            args=[mdot], 
            method='Radau') # default method is 'RK45'
        return sol.y[0]

    # similar to the find_final_gradient function, but this function returns the solution object from the solve_ivp function, which is useful for plotting
    def calculate_solution(mdot_over_mdot_crit):
        mdot                = mdot_crit * mdot_over_mdot_crit
        T_initial           = T_hot
        vz_initial          = mdot/rho_hot
        dT_dz_initial       = -1e-6 # default -1e-6
        dvz_dz_initial      = 1e-6
        initial_conditions  = [dT_dz_initial, T_initial, dvz_dz_initial, vz_initial]
        stop_distance       = 10**4
        sol = solve_ivp(integrator, [0, stop_distance], initial_conditions, 
            dense_output=True, 
            events=[dip, bump, length],
            rtol=3e-14, atol=[1e-9,1e-11,1e-9,1e-11],
            # rtol=3e-10, atol=[1e-5,1e-7,1e-5,1e-7],
            args=[mdot], 
            method='Radau') # default method is 'RK45'
        return sol

    # Initialize the iteration counter
    iter_count = 0
    # We perform our bisection to find the eigenvalue of the mass flux mdot here
    # Our first guess for mdot is the maximum possible value mdot_crit, 
    # which corresponds to a vz_initial that is equal to c_s,hot
    mdot_over_mdot_crit = 1.0
    final_gradient = find_final_gradient(mdot_over_mdot_crit)
    all_gradients = find_gradient(mdot_over_mdot_crit)
    if np.max(all_gradients) > 0:
        factor = 1.1 
    else:
        factor = 0.1 
        
    # We record the sign of the final T gradient of this solution, 
    # decrease our guess of mdot by factor*100%, and repeat this process
    positive = 1
    # Save all mdot_over_mdot_crit values from these iterations
    mdot_over_mdot_crit_lst = []
    dT_dz_lst = []
    # We repeat the above procedure until the final T gradient changes sign
    # or until the maximum number of iterations is reached
    while (positive > 0) and (iter_count < max_iter):
        # current mdot_over_mdot_crit
        mdot_over_mdot_crit *= factor
        mdot_over_mdot_crit_lst.append(mdot_over_mdot_crit)
        # current dT_dz
        next_final_gradient = find_final_gradient(mdot_over_mdot_crit)
        all_gradients = find_gradient(mdot_over_mdot_crit)
        dT_dz_lst.append(next_final_gradient)
        print(f"Iteration {iter_count+1}, mdot_over_mdot_crit: {mdot_over_mdot_crit}, dT/dz: {next_final_gradient}")
        positive = next_final_gradient * final_gradient
        final_gradient = next_final_gradient
        iter_count += 1

    # Output the final gradient after exiting the loop
    print(f"Final dT/dz after {iter_count} iterations: {final_gradient:.4f}")
    if iter_count >= max_iter:
        chosen_indx = np.argmin(np.abs(np.array(dT_dz_lst)))  # when dT_dz closet to 0
        print(f"Chosen mdot_over_mdot_crit after {iter_count} iterations: {np.array(mdot_over_mdot_crit_lst)[chosen_indx]:.4f}") # when dT_dz closet to 0
        
    # now we know that the eigenvalue of mdot must be sandwiched between the two most recent guesses of mdot
    # given this information, we can use scipy's root finder optimize.root_scalar to find the eigenvalue of mdot (mdot that corresponds to final T gradient=0)
    # note that we set rtol and xtol to be very small here so that we can resolve the eigenvalue as accurately as possible
    if iter_count < max_iter:
        target = scipy.optimize.root_scalar(find_final_gradient, 
                                            bracket = [mdot_over_mdot_crit,mdot_over_mdot_crit/factor], 
                                            xtol=1e-14, rtol=1e-14)
        sol = calculate_solution(target.root)
    else:
        sol = calculate_solution(mdot_over_mdot_crit)

    # finally we plot the solution we obtained using the pre-defined plotting function
    # Create the directory if it doesn't exist
    if not os.path.exists(f'result_plots_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}'):
        os.makedirs(f'result_plots_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}')
    name = f'result_plots_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}/result_plots_T_hot={T_hot_in_K:.1e}_P={P_over_kb:.1e}_mach_rel={mach_rel:.2f}_tau={tau:.2f}'
    if iter_count < max_iter:
        plot_solution(sol, mdot_crit, target.root, h, mach_rel, f_nu, kappa_0, Prandtl, edot_cool_double_equilibrium, T_cold, T_hot, name = name)
        return sol, target, (tau, P_over_kb, e_dot_cool_norm, edot_cool_double_equilibrium, iter_count)
    else:
        plot_solution(sol, mdot_crit, mdot_over_mdot_crit, h, mach_rel, f_nu, kappa_0, Prandtl, edot_cool_double_equilibrium, T_cold, T_hot, name = name)
        return sol, mdot_over_mdot_crit, (tau, P_over_kb, e_dot_cool_norm, edot_cool_double_equilibrium, iter_count)

if __name__ == "__main__":
    #### plot the results
    plot_solution_given_mdot(alpha = 1, beta = 1, mach_rel = 0.75, h = 1, f_nu = 0.01, kappa_0 = 1e-3, Prandtl = 0.1, tau = 10**(-1), T_cold_in_K = 1e4, T_hot_in_K = 10**(5.5),
                             max_iter = 15)

