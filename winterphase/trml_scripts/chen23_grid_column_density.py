import numpy as np
import re
import math
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
from scipy.signal import savgol_filter
from scipy import interpolate
from matplotlib import rc
from matplotlib import cm
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys, os
import h5py
import cmasher as cmr
from skimage import measure
import csv
from pathlib import Path

# Add the path to the winterphase module
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

from utils import * 
from chen23_solve_trml import *
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


def best_fit_fnu(mach_rel = 0.75, chi = 100, tau = 10**(-1.5)):
    '''
    Best-fitting fnu of rho-vx cosine profile based on Equation (45) in Chen et al. (2023).
    '''
    fnu = 0.00065 * (mach_rel / 0.75) * ((chi / 100)**(-0.5)) * ((tau / 10**(-1.5))**(1.5))
    return fnu

def best_fit_Pr(tau = 10**(-1.5)):
    '''
    Best-fitting Pr of rho-vx cosine profile based on Equation (44) in Chen et al. (2023).
    '''
    Pr = 0.4 * ((tau / 10**(-1.5))**(0.15))
    return Pr

def return_column_density(norm_cooling_curve, mach_rel = 0.75, h = 1, f_nu = 0.01, kappa_0 = 1e-3, 
                          Prandtl = 0.1, tau = 10**(-1), Z = 1.0, T_cold_in_K = 1e4, T_hot_in_K = 10**(5.5), max_iter = 10):

    # check whether Prandtl and f_nu is None or not; if the inputs are None, use the best-fit values
    if Prandtl == None:
        Prandtl = best_fit_Pr(tau)
    if f_nu == None:
        chi = T_hot_in_K / T_cold_in_K
        f_nu = best_fit_fnu(mach_rel, chi, tau)

    sol, target, pars = plot_solution_given_mdot(norm_cooling_curve, mach_rel = mach_rel, h = h, f_nu = f_nu, kappa_0 = kappa_0, 
                                                 Prandtl = Prandtl, tau = tau, T_cold_in_K = T_cold_in_K, T_hot_in_K = T_hot_in_K, max_iter = max_iter)
    tau, p_hot_over_kb, e_dot_cool_norm, edot_cool_double_equilibrium, iter_count = pars

    # now that we have a 1D solution of TRML with the real cooling curve, we use the data from it to calculate column densities of ions
    # enter the desired value of p_hot/kb, the fiducial value is 1000 K cm^(-3)
    # p_hot_over_kb = 1e3

    # solar abundances of elements, given as number of atoms per 1 hydrogen atom
    solar_abundance = {
        'H' : 1.00e+00, 'He': 1.00e-01, 'Li': 2.04e-09,
        'Be': 2.63e-11, 'B' : 6.17e-10, 'C' : 2.45e-04,
        'N' : 8.51e-05, 'O' : 4.90e-04, 'F' : 3.02e-08,
        'Ne': 1.00e-04, 'Na': 2.14e-06, 'Mg': 3.47e-05,
        'Al': 2.95e-06, 'Si': 3.47e-05, 'P' : 3.20e-07,
        'S' : 1.84e-05, 'Cl': 1.91e-07, 'Ar': 2.51e-06,
        'K' : 1.32e-07, 'Ca': 2.29e-06, 'Sc': 1.48e-09,
        'Ti': 1.05e-07, 'V' : 1.00e-08, 'Cr': 4.68e-07,
        'Mn': 2.88e-07, 'Fe': 2.82e-05, 'Co': 8.32e-08,
        'Ni': 1.78e-06, 'Cu': 1.62e-08, 'Zn': 3.98e-08}

    # atomic mass of elements in amu
    atomic_mass = {
        'H' : 1.00794,   'He': 4.002602,  'Li': 6.941,
        'Be': 9.012182,  'B' : 10.811,    'C' : 12.0107,
        'N' : 14.0067,   'O' : 15.9994,   'F' : 18.9984032,
        'Ne': 20.1797,   'Na': 22.989770, 'Mg': 24.3050,
        'Al': 26.981538, 'Si': 28.0855,   'P' : 30.973761,
        'S' : 32.065,    'Cl': 35.453,    'Ar': 39.948,
        'K' : 39.0983,   'Ca': 40.078,    'Sc': 44.955910,
        'Ti': 47.867,    'V' : 50.9415,   'Cr': 51.9961,
        'Mn': 54.938049, 'Fe': 55.845,    'Co': 58.933200,
        'Ni': 58.6934,   'Cu': 63.546,    'Zn': 65.409}

    if iter_count < max_iter:
        j_eigen = target.root
    else:
        j_eigen = target
    z = sol.t
    T = sol.y[1]
    dT_dz = sol.y[0]
    vz = sol.y[3]
    rho = j_eigen/sol.y[3]

    # read in data from the 1D solution with the real cooling curve
    # remember that this data is still in code units, 
    # we need to convert to cgs units before we can properly calculate physical quantities like column density
    # The detailed procedure of how to do that can be found in our paper
    pc_to_cm = 3.086e18
    kb = 1.38e-16
    mH = 1.66054e-24 # in grams
    gamma = 5.0/3
    mu = 1.0
    P_hot = p_hot_over_kb*kb
    rho_hot = (P_hot / T_hot_in_K) / (kb / (mu*mH))
    L_0 = 378.4*(tau/10**(-1.5))**(-1)*(p_hot_over_kb/1000)**(-1)*(T_hot_in_K/10**6)**0.5*pc_to_cm*sol.t[-1]
    cs_hot = math.sqrt(gamma*P_hot / rho_hot)

    gram_to_msun = 5.02785e-34

    # take arrays of distance, temperature, pressure from the 1D solution, and calculate number density
    # sol.t is the array of positions
    # sol.y[1] is the array of temperatures
    # target.root is the eigenvalue of the mass flux mdot
    # sol.y[3] is the array of z velocities
    zs           = L_0*z
    Ts           = T_hot_in_K*T # T_hot
    Ps           = p_hot_over_kb*(T*rho)# P/kb=1000 at the hot phase
    ns = Ps/Ts

    # take arrays of the mass and cooling distributions and convert to physical units
    M_0 = rho_hot*L_0**3*gram_to_msun
    dM_dlogT_in_code_units    = (T/-dT_dz)*(rho)
    dM_dlogT    = M_0*dM_dlogT_in_code_units # in solar masses

    # convert mass distribution to column density distribution by dividing through by L_0**2*mH
    dSigma_dlogT = (dM_dlogT / L_0**2)
    dN_dlogT = dSigma_dlogT / (mH*gram_to_msun)

    # convert cooling distribution to cooling flux distribution by dividing through by L_0**2
    # dFcool_dlogT = (dEdot_dlogT / L_0**2) * L_0
    # define the T_min & T_max for integration over the temperature space
    # T_min = 1e4
    # T_max = T_hot_in_K
    # temp_mask = (Ts >= T_min) & (Ts <= T_max)
    # integrate dFcool_dlogT over the temperature space within the range T_min and T_max
    # dFcool_dlogT_int = integrate.simps(dFcool_dlogT, Ts)

    # calculate the number density of H using the total number density and fraction of H
    total_ion_count = 0
    for i in solar_abundance:
        total_ion_count += solar_abundance[i]

    # calculate ion fractions and column densities
    elements = []
    ionizations = []
    ion_fractions = []
    column_densities = []

    for i in ions_of_interest: # loop over each element
        for j in range(len(ions_of_interest[i])): # loop over each ion number
            n_of_element = ns*(solar_abundance[i] / (total_ion_count)) # number density of the element of interest (all ions combined)
            fraction_of_ion = IonFractions[i+str(ions_of_interest[i][j])]((np.log10(Ts), np.log10(ns), np.log10(Z)))
            # fraction_of_ion = 10**LogIonFractions[i+str(ions_of_interest[i][j])]((np.log10(Ps/Ts), 0, np.log10(Ts))) # fraction of the ion of interest
            n_of_ion = fraction_of_ion*n_of_element # calculate number density of the ion using the two previous results
            column_density_of_ion = integrate.simps(n_of_ion, zs) # calculate column density by integrating across the mixing layer
            elements.append(i)
            ionizations.append(ions_of_interest[i][j])
            ion_fractions.append(fraction_of_ion)
            column_densities.append(column_density_of_ion)

    # calculate the column density of each ion
    column_density_dict = {}
    for i, (ion, col_den) in enumerate(zip(ionizations, column_densities)):
        ion_name = elements[i]+str(int_to_roman(ion))
        column_density_dict[ion_name] = col_den 

    # save the column densities dictionary
    # Create the directory if it doesn't exist
    if not os.path.exists(f'column_density_dicts_tau={tau:.2f}'):
        os.makedirs(f'column_density_dicts_tau={tau:.2f}')
    eq_suffix = "" if eq_or_noneq == 'eq' else f"_noneq_{noneq_time}Myr"
    col_den_dict_name = f'column_density_dicts_tau={tau:.2f}/column_density_T_hot={T_hot_in_K:.1e}_P={p_hot_over_kb:.1e}_mach_rel={mach_rel:.2f}_tau={tau:.2f}_Z={Z:.4f}{eq_suffix}.csv'

    with open(col_den_dict_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for ion_name, value in column_density_dict.items():
            writer.writerow([ion_name, value])

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    gridspec = dict(hspace=0.03, height_ratios=[1, 1, 0.15, 1])
    fig, ((ax1,ax2,ax3,ax4)) = plt.subplots(4,1, gridspec_kw=gridspec)

    ax3.set_visible(False)

    ax1.loglog(Ts,  dN_dlogT, color='#743023')
    ax1.set_ylim(top=1e21)
    ax1.set_ylabel(r'$\frac{d N}{d logT}$ $[{\rm cm}^{-2}$ ${\rm dex}^{-1}] $', fontsize=8)

    plt.setp(ax1.get_xminorticklabels(), visible=False)
    plt.setp(ax1.get_xmajorticklabels(), visible=False)
    ion_colors = cmr.take_cmap_colors('cmr.infinity_s', len(column_densities), cmap_range=(0.1, 0.9), return_fmt='hex')


    for i in range(len(column_densities)):
        ax2.loglog(Ts, ion_fractions[i], color=ion_colors[i], label=elements[i]+str(int_to_roman(ionizations[i])))
    ax2.set_ylabel(r'$f_{\rm ion}$')
    ax2.set_xlabel(r'$T$ $[{\rm K}]$', fontsize=8)

    ax2.set_ylim(2e-4,2)
    ax2.legend(loc='best', ncol = 1, fontsize=6, handlelength=0.9, labelspacing=0.25, columnspacing=0.7)

    ion_ids = []
    for i in range(len(column_densities)):
        ion_ids.append(elements[i]+str(int_to_roman(ionizations[i])))

    for i in range(len(column_densities)):
        ax4.scatter(i, column_densities[i], marker='o', color=ion_colors[i])

    ax4.xaxis.set_ticks(np.arange(len(column_densities)))
    ax4.xaxis.set_ticklabels(ion_ids, rotation=30)
    ax4.set_ylabel(r'$N_{\rm ion}$ $[{\rm cm}^{-2}]$', fontsize=8)
    ax4.set_yscale('log')

    ax1.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax4.tick_params(axis='both', labelsize=8)

    ax4.tick_params(axis='x', which='minor', bottom=False)
    ax4.tick_params(axis='x', which='minor', top=False)
    ax4.tick_params(axis='y', which='minor', left=True)
    ax4.tick_params(axis='y', which='minor', right=True)

    fig.set_size_inches(3.39375, 7.1)

    # Create the directory if it doesn't exist
    if not os.path.exists(f'column_density_plots_tau={tau:.2f}'):
        os.makedirs(f'column_density_plots_tau={tau:.2f}')
    filename = f'column_density_plots_tau={tau:.2f}/column_density_T_hot={T_hot_in_K:.1e}_P={p_hot_over_kb:.1e}_mach_rel={mach_rel:.2f}_tau={tau:.2f}_Z={Z:.4f}{eq_suffix}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    # list out the ions that you want to plot the ion fraction and column densities of as a dictionary
    # for example, if you are interested in NV and OVI, then the dictionary should look like:
    # ions_of_interest = {'N': [5], 'O':[6]}
    # if you are interested in more than one ion of the same element, simply enter more than one number in the corresponding list
    # for example, if you are interested in OVI and OVII, then the dictionary should look like:
    # ions_of_interest = {'O':[6, 7]}
    # ions_of_interest = {'Mg':[2], 'Si':[2], 'C':[4], 'N':[5], 'O':[6, 7], 'Ne':[8]}
    ions_of_interest = {'Mg':[2], 'Si':[2, 3, 4], 'C':[2, 4], 'N':[5], 'O':[6, 7], 'Ne':[8]}

    # set the input values
    # fixed parameters
    T_hot_in_K = 10**(6)
    T_cold_in_K = 1e4 # set the T cold in physical units
    h = 1 
    kappa_0 = 1e-3
    # grid parameters 
    mach_rel_arr = np.arange(0.75, 2.25, 0.25) # mach_rel array
    tau = np.array([10**(-1)]) # tau array
    # tau = np.array([10**(-2), 10**(-1.5), 10**(-1)]) # tau array
    # Z_arr = 10**(np.arange(-4.0, 0.6, 0.5)) # metallicity array (should not sensitive to Z)
    Z_arr = np.array([1.0])
    # Creating a meshgrid
    mach_rel_mesh, tau_mesh, Z_mesh = np.meshgrid(mach_rel_arr, tau, Z_arr)
    Prandtl = None 
    f_nu = None 
    # creating the maximum iteration number to find mdot_crit
    max_iter = 10

    # CHIMES ion-fraction interpolators for the chosen line
    eq_or_noneq = 'noneq' # whether to import equilibrium ('eq') or non-equilibrium ('noneq') CHIMES grid
    noneq_time = 10 # nonequilibrium time in Myr
    t_req_seconds = noneq_time * Myr # in seconds
    redshift_chimes = 2.0 # redshift of the CHIMES grid

    # define the ion fractions for the ions of interest
    IonFractions = {}
    for i in ions_of_interest: # loop over each element
        for j in range(len(ions_of_interest[i])): # loop over each ion number
            chimes_ionfracs = build_ion_fraction_interpolators(line_id=None, elem = i, ion_n = ions_of_interest[i][j],
                                                               chimes_dict = chimes_dict, neq_times_seconds=(t_req_seconds,))
            if eq_or_noneq == 'eq': # equilibrium solution
                chimes_eq_interp = chimes_ionfracs["eq"]
                IonFractions[i+str(ions_of_interest[i][j])] = chimes_eq_interp
            if eq_or_noneq == 'noneq': # non-equilibrium solution
                # get the (single) nearest available time used
                (t_used,) = tuple(chimes_ionfracs["neq"].keys())
                chimes_neq_interp = chimes_ionfracs["neq"][t_used]
                IonFractions[i+str(ions_of_interest[i][j])] = chimes_neq_interp

    # Assuming mach_rel_mesh and tau_mesh are your meshgrid arrays
    for i in range(mach_rel_mesh.shape[0]):
        for j in range(mach_rel_mesh.shape[1]):
            for k in range(mach_rel_mesh.shape[2]):
                # Retrieve the current combination of mach_rel, tau, and Z
                mach_rel = mach_rel_mesh[i, j, k]
                tau = tau_mesh[i, j, k]
                Z = Z_mesh[i, j, k]
                # get the cooling curves at constant pressures
                norm_cooling_curve_dir = 'cooling_curves_const_P'
                cooling_curve_files = sorted(os.listdir(norm_cooling_curve_dir))
                for cooling_curve in cooling_curve_files:
                    if ".npz" in cooling_curve:
                        cooling_curve_path = os.path.join(norm_cooling_curve_dir, cooling_curve)
                        print(f"The current relative Mach number is: {mach_rel:.2f}; tau value is : {tau:.2f}; Z value is: {Z:.4f}")
                        print("The current cooling curve is: " + cooling_curve)
                        return_column_density(cooling_curve_path, mach_rel = mach_rel, h = h, f_nu = f_nu, kappa_0 = kappa_0, Prandtl = Prandtl, tau = tau, Z = Z,
                                              T_cold_in_K = T_cold_in_K, T_hot_in_K = T_hot_in_K, max_iter = max_iter)


