################################################################
# Based on the Chen et al. (2023) TRML solution to derive the total cooling flux and the flux fraction of each emission line (Peng et al. 2025)
################################################################
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

# read in the emissivity tables from Ploeckinger & Schaye
cooling_table_ps20 = '../../ps20_grids/UVB_dust1_CR1_G1_shield1_lines.hdf5'
f = h5py.File(cooling_table_ps20, 'r')
T_bins = np.array(f['TableBins']['TemperatureBins']) # temperature, given in logT
nH_bins = np.array(f['TableBins']['DensityBins']) # number density of H, given in log(nH)
Z_bins = np.array(f['TableBins']['MetallicityBins']) # metallicity, given in log(Z/Z_solar)
z_bins = np.array(f['TableBins']['RedshiftBins']) # redshift
IDs = np.array(f['IdentifierLines']) # list of emission line ids

# enter a list of emission line ids. For a list of all emission lines, please refer to Table 9 of Ploeckinger & Schaye (2020)
emission_line_ids = [b'H  1      6562.81A', b'H  1      4861.33A', b'O  1      6300.30A', b'S  2      4068.60A', b'Si 3      1206.50A', 
                     b'O  2      3726.03A', b'O  2      3728.81A', b'Blnd      1397.00A', b'O  3      5006.84A', b'O  3      4958.91A', 
                     b'O  3      4363.21A', b'Blnd      1549.00A', b'O  6      1031.91A', b'O  6      1037.62A', b'N  2      6583.45A']

ROMANS = ["","I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX","XXI","XXII","XXIII","XXIV","XXV","XXVI","XXVII"]
def int_to_roman(n: int) -> str:
    if n < 0 or n >= len(ROMANS):
        raise ValueError(f"Cannot convert {n} to Roman numeral within supported range.")
    return ROMANS[n]

def make_emission_label(line_id):
    """Minimal: bytes/str CHIMES ID -> 'SPECIES λA'; keeps 'Blnd' as 'Blnd'."""
    s = line_id.decode("ascii") if isinstance(line_id, (bytes, bytearray)) else str(line_id)
    s = s.strip().replace("A", "")
    parts = s.split()

    # Blends: keep as 'Blnd λA'
    if parts[0].lower().startswith("blnd"):
        w = float(parts[-1])
        return f"Blnd {w:.2f}A"

    # Regular lines: 'Elem Ion Wavelength'
    elem, ion_str, wave_str = parts[0], parts[1], parts[-1]
    species = f"{elem}{int_to_roman(int(ion_str))}"
    w = float(wave_str)
    w_key = int(round(w))
    return f"{species} {w_key:.2f}A"

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

def return_flux_fraction(norm_cooling_curve, mach_rel = 0.75, h = 1, f_nu = 0.01, kappa_0 = 1e-3, 
                         Prandtl = 0.1, tau = 10**(-1), T_cold_in_K = 1e4, T_hot_in_K = 10**(5.5), max_iter = 10):

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
    # Edot_0 = rho_hot*cs_hot**3*L_0**2
    Edot_0 = e_dot_cool_norm*L_0**2

    dEdot_dlogT_in_code_units = (T/-dT_dz)*edot_cool_double_equilibrium(T)
    dM_dlogT_in_code_units    = (T/-dT_dz)*(rho)

    dM_dlogT    = M_0*dM_dlogT_in_code_units # in solar masses
    dEdot_dlogT = Edot_0*dEdot_dlogT_in_code_units # in erg/s

    # convert mass distribution to column density distribution by dividing through by L_0**2*mH
    dSigma_dlogT = (dM_dlogT / L_0**2)
    dN_dlogT = dSigma_dlogT / (mH*gram_to_msun)
    # convert cooling distribution to cooling flux distribution by dividing through by L_0**2
    dFcool_dlogT = (dEdot_dlogT / L_0**2) * L_0
    # define the T_min & T_max for integration over the temperature space
    # T_min = 1e4
    # T_max = T_hot_in_K
    # temp_mask = (Ts >= T_min) & (Ts <= T_max)
    # integrate dFcool_dlogT over the temperature space within the range T_min and T_max
    dFcool_dlogT_int = integrate.simps(dFcool_dlogT, Ts)

    # calculate the number density of H using the total number density and fraction of H
    total_ion_count = 0
    for i in solar_abundance:
        total_ion_count += solar_abundance[i]

    n_H = ns*(solar_abundance['H'] / (total_ion_count))

    # find the index correponding to these emission lines so that we can extract them from the table
    emission_line_indexes = [0]*len(emission_line_ids)
    for i in range(len(emission_line_indexes)):
        emission_line_indexes[i] = np.where(np.array(IDs == emission_line_ids[i]))[0][0]
        
    # to calculate the surface brightness, we first extract out the emissivity table of a given emission line and interpolate in between.
    # then we plug in the temperature and density profiles from the analytic solution to obtain the emissivity as a function of position
    # in the mixing layer
    # finally, we integrate across the mixing layer to find the surface brightness
    # emissivities = {}
    surface_brightnesses = {}
    df_line_dlogT = {}
    df_line_dlogT_int = {} # emissivity * T / (dT/dz)
    # surface_brightnesses_temp = []

    # we set the metallicity to be solar (which means log(Z/Z_solar)=0) and redshift to be 0
    # here we find the corresponding indexes
    index_z_0 = np.argmin(abs(z_bins-0))
    index_Z_0 = np.argmin(abs(Z_bins-0))
    for i in range(len(IDs)):
        emissivity_grid = interpolate.RegularGridInterpolator((np.array(f['TableBins']['TemperatureBins']), np.array(f['TableBins']['DensityBins'])),
                                                               np.array(f['Tdep']['EmissivitiesVol'][index_z_0,:,index_Z_0,:,i]),
                                                               bounds_error=False, fill_value=0.0)
        if IDs[i] in emission_line_ids:
            emissivity_in_mixing_layer = emissivity_grid((np.log10(Ts), np.log10(n_H)))
            # emissivities[IDs[i]] = emissivity_in_mixing_layer
            df_line_dlogT[IDs[i]] = (T/-dT_dz) * (10**emissivity_in_mixing_layer) * L_0
            df_line_dlogT_int[IDs[i]] = integrate.simps((T/-dT_dz) * (10**emissivity_in_mixing_layer) * L_0, Ts)
            surface_brightnesses[IDs[i]] = integrate.simps(10**emissivity_in_mixing_layer, zs) / (4*math.pi)

    # calculate the flux fraction of each line 
    flux_fraction_dict = {}
    for line, value in df_line_dlogT_int.items():
        flux_fraction_dict[line] = np.abs(value / dFcool_dlogT_int)

    # save the flux fractions dictionary
    # Create the directory if it doesn't exist
    if not os.path.exists(f'flux_fraction_dicts_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}'):
        os.makedirs(f'flux_fraction_dicts_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}')
    flux_frac_dict_name = f'flux_fraction_dicts_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}/flux_fractions_T_hot={T_hot_in_K:.1e}_P={p_hot_over_kb:.1e}_mach_rel={mach_rel:.2f}_tau={tau:.2f}.csv'
    with open(flux_frac_dict_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for line, value in flux_fraction_dict.items():
            writer.writerow([line, value])

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })


    gridspec = dict(hspace=0.25, height_ratios=[1, 1, 1, 1])
    fig, ((ax1,ax2,ax3,ax4)) = plt.subplots(4,1, gridspec_kw=gridspec)

    ax1.loglog(Ts,  dFcool_dlogT, color='#118AB2')

    ax1.set_ylim(bottom=1e-10)
    ax1.set_ylabel(r'$\frac{d F_{\rm cool} }{d logT}$ $[{\rm erg}$ ${\rm cm}^{-2}$ ${\rm s}^{-1} {\rm dex}^{-1}]$', fontsize=8)

    plt.setp(ax1.get_xminorticklabels(), visible=False)
    plt.setp(ax1.get_xmajorticklabels(), visible=False)

    # enter a list of emission line labels corresponding to the emission lines that was selected
    # emission_line_labels = []
    # for line_id in emission_line_ids:
    #     emission_line_labels.append(make_emission_label(line_id))
    emission_line_labels = list(df_line_dlogT_int.keys())

    emission_line_colors = cmr.take_cmap_colors('cmr.tropical', len(emission_line_labels), cmap_range=(0.1, 0.9), return_fmt='hex')

    for i, (ID, line_flux_in_Tspace) in enumerate(df_line_dlogT.items()):
        ax2.loglog(Ts, line_flux_in_Tspace, color=emission_line_colors[i], label=ID)

    ax2.set_ylim(bottom=1e-34)
    ax2.set_ylabel(r'$\frac{d f_{\rm cool} }{d logT}$ $[{\rm erg}$ ${\rm cm}^{-2}$ ${\rm s}^{-1} {\rm dex}^{-1}]$', fontsize=8)
    ax2.set_xlabel(r'$T$ $[{\rm K}]$', fontsize=8)

    ax2.set_ylim(bottom=1e-32)

    for i, (ID, line_flux) in enumerate(df_line_dlogT_int.items()):
        ax3.scatter(i, np.abs(line_flux / dFcool_dlogT_int), marker='o', color=emission_line_colors[i])

    ax3.set_ylabel(r'Flux Fraction', fontsize=8)
    ax3.xaxis.set_ticks(np.arange(len(emission_line_labels)))
    ax3.xaxis.set_ticklabels([])
    ax3.set_yscale('log')

    ax3.tick_params(axis='x', which='minor', bottom=False)
    ax3.tick_params(axis='x', which='minor', top=False)
    ax3.tick_params(axis='y', which='minor', left=True)
    ax3.tick_params(axis='y', which='minor', right=True)

    for i, (ID, surface_brightness) in enumerate(surface_brightnesses.items()):
        ax4.scatter(i, surface_brightness, marker='o', color=emission_line_colors[i])

    ax4.xaxis.set_ticks(np.arange(len(emission_line_labels)))
    ax4.xaxis.set_ticklabels(emission_line_labels, rotation=90)
    ax4.set_ylabel(r'SB $[{\rm erg}$ ${\rm cm}^{-2}$ ${\rm s}^{-1}$ ${\rm sr}^{-1}]$', fontsize=8)
    ax4.set_yscale('log')

    ax1.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax3.tick_params(axis='both', labelsize=8)
    ax4.tick_params(axis='both', labelsize=8)

    ax4.tick_params(axis='x', which='minor', bottom=False)
    ax4.tick_params(axis='x', which='minor', top=False)
    ax4.tick_params(axis='y', which='minor', left=True)
    ax4.tick_params(axis='y', which='minor', right=True)

    fig.set_size_inches(3.39375, 9.5)

    # Create the directory if it doesn't exist
    if not os.path.exists(f'flux_fraction_plots_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}'):
        os.makedirs(f'flux_fraction_plots_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}')
    plt.savefig(f'flux_fraction_plots_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}/emissivity_and_surface_brightness_T_hot={T_hot_in_K:.1e}_P={p_hot_over_kb:.1e}_mach_rel={mach_rel:.2f}_tau={tau:.2f}.pdf', 
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    # set the input values
    # fixed parameters
    T_hot_in_K = 10**(6)
    T_cold_in_K = 1e4 # set the T cold in physical units (don't change this value)
    h = 1 
    kappa_0 = 1e-3
    # grid parameters 
    mach_rel_arr = np.arange(0.75, 2.25, 0.25) # mach_rel array
    tau = np.array([10**(-1)]) # tau array
    # Creating a meshgrid
    mach_rel_mesh, tau_mesh = np.meshgrid(mach_rel_arr, tau)
    Prandtl = None 
    f_nu = None 
    # creating the maximum iteration number to find mdot_crit
    max_iter = 10

    # Assuming mach_rel_mesh and tau_mesh are your meshgrid arrays
    for i in range(mach_rel_mesh.shape[0]):
        for j in range(mach_rel_mesh.shape[1]):
            # Retrieve the current combination of mach_rel and tau
            mach_rel = mach_rel_mesh[i, j]
            tau = tau_mesh[i, j]
            # get the cooling curves at constant pressures based on PS20 
            # current P/kb value varies from 10**(0.5) to 10**(9) cm-3 K
            norm_cooling_curve_dir = 'ps20_cooling_curves_const_P'
            cooling_curve_files = sorted(os.listdir(norm_cooling_curve_dir))
            for cooling_curve in cooling_curve_files:
                if ".npz" in cooling_curve:
                    cooling_curve_path = os.path.join(norm_cooling_curve_dir, cooling_curve)
                    print(f"The current relative Mach number is: {mach_rel:.2f}; tau value is : {tau:.2f}")
                    print("The current cooling curve is: " + cooling_curve)
                    return_flux_fraction(cooling_curve_path, mach_rel = mach_rel, h = h, f_nu = f_nu, kappa_0 = kappa_0, Prandtl = Prandtl, tau = tau, 
                                         T_cold_in_K = T_cold_in_K, T_hot_in_K = T_hot_in_K, max_iter = max_iter)


