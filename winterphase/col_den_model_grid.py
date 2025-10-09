import numpy as np
import time
import os, sys
import pandas as pd
from astropy import constants as const
from scipy.integrate import quad, dblquad
from scipy.integrate import simps, trapz
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import cumtrapz
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from IPython import embed
from constants import *
from utils import *
from input_params import *

"""
MAIN function to derive the ion column density at each impact parameter (b)
"""
def calc_column_density_at_b(b_eval, xi, R, etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, M_dot_star, 
                             master_folder_name = 'hayes_2016_setups', which_setup = 'default_setup', 
                             which_elem = 'Si', ion_num = 4, T_min = 1e4, T_max = 1e6):
    """
    Calculates the column density of a chosen ion at each evaluated impact parameter based on the given FB22 solution.
    """
    # define the radius array in kpc
    r_model = xi * R
    r_model_kpc = xi * R / kpc

    # get the 1D interpolation functions given the etaE, etaM and v_e values
    v_c_km_per_s = v_c / kms
    # for clouds
    _, _, rho_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'rho', master_setup_name = master_folder_name, which_setup = which_setup)
    _, _, u_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'u', master_setup_name = master_folder_name, which_setup = which_setup)
    _, _, T_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'T', master_setup_name = master_folder_name, which_setup = which_setup)
    _, _, Z_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'Z', master_setup_name = master_folder_name, which_setup = which_setup)
    _, _, Mcl_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'Mcl', master_setup_name = master_folder_name, which_setup = which_setup)
    # for winds
    _, _, rho_w_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'rho', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')
    _, _, u_w_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'u', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')
    _, _, T_w_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'T', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')
    _, _, P_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'P', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')
    _, _, Z_w_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'Z', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')

    # define the E_dot, M_dot, and the corresponding v0, n0, and T0 in cgs units
    E_dot = 3e41 * etaE * M_dot_star # erg / s 
    M_dot = etaM * M_dot_star * Msun / yr # M_sun yr-1
    v0 = v0_func(M_dot, E_dot) 
    n0 = n0_func(M_dot, E_dot, R, mu, mp)
    T0 = T0_func(M_dot, E_dot, mu, mp, kB)
    P0 = P0_func(M_dot, E_dot, R)

    # define the temperature cut based on T_min and T_max
    T_mix_r = T0 * T_x_func1d(xi)
    T_mask = (T_mix_r >= T_min) & (T_mix_r <= T_max)
    R_min_Tmask = np.nanmin(r_model[T_mask])
    R_max_Tmask = np.nanmax(r_model[T_mask])

    # ion name 
    ion_name = which_elem+int_to_roman(ion_num)
        
    # Define the integrand for the column density integral
    def cd_integrand(l, which_phase = 'hot'):
        r = np.hypot(b_eval, l)  # sqrt(b^2 + l^2)
        r_star = r / R
        v_r = v0 * u_x_func1d(r_star) # v_cloud
        rho_r = mu * mp * n0 * rho_x_func1d(r_star) # rho_cloud
        n_r = rho_r / (mu * mp)
        # avoid the cloud density too large or too small 
        # (outside the PS20 cooling table limit)
        if n_r > 1e6:
            n_r = 1e6
        elif n_r < 1e-8:
            n_r = 1e-8
        T_r = T0 * T_x_func1d(r_star) # T_mix
        T_cl = 1e4 # K
        Z_r = Z_x_func1d(r_star) # Z_cloud
        Mcl_r = Mcl_x_func1d(r_star) # M_cloud

        v_w_r = v0 * u_w_x_func1d(r_star) # v_wind
        rho_w_r = mu * mp * n0 * rho_w_x_func1d(r_star) # rho_wind
        n_w_r = rho_w_r / (mu * mp)
        # avoid the wind density too large or too small 
        # (outside the PS20 cooling table limit)
        if n_w_r > 1e6:
            n_w_r = 1e6
        elif n_w_r < 1e-8:
            n_w_r = 1e-8
        T_w_r = T0 * T_w_x_func1d(r_star) # T_wind
        P_r = P_x_func1d(r_star) * P0 # pressure for winds and clouds
        Z_w_r = Z_w_x_func1d(r_star) # Z_wind

        cs_sq_w = (gamma * P_r / rho_w_r) # sound speed square for winds
        cs_sq_cl = gamma * kB * T_cl / (mu * mp) # sound speed square for clouds
        v_rel = (v_w_r - v_r)
        M_rel = v_rel / np.sqrt(cs_sq_w) # relative mach number 

        # cloud number density 
        M_cloud_init = 10**log_Mcl * Msun
        SFR = M_dot_star * Msun / yr # M_sun yr-1
        Mdot_cold_init  = etaM_cold * SFR              ## mass flux in cold clouds
        Ndot_cloud_init = Mdot_cold_init / M_cloud_init ## number flux in cold clouds
        # cold_cloud_injection_radial_power   = np.inf
        cold_cloud_injection_radial_power   = 6 
        cold_cloud_injection_radial_extent  = 1.33*R
        Ndot_cloud              = Ndot_cloud_init * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
        number_density_cloud    = Ndot_cloud/(4 * np.pi * v_r * r**2) 

        if which_phase == 'hot':
            # hot phase contribution
            n_of_element_w = n_w_r*(solar_abundance[which_elem] / (total_ion_count)) # number density of the element of interest (all ions combined)
            fraction_of_ion_w = IonFractions[ion_name]((np.log10(T_w_r), np.log10(n_w_r), np.log10(Z_w_r)))
            n_of_ion_w = fraction_of_ion_w*n_of_element_w # calculate number density of the ion using the two previous results
            return n_of_ion_w

        elif which_phase == 'mix':
            R_cl = (Mcl_r / (4 * np.pi * rho_r / 3))**(1/3) # TODO: assume a spherical cloud
            f_cl = number_density_cloud * np.pi * R_cl**2 # (i.e., number of hits with clouds for a unit length)
            # C_f = np.pi * number_density_cloud * ((R_cl**2 * d_eval) - ((d_eval**3) / 12)) # based on Eq. 3 in Li et al. (2024)
            col_den = 10**eval_col_den_log(ion_name, np.log10(P_r / kB), M_rel, np.log10(T_w_r))
            # returns the interpolated line flux fraction at certain pressure, mach_rel, and T_hot
            return col_den * f_cl

    L_max = np.sqrt(R_max_Tmask**2 - b_eval**2)
    cd_mix, _ = quad(lambda l: cd_integrand(l, 'mix'), 0.0, L_max, points=[0.0])
    cd_hot, _ = quad(lambda l: cd_integrand(l, 'hot'), 0.0, L_max, points=[0.0])
    cd = cd_mix + cd_hot

    return 2 * cd

if __name__ == "__main__":
    v_c_km_per_s = v_circ0 / 1e5 # in km / s
    Z_wind_init_solar = Z_wind_init / Z_solar

    # initialize the SB profile
    R_eval_arr = R_eval_arr * r_sonic
    R_eval_arr_kpc = R_eval_arr / kpc

    # define the ion fractions for the ions of interest
    IonFractions = {}
    for i in ions_of_interest: # loop over each element
        for j in range(len(ions_of_interest[i])): # loop over each ion number
            chimes_ionfracs = build_ion_fraction_interpolators(line_id=None, elem = i, ion_n = ions_of_interest[i][j],
                                                               chimes_dict = chimes_dict, neq_times_seconds=(noneq_time * Myr,))
            if eq_or_noneq == 'eq': # equilibrium solution
                chimes_eq_interp = chimes_ionfracs["eq"]
                IonFractions[i+int_to_roman(ions_of_interest[i][j])] = chimes_eq_interp
            if eq_or_noneq == 'noneq': # non-equilibrium solution
                # get the (single) nearest available time used
                (t_used,) = tuple(chimes_ionfracs["neq"].keys())
                chimes_neq_interp = chimes_ionfracs["neq"][t_used]
                IonFractions[i+int_to_roman(ions_of_interest[i][j])] = chimes_neq_interp

    def run_cd_model(etaE_val, etaM_val, SFR_input):
        etaE = etaE_val
        etaM = etaM_val

        # define the folder name
        setup_name = f'{folder_name_base}_etaMcl={etaM_cold:.{num_decimal}f}_logMcl={log_Mcl:.{num_decimal}f}_sfr={SFR_input:.{num_decimal}f}_setup'
        print('-'*30)
        print(f'etaE={etaE:.{num_decimal}f}, etaM={etaM:.{num_decimal}f}, sfr={SFR_input:.{num_decimal}f}, setup_name={setup_name}')
        start_time = time.time()  # Start time for this setup

        # For plotting the dL/dlnT over T curves (and to get xi grid)
        xi_arr = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Z_wind_init_solar, which = 'u', 
                                       master_setup_name = master_folder_name, which_setup = setup_name)[0]

        # define the evaluated radius (starting from R_SF to R_max)
        col_den_ions = dict()
        col_den_ions['b_kpc'] = R_eval_arr_kpc

        for elem in ions_of_interest: # loop over each element
            for ion_indx in range(len(ions_of_interest[elem])): # loop over each ion number
                ion_num = ions_of_interest[elem][ion_indx]
                col_den_eval_arr = np.zeros(len(R_eval_arr))
                for i, R_eval in enumerate(R_eval_arr):
                    # d_eval = R_eval_diff_model[i]
                    col_den_at_b = calc_column_density_at_b(R_eval, xi_arr, r_sonic, etaE, etaM, etaM_cold, log_Mcl,
                                                            v_circ0, Z_wind_init_solar, SFR_input, master_folder_name = master_folder_name,
                                                            which_setup = setup_name, which_elem = elem, ion_num = ion_num, 
                                                            T_min = 1e4, T_max = 1e6)
                    col_den_eval_arr[i] = col_den_at_b  
                col_den_ions[elem+int_to_roman(ion_num)] = col_den_eval_arr

        # Save
        base_root = "../cd_model_outputs"
        tag = make_tag(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Z_wind_init_solar, num_decimal)
        # append noneq label if needed
        if eq_or_noneq == 'noneq':
            tag = f"{tag}_noneq={noneq_time}Myr"
        save_items(base_root, master_folder_name, setup_name, tag, items=[("cd_arr", "cd_dict", col_den_ions)])

        elapsed_time = time.time() - start_time
        print(f"Computation time: {elapsed_time:.2f} seconds")
        print('-'*30 + '\n')

    # iterate according to vary_which_eta
    SFR_inputs_Msun_yr = SFR_inputs / (Msun / yr)
    for SFR_input in SFR_inputs_Msun_yr:
        if vary_which_eta == 'etaM':
            for etaM_val in etaM_grid:
                run_cd_model(etaE, etaM_val, SFR_input)
        elif vary_which_eta == 'etaE':
            for etaE_val in etaE_grid:
                run_cd_model(etaE_val, etaM, SFR_input)
        else:
            raise ValueError("vary_which_eta must be 'etaM' or 'etaE'.")



