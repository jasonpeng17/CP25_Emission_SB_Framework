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
MAIN function to derive the SB
"""
def calc_surface_brightness(R_eval, xi, R, etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, M_dot_star, 
                            master_folder_name = 'hayes_2016_setups', which_setup = 'default_setup', which = 'line', 
                            line_key = b'O  6      1031.91A', T_min = 1e4, T_max = 1e6, eq_or_noneq = 'eq', noneq_time = 1):
    """
    Calculates the surface brightness of a chosen line at each evaluated radius based on the given FB22 solution.
    """
    # define the radius array in kpc
    r_model = xi * R
    r_model_kpc = xi * R / kpc
    R_max = np.nanmax(r_model)
    
    # get the 1D interpolation functions given the etaE, etaM and v_e values
    # gamma = 5. / 3.
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

    # CHIMES ion-fraction interpolators for the chosen line
    if eq_or_noneq == 'noneq':
        t_req_seconds = noneq_time * Myr
        chimes_ionfracs = build_ion_fraction_interpolators(line_key, chimes_dict, neq_times_seconds=(t_req_seconds,))
        # get the (single) nearest available time used
        (t_used,) = tuple(chimes_ionfracs["neq"].keys())
        chimes_neq_interp = chimes_ionfracs["neq"][t_used]
        chimes_eq_interp  = chimes_ionfracs["eq"]
        
    # Define the integrand for the line surface brightness integral
    def sb_integrand(r, Rprime, which = 'line', which_phase = 'hot'):
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

        # define parameters for M_cl_grow
        f_turb0      = 10**-1.0
        TurbulentVelocityChiPower   =  0.0 
        CoolingAreaChiPower         =  0.5
        ColdTurbulenceChiPower      = -0.5
        Mdot_coefficient            = 2.0 # consistent with public version of FB22 (online github version: 1.0 / 3.0)
        M_cloud_min = 1e-2*const.M_sun.cgs.value     ## minimum mass of clouds
        Myr          = 3.154e13
        chi          = rho_r / rho_w_r # density contrast
        r_cl      = (Mcl_r / ( 4*np.pi/3. * rho_r))**(1/3.) 
        v_rel        = (v_w_r - v_r)
        M_rel = v_rel / np.sqrt(cs_sq_w) # relative mach number 
        v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
        T_mix        = (T_w_r*T_cl)**0.5
        Z_mix        = (Z_w_r*Z_r)**0.5
        n_mix        = chi**0.5 * n_w_r
        t_cool_layer = tcool_P(T_mix, P_r/kB, Z_mix)[()] 
        t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
        ksi          = r_cl / (v_turb * t_cool_layer)
        AreaBoost    = chi**CoolingAreaChiPower
        v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
        Mdot_grow    = Mdot_coefficient * 3.0 * Mcl_r * v_turb * AreaBoost / (r_cl * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
        Mdot_loss    = Mdot_coefficient * 3.0 * Mcl_r * v_turb_cold / r_cl
        Mdot_cloud   = np.where(Mcl_r > M_cloud_min, Mdot_grow - Mdot_loss, 0)

        # cloud number density 
        M_cloud_init = 10**log_Mcl * const.M_sun.cgs.value
        SFR = M_dot_star * const.M_sun.cgs.value / (3.154e7) # M_sun yr-1
        Mdot_cold_init  = etaM_cold * SFR              ## mass flux in cold clouds
        Ndot_cloud_init = Mdot_cold_init / M_cloud_init ## number flux in cold clouds
        # cold_cloud_injection_radial_power   = np.inf
        cold_cloud_injection_radial_power   = 6 
        cold_cloud_injection_radial_extent  = 1.33*R
        Ndot_cloud              = Ndot_cloud_init * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
        number_density_cloud    = Ndot_cloud/(4 * np.pi * v_r * r**2) 
        drhodt_plus  = (number_density_cloud * Mdot_loss)
        drhodt_minus = (number_density_cloud * Mdot_grow) 

        # total advection enthalpy (compute each term)
        M_w = v_w_r / np.sqrt(cs_sq_w) # define the Mach number
        cooling_factor = (gamma - (M_w**(-2))) / (1. - (M_w**(-2)))
        
        # define the emissivity for the TRMLs
        epsilon_dot_mixing = (((cs_sq_w - cs_sq_cl) / (gamma - 1.)) + (0.5 * v_rel**2)) * drhodt_minus
        emis_mixing = cooling_factor * epsilon_dot_mixing

        if which == 'line':
            # get the emissivity grid interpolator for a specific line
            if which_phase == 'hot':
                emissivity_grid_line = emissivity_grids[line_key]
                emissivity_line = 10**emissivity_grid_line((np.log10(T_w_r), np.log10(Z_w_r), np.log10(n_w_r)))
                emis_line_cooling = cooling_factor * emissivity_line
                return np.abs(emis_line_cooling) * r / np.sqrt(r**2 - Rprime**2)

            elif which_phase == 'mix':
                flux_frac = 10**(flux_frac_line_func2d[line_key](np.log10(P_r / kB), M_rel, grid = False))
                emis_line_mixing = emis_mixing * flux_frac
                return np.abs(emis_line_mixing) * r / np.sqrt(r**2 - Rprime**2)
        
        if which == 'total':
            if which_phase == 'hot':
                epsilon_dot_cooling = n_w_r**2 * Lambda_P_rho(P_r, rho_w_r, Z_w_r)
                emis_cooling = cooling_factor * epsilon_dot_cooling
                return np.abs(emis_cooling) * r / np.sqrt(r**2 - Rprime**2)

            elif which_phase == 'mix':
                return np.abs(emis_mixing) * r / np.sqrt(r**2 - Rprime**2)

    # define the temperature, density, and metallicity for clouds, mixing layers, and hot winds at this radius
    r_star = R_eval / R
    v_r = v0 * u_x_func1d(r_star) # v_cloud
    rho_r = mu * mp * n0 * rho_x_func1d(r_star) # rho_cloud
    n_r = rho_r / (mu * mp)
    T_r = T0 * T_x_func1d(r_star) # T_mix
    T_cl = 1e4 # K
    Z_r = Z_x_func1d(r_star) # Z_cloud

    v_w_r = v0 * u_w_x_func1d(r_star) # v_wind
    rho_w_r = mu * mp * n0 * rho_w_x_func1d(r_star) # rho_wind
    n_w_r = rho_w_r / (mu * mp)
    T_w_r = T0 * T_w_x_func1d(r_star) # T_wind
    Z_w_r = Z_w_x_func1d(r_star) # Z_wind

    chi = rho_r / rho_w_r # density contrast
    # T_mix = (T_w_r*T_cl)**0.5
    Z_mix = (Z_w_r*Z_r)**0.5
    n_mix = chi**0.5 * n_w_r

    sb_mix, _ = quad(sb_integrand, R_eval, R_max_Tmask, args=(R_eval, which, 'mix'))
    sb_hot, _ = quad(sb_integrand, R_eval, R_max_Tmask, args=(R_eval, which, 'hot'))

    if eq_or_noneq == 'noneq':
        # eq fractions at R_eval
        ion_eq_mix = chimes_eq_interp((np.log10(T_r), np.log10(n_mix), np.log10(Z_mix)))
        ion_eq_w   = chimes_eq_interp((np.log10(T_w_r), np.log10(n_w_r), np.log10(Z_w_r)))
        # noneq fractions at nearest snapshot
        ion_neq_mix = chimes_neq_interp((np.log10(T_r), np.log10(n_mix), np.log10(Z_mix)))
        ion_neq_w   = chimes_neq_interp((np.log10(T_w_r), np.log10(n_w_r), np.log10(Z_w_r)))

        # safe ratios
        mix_ratio = (ion_neq_mix / ion_eq_mix) if ion_eq_mix > 0 else 1.0
        hot_ratio = (ion_neq_w   / ion_eq_w)   if ion_eq_w   > 0 else 1.0

        sb_mix *= mix_ratio
        sb_hot *= hot_ratio

    sb = sb_mix + sb_hot

    return 2 * sb

if __name__ == "__main__":
    v_c_km_per_s = v_circ0 / 1e5 # in km / s
    Z_wind_init_solar = Z_wind_init / Z_solar

    # Set up necessary variables for cosmological calculations.
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3) # initializing the flat-LambdaCDM cosmology
    D_L_at_z = cosmo.luminosity_distance(z_galaxy).value # luminosity distance in Mpc
    D_L_cm = D_L_at_z * 1e3 * kpc # in cm
    D_A_at_z = cosmo.angular_diameter_distance(z_galaxy).to(u.kpc) # angular diameter distance in kpc
    pixel_scale = (D_A_at_z / u.radian).to(u.kpc/u.arcsec).value # pixel scale in kpc/arcsec

    # initialize the SB profile
    R_eval_arr = R_eval_arr * r_sonic
    R_eval_arr_kpc = R_eval_arr / kpc
    shell_dr = np.diff(R_eval_arr, prepend=0)
    shell_dr_kpc = shell_dr / kpc

    def run_cp25_sb_model(etaE_val, etaM_val, SFR_input):
        etaE = etaE_val
        etaM = etaM_val

        # define the folder name
        setup_name = f'{folder_name_base}_etaMcl={etaM_cold:.{num_decimal}f}_logMcl={log_Mcl:.{num_decimal}f}_sfr={SFR_input:.{num_decimal}f}_setup'
        print('-'*30)
        print(f'etaE={etaE:.{num_decimal}f}, etaM={etaM:.{num_decimal}f}, sfr={SFR_input:.{num_decimal}f}, setup_name={setup_name}')
        start_time = time.time()  # Start time for this setup

        # For plotting the dL/dlnT over T curves (and to get xi grid)
        xi_arr = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Z_wind_init_solar, which = 'u', master_setup_name = master_folder_name, which_setup = setup_name)[0]

        # define the evaluated radius (starting from R_SF to R_max)
        sb_lines = dict()
        sb_lines['r_kpc'] = R_eval_arr_kpc[:-1]

        for indx, line in enumerate(which_lines): # iterate over each line
            # calculate SB at each radius
            SB_eval_arr = np.zeros(len(R_eval_arr))
            for i, R_eval in enumerate(R_eval_arr):
                sb_tot_Reval = calc_surface_brightness(R_eval, xi_arr, r_sonic, etaE, etaM, etaM_cold, log_Mcl,
                                                       v_circ0, Z_wind_init_solar, SFR_input, master_folder_name = master_folder_name,
                                                       which_setup = setup_name, which = 'line', line_key = line, T_min = 1e4, T_max = 1e6, 
                                                       eq_or_noneq = eq_or_noneq, noneq_time = noneq_time)
                SB_eval_arr[i] = sb_tot_Reval  

            # convert to SB per arcsec^2
            L_cum_lum = cumtrapz(2 * np.pi * R_eval_arr * SB_eval_arr, R_eval_arr, initial=0) # cumulative luminosity up to R
            L_lum_dr = L_cum_lum[1:] - L_cum_lum[:-1]
            F_eval_dr = L_lum_dr / (4 * np.pi * D_L_cm**2)
            SB_eval_arcsec = F_eval_dr / (2 * np.pi * R_eval_arr_kpc[:-1] * shell_dr_kpc[:-1] / (pixel_scale)**2)

            sb_lines[line] = SB_eval_arcsec 

        # Save
        base_root = "../cp25_model_outputs"
        tag = make_tag(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Z_wind_init_solar, num_decimal)
        # append noneq label if needed
        if eq_or_noneq == 'noneq':
            tag = f"{tag}_noneq={noneq_time}Myr"
        save_items(base_root, master_folder_name, setup_name, tag, items=[("sb_arr", "sb_dict", sb_lines)])

        elapsed_time = time.time() - start_time
        print(f"Computation time: {elapsed_time:.2f} seconds")
        print('-'*30 + '\n')

    # iterate according to vary_which_eta
    SFR_inputs_Msun_yr = SFR_inputs / (Msun / yr)
    for SFR_input in SFR_inputs_Msun_yr:
        if vary_which_eta == 'etaM':
            for etaM_val in etaM_grid:
                run_cp25_sb_model(etaE, etaM_val, SFR_input)
        elif vary_which_eta == 'etaE':
            for etaE_val in etaE_grid:
                run_cp25_sb_model(etaE_val, etaM, SFR_input)
        else:
            raise ValueError("vary_which_eta must be 'etaM' or 'etaE'.")



