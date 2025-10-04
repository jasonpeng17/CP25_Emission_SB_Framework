import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
from matplotlib import cm
from matplotlib import rc
from matplotlib.ticker import NullLocator, LogLocator
import matplotlib.colors as mcolors
import os, sys
from pathlib import Path

from IPython import embed
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from constants import *
from utils import phys_star_interp_fb22, make_tag

def pressure_Mrel_fb22(R_eval_arr, R, etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, M_dot_star, master_folder_name, which_setup):
    """
    Compute dynamical/thermo profiles (thermal pressure, ram pressure, relative Mach number)
    at radii R_eval_arr for an FB22 solution identified by (master_folder_name, which_setup).

    Returns
    -------
    P_r : ndarray [erg/cm^3]
        Thermal pressure of the wind/cloud (pressure equilibrium branch).
    P_ram_r : ndarray [erg/cm^3]
        Ram pressure of the wind on the clouds, rho_w * (v_w - v_cl)^2.
    M_rel_r : ndarray
        Relative Mach number, (v_w - v_cl) / c_s,wind (dimensionless).

    Notes
    -----
    - Pulls dimensionless FB22 solutions (rho,u,T,P) via `phys_star_interp_fb22` for both cloud and wind,
      converts them to physical units with E_dot, M_dot, and the base scales (v0, n0, T0, P0).
    """
    # define the dimensionless radius array
    xi = R_eval_arr / R
    
    # get the 1D interpolation functions given the etaE, etaM and v_e values
    v_c_km_per_s = v_c / kms
    # for clouds
    _, _, rho_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'rho', master_setup_name = master_folder_name, which_setup = which_setup)
    _, _, u_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'u', master_setup_name = master_folder_name, which_setup = which_setup)
    _, _, T_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'T', master_setup_name = master_folder_name, which_setup = which_setup)
    # for winds
    _, _, rho_w_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'rho', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')
    _, _, u_w_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'u', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')
    _, _, T_w_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'T', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')
    _, _, P_x_func1d = phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, which = 'P', master_setup_name = master_folder_name, which_setup = which_setup, wind_or_cloud = 'wind')

    # define the E_dot, M_dot, and the corresponding v0, n0, and T0 in cgs units
    E_dot = 3e41 * etaE * M_dot_star # erg / s 
    M_dot = etaM * M_dot_star * Msun / yr # M_sun yr-1
    v0 = v0_func(M_dot, E_dot) 
    n0 = n0_func(M_dot, E_dot, R, mu, m_p)
    T0 = T0_func(M_dot, E_dot, mu, m_p, k_B)
    P0 = P0_func(M_dot, E_dot, R)

    # v cloud
    v_cl_r = v0 * u_x_func1d(xi) # v_cloud
    # v wind 
    v_w_r = v0 * u_w_x_func1d(xi) # v_wind
    # v_rel
    v_rel_r = v_w_r - v_cl_r
    # rho_wind
    rho_w_r = mu * m_p * n0 * rho_w_x_func1d(xi) 
    n_w_r = rho_w_r / (mu * m_p)
    # thermal pressure for winds and clouds
    P_r = P_x_func1d(xi) * P0 
    # sound speed square for wind
    cs_sq_w_r = (gamma * P_r / rho_w_r) 
    # relative mach number 
    M_rel_r = v_rel_r / np.sqrt(cs_sq_w_r) 
    # ram pressure
    P_ram_r = rho_w_r * v_rel_r**2

    return P_r, P_ram_r, M_rel_r

def extract_ovi_profile(etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, master_setup_name, which_setup, eq_or_noneq = 'noneq', noneq_time = 10, eta_num_decimal = 2):
    """
    Load the O VI surface-brightness radial profile for a given setup.

    Returns
    -------
    ovi_profile_rkpc : ndarray [kpc]
        Radial bins (kpc) at which the SB is reported (excluding first bin).
    ovi_tot_profile : ndarray [erg/s/cm²/arcsec²]
        Total O VI SB (sum of 1031.91 Å and 1037.62 Å components) per radial bin.
    """

    v_c_km_per_s = v_c / 1e5

    # Build the tag once
    tag = make_tag(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind, eta_num_decimal)
    if eq_or_noneq == 'noneq':
        tag = f"{tag}_noneq={noneq_time}Myr"

    # Compose the filename and full path
    filename = f"sb_dict_{tag}.npy"
    ovi_profile_path = PARENT / f'cp25_model_outputs/sb_arr/{master_setup_name}/{which_setup}' / filename
    # Load
    ovi_profile_dict = np.load(ovi_profile_path, allow_pickle=True).item()

    # get the radius and ovi sb arrays
    ovi_profile_rkpc = ovi_profile_dict['r_kpc'][1:]
    ovi_1032_profile = ovi_profile_dict[b'O  6      1031.91A'][1:]
    ovi_1038_profile = ovi_profile_dict[b'O  6      1037.62A'][1:]
    ovi_tot_profile = ovi_1032_profile + ovi_1038_profile

    return ovi_profile_rkpc, ovi_tot_profile


def find_rkpc_obs_limit(R, etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, M_dot_star, master_setup_name, 
                        which_setup, eq_or_noneq = 'noneq', noneq_time = 10, eta_num_decimal = 2, ram_or_therm = 'ram', sb_limit = 1e-18):
    """
    Find the largest radius where the O VI surface brightness reaches an observational
    threshold, optionally applying a ram/thermal pressure scaling.

    Parameters
    ----------
    ram_or_therm : {'ram','therm'}
        If 'ram', scale SB by (P_ram/P_therm)^2 using `pressure_Mrel_fb22`; if 'therm', use SB as-is.
    sb_limit : float [erg/s/cm²/arcsec²]
        Observational SB threshold for the crossing.

    Returns
    -------
    rkpc_intersect : float [kpc]
        Largest radius where the profile crosses the threshold.
    intersect_idx : tuple (array-of-int,)
        Index/indices in the radial grid corresponding to the crossing bin(s).

    Notes
    -----
    - Crossing is found via sign-change of (SB - SB_limit). For sub-bin precision, linearly
      interpolate between bracketing bins if needed.
    """

    # get the ovi kpc and profile arrays
    ovi_profile_rkpc, ovi_tot_profile = extract_ovi_profile(etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, master_setup_name,
                                                            which_setup, eq_or_noneq, noneq_time, eta_num_decimal)


    # ram or thermal pressure equilibrium
    if ram_or_therm == 'ram':
        P_r, P_ram_r, M_rel_r = pressure_Mrel_fb22(ovi_profile_rkpc * kpc, R, etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, M_dot_star, master_setup_name, which_setup)
        ovi_tot_profile = ovi_tot_profile * (P_ram_r / P_r)**2

    # initialize the sb limit with the same length as the ovi profile
    sb_limit_arr = np.ones(len(ovi_profile_rkpc)) * sb_limit

    # find the intersection points between the ovi_profile and the sb_limit_arr
    ovi_sb_limit_diff = ovi_tot_profile - sb_limit_arr
    # find the index around the intersection point (usually the grid is fined enough so this approch is fine)
    intersect_idx = np.argwhere(np.diff(np.sign(ovi_sb_limit_diff[~np.isnan(ovi_sb_limit_diff)])) != 0) 

    try:
        rkpc_intersect = np.nanmax(ovi_profile_rkpc[intersect_idx])
        intersect_idx = np.where(ovi_profile_rkpc == rkpc_intersect)
    except: # when there is no intersection between the ovi profile and the sb limit 
        rkpc_intersect = np.nan 
        intersect_idx = np.nan

    return rkpc_intersect, intersect_idx


