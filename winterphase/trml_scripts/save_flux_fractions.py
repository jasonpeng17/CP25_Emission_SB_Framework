# plot flux fraction for different parameter values (e.g., pressure)

import numpy as np
import matplotlib.pyplot as plt
import os
from IPython import embed
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import RectBivariateSpline
from matplotlib import rc

# vary pressure and mach_rel (tau = 0.1; fnu and Pr based on Chen et al. (2023))
T_hot_in_K_arr = 10**(np.arange(5, 8.5, 0.5))
tau = 10**(-1)

############ grid 
for indx, T_hot_in_K in enumerate(T_hot_in_K_arr):
    flux_frac_dir = f'/Users/zixuanpeng/Desktop/Prof.Martin/galactic_wind/zirui_fluid_notes/flux_fraction_dicts_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}'
    flux_frac_files = sorted(os.listdir(flux_frac_dir))
    flux_frac_tables = [file for file in flux_frac_files if f'tau={tau:.2f}' in file]

    # mach_rel_arr = np.arange(0.75, 6.00, 0.50)
    # mach_rel_arr = np.arange(0.75, 2.50, 0.25)
    mach_rel_arr = np.arange(0.75, 2.25, 0.25)
    p_arr = np.power(10, np.arange(0.5, 9.5, 0.5))
    # Creating a meshgrid
    mach_rel_mesh, p_mesh = np.meshgrid(mach_rel_arr, p_arr)
    ############

    # iterate through each table
    for i, table in enumerate(flux_frac_tables):

        # get the file path
        file_path = os.path.join(flux_frac_dir, table)
        df = pd.read_csv(file_path, index_col=0, header=None)  # Set the first column as the index (row names)

        if i == 0:
            line_keys = df.index.tolist()
            print(line_keys)
            flux_frac_line_dict = {line: np.zeros((len(p_arr), len(mach_rel_arr))) for line in line_keys}

        # Extract pressure value from the file name
        pressure_str = table.split('_')[4]  # This splits the filename and picks the part with pressure
        pressure_value = float(pressure_str.split('=')[1])  # This splits at '=' and converts the value to float

        # Extract mach_rel value from the file name
        mach_rel_str = table.split('_')[6]
        mach_rel_value = float(mach_rel_str.split('=')[1])

        # Indices for the current parameter values
        p_idx = np.argmin(np.abs(p_arr - pressure_value))
        mach_idx = np.argmin(np.abs(mach_rel_arr - mach_rel_value))

        # Save emissivities for all lines
        for line in line_keys:
            flux_frac_line_dict[line][p_idx, mach_idx] = df.loc[line].iloc[0]

        # Extracting the flux fraction for 'O 3 5006.84A'
        # flux_fraction_5007 = df.loc["b'O  3      5006.84A'"].iloc[0]
        # flux_frac_5007_dict[f"p={pressure_value:.1e}_mach_rel={mach_rel_value:.2f}"] = flux_fraction_5007

    output_file_path = f"flux_fractions_T_hot={T_hot_in_K:.1e}_tau={tau:.2f}_rho_vx_cosine_grid.npz"
    np.savez(output_file_path, flux=flux_frac_line_dict, Ps=p_arr, Mrels=mach_rel_arr, tau = tau, Thot = T_hot_in_K)


