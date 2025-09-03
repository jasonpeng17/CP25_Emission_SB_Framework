# CP25 Emission-Line Surface Brightness Modeling Suite

## Introduction

This repository provides a **Python-based pipeline** for computing **emission-line surface brightness (SB) profiles** by combining the multiphase galactic wind model developed by [Fielding & Bryan (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...924...82F/abstract) with the mixing layer model developed by [Chen et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...950...91C/abstract).  

The suite is composed of two complementary modules:  

* **FB22 Model Grid (`fb22_model_grid.py`)**: Implements and extends the multiphase wind model of [Fielding & Bryan (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...924...82F/abstract). It produces the radial structure of hot winds and cold clouds, including velocity, density, pressure, metallicity, and cloud growth/loss.  
* **CP25 SB Model Grid (`cp25_sb_model_grid.py`)**: Builds on FB22 outputs and computes **emission-line SB profiles** using equilibrium and/or non-equilibrium ionization fractions from **CHIMES** and line emissivities from **PS20 cooling/emissivity tables**.  

**Key Features**:  
- Self-consistent treatment of **hot wind phase, mixing layers, and cold clouds**.  
- Incorporates **non-equilibrium ionization** from CHIMES.  
- Computes SB profiles for emission lines (e.g., H I Lyα, [O III] 5007, O VI 1032).  
- Flexible grid setup via `input_params.py` (star formation rate, ηE, ηM, cloud mass, metallicity, etc.).  
- Automated execution pipeline (`run.sh`) with logging and error handling.  
- Outputs include **model solution arrays and emission-line SB arrays**.  

---

## Installation

1. Clone or download the repository.  
2. Ensure you have a Python environment (conda/venv) with the required packages:
   ```
   numpy, scipy, matplotlib, astropy, h5py, pandas, cmasher
   ```
3. Activate your environment before running scripts:
   ```bash
   conda activate myenv
   ```
4. External grids (place in `../ps20_grids`, `../chen23_grids`, and `../chimes_grids`):  
To run the models, you must download several large external datasets. 
We provide them via Google Drive here:  
👉 [Download Grids from Google Drive](https://drive.google.com/drive/folders/1m2wRyZ6dbtOACDnK1PJaEXcxGoXijNLl?usp=sharing)  
   - **[Ploeckinger & Schaye (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4857P/abstract) cooling/emissivity tables**  
   - **Chen+23 TRML flux fraction grids**  
   - **CHIMES equilibrium & non-equilibrium outputs**  

---

## Running

The pipeline is executed in two stages using `run.sh`:  

```bash
./run.sh
```

### Options:
- `--logdir logs` → directory for log files (default: `logs/`)  
- `--dry-run` → print commands without running  
- `--skip-fb22` → skip FB22 grid stage  
- `--skip-cp25` → skip CP25 SB stage  

### Workflow:
1. **FB22 Stage** (`fb22_model_grid.py`):  
   Generates multiphase wind solutions across the chosen ηE/ηM grid and saves them into `../fb22_model_outputs`.  

2. **CP25 Stage** (`cp25_sb_model_grid.py`):  
   Uses FB22 outputs + CHIMES/PS20 grids to compute emission-line SB profiles, saved into `../cp25_model_outputs`.  

---

## Configuration

All model parameters are set in `input_params.py`:

- **Global settings**:  
  - `master_folder_name` → master folder name for a given set of runs  
  - `folder_name_base` → base of each folder name for a particular run within a set  
  - `num_decimal` → number of decimal points for parameter values when saving outputs  

- **Wind parameters (FB22 main inputs)**:  
  - `SFR_inputs` → input star formation rates (in M⊙/yr)  
  - `etaE` → fixed thermalization efficiency factor value (used when `vary_which_eta = 'etaM'`)  
  - `etaM` → fixed initial hot-phase mass loading factor (used when `vary_which_eta = 'etaE'`)  
  - `vary_which_eta` → choose which parameter to vary in grid runs: `'etaM'` or `'etaE'`  
  - `etaM_grid` → grid of initial hot-phase mass loading factor values (used when `vary_which_eta = 'etaM'`)  
  - `etaE_grid` → grid of thermalization efficiency factor values (used when `vary_which_eta = 'etaE'`)  
  - `etaM_cold` → initial cold-phase mass loading factor  
  - `log_Mcl` → initial cold cloud mass (log10 scale, in M⊙)  
  - `v_cloud_init` → initial cloud velocity (cm/s)  
  - `r_sonic` → sonic radius (cm)  
  - `Z_wind_init` → initial wind metallicity (in units of solar metallicity)  
  - `Z_cloud_init` → initial cloud metallicity (in units of solar metallicity)  
  - `v_circ0` → circular velocity of the external isothermal gravitational potential (cm/s)  
  - `half_opening_angle` → half-opening angle of the spherical symmetric wind (radians)  
  - `Omwind` → solid angle of the wind, derived from `half_opening_angle`  

- **Cooling/ionization parameters**:  
  - `redshift_ps20` → redshift for the Ploeckinger & Schaye (2020) cooling tables  
  - `eq_or_noneq` → whether to use equilibrium (`'eq'`) or non-equilibrium (`'noneq'`) CHIMES solutions  
  - `noneq_time` → if `eq_or_noneq = 'noneq'`, select the time snapshot of the non-equilibrium solution (in Myr)  

- **Emission-line SB parameters**:  
  - `which_lines` → list of emission lines to compute surface brightness for (PS20 format, e.g., `b'O  6      1031.91A'`)  
  - `R_eval_arr` → radii (in units of `r_sonic`) at which to evaluate SB profiles  
  - `z_galaxy` → redshift of the galaxy for simulating the SB profile 

---

## Outputs

- **FB22 (`../fb22_model_outputs`)**:  
  - Cloud and wind profiles (stored as `.npy`): velocity, density, temperature, metallicity, cloud mass.  
  - Extra diagnostics (stored as `.npy`): Mach numbers, pressure, overdensity, etc.  
  - Diagnostic plots (stored as `.pdf`) saved to `../fb22_model_outputs/fb22_solution_plots`.  

- **CP25 (`../cp25_model_outputs`)**:  
  - Radial SB profiles for specified emission lines.  
  - Arrays stored as `.npy` in structured subfolders.  

---

## Post-Processing Utilities (Optional)

Beyond generating wind solutions and emission-line SB profiles, this repository (`../cp25/post_process`) also provides helper routines for analyzing the outputs of FB22 and CP25 models. 

- **`pressure_Mrel_fb22`**  
  Computes the radial profiles of ram pressure, thermal pressure, and relative Mach number from FB22 solutions:  
  - `P_r` → thermal pressure of the wind/cloud (in pressure equilibrium)  
  - `P_ram_r` → ram pressure of the wind on clouds (ρ_w (v_w − v_cl)²)  
  - `M_rel_r` → relative Mach number ((v_w − v_cl)/c_s,wind)  

- **`extract_ovi_profile`**  
  Loads the **O VI radial SB profile** from CP25 outputs, combining the λ1031.91 and λ1037.62 Å doublet into a total profile.  
  Returns:  
  - `ovi_profile_rkpc` → radial bins in kpc  
  - `ovi_tot_profile` → summed O VI SB (erg/s/cm²/arcsec²)  

- **`find_rkpc_obs_limit`**  
  Determines the **largest radius** at which the O VI SB profile falls below an observational SB limit.  
  - `ram_or_therm` → choose whether to scale SB using ram/thermal pressure (`'ram'` vs `'therm'`).  
  - `sb_limit` → SB detection threshold (default: 1e−18 erg/s/cm²/arcsec²).  
  Returns:  
  - `rkpc_intersect` → radius (kpc) where the profile crosses the threshold  
  - `intersect_idx` → index of the corresponding radial bin(s)  

---

## Citation

If you use this code, please cite:  
- Chen & Peng et al. (2025, in prep.)  

---

## Contact

Developed and maintained by [Zixuan Peng](mailto:zixuanpeng@ucsb.edu) and [Zirui Chen](mailto:ziruichen@ucsb.edu). For bug reports, questions, or feature requests, please contact us via email.





