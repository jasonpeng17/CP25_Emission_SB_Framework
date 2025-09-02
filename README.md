# FB22–CP25 Emission-Line Surface Brightness Modeling Suite

## Introduction

This repository provides a **Python-based pipeline** for computing **emission-line surface brightness (SB) profiles** by combining the multiphase galactic wind model developed by [Fielding & Bryan (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...924...82F/abstract) with the mixing layer model developed by [Chen et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...950...91C/abstract).  

The suite is composed of two complementary modules:  

* **FB22 Model Grid (`fb22_model_grid.py`)**: Implements and extends the multiphase wind model of [Fielding & Bryan (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...924...82F/abstract). It produces the radial structure of hot winds and cold clouds, including velocity, density, pressure, metallicity, and cloud growth/loss.  
* **CP25 SB Model Grid (`cp25_sb_model_grid.py`)**: Builds on FB22 outputs and computes **emission-line SB profiles** using equilibrium and/or non-equilibrium ionization fractions from **CHIMES** and line emissivities from **PS20 cooling/emissivity tables**.  

**Key Features**:  
- Self-consistent treatment of **hot wind phase, mixing layers, and cold clouds**.  
- Incorporates **non-equilibrium ionization** from CHIMES.  
- Computes SB profiles for emission lines (e.g. H I Lyα, [O III] 5007, O VI 1032).  
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
   - **PS20 cooling/emissivity tables**  
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
  `master_folder_name`, `folder_name_base`, `num_decimal`  

- **Wind parameters**:  
  `SFR_inputs`, `etaE`, `etaM`, `etaM_grid`, `etaE_grid`, `etaM_cold`, `log_Mcl`, `v_cloud_init`, `r_sonic`, `Z_wind_init`, `Z_cloud_init`, `v_circ0`, `half_opening_angle`  

- **Cooling/ionization**:  
  `redshift_ps20`, `eq_or_noneq`, `noneq_time`  

- **Emission-line SB**:  
  `which_lines`, `R_eval_arr`, `z_galaxy`  

Example (from default setup):
```python
SFR_inputs = np.arange(5, 101, 5) * Msun / yr
etaE = 1.0
etaM = 0.2
vary_which_eta = 'etaM'
etaM_grid = np.arange(0.1, 1.01, 0.05)
which_lines = np.array([b'O  6      1031.91A', b'O  6      1037.62A'])
R_eval_arr = np.arange(1.01, 100, 0.1)
z_galaxy = 0.2
```

---

## Outputs

- **FB22 (`../fb22_model_outputs`)**:  
  - Cloud and wind profiles: velocity, density, temperature, metallicity, cloud mass.  
  - Extra diagnostics: Mach numbers, pressure, overdensity, etc.  
  - Diagnostic plots saved to `../fb22_model_outputs/fb22_solution_plots`.  

- **CP25 (`../cp25_model_outputs`)**:  
  - Radial SB profiles for specified emission lines.  
  - Arrays stored as `.npy` in structured subfolders.  

---

## Citation

If you use this code, please cite:  
- Chen & Peng et al. (2025, in prep.)  

---

## Contact

Developed and maintained by [Zixuan Peng](mailto:zixuanpeng@ucsb.edu) and [Zirui Chen](mailto:ziruichen@ucsb.edu). For bug reports, questions, or feature requests, please contact us via email.





