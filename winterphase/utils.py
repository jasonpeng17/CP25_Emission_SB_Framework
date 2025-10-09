################################################################
# Define some useful utils/helper functions for calculations
################################################################

import numpy as np
import re
import h5py 
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from pathlib import Path
import os, sys
import glob
from pathlib import Path
from collections import defaultdict

from IPython import embed
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from constants import *
from input_params import *


"""
CHIMES species dictionary. This dictionary maps the species names to their position in the full CHIMES abundance array. 
"""
chimes_dict = {"elec": 0,
               "HI": 1,
               "HII": 2,
               "Hm": 3,
               "HeI": 4,
               "HeII": 5,
               "HeIII": 6,
               "CI": 7,
               "CII": 8,
               "CIII": 9,
               "CIV": 10,
               "CV": 11,
               "CVI": 12,
               "CVII": 13,
               "Cm": 14,
               "NI": 15,
               "NII": 16,
               "NIII": 17,
               "NIV": 18,
               "NV": 19,
               "NVI": 20,
               "NVII": 21,
               "NVIII": 22,
               "OI": 23,
               "OII": 24,
               "OIII": 25,
               "OIV": 26,
               "OV": 27,
               "OVI": 28,
               "OVII": 29,
               "OVIII": 30,
               "OIX": 31,
               "Om": 32,
               "NeI": 33,
               "NeII": 34,
               "NeIII": 35,
               "NeIV": 36,
               "NeV": 37,
               "NeVI": 38,
               "NeVII": 39,
               "NeVIII": 40,
               "NeIX": 41,
               "NeX": 42,
               "NeXI": 43,
               "MgI": 44,
               "MgII": 45,
               "MgIII": 46,
               "MgIV": 47,
               "MgV": 48,
               "MgVI": 49,
               "MgVII": 50,
               "MgVIII": 51,
               "MgIX": 52,
               "MgX": 53,
               "MgXI": 54,
               "MgXII": 55,
               "MgXIII": 56,
               "SiI": 57,
               "SiII": 58,
               "SiIII": 59,
               "SiIV": 60,
               "SiV": 61,
               "SiVI": 62,
               "SiVII": 63,
               "SiVIII": 64,
               "SiIX": 65,
               "SiX": 66,
               "SiXI": 67,
               "SiXII": 68,
               "SiXIII": 69,
               "SiXIV": 70,
               "SiXV": 71,
               "SI": 72,
               "SII": 73,
               "SIII": 74,
               "SIV": 75,
               "SV": 76,
               "SVI": 77,
               "SVII": 78,
               "SVIII": 79,
               "SIX": 80,
               "SX": 81,
               "SXI": 82,
               "SXII": 83,
               "SXIII": 84,
               "SXIV": 85,
               "SXV": 86,
               "SXVI": 87,
               "SXVII": 88,
               "CaI": 89,
               "CaII": 90,
               "CaIII": 91,
               "CaIV": 92,
               "CaV": 93,
               "CaVI": 94,
               "CaVII": 95,
               "CaVIII": 96,
               "CaIX": 97,
               "CaX": 98,
               "CaXI": 99,
               "CaXII": 100,
               "CaXIII": 101,
               "CaXIV": 102,
               "CaXV": 103,
               "CaXVI": 104,
               "CaXVII": 105,
               "CaXVIII": 106,
               "CaXIX": 107,
               "CaXX": 108,
               "CaXXI": 109,
               "FeI": 110,
               "FeII": 111,
               "FeIII": 112,
               "FeIV": 113,
               "FeV": 114,
               "FeVI": 115,
               "FeVII": 116,
               "FeVIII": 117,
               "FeIX": 118,
               "FeX": 119,
               "FeXI": 120,
               "FeXII": 121,
               "FeXIII": 122,
               "FeXIV": 123,
               "FeXV": 124,
               "FeXVI": 125,
               "FeXVII": 126,
               "FeXVIII": 127,
               "FeXIX": 128,
               "FeXX": 129,
               "FeXXI": 130,
               "FeXXII": 131,
               "FeXXIII": 132,
               "FeXXIV": 133,
               "FeXXV": 134,
               "FeXXVI": 135,
               "FeXXVII": 136,
               "H2": 137,
               "H2p": 138,
               "H3p": 139,
               "OH": 140,
               "H2O": 141,
               "C2": 142,
               "O2": 143,
               "HCOp": 144,
               "CH": 145,
               "CH2": 146,
               "CH3p": 147,
               "CO": 148,
               "CHp": 149,
               "CH2p": 150,
               "OHp": 151,
               "H2Op": 152,
               "H3Op": 153,
               "COp": 154,
               "HOCp": 155,
               "O2p": 156}


ROMANS = ["","I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX","XXI","XXII","XXIII","XXIV","XXV","XXVI","XXVII"]
def int_to_roman(n: int) -> str:
    if n < 0 or n >= len(ROMANS):
        raise ValueError(f"Cannot convert {n} to Roman numeral within supported range.")
    return ROMANS[n]

def parse_emission_id(line_id: bytes):
    """
    Parse a CHIMES/Cloudy-style line id like b'O  6      1031.91A' into
    (element_symbol, ion_stage_int, species_key), e.g. ('O', 6, 'OVI').
    """
    s = line_id.decode("ascii")
    # Typical formats: 'O  6      1031.91A', 'C  3      977.020A', 'H  1 1215.67A', 'Si 3 1206.50A'
    m = re.match(r"\s*([A-Za-z]{1,2})\s+(\d+)\b", s)
    if not m:
        raise ValueError(f"Cannot parse line id: {s}")
    elem = m.group(1)  # 'O', 'C', 'Si', 'H', ...
    ion_n = int(m.group(2))  # 1 -> I (neutral), 2 -> II, ...
    species_key = f"{elem}{int_to_roman(ion_n)}"  # e.g. 'OVI', 'CIII', 'OI'
    # Special case: Sulfur is labeled 'S' in lines but 'SI'.. 'SXVII' in chimes_dict.
    # Our construction 'SI', 'SII', ... already matches this, so nothing extra needed.
    return elem, ion_n, species_key

def element_total_keys(chimes_dict, elem: str):
    """
    Return the list of species keys in chimes_dict that correspond to *this element's*
    ionic stages (positive stages only, i.e., roman numerals). Excludes negative ions like 'Om', 'Cm', 'Hm'.
    """
    keys = []
    roman_pat = re.compile(r"^(%s)([IVX]+)$" % re.escape(elem))
    # Special: for Sulfur, keys indeed are 'SI', 'SII', ... (same pattern).
    for k in chimes_dict.keys():
        if roman_pat.match(k):
            keys.append(k)
    if not keys:
        raise ValueError(f"No ionic species keys found for element '{elem}' in chimes_dict.")
    # Sort by ion stage (I, II, III, ...), so totals are reproducible (order doesn't matter for sum)
    roman_to_int = {ROMANS[i]: i for i in range(len(ROMANS))}
    keys.sort(key=lambda kk: roman_to_int[kk[len(elem):]])
    return keys

def species_index(chimes_dict, key: str) -> int:
    """
    Return the index for each ion based on the chimes_dict.
    """
    if key not in chimes_dict:
        raise KeyError(f"Species '{key}' not found in chimes_dict.")
    return chimes_dict[key]

def build_ion_fraction_interpolators(line_id: bytes,
                                     chimes_dict: dict,
                                     neq_times_seconds=(1 * Myr, 10 * Myr)):
    """
    Given a line_id (bytes), return a dict with:
      - 'species_key': e.g. 'OVI'
      - 'eq': RegularGridInterpolator for equilibrium ion fraction
      - 'neq': dict mapping chosen time (seconds) -> RegularGridInterpolator for ion fraction
    For equilibrium, CHIMES stores log10 abundances.
    For non-equilibrium, CHIMES stores linear abundances over time.
    """
    elem, ion_n, sp_key = parse_emission_id(line_id)

    # Which species index for the *ion of interest*?
    sp_idx = species_index(chimes_dict, sp_key)

    # Build element-total indices (sum over all positive ionic stages for that element)
    elem_keys = element_total_keys(chimes_dict, elem)
    elem_indices = [species_index(chimes_dict, k) for k in elem_keys]

    # Equilibrium ion fraction
    # Convert species abundances to linear before summing/ratio
    # abundance_eq shape: (len(T_eq), len(n_eq), len(Z_eq), n_species) in log10
    ion_lin_eq = 10.0**(abundance_eq[:, :, :, sp_idx])
    elem_tot_lin_eq = np.zeros_like(ion_lin_eq)
    for idx in elem_indices:
        elem_tot_lin_eq += 10.0**(abundance_eq[:, :, :, idx])

    # Avoid division by zero; use a tiny floor 
    with np.errstate(divide='ignore', invalid='ignore'):
        ionfrac_eq = np.where(elem_tot_lin_eq > 0.0, ion_lin_eq / elem_tot_lin_eq, 0.0)

    eq_interp = interpolate.RegularGridInterpolator((T_eq, n_eq, Z_eq), ionfrac_eq, bounds_error=False, fill_value=-1e-30)

    # Non-equilibrium ion fraction(s)
    # abundance_neq shape: (len(T), len(n), len(Z), n_species, n_times) in *linear* units
    # For requested times, pick nearest time index and build an interpolator for each.
    neq_interps = {}
    for t_req in neq_times_seconds:
        it = int(np.argmin(np.abs(time_array_neq - t_req)))
        ion_lin_neq = abundance_neq[:, :, :, sp_idx, it]
        elem_tot_lin_neq = np.zeros_like(ion_lin_neq)
        for idx in elem_indices:
            elem_tot_lin_neq += abundance_neq[:, :, :, idx, it]
        with np.errstate(divide='ignore', invalid='ignore'):
            ionfrac_neq = np.where(elem_tot_lin_neq > 0.0, ion_lin_neq / elem_tot_lin_neq, 0.0)

        neq_interps[float(time_array_neq[it])] = interpolate.RegularGridInterpolator((T_neq, n_neq, Z_neq), ionfrac_neq, bounds_error=False, fill_value=-1e-30)

    return {
        "species_key": sp_key,
        "eq": eq_interp,
        "neq": neq_interps,  # keys are the *actual* nearest times in seconds
        "grids": {
            "eq_axes": (T_eq, n_eq, Z_eq),
            "neq_axes": (T_neq, n_neq, Z_neq),
            "neq_times": time_array_neq
        }
    }

def make_tag(etaE, etaM, etaM_cold, log_Mcl, v_c_km_per_s, Zwind_over_Zsol, num_decimal):
    return (
        f"etaE={etaE:.{num_decimal}f}_"
        f"etaM={etaM:.{num_decimal}f}_"
        f"etaMcl={etaM_cold:.{num_decimal}f}_"
        f"logMcl={log_Mcl:.{num_decimal}f}_"
        f"vc={v_c_km_per_s:.{num_decimal}f}_"
        f"Zwind={Zwind_over_Zsol:.{num_decimal}f}"
    )

def save_items(base_root, master_folder_name, folder_name, tag, items):
    """
    items: iterable of (subdir, filename_prefix, array)
           e.g. ("u_arr", "fb22_u_arr", v_star)
    """
    for subdir, prefix, arr in items:
        outdir = Path(base_root) / subdir / master_folder_name / folder_name
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / f"{prefix}_{tag}.npy", arr)

def v0_func(M_dot, E_dot):
    """
    The normalization factor for velocity based on Chevalier & Chegg (1985).
    """
    # v0 = M_dot**(-1/2) * E_dot**(1/2)
    v0 = (M_dot**(-1/2)) * (E_dot**(1/2))
    return v0

def n0_func(M_dot, E_dot, R_SF, mu, mp):
    """
    The normalization factor for density based on Chevalier & Chegg (1985).
    """
    rho_0 = (M_dot**(3/2)) * (E_dot**(-1/2)) * (R_SF**(-2))
    n0 = rho_0 / (mu * mp)
    return n0

def P0_func(M_dot, E_dot, R_SF):
    """
    The normalization factor for pressure based on Chevalier & Chegg (1985).
    """
    P0 = (M_dot**(1/2)) * (E_dot**(1/2)) * (R_SF**(-2))
    return P0

def T0_func(M_dot, E_dot, mu, mp, kB):
    """
    The normalization factor for temperature based on Chevalier & Chegg (1985).
    """
    T0 = (E_dot / M_dot) * (mp / kB) * mu
    return T0

# new cooling curve related functions based on the Ploeckinger & Schaye 2020 cooling tables
def tcool_P(T, P, metallicity):
    """
    cooling time function
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    """
    T = np.where(T>10**9.5, 10**9.5, T)
    T = np.where(T<10**1, 10**1, T)
    nH_actual = P/T*(mu/muH)
    nH = np.where(nH_actual>1, 1, nH_actual)
    nH = np.where(nH<10**-8, 10**-8, nH)
    # calculate the total cooling and heating rate
    Lambda_z0_c = 10**log_cooling_curve_func2d((np.log10(T), np.log10(metallicity), np.log10(nH)))
    Lambda_z0_h = 10**log_heating_curve_func2d((np.log10(T), np.log10(metallicity), np.log10(nH)))
    # calculate the net cooling/heating rate
    Lambda_z0 = Lambda_z0_c - Lambda_z0_h
    return (1./(gamma-1.)) * (muH/mu) * kB * T / ( nH_actual * Lambda_z0)

def Lambda_T_P(T, P, metallicity):
    """
    cooling curve function as a function of
    T in units of K
    P in units of K * cm**-3
    metallicity in units of solar metallicity
    above nH = 1e6 * cm**-3 there is no more density dependence 
    """
    nH = P/T*(mu/muH)
    if nH > 1e6:
        nH = 1e6
    elif nH < 1e-8:
        nH = 1e-8
    # calculate the total cooling and heating rate
    Lambda_z0_c = 10**log_cooling_curve_func2d((np.log10(T), np.log10(metallicity), np.log10(nH)))
    Lambda_z0_h = 10**log_heating_curve_func2d((np.log10(T), np.log10(metallicity), np.log10(nH)))
    # calculate the net cooling/heating rate
    Lambda_z0 = Lambda_z0_c - Lambda_z0_h
    return Lambda_z0
Lambda_T_P = np.vectorize(Lambda_T_P)

def Lambda_P_rho(P, rho, metallicity):
    """
    cooling curve function as a function of
    P in units of erg * cm**-3
    rho in units of g * cm**-3
    metallicity in units of solar metallicity
    above nH = 1e6 * cm**-3 there is no more density dependence 
    """
    nH = rho / (muH * mp)
    T  = P/kB / (rho/(mu*mp))
    if nH > 1e6:
        nH = 1e6
    elif nH < 1e-8:
        nH = 1e-8
    # calculate the total cooling and heating rate
    Lambda_z0_c = 10**log_cooling_curve_func2d((np.log10(T), np.log10(metallicity), np.log10(nH)))
    Lambda_z0_h = 10**log_heating_curve_func2d((np.log10(T), np.log10(metallicity), np.log10(nH)))
    # calculate the net cooling/heating rate
    Lambda_z0 = Lambda_z0_c - Lambda_z0_h
    return Lambda_z0
Lambda_P_rho = np.vectorize(Lambda_P_rho)


def _load_matching_array(base_dir, master_setup_name, which_setup, tag):
    """Load the first .npy that contains the tag; return None if not found."""
    target_dir = os.path.join(base_dir, master_setup_name, which_setup)
    if not os.path.isdir(target_dir):
        return None
    # Find the first file containing the tag
    matches = sorted(glob.glob(os.path.join(target_dir, f"*{tag}*.npy")))
    if not matches:
        # fall back to scanning non-.npy too (in case arrays saved without suffix)
        matches = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if tag in f]
        if not matches:
            return None
    try:
        return np.load(matches[0])
    except Exception:
        return None

def phys_star_interp_fb22(etaE, etaM, etaM_cold, log_Mcl, v_c, Zwind, which, master_setup_name,
                          which_setup, wind_or_cloud='cloud', eta_num_decimal=2):
    """
    Load x, y arrays for the requested solution 'which' and return (x, y, interp1d).
    which: one of {'rho', 'u', 'T', 'P', 'Mcl', 'Z'}
    wind_or_cloud: 'cloud' or 'wind' (note: 'Mcl' available only for 'cloud')
    """
    # Map which -> subdirectory names for (cloud, wind). Identical subdir for both when in pressure eq (P_arr).
    SUBDIR_MAP = {
        'rho': {'cloud': 'rho_arr', 'wind': 'rho_w_arr'},
        'u'  : {'cloud': 'u_arr',   'wind': 'u_w_arr'},
        'T'  : {'cloud': 'T_arr',   'wind': 'T_w_arr'},
        'P'  : {'cloud': 'P_arr',   'wind': 'P_arr'},
        'Mcl': {'cloud': 'M_cl_arr'},
        'Z'  : {'cloud': 'Z_arr',   'wind': 'Z_w_arr'},
    }

    if which not in SUBDIR_MAP:
        raise ValueError(f"'which' must be one of {list(SUBDIR_MAP.keys())}, got {which!r}")

    if which == 'Mcl' and wind_or_cloud != 'cloud':
        raise ValueError("Cloud mass ('Mcl') is only defined for wind_or_cloud='cloud'.")

    # Build filename tag once
    tag = (
        f"etaE={etaE:.{eta_num_decimal}f}_"
        f"etaM={etaM:.{eta_num_decimal}f}_"
        f"etaMcl={etaM_cold:.{eta_num_decimal}f}_"
        f"logMcl={log_Mcl:.{eta_num_decimal}f}_"
        f"vc={v_c:.{eta_num_decimal}f}_"
        f"Zwind={Zwind:.{eta_num_decimal}f}"
    )

    # Base outputs root
    cwd = os.getcwd()
    base_root = os.path.join(cwd, "..", "fb22_model_outputs")

    # Always load x* from x_arr
    x_dir = os.path.join(base_root, "x_arr")
    x_star_arr = _load_matching_array(x_dir, master_setup_name, which_setup, tag)

    # Select subdir for the requested 'which'
    subdir_name = SUBDIR_MAP[which]['cloud'] if wind_or_cloud == 'cloud' else SUBDIR_MAP[which]['wind']
    y_dir = os.path.join(base_root, subdir_name)
    y_arr = _load_matching_array(y_dir, master_setup_name, which_setup, tag) 

    # If either array is missing, return gracefully with None interpolator (keeps your original behavior)
    if x_star_arr is None or y_arr is None:
        return x_star_arr, y_arr, None

    # Build safe 1D interpolator
    try:
        y_x_func1d = interp1d(x_star_arr, y_arr, kind='linear', bounds_error=False, fill_value=-1e-30)
    except Exception:
        y_x_func1d = None

    return x_star_arr, y_arr, y_x_func1d

def available_chimes_redshifts(root=PARENT / "chimes_grids", require_both=True):
    """
    Scan the CHIMES grids folder and return the list of available redshifts.

    Parameters
    ----------
    root : path-like
        Directory that contains the CHIMES HDF5 files.
    require_both : bool
        If True, only return redshifts for which BOTH the equilibrium file
        (eqm_table_colibre_z=*.hdf5) and the non-equilibrium file
        (grid_noneq_evolution_colibre_1Myr_noTherm_z=*.hdf5) are present.
        If False, return the union (any of the two file types).

    Returns
    -------
    list[float]
        Sorted list of redshifts found (as floats). Example: [0.2, 2.0, 6.0]
    """
    root = Path(root)
    if not root.exists():
        return []

    eq_pat  = re.compile(r"^eqm_table_colibre_z=(\d+(?:\.\d+)?)\.hdf5$")
    neq_pat = re.compile(r"^grid_noneq_evolution_colibre_1Myr_noTherm_z=(\d+(?:\.\d+)?)\.hdf5$")

    eq_z, neq_z = set(), set()
    for name in os.listdir(root):
        m = eq_pat.match(name)
        if m:
            eq_z.add(float(m.group(1)))
            continue
        m = neq_pat.match(name)
        if m:
            neq_z.add(float(m.group(1)))

    zs = (eq_z & neq_z) if require_both else (eq_z | neq_z)
    return sorted(zs)


def validate_chimes_redshift(redshift_chimes, root=PARENT / "chimes_grids", require_both=True):
    """
    Ensure `redshift_chimes` matches an available grid; raise ValueError otherwise.

    Parameters
    ----------
    redshift_chimes : float
        Desired CHIMES redshift parameter.
    root : path-like
        Directory with CHIMES HDF5 files.
    require_both : bool
        If True, require both eq and non-eq files to exist for the redshift.

    Returns
    -------
    float
        The validated redshift rounded to one decimal (to match filename format).

    Raises
    ------
    ValueError
        If no grids are found or the requested redshift is not available.
    """
    # Filenames are formatted with one decimal (e.g., z=0.2, 2.0, 6.0)
    z_req = float(f"{redshift_chimes:.1f}")
    zs = available_chimes_redshifts(root=root, require_both=require_both)

    if not zs:
        raise ValueError(f"No CHIMES grids found in {Path(root).resolve()}.")

    if z_req not in zs:
        nearest = min(zs, key=lambda z: abs(z - z_req))
        avail_str = ", ".join(f"{z:.1f}" for z in zs)
        raise ValueError(
            f"CHIMES grids not found for z={z_req:.1f}. "
            f"Available redshifts: {avail_str}. "
            f"Closest available is z={nearest:.1f}."
        )
    return z_req


def get_chimes_grid_paths(redshift_chimes, root=PARENT / "chimes_grids"):
    """
    Return Path objects for the eq and non-eq CHIMES HDF5 files at a given redshift.

    Parameters
    ----------
    redshift_chimes : float
        Desired redshift (will be rounded to one decimal to match filenames).
    root : path-like
        CHIMES grids directory.

    Returns
    -------
    (Path, Path)
        (eq_path, neq_path) for the selected redshift.
    """
    z = validate_chimes_redshift(redshift_chimes, root=root, require_both=True)
    root = Path(root)
    eq  = root / f"eqm_table_colibre_z={z:.1f}.hdf5"
    neq = root / f"grid_noneq_evolution_colibre_1Myr_noTherm_z={z:.1f}.hdf5"
    return eq, neq

"""
Flux fraction of line emissivities within the TRML (Chen et al. 2023; Peng et al. 2025).
This version loads *all* available T_hot grids in `chen23_grids/` and builds
interpolators for each one.

Creates:
    - `flux_frac_line_func2d_by_Thot`: dict[float T_hot] -> dict[line_id] -> RectBivariateSpline
    - `available_Thot_chen23`: sorted list of T_hot values present on disk
"""
grids_dir = PARENT / "chen23_grids"
_fname_re = re.compile(r"^flux_fractions_T_hot=([0-9.eE+\-]+)_tau=([0-9.]+)_rho_vx_cosine_grid\.npz$")
line_to_Thot_to_logfrac = defaultdict(dict)
available_Thot_chen23 = []
Ps_master = None
Mrels_master = None

for fpath in sorted(grids_dir.glob("flux_fractions_T_hot=*_*_rho_vx_cosine_grid.npz")):
    m = _fname_re.match(fpath.name)
    if not m:
        continue
    data = np.load(str(fpath), allow_pickle=True)
    flux_dict = data["flux"].item() 
    Ps = data["Ps"]
    Mrels = data["Mrels"]
    T_hot = float(data["Thot"])

    if Ps_master is None:
        Ps_master = np.asarray(Ps)
        Mrels_master = np.asarray(Mrels)
    else:
        if Ps_master.shape != np.asarray(Ps).shape or not np.allclose(Ps_master, Ps):
            raise ValueError(f"Inconsistent Ps grid in {fpath.name}")
        if Mrels_master.shape != np.asarray(Mrels).shape or not np.allclose(Mrels_master, Mrels):
            raise ValueError(f"Inconsistent Mrels grid in {fpath.name}")

    for line_key, frac_arr in flux_dict.items():
        log_frac = np.log10(frac_arr)
        line_to_Thot_to_logfrac[eval(line_key)][T_hot] = np.asarray(log_frac)
    available_Thot_chen23.append(T_hot)

available_Thot_chen23 = sorted(set(available_Thot_chen23))
logT_master = np.log10(np.array(available_Thot_chen23, dtype=float))
logP_master = np.log10(Ps_master)
Mrels_master = np.asarray(Mrels_master)

# Build 3D cubes and interpolators per line
flux_frac_interp3d = {}   # line_id -> RegularGridInterpolator over (logP, Mrel, logT)
for line_id, Thot_map in line_to_Thot_to_logfrac.items():
    # Ensure we have all T_hot slices; if not, clip by using nearest along T (build by nearest fill)
    cube = np.empty((logP_master.size, Mrels_master.size, logT_master.size), dtype=float)

    for k, T_hot in enumerate(available_Thot_chen23):
        try:
            slab = Thot_map[T_hot]
        except KeyError:
            continue
        cube[:, :, k] = slab

    # Build linear interpolator on a regular grid — returns log10(frac)
    flux_frac_interp3d[line_id] = interpolate.RegularGridInterpolator(
        (logP_master, Mrels_master, logT_master),
        cube,
        method="linear",
        bounds_error=False,   # we will clip manually to nearest edge
        fill_value=None       # (unused because we clamp inputs)
    )

# Helpers to evaluate with nearest-edge behavior outside the grid
def _clip_to_grid(logP, Mrel, logT):
    lp = np.clip(np.asarray(logP), logP_master.min(), logP_master.max())
    mr = np.clip(np.asarray(Mrel), Mrels_master.min(), Mrels_master.max())
    lt = np.clip(np.asarray(logT), logT_master.min(), logT_master.max())
    # Broadcast to common shape
    lp, mr, lt = np.broadcast_arrays(lp, mr, lt)
    pts = np.stack([lp, mr, lt], axis=-1)  
    return pts

def eval_flux_frac_log(line_id, logP, Mrel, logT_hot):
    """
    Evaluate log10(flux fraction) at (logP, Mrel,logT_hot).
    Inputs outside the tabulated ranges are clipped to nearest grid values.
    """
    if line_id not in flux_frac_interp3d:
        raise KeyError(f"Line key not found in TRML flux fractions: {line_id!r}")
    interp = flux_frac_interp3d[line_id]
    pts = _clip_to_grid(logP, Mrel, logT_hot)
    return interp(pts)

"""
Cooling curve as a function of density, temperature, metallicity (Ploeckinger & Schaye 2020) (assume a certain redshift)
"""
# define the cooling curve based on Ploeckinger and Schaye 2020
table_dust1_CR1_G1_shield1 = PARENT / "ps20_grids/UVB_dust1_CR1_G1_shield1.hdf5"

# total cooling & heating
with h5py.File(table_dust1_CR1_G1_shield1, 'r') as f:
    # List all groups
    table_bins_dust1_CR1_G1_shield1 = f['TableBins']
    # five dependencies of cooling & heating rates
    density_bin_dust1_CR1_G1_shield1 = table_bins_dust1_CR1_G1_shield1['DensityBins'][:]
    temperature_bin_dust1_CR1_G1_shield1 = table_bins_dust1_CR1_G1_shield1['TemperatureBins'][:]
    metallicity_bin_dust1_CR1_G1_shield1 = table_bins_dust1_CR1_G1_shield1['MetallicityBins'][:]
    redshift_bin_dust1_CR1_G1_shield1 = table_bins_dust1_CR1_G1_shield1['RedshiftBins'][:]
    u_bin_dust1_CR1_G1_shield1 = table_bins_dust1_CR1_G1_shield1['InternalEnergyBins'][:]
    # cooling & heating rates
    cooling_rate_dust1_CR1_G1_shield1 = f['Tdep']['Cooling'][:]
    heating_rate_dust1_CR1_G1_shield1 = f['Tdep']['Heating'][:]

# total cooling & heating
# redshift_ps20 = 0.2 # define the redshift of the ps tables 

cooling_curve_total_metals = cooling_rate_dust1_CR1_G1_shield1[np.where(redshift_bin_dust1_CR1_G1_shield1 == redshift_ps20)[0][0], :, 
                                                               :, :, -1] # cooling rate of total metals
cooling_curve_total_prim = cooling_rate_dust1_CR1_G1_shield1[np.where(redshift_bin_dust1_CR1_G1_shield1 == redshift_ps20)[0][0], :,
                                                             :, :, -2] # cooling rate except metals
heating_curve_total_metals = heating_rate_dust1_CR1_G1_shield1[np.where(redshift_bin_dust1_CR1_G1_shield1 == redshift_ps20)[0][0], :,
                                                               :, :, -1] # heating rate of total metals
heating_curve_total_prim = heating_rate_dust1_CR1_G1_shield1[np.where(redshift_bin_dust1_CR1_G1_shield1 == redshift_ps20)[0][0], :,
                                                             :, :, -2] # heating rate except metals

# total cooling and heating rates --> interpolation functions as 
cooling_tot = 10**(cooling_curve_total_metals) + 10**(cooling_curve_total_prim)
log_cooling_tot = np.log10(cooling_tot)
log_cooling_curve_func2d = interpolate.RegularGridInterpolator((temperature_bin_dust1_CR1_G1_shield1, metallicity_bin_dust1_CR1_G1_shield1, density_bin_dust1_CR1_G1_shield1), 
                                                                log_cooling_tot, bounds_error=False, fill_value=-1e-30)
heating_tot = 10**(heating_curve_total_metals) + 10**(heating_curve_total_prim)
log_heating_tot = np.log10(heating_tot)
log_heating_curve_func2d = interpolate.RegularGridInterpolator((temperature_bin_dust1_CR1_G1_shield1, metallicity_bin_dust1_CR1_G1_shield1, density_bin_dust1_CR1_G1_shield1), 
                                                                log_heating_tot, bounds_error=False, fill_value=-1e-30)


"""
Cooling emissivity as a function of density, temperature, metallicity (Ploeckinger & Schaye 2020) 
"""
# ploeckinger & shaye 2020 cooling table for lines
table_dust1_CR1_G1_shield1_lines = PARENT / "ps20_grids/UVB_dust1_CR1_G1_shield1_lines.hdf5"
# define the IDs for different lines
with h5py.File(table_dust1_CR1_G1_shield1_lines, 'r') as f:
    line_IDs = f['IdentifierLines'][:] # list of emission line ids
    # List all groups
    table_bins_dust1_CR1_G1_shield1_lines = f['TableBins']
    density_bin_dust1_CR1_G1_shield1_lines = table_bins_dust1_CR1_G1_shield1_lines['DensityBins'][:]
    temperature_bin_dust1_CR1_G1_shield1_lines = table_bins_dust1_CR1_G1_shield1_lines['TemperatureBins'][:]
    metallicity_bin_dust1_CR1_G1_shield1_lines = table_bins_dust1_CR1_G1_shield1_lines['MetallicityBins'][:]
    redshift_bin_dust1_CR1_G1_shield1_lines = table_bins_dust1_CR1_G1_shield1_lines['RedshiftBins'][:]
    # emissivities for different lines
    emissivities_vol = f['Tdep']['EmissivitiesVol'][:] # emissivity at the last CLOUDY zone
    emissivities_col = f['Tdep']['EmissivitiesCol'][:] # average emissivity of the shielding column
    
# emission line ids based on PS20 (should be consistent with the available lines in your chen23 flux fraction grids)
emission_line_ids = list(flux_frac_interp3d.keys())
emission_line_indexes = {}
emissivity_grids = {}
for line_id in emission_line_ids:
    index = np.where(np.array(line_IDs == line_id))[0][0]
    emission_line_indexes[line_id] = index
    # for zero redshift z = 0
    emissivity_grid = interpolate.RegularGridInterpolator((temperature_bin_dust1_CR1_G1_shield1_lines, metallicity_bin_dust1_CR1_G1_shield1_lines, density_bin_dust1_CR1_G1_shield1_lines),
                                                           emissivities_vol[np.where(redshift_bin_dust1_CR1_G1_shield1_lines == 0.0)[0][0], :, 
                                                                            :, :, index],
                                                           bounds_error=False, fill_value=-1e-30)
    emissivity_grids[line_id] = emissivity_grid


"""
Equilibrium and non-equilibrium chemistry solutions based on CHIMES (Richings et al. 2014a,b)
"""
# open the chimes equilibrium and non-equilibrium solutions
file_eq, file_neq = get_chimes_grid_paths(redshift_chimes, root=PARENT / "chimes_grids")
# equilibrium
h5file_eq  = h5py.File(file_eq, "r")
T_eq = np.array(h5file_eq["TableBins/Temperatures"]) 
n_eq = np.array(h5file_eq["TableBins/Densities"]) 
Z_eq = np.array(h5file_eq["TableBins/Metallicities"])         
abundance_eq = np.array(h5file_eq["Abundances"]) 

# non-equilibrium
h5file_neq = h5py.File(file_neq, "r")
T_neq = np.array(h5file_neq["TableBins/Temperatures"]) 
n_neq = np.array(h5file_neq["TableBins/Densities"]) 
Z_neq = np.array(h5file_neq["TableBins/Metallicities"])         
time_array_neq = np.array(h5file_neq["TimeArray_seconds"]) 
abundance_neq = np.array(h5file_neq["AbundanceEvolution"]) 











