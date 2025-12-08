import pandas as pd
import ast
import math
import re
import numpy as np

# def identify_class_by_mass(precursor_mz, adduct, tolerance_ppm=15.0):
#     """
#     Identifies potential lipid classes and compositions by matching the precursor m/z.

#     This function calculates the neutral mass from the precursor m/z and adduct, then
#     iterates through possible lipid compositions. If a theoretical mass matches the
#     neutral mass within the specified ppm tolerance, the class, its specific
#     composition, and the ppm error are recorded. Results are sorted by ppm error.

#     Args:
#         precursor_mz (float): The experimentally observed mass-to-charge ratio.
#         adduct (str): The adduct type (e.g., '[M+H]+', '[M-H]-').
#         tolerance_ppm (float): The mass tolerance in parts-per-million (ppm).

#     Returns:
#         tuple: A tuple containing two lists:
#             - A sorted list of unique potential lipid class abbreviations based on ppm error.
#             - A sorted list of unique potential compositions based on ppm error.
#             Returns ([], []) if no match is found.
#     """
    
#     # --- Foundational Constants ---
#     ADDUCT_MASSES = {
#         '[M+H]+': 1.007276,
#         '[M+Na]+': 22.989218,
#         '[M+NH4]+': 18.033823,
#         '[M-H]-': -1.007276,
#         '[M+HCOO]-': 44.998201,
#         '[M+CH3COO]-': 59.013851,
#         '[M+Cl]-': 34.969402,
#         '[M-OH]+': -17.00328866 
#     }

#     # --- Lipid Class Formulas ---
#     LIPID_RULES = [
#         {'class': 'CE',    'constant_mass': 400.3300, 'tails': 1},
#         {'class': 'DG',    'constant_mass': 120.0100, 'tails': 2},
#         {'class': 'FA',    'constant_mass': 31.9898,  'tails': 1},
#         {'class': 'PC',    'constant_mass': 285.0614, 'tails': 2},
#         {'class': 'LPC',   'constant_mass': 271.0800, 'tails': 1},
#         {'class': 'LPC-P',   'constant_mass': 486.3537, 'tails': 1},
#         {'class': 'PE',    'constant_mass': 243.0143, 'tails': 2},
#         {'class': 'PE-O',  'constant_mass': 229.0351, 'tails': 2},
#         {'class': 'LPC-O', 'constant_mass': 257.1028, 'tails': 1},
#         {'class': 'LPE',   'constant_mass': 229.0400, 'tails': 1},
#         {'class': 'LPE-P',  'constant_mass': 447.3213, 'tails': 1},
#         {'class': 'MG',    'constant_mass': 106.0266, 'tails': 1},
#         {'class': 'NAE',   'constant_mass': 75.0317,  'tails': 1},
#         {'class': 'PA',    'constant_mass': 199.9722, 'tails': 2},
#         {'class': 'PG',    'constant_mass': 274.0090, 'tails': 2},
#         {'class': 'PI',    'constant_mass': 362.0250, 'tails': 2},
#         {'class': 'PS',    'constant_mass': 287.0042, 'tails': 2},
#         {'class': 'PS',    'constant_mass': 269.9800, 'tails': 2},
#     ]

#     MASS_PER_CARBON = 14.01565006
#     MASS_PER_2H = 2.01565006

#     # 1. Calculate the neutral mass
#     if adduct not in ADDUCT_MASSES:
#         return [f"Adduct '{adduct}' not recognized."], []
#     neutral_mass = precursor_mz - ADDUCT_MASSES[adduct]

#     # Store results as a list of tuples: (ppm_error, class, composition)
#     found_results = []

#     # 2. Iterate through each lipid class rule
#     for rule in LIPID_RULES:
#         c_range = range(6, 27) if rule['tails'] == 1 else range(10, 51)
#         db_range = range(0, 8) if rule['tails'] == 1 else range(0, 13)

#         # 3. Iterate through possible chain compositions
#         for C in c_range:
#             for DB in db_range:
#                 # 4. Calculate theoretical mass and ppm error
#                 theoretical_mass = rule['constant_mass'] + (MASS_PER_CARBON * C) - (MASS_PER_2H * DB)
                
#                 if neutral_mass == 0:
#                     continue
                
#                 mass_difference = abs(theoretical_mass - neutral_mass)
#                 ppm_error = (mass_difference / neutral_mass) * 1e6

#                 # 5. Check if it matches within tolerance
#                 if ppm_error <= tolerance_ppm:
#                     composition_str = f"{rule['class']} {C}:{DB}"
#                     found_results.append((ppm_error, rule['class'], composition_str))
    
#     if not found_results:
#         return [], []
    
#     # 6. Sort results by ppm_error (the first element of the tuple)
#     found_results.sort(key=lambda x: x[0])
    
#     # Extract unique classes and compositions while preserving order
#     sorted_classes = list(dict.fromkeys([res[1] for res in found_results]))
#     sorted_compositions = list(dict.fromkeys([res[2] for res in found_results]))
    
#     return sorted_classes, sorted_compositions


# def batch_identify_by_mass(df, tolerance_ppm=5.0):
#     """
#     Processes a DataFrame to identify lipid classes and compositions for each row,
#     sorted by mass accuracy (ppm).

#     Args:
#         df (pd.DataFrame): A DataFrame containing 'precursor_mz' and 'adduct' columns.
#         tolerance_ppm (float): The mass tolerance in ppm.

#     Returns:
#         pd.DataFrame: The input DataFrame with two new columns:
#             - 'classes_mz': A list of possible lipid classes, sorted by ppm.
#             - 'possible_name': A list of possible lipid compositions, sorted by ppm.
#     """
#     if not all(col in df.columns for col in ['precursor_mz', 'adduct']):
#         raise ValueError("Input DataFrame must contain 'precursor_mz' and 'adduct' columns.")

#     def process_row(row):
#         return identify_class_by_mass(row['precursor_mz'], row['adduct'], tolerance_ppm)

#     # Apply the function and expand the result into two new, renamed columns
#     df[['classes_mz', 'possible_name']] = df.apply(
#         process_row, axis=1, result_type='expand'
#     )
    
#     return df

import pandas as pd

# --- Constants ---
M_C = 12.00000  # Carbon-12 exact mass
M_H = 1.00783   # Hydrogen-1 exact mass

# Adduct masses
ADDUCT_MASSES = {
    '[M+H]+': 1.007276,
    '[M+Na]+': 22.989218,
    '[M+NH4]+': 18.033823,
    '[M-H]-': -1.007276,
    '[M+HCOO]-': 44.998201,
    '[M+CH3COO]-': 59.013851,
    '[M+Cl]-': 34.969402,
    '[M-OH]+': -17.00328866
}

# Lipid class rules with constant (head) masses
LIPID_RULES = [
    {'class': 'BMP',    'constant_mass': 273.0},
    {'class': 'CAR',    'constant_mass': 175.08},
    {'class': 'CE',     'constant_mass': 399.33},
    {'class': 'CL',     'constant_mass': 454.96},
    {'class': 'DG',     'constant_mass': 119.0},
    {'class': 'DG-O',   'constant_mass': 105.02},
    {'class': 'DG-P',   'constant_mass': 103.0},
    {'class': 'DGCC',   'constant_mass': 278.09},
    {'class': 'DGDG',   'constant_mass': 443.1},
    {'class': 'DGGA',   'constant_mass': 295.03},
    {'class': 'DGTS',   'constant_mass': 262.09},
    {'class': 'FA',     'constant_mass': 30.98},
    {'class': 'LDGCC',  'constant_mass': 264.11},
    {'class': 'LDGTS',  'constant_mass': 248.11},
    {'class': 'LPA',    'constant_mass': 184.99},
    {'class': 'LPC',    'constant_mass': 270.07},
    {'class': 'LPC-O',  'constant_mass': 256.09},
    {'class': 'LPE',    'constant_mass': 228.03},
    {'class': 'LPE-O',  'constant_mass': 214.05},
    {'class': 'LPG',    'constant_mass': 259.02},
    {'class': 'LPI',    'constant_mass': 347.04},
    {'class': 'LPS',    'constant_mass': 272.02},
    {'class': 'MG',     'constant_mass': 105.02},
    {'class': 'MG-O',   'constant_mass': 91.04},
    {'class': 'MG-P',   'constant_mass': 89.02},
    {'class': 'MGDG',   'constant_mass': 281.05},
    {'class': 'NAE',    'constant_mass': 74.02},
    {'class': 'PA',     'constant_mass': 198.96},
    {'class': 'PA-O',   'constant_mass': 184.98},
    {'class': 'PA-P',   'constant_mass': 182.97},
    {'class': 'PC',     'constant_mass': 284.05},
    {'class': 'PC-O',   'constant_mass': 270.07},
    {'class': 'PC-P',   'constant_mass': 268.06},
    {'class': 'PE',     'constant_mass': 242.01},
    {'class': 'PE-O',   'constant_mass': 228.03},
    {'class': 'PE-P',   'constant_mass': 226.01},
    {'class': 'PG',     'constant_mass': 273.0},
    {'class': 'PG-O',   'constant_mass': 259.02},
    {'class': 'PG-P',   'constant_mass': 257.01},
    {'class': 'PI',     'constant_mass': 361.02},
    {'class': 'PI-O',   'constant_mass': 347.04},
    {'class': 'PI-P',   'constant_mass': 345.02},
    {'class': 'PMeOH',  'constant_mass': 212.98},
    {'class': 'PS',     'constant_mass': 286.0},
    {'class': 'PS-O',   'constant_mass': 272.02},
    {'class': 'PS-P',   'constant_mass': 270.0},
    {'class': 'SE',     'constant_mass': 22.92},
    {'class': 'SM-d',   'constant_mass': 227.04},
    {'class': 'SM-t',   'constant_mass': 243.04},
    {'class': 'SQDG',   'constant_mass': 345.01},
    {'class': 'TG',     'constant_mass': 132.98},
    {'class': 'TG-O',   'constant_mass': 119.0},
    {'class': 'WE',     'constant_mass': 30.98}
]


def identify_class_by_mass(precursor_mz, adduct, tolerance_ppm=15.0):
    """
    Identifies potential lipid classes and compositions by matching the precursor m/z
    using exact atomic masses for Carbon and Hydrogen.
    
    Args:
        precursor_mz (float): The measured precursor m/z value.
        adduct (str): The adduct type (e.g., '[M+H]+', '[M-H]-').
        tolerance_ppm (float): Mass tolerance in ppm (default: 15.0).
    
    Returns:
        tuple: (sorted_classes, sorted_compositions) - lists sorted by mass accuracy (ppm).
               sorted_classes: List of lipid class names.
               sorted_compositions: List of lipid compositions (e.g., 'PC 34:1').
    """
    # Validate adduct
    if adduct not in ADDUCT_MASSES:
        raise ValueError(f"Unknown adduct: {adduct}. Available adducts: {list(ADDUCT_MASSES.keys())}")
    
    # Calculate neutral mass from precursor m/z
    adduct_mass = ADDUCT_MASSES[adduct]
    neutral_mass = precursor_mz - adduct_mass
    
    matches = []
    
    for rule in LIPID_RULES:
        lipid_class = rule['class']
        head_mass = rule['constant_mass']
        
        # Calculate expected tail mass
        target_tail_mass = neutral_mass - head_mass
        
        # Skip if tail mass is unreasonable (too small)
        if target_tail_mass < 50:
            continue
        
        # Estimate reasonable carbon range based on target tail mass
        # Approximation: tail_mass ≈ 14.02 * num_c (average CH2 mass)
        min_c = max(2, int(target_tail_mass / 16) - 5)
        max_c = min(100, int(target_tail_mass / 12) + 5)
        
        for num_c in range(min_c, max_c + 1):
            # Max double bonds: limited by chemistry (typically ≤ num_c/2)
            max_db = min(num_c // 2, 15)
            
            for num_db in range(0, max_db + 1):
                # Calculate number of hydrogens using the formula:
                # num_h = (2 * num_c + 1) - (2 * num_db)
                num_h = (2 * num_c + 1) - (2 * num_db)
                
                if num_h < 1:  # Invalid composition
                    continue
                
                # Calculate theoretical tail mass
                theoretical_tail_mass = (num_c * M_C) + (num_h * M_H)
                
                # Calculate theoretical neutral mass
                theoretical_neutral_mass = head_mass + theoretical_tail_mass
                
                # Calculate ppm error
                ppm_error = abs((neutral_mass - theoretical_neutral_mass) / theoretical_neutral_mass * 1e6)
                
                if ppm_error <= tolerance_ppm:
                    composition = f"{lipid_class} {num_c}:{num_db}"
                    matches.append({
                        'class': lipid_class,
                        'composition': composition,
                        'ppm_error': ppm_error,
                        'num_c': num_c,
                        'num_db': num_db,
                        'theoretical_mass': theoretical_neutral_mass + adduct_mass
                    })
    
    # Sort by ppm error (best matches first)
    matches.sort(key=lambda x: x['ppm_error'])
    
    # Extract sorted lists
    sorted_classes = [m['class'] for m in matches]
    sorted_compositions = [m['composition'] for m in matches]
    
    return (sorted_classes, sorted_compositions)


def batch_identify_by_mass(df, tolerance_ppm=5.0):
    """
    Processes a DataFrame to identify lipid classes and compositions for each row,
    sorted by mass accuracy (ppm).
    
    Args:
        df (pd.DataFrame): A DataFrame containing 'precursor_mz' and 'adduct' columns.
        tolerance_ppm (float): The mass tolerance in ppm (default: 15.0).
    
    Returns:
        pd.DataFrame: The input DataFrame with two new columns:
            - 'classes_mz': A list of possible lipid classes, sorted by ppm.
            - 'possible_name': A list of possible lipid compositions, sorted by ppm.
    """
    result_df = df.copy()
    
    classes_list = []
    compositions_list = []
    
    for idx, row in df.iterrows():
        try:
            precursor_mz = row['precursor_mz']
            adduct = row['adduct']
            
            # Skip rows with missing data
            if pd.isna(precursor_mz) or pd.isna(adduct):
                classes_list.append([])
                compositions_list.append([])
                continue
            
            # Identify classes and compositions
            classes, compositions = identify_class_by_mass(
                precursor_mz,
                adduct,
                tolerance_ppm
            )
            classes_list.append(classes)
            compositions_list.append(compositions)
            
        except (ValueError, KeyError) as e:
            # Handle errors gracefully
            classes_list.append([])
            compositions_list.append([])
    
    result_df['classes_mz'] = classes_list
    result_df['possible_name'] = compositions_list
    
    return result_df


# --- MS2 Identification Section ---

# Knowledge base for MS2 fragment identification
MS2_LIPID_RULES = [
    # Rule format: {'class': Lipid Class, 'adduct': Adduct Type, 'type': Scan Type (PIS/NLS), 
    #               'value': m/z or Da, 'comment': Description of the fragment}
    {'class': 'PC', 'adduct': '[M+HC2OO]-', 'type': 'NLS', 'value': 60.0211, 'comment': 'Loss of methyl group + formic acid'},
    {'class': 'PC', 'adduct': '[M+CH3C2OO]-', 'type': 'NLS', 'value': 74.0368, 'comment': 'Loss of methyl group + acetic acid'},
    {'class': 'PC', 'adduct': '[M+HCOO]-', 'type': 'NLS', 'value': 60.0211, 'comment': 'Loss of methyl group + formic acid'},
    {'class': 'PC', 'adduct': '[M+CH3C2OO]-', 'type': 'NLS', 'value': 74.0368, 'comment': 'Loss of methyl group + formic acid'},
    {'class': 'PC', 'adduct': '[M+H]+', 'type': 'PIS', 'value': 184.0733, 'comment': 'Protonated phosphocholine headgroup'},
    {'class': 'LPC', 'adduct': '[M+H]+', 'type': 'PIS', 'value': 184.0733, 'comment': 'Protonated phosphocholine headgroup'}, 
    {'class': 'LPC', 'adduct': '[M+H]+', 'type': 'PIS', 'value': 104.107, 'comment': '[C5H14NO+] fragment, differentiates from PC'},     # This rule is a key differentiator for LPCs vs PCs in positive mode.
    {'class': 'PE', 'adduct': '[M+H]+', 'type': 'NLS', 'value': 141.0194, 'comment': 'Loss of phosphoethanolamine headgroup'},
    {'class': 'LPE', 'adduct': '[M+H]+', 'type': 'NLS', 'value': 141.0194, 'comment': 'Loss of phosphoethanolamine headgroup'},
    {'class': 'PE', 'adduct': '[M-H]-', 'type': 'PIS', 'value': 140.0188, 'comment': 'Deprotonated phosphoethanolamine headgroup'},
    {'class': 'PE', 'adduct': '[M-H]-', 'type': 'PIS', 'value': 196.0380, 'comment': 'C5H11NO5P−'},
    {'class': 'PS', 'adduct': '[M+H]+', 'type': 'NLS', 'value': 185.0089, 'comment': 'Loss of phosphoserine headgroup'},
    {'class': 'PS', 'adduct': '[M-H]-', 'type': 'NLS', 'value': 87.0320, 'comment': 'Loss of serine moiety'},
    {'class': 'PG', 'adduct': '[M+H]+', 'type': 'NLS', 'value': 172.0135, 'comment': 'Loss of glycerol phosphate headgroup'},
    {'class': 'PG', 'adduct': '[M+NH4]+', 'type': 'NLS', 'value': 189.0402, 'comment': 'Loss of glycerol phosphate head group + NH4'},
    {'class': 'PI', 'adduct': '[M-H]-', 'type': 'PIS', 'value': 241.0119, 'comment': 'Inositol-1,2-cyclic monophosphate anion'},
    {'class': 'PI', 'adduct': '[M+NH4]+', 'type': 'NL', 'value': 277.0563, 'comment': 'Inositol-1,2-cyclic monophosphate cation with NH4'},
    {'class': 'PA', 'adduct': '[M+NH4]+', 'type': 'NLS', 'value': 115.0262, 'comment': 'Loss of NH3 + H3PO4'},
    {'class': 'MG', 'adduct': '[M+H]+', 'type': 'NLS', 'value': 18.0106, 'comment': 'Neutral loss of water (H2O)'},
    {'class': 'MG', 'adduct': '[M+H]+', 'type': 'NLS', 'value': 92.0470, 'comment': 'Neutral loss of glycerol headgroup'},
    {'class': 'CE', 'adduct': '[M+Na]+', 'type': 'NLS', 'value': 368.3441, 'comment': 'Loss of neutral cholestane molecule'},
    {'class': 'CE', 'adduct': '[M+H]+', 'type': 'PIS', 'value': 369.3516, 'comment': 'Dehydrated cholesterol fragment'},
    {'class': 'FA', 'adduct': '[M-H]-', 'type': 'NLS', 'value': 43.9898, 'comment': 'Loss of carbon dioxide (CO2)'},
    {'class': 'NAE', 'adduct': '[M+H]+', 'type': 'PIS', 'value': 62.0600, 'comment': 'Protonated ethanolamine fragment'},
    {'class': 'PC-O', 'adduct': '[M+H]+', 'type': 'PIS', 'value': 184.0733, 'comment': 'Protonated phosphocholine headgroup (indistinguishable from PC/LPC)'},
    {'class': 'PE-O', 'adduct': '[M+H]+', 'type': 'NLS', 'value': 141.0194, 'comment': 'Loss of phosphoethanolamine (indistinguishable from PE/LPE)'},
    

]

def add_num_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'num_chain' column to a DataFrame based on a predefined mapping.

    This function maps the 'class' column to a 'num_chain' value using a 
    predefined dictionary.

    Args:
        df: The input pandas DataFrame containing a 'class' column.

    Returns:
        The DataFrame with a new 'num_chain' column added.
    """
    
    # Predefined mapping of 'class' values to their corresponding number of chains.
    num_chain_map = {
        "BMP": 2, "ST": 1, "CE": 1, "DG": 2, "DGCC": 2, "DGDG": 2, "DGTS": 2, 
        "PC-O": 2, "PE-O": 2, "LPC-O": 1, "LPE-O": 1, "FA": 1, "DGGA": 2, 
        "LDGCC": 1, "LDGTS": 1, "LPA": 1, "LPC": 1, "LPE": 1, "LPG": 1, "LPI": 1, 
        "LPS": 1, "MG": 1, "MGDG": 2, "PA": 2, "PC": 2, "PE": 2, "PG": 2, "PI": 2, 
        "PMeOH": 2, "PS": 2, "SQDG": 2, "TG": 3, "GalCer": 2, "Cer": 2, "PE-P": 2, 
        "SM": 2, "PC-P": 2, "LacCer": 2, "PE_Cer": 3, "PI_Cer": 3, "PG-O": 2, 
        "CL": 4, "PG-P": 2, "GlcCer": 2, "NAE": 1, "CAR": 1, "WE": 1, "HexCer": 2, 
        "MG-O": 1
    }

    # Use the map to create the new 'num_chain' column
    df['num_chain'] = df['class'].map(num_chain_map)
    
    # Return the modified DataFrame
    return df

# --- FUNCTION MODIFIED ---
def identify_lipid_class_from_ms2(precursor_mz, adduct, ms2_spectrum, tolerance=0.5):
    """
    Identifies potential lipid classes based on an MS2 spectrum.

    This function first identifies all possible lipid classes by matching fragments
    from the MS2 spectrum against the MS2_LIPID_RULES. It then applies a special
    post-processing step to differentiate between PC and LPC classes in positive
    mode, using the presence of the 104.107 m/z fragment as a specific marker for LPC.

    Args:
        precursor_mz (float): The m/z of the precursor ion.
        adduct (str): The adduct type.
        ms2_spectrum (list of lists): The MS2 spectrum, [[m/z, intensity], ...].
        tolerance (float): The mass tolerance in Daltons for matching fragments.

    Returns:
        list: A sorted list of unique potential lipid classes, or an empty list if none found.
    """
    potential_classes = set()
    effective_adduct = '[M+H]+' if adduct == '[M-OH]+' else adduct

    # Step 1: Gather all potential classes based on the rules.
    for rule in MS2_LIPID_RULES:
        if effective_adduct != rule['adduct']:
            continue

        if rule['type'] == 'PIS':
            # Check if any m/z in the spectrum matches the rule's value.
            if any(abs(mz - rule['value']) <= tolerance for mz, _ in ms2_spectrum):
                potential_classes.add(rule['class'])
        
        elif rule['type'] == 'NLS':
            if adduct == '[M-OH]+' and rule['value'] != 18.0106:
                continue
            # Check if any neutral loss matches the rule's value.
            if any(abs((precursor_mz - mz) - rule['value']) <= tolerance for mz, _ in ms2_spectrum):
                potential_classes.add(rule['class'])

    # # Step 2: Apply differentiation logic if ambiguity between PC and LPC exists.
    # # This ambiguity arises when the 184.0733 fragment causes both to be identified.
    # if 'PC' in potential_classes and 'LPC' in potential_classes and effective_adduct == '[M+H]+':
    #     # Check for the presence of the differentiating LPC-specific ion.
    #     has_104_ion = any(abs(mz - 104.107) <= tolerance for mz, _ in ms2_spectrum)
        
    #     if has_104_ion:
    #         # The 104.107 ion is specific to LPC. Its presence resolves the ambiguity.
    #         # We can confidently remove PC from the set of possibilities.
    #         potential_classes.remove('PC')
    #     # If the 104 ion is NOT present, we leave the ambiguity, as an LPC might not
    #     # always produce the 104 ion. In that case, both PC and LPC remain valid possibilities.

    return sorted(list(potential_classes)) if potential_classes else []

def batch_identify_by_ms2(df):
    """
    Processes a DataFrame to identify lipid classes based on MS2 spectra.

    Args:
        df (pd.DataFrame): DataFrame with 'precursor_mz', 'adduct', and 'MS2_norm' columns.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'classes_ms2' column.
    """
    
    def process_row(row):
        """Helper function to process each row."""
        try:
            # Ensure 'MS2_norm' is a string representation of a list before evaluation
            if not isinstance(row['MS2_norm'], str):
                return ['MS2_norm is not a string']
            ms2_spectrum = ast.literal_eval(row['MS2_norm'])
            if not isinstance(ms2_spectrum, list) or not all(isinstance(i, list) and len(i) == 2 for i in ms2_spectrum):
                return ['Invalid MS2 format']
            return identify_lipid_class_from_ms2(row['precursor_mz'], row['adduct'], ms2_spectrum)
        except (ValueError, SyntaxError) as e:
            return [f'Error parsing MS2: {e}']
        except Exception as e:
            return [f'An error occurred: {e}']

    # Apply the processing function and name the new column 'classes_ms2'
    df['classes_ms2'] = df.apply(process_row, axis=1)
    
    return df

def find_class_overlap(df):
    """
    Finds the overlap between mz-based and MS2-based class identifications.

    The logic is as follows:
    - If one identification method yields an empty list, use the first result from the other.
    - If both are non-empty, find their intersection and use the first result.
    - If there is no overlap or both are empty, the result is an empty string.
    The final result is a single string, not a list.

    Args:
        df (pd.DataFrame): DataFrame must contain 'classes_mz' and 'classes_ms2' columns.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'final_class' column (string).
    """
    if not all(col in df.columns for col in ['classes_mz', 'classes_ms2']):
        raise ValueError("Input DataFrame must contain 'classes_mz' and 'classes_ms2' columns.")

    # def get_overlap(row):
    #     classes_mz = row['classes_mz']
    #     classes_ms2 = row['classes_ms2']

    #     # If both are empty, return empty string
    #     if not classes_mz and not classes_ms2:
    #         return ''
    #     # If one is empty, use the first element of the other one
    #     if not classes_mz:
    #         return classes_ms2[0] if classes_ms2 else ''
    #     if not classes_ms2:
    #         return classes_mz[0] if classes_mz else ''

    #     # Both lists have values, find the intersection
    #     set_mz = set(classes_mz)
    #     set_ms2 = set(classes_ms2)
    #     overlap = sorted(list(set_mz.intersection(set_ms2)))
        
    #     # Return the first element of the overlap, or empty string if no overlap
    #     return overlap[0] if overlap else ''

    # df['class'] = df.apply(get_overlap, axis=1)
    # return df
    
    def get_overlap(row):
        classes_mz = row['classes_mz']
        classes_ms2 = row['classes_ms2']

        # If both are empty, return empty string
        if not classes_mz and not classes_ms2:
            return ''
        # If MS1 results are empty, use the first from MS2
        if not classes_mz:
            return classes_ms2[0] if classes_ms2 else ''
        # If MS2 results are empty, use the first from MS1
        if not classes_ms2:
            return classes_mz[0] if classes_mz else ''

        # Both lists have values, find their overlap while preserving MS1 order
        set_ms2 = set(classes_ms2) # Use a set for efficient lookup
        
        # Create a new list of overlapping classes in the order of mass accuracy
        ordered_overlap = [cls for cls in classes_mz if cls in set_ms2]
        
        # Return the first element of the ordered overlap, or empty string if none
        return ordered_overlap[0] if ordered_overlap else ''
    
    df['class'] = df.apply(get_overlap, axis=1)
    return df

# --- Category Mapping Section ---
LIPID_CATEGORY_MAP = {
    # FA - Fatty Acyls
    'FA': 'FA', 'NAE': 'FA', 'CAR': 'FA', 'WE': 'FA',
    # GL - Glycerolipids
    'TG': 'GL', 'DG': 'GL', 'MGDG': 'GL', 'DGDG': 'GL', 'SQDG': 'GL', 'MG': 'GL', 'MG-O': 'GL',
    # GP - Glycerophospholipids
    'PC': 'GP', 'PE': 'GP', 'PG': 'GP', 'PS': 'GP', 'PI': 'GP', 'PA': 'GP', 'LPC': 'GP', 
    'LPE': 'GP', 'LPG': 'GP', 'LPS': 'GP', 'LPI': 'GP', 'LPA': 'GP', 'CL': 'GP', 'BMP': 'GP', 
    'PMeOH': 'GP', 'PC-O': 'GP', 'PE-O': 'GP', 'PG-O': 'GP', 'PC-P': 'GP', 'PE-P': 'GP', 
    'PG-P': 'GP', 'LPC-O': 'GP', 'LPE-O': 'GP', 'LPE-P' : 'GP', 'LPC-P' : 'GP',
    # SP - Sphingolipids
    'Cer': 'SP', 'SM': 'SP', 'GalCer': 'SP', 'GlcCer': 'SP', 'LacCer': 'SP', 'HexCer': 'SP', 
    'PI_Cer': 'SP', 'PE_Cer': 'SP',
    # SL - Saccharolipids
    'DGCC': 'SL', 'DGTS': 'SL', 'DGGA': 'SL', 'LDGCC': 'SL', 'LDGTS': 'SL',
    # ST - Sterol Lipids
    'ST': 'ST', 'CE': 'ST',
}

def add_final_category(df):
    """
    Adds a 'final_category' column to the DataFrame based on the 'final_class'.

    Args:
        df (pd.DataFrame): DataFrame must contain a 'final_class' column.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'final_category' column.
    """
    if 'class' not in df.columns:
        raise ValueError("Input DataFrame must contain 'final_class' column.")

    def map_class_to_category(final_class):
        """Helper function to map a single class string to its category."""
        if not final_class:
            return '' # Return empty string if class is empty
        # Get the category from the map, default to 'Unknown' if not found
        return LIPID_CATEGORY_MAP.get(final_class, 'Unknown')

    df['category'] = df['class'].apply(map_class_to_category)
    return df





def get_lipid_categories(formula: str):
    """
    Classifies a chemical formula into possible lipid categories based on
    its elemental composition.

    The classification rules are based on the Consolidated Elemental Signatures
    of the Eight Lipid Categories.

    Args:
        formula: A string representing a chemical formula (e.g., "C27H46O").

    Returns:
        A list of strings with the short names (e.g., 'ST', 'GP') of all
        possible lipid categories. Returns an empty list if the formula is
        invalid or contains no elements.
    """
    # Rules based on elemental presence/absence for each category.
    # 'required': a set of elements that MUST be in the formula.
    # 'forbidden': a set of elements that MUST NOT be in the formula.
    LIPID_CATEGORIES_RULES = {
        'Glycerophospholipids [GP]': {'required': {'C', 'H', 'O', 'P'}, 'forbidden': set()},
        'Sphingolipids [SP]':        {'required': {'C', 'H', 'O', 'N'}, 'forbidden': set()},
        'Saccharolipids [SL]':       {'required': {'C', 'H', 'O', 'N'}, 'forbidden': {'P'}},
        'Sterol Lipids [ST]':        {'required': {'C', 'H'}, 'forbidden': {'N', 'P', 'S'}},
        'Prenol Lipids [PR]':        {'required': {'C', 'H'}, 'forbidden': {'N', 'P', 'S'}},
        'Polyketides [PK]':          {'required': {'C', 'H', 'O'}, 'forbidden': {'P'}},
        'Fatty Acyls [FA]':          {'required': {'C', 'H', 'O'}, 'forbidden': {'P'}},
        'Glycerolipids [GL]':        {'required': {'C', 'H', 'O'}, 'forbidden': set()},
    }

    # 1. Parse the formula to find all unique elements present.
    # This regex finds all capital letters followed by zero or one lowercase letter.
    elements_present = set(re.findall(r'[A-Z][a-z]?', formula))
    
    if not elements_present:
        return []

    # 2. Check the elements against the rules for each category.
    matched_categories = []
    for category_full_name, rules in LIPID_CATEGORIES_RULES.items():
        # A formula matches if:
        # a) All 'required' elements are present.
        # b) No 'forbidden' elements are present.
        if rules['required'].issubset(elements_present) and \
           rules['forbidden'].isdisjoint(elements_present):
            # Extract the short name from the full name string (e.g., 'GP' from '[GP]')
            short_name_match = re.search(r'\[(.*?)\]', category_full_name)
            if short_name_match:
                matched_categories.append(short_name_match.group(1))

    return matched_categories



def process_lipid_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a classification function to formula columns in a DataFrame,
    filters them based on a 'category' column, and left-shifts the results.

    If all formulas for a row are filtered out (i.e., none match the category),
    this function will use the first formula ('formula_rank_1') as a fallback.

    Args:
        df: A pandas DataFrame with a 'category' column and columns named
            'formula_rank_1' through 'formula_rank_5'.

    Returns:
        A pandas DataFrame with the formula columns processed.
    """
    formula_cols = [f'formula_rank_{i}' for i in range(1, 6)]

    # This helper function will be applied to each row of the DataFrame.
    def process_row(row):
        target_category = row['category']
        
        valid_formulas = []
        # Iterate through the formula columns for the current row
        for col in formula_cols:
            formula = row[col]
            
            # Check if the formula is a non-empty string
            if isinstance(formula, str) and formula:
                classification_result = get_lipid_categories(formula)
                # Check if the classification is a list and contains the target category
                if isinstance(classification_result, list) and target_category in classification_result:
                    valid_formulas.append(formula)
        
        # --- MODIFICATION START ---
        # If no formulas matched the category, use the first formula as a fallback.
        if not valid_formulas:
            first_formula = row[formula_cols[0]]  # This is 'formula_rank_1'
            if isinstance(first_formula, str) and first_formula:
                valid_formulas.append(first_formula)
        # --- MODIFICATION END ---

        # Create the new row with valid formulas shifted to the left, padded with NaNs.
        # The final list of formulas will have a length of 5.
        padded_formulas = valid_formulas + [np.nan] * (len(formula_cols) - len(valid_formulas))
        
        # Return the new values for the formula columns as a Series.
        return pd.Series(padded_formulas, index=formula_cols)

    # Apply the processing function row-by-row and update the DataFrame.
    df[formula_cols] = df.apply(process_row, axis=1)
    
    return df



# --- Example Usage ---
if __name__ == '__main__':
    # This example assumes you have a CSV file at the specified path.
    # You will need to change the path to your actual data file.
    try:
        data = pd.read_csv('/Users/bowen/Desktop/DeepLipid/datasets/Li_Lab_Lipid_Stanard/standard_df_new.csv')
        
        # 1. Identify by precursor mass, sorting by ppm
        print("--- Step 1: Identifying by mass (mz) ---")
        with_class_mz = batch_identify_by_mass(data, tolerance_ppm=20)

        # 2. Identify by MS2 fragments
        print("--- Step 2: Identifying by MS2 fragments ---")
        with_class_ms2 = batch_identify_by_ms2(with_class_mz)
        
        # 3. Find the overlap for final identification
        print("--- Step 3: Finding overlap for final class ---")
        with_final_class = find_class_overlap(with_class_ms2)

        # 4. Add final category
        print("--- Step 4: Adding final category ---")
        final_results = add_final_category(with_final_class)
        
        print("--- Writing results to test_result.csv ---")
        final_results.to_csv('test_result.csv', index=False)
        print("--- Done ---")

    except FileNotFoundError:
        print("Error: The example data file was not found.")
        print("Please update the path in the 'if __name__ == '__main__':' block.")
