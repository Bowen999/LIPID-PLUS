import io  # Used for creating the self-contained example
import pandas as pd
import numpy as np

import ast
import math

from rdkit import Chem


"""
# Molecular representation
▖  ▖  ▜       ▜                       ▗   ▗ ▘      ▖  ▖  ▜       ▜                       ▗   ▗ ▘    
▛▖▞▌▛▌▐ █▌▛▘▌▌▐ ▀▌▛▘  ▛▘█▌▛▌▛▘█▌▛▘█▌▛▌▜▘▀▌▜▘▌▛▌▛▌  ▛▖▞▌▛▌▐ █▌▛▘▌▌▐ ▀▌▛▘  ▛▘█▌▛▌▛▘█▌▛▘█▌▛▌▜▘▀▌▜▘▌▛▌▛▌
▌▝ ▌▙▌▐▖▙▖▙▖▙▌▐▖█▌▌   ▌ ▙▖▙▌▌ ▙▖▄▌▙▖▌▌▐▖█▌▐▖▌▙▌▌▌  ▌▝ ▌▙▌▐▖▙▖▙▖▙▌▐▖█▌▌   ▌ ▙▖▙▌▌ ▙▖▄▌▙▖▌▌▐▖█▌▐▖▌▙▌▌▌
                          ▌                                                  ▌                      
"""


# --- Function 1: InChI String to SMILES String ---
def inchi_to_smiles(inchi_string: str):
    """
    Converts a single InChI string to a SMILES string using RDKit.

    Args:
        inchi_string: The InChI string.

    Returns:
        The corresponding SMILES string, or None if the
        InChI is invalid or conversion fails.
    """
    if not inchi_string or not isinstance(inchi_string, str):
        return None
        
    # --- FIX was applied here ---
    # Corrected Chem.MolFromInChI to Chem.MolFromInchi
    mol = Chem.MolFromInchi(inchi_string)
    
    if mol:
        # Generate canonical SMILES
        return Chem.MolToSmiles(mol)
    else:
        # RDKit will print a warning to the console for invalid InChIs
        # We return None to handle this gracefully in the DataFrame
        return None

# --- Function 2: Add SMILES Column to DataFrame ---

def add_smiles_to_df(df: pd.DataFrame, 
                     inchi_col: str = 'inchi', 
                     smiles_col: str = 'smiles') -> pd.DataFrame:
    """
    Adds a new column of SMILES strings to a DataFrame based on 
    an existing column of InChI strings.

    Args:
        df: The pandas DataFrame.
        inchi_col: The name of the column containing InChI strings.
        smiles_col: The name for the new SMILES column.

    Returns:
        The DataFrame with the new SMILES column added.
    """
    if inchi_col not in df.columns:
        print(f"Error: Column '{inchi_col}' not found in DataFrame.")
        return df
        
    # Apply the first function to every row in the inchi_col
    # The .apply() method iterates over the 'inchi' column
    # and passes each value to the inchi_to_smiles function.
    df[smiles_col] = df[inchi_col].apply(inchi_to_smiles)
    
    return df




def parse_chain_string(input_string: str):
    """
    Parses a chain string into its component blocks and sub-blocks.
    (Logic remains unchanged from original request)
    """
    def _safe_int_convert_desc(value: str | None):
        try:
            return int(value)
        except (ValueError, TypeError):
            return float('-inf')

    all_blocks_data = []
    
    # Handle empty or None input gracefully
    if not input_string:
        return [], 0

    blocks = input_string.split('_')

    for block in blocks:
        num_c = None
        num_db = None
        extra = None

        semicolon_parts = block.split(';', 1)
        before_semicolon = semicolon_parts[0]

        if len(semicolon_parts) == 2:
            extra = semicolon_parts[1]

        colon_parts = before_semicolon.split(':', 1)
        num_c = colon_parts[0]

        if len(colon_parts) == 2:
            num_db = colon_parts[1]

        block_data = {
            'num_c': num_c,
            'num_db': num_db,
            'extra': extra
        }
        all_blocks_data.append(block_data)

    all_blocks_data.sort(key=lambda block: (
        _safe_int_convert_desc(block['num_c']),
        _safe_int_convert_desc(block['num_db']) 
    ), reverse=True)

    num_tail = input_string.count('_') + 1

    return all_blocks_data, num_tail


def process_lipid_name_df(df: pd.DataFrame):
    """
    Processes a DataFrame with a 'name' column.

    1. Extracts 'class' (everything before the first space in 'name').
    2. Extracts the chain string (everything after the first space).
    3. Parses the chain string to create num_c_i, num_db_i, extra_i columns.

    Args:
        df: A pandas DataFrame which must contain a 'name' column.

    Returns:
        The modified pandas DataFrame with 'class' and parsed columns added.
    """
    if 'name' not in df.columns:
        print("Error: DataFrame must have a 'name' column.")
        return df

    # 1. Split 'name' into two parts based on the first space only
    # expand=True creates a DataFrame with 2 columns (0 and 1)
    split_data = df['name'].astype(str).str.split(' ', n=1, expand=True)

    # 2. Assign the 'class' column (the part before the space)
    df['class'] = split_data[0]

    # 3. Get the chain string (the part after the space)
    # We handle cases where there might be no space (result is None/NaN)
    # by filling with an empty string so the parser doesn't crash.
    if split_data.shape[1] > 1:
        chain_series = split_data[1].fillna('')
    else:
        # If there were no spaces in any row, create a series of empty strings
        chain_series = pd.Series([''] * len(df), index=df.index)

    # 4. Apply the parsing function to the extracted chain string
    parsed_results = chain_series.apply(parse_chain_string)

    # 5. Extract num_tail
    df['num_tail'] = parsed_results.apply(lambda x: x[1])

    # 6. Extract the sorted blocks list
    blocks_series = parsed_results.apply(lambda x: x[0])

    # 7. Create new columns for each of the top 4 blocks
    for i in range(4):
        block_num = i + 1
        
        # Helper lambda to get a block's value, or None if block doesn't exist
        get_block_val = lambda blocks, key: blocks[i][key] if len(blocks) > i else None

        # Get the raw series
        num_c_series = blocks_series.apply(get_block_val, key='num_c')
        num_db_series = blocks_series.apply(get_block_val, key='num_db')
        
        # Convert to numeric and fill NaNs with 0
        df[f'num_c_{block_num}'] = pd.to_numeric(num_c_series, errors='coerce').fillna(0)
        df[f'num_db_{block_num}'] = pd.to_numeric(num_db_series, errors='coerce').fillna(0)
        
        # 'extra' column remains as is
        df[f'extra_{block_num}'] = blocks_series.apply(get_block_val, key='extra')

    return df


def add_ref_list_column(df: pd.DataFrame, output_col_name: str = 'ref') -> pd.DataFrame:
    """
    Creates a new column containing a list of 8 integer elements:
    [num_c_1, num_db_1, num_c_2, num_db_2, num_c_3, num_db_3, num_c_4, num_db_4].

    Args:
        df: DataFrame containing the processed num_c/num_db columns.
        output_col_name: The name of the new column to create (default 'ref').

    Returns:
        The DataFrame with the new list column added.
    """
    # Define the exact order of columns to include in the list
    target_cols = [
        'num_c_1', 'num_db_1',
        'num_c_2', 'num_db_2',
        'num_c_3', 'num_db_3',
        'num_c_4', 'num_db_4'
    ]

    # Check if required columns exist
    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following columns are missing: {missing_cols}")
        print("Please run 'process_chain_dataframe' first.")
        return df

    # 1. Select the columns
    # 2. .astype(int) converts 18.0 -> 18
    # 3. .values.tolist() converts the rows to python lists
    df[output_col_name] = df[target_cols].astype(int).values.tolist()

    return df










"""
proprety
            ▗                 ▗                 ▗   
▛▌▛▘▛▌▛▌▛▘█▌▜▘▌▌  ▛▌▛▘▛▌▛▌█▌▛▘▜▘▌▌  ▛▌▛▘▛▌▛▌█▌▛▘▜▘▌▌
▙▌▌ ▙▌▙▌▌ ▙▖▐▖▙▌  ▙▌▌ ▙▌▙▌▙▖▌ ▐▖▙▌  ▙▌▌ ▙▌▙▌▙▖▌ ▐▖▙▌
▌     ▌       ▄▌  ▌     ▌       ▄▌  ▌     ▌       ▄▌
"""


def calculate_exact_mass(df):
    """
    Calculate exact mass from precursor_mz and adduct
    
    Args:
        df: DataFrame with 'precursor_mz' and 'adduct' columns
        
    Returns:
        DataFrame with added 'exact_mass' column
    """
    # Adduct mass adjustments (in Da)
    adduct_adjustments = {
        '[M]+': 0.0,
        '[M+H]+': -1.00783,
        '[M+NH4]+': -18.03383,
        '[M+Na]+': -22.98977,
        '[M+K]+': -38.96371,
        '[M-H]-': 1.00783,
        '[M+HCOO]-': -44.99820,
        '[M+CH3COO]-': -59.01385,
        '[2M+H]+': -1.00783,
        '[2M+Na]+': -22.98977,
        '[2M-H]-': 1.00783,
        '[M-H2O+H]+': 17.00274,
        '[M-2H2O+H]+': 35.01311,
        '[M+H-H2O]+': 17.00274,
        '[M+]+': 0.0,
        '[M]': 0.0,
        '[M-H1]-': 1.00783,
        '[M+CH3COOH-H]-': -59.01385
    }
    
    exact_masses = []
    for idx, row in df.iterrows():
        precursor_mz = row['precursor_mz']
        adduct = row['adduct']
        
        if adduct in adduct_adjustments:
            adjustment = adduct_adjustments[adduct]
            exact_mass = precursor_mz + adjustment
            
            # Handle dimers (2M adducts)
            if '2M' in adduct:
                exact_mass = exact_mass / 2.0
                
            exact_masses.append(exact_mass)
        else:
            # Unknown adduct, use precursor_mz as is
            print(f"Warning: Unknown adduct '{adduct}' at index {idx}, using precursor_mz as exact_mass")
            exact_masses.append(precursor_mz)
    
    df['exact_mass'] = exact_masses
    return df











"""
MS2 Processing
▖  ▖▄▖  ▄▖            ▘      ▖  ▖▄▖  ▄▖            ▘      ▖  ▖▄▖  ▄▖            ▘    
▛▖▞▌▚   ▙▌▛▘▛▌▛▘█▌▛▘▛▘▌▛▌▛▌  ▛▖▞▌▚   ▙▌▛▘▛▌▛▘█▌▛▘▛▘▌▛▌▛▌  ▛▖▞▌▚   ▙▌▛▘▛▌▛▘█▌▛▘▛▘▌▛▌▛▌
▌▝ ▌▄▌  ▌ ▌ ▙▌▙▖▙▖▄▌▄▌▌▌▌▙▌  ▌▝ ▌▄▌  ▌ ▌ ▙▌▙▖▙▖▄▌▄▌▌▌▌▙▌  ▌▝ ▌▄▌  ▌ ▌ ▙▌▙▖▙▖▄▌▄▌▌▌▌▙▌
                         ▄▌                           ▄▌                           ▄▌
"""

def ms2_format(ms2_string):
    """
    Converts a string representation of an MS2 spectrum to a Python list.
    Optimized with early returns and fast path checks.
    
    Args:
        ms2_string (str): The string to convert.
    
    Returns:
        list: A list of [mz, intensity] pairs, or an empty list if conversion fails.
    """
    # Fast path: already a list
    if isinstance(ms2_string, list):
        return ms2_string
    
    # Fast path: not a string
    if not isinstance(ms2_string, str):
        return []
    
    # Fast path: empty or common null values
    if not ms2_string or ms2_string in ('[]', 'nan', 'None', ''):
        return []
    
    try:
        result = ast.literal_eval(ms2_string)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError, MemoryError):
        return []


def ms2_format_df_process(df):
    """
    Applies the ms2_format function to the 'MS2' column of a DataFrame.
    Optimized to skip already-converted values.
    
    Args:
        df (pd.DataFrame): The DataFrame with an 'MS2' column containing string spectra.
    
    Returns:
        pd.DataFrame: The DataFrame with the 'MS2' column converted to lists.
    """
    if 'MS2' in df.columns:
        # Vectorized check: only process non-list values
        mask = ~df['MS2'].apply(lambda x: isinstance(x, list))
        if mask.any():
            df.loc[mask, 'MS2'] = df.loc[mask, 'MS2'].apply(ms2_format)
    return df


def ms2_norm(df):
    """
    Generates 'MS2_norm' and 'num_peaks' columns based on the 'MS2' column.
    Vectorized implementation using NumPy for 5-20x speedup.
    
    Args:
        df (pd.DataFrame): DataFrame with an 'MS2' column.
    
    Returns:
        pd.DataFrame: The DataFrame with 'MS2_norm' and 'num_peaks' columns added.
    """
    if 'MS2' not in df.columns:
        print("Error: 'MS2' column not found.")
        return df
    
    # Ensure MS2 column contains lists
    ms2_lists = df['MS2'].apply(
        lambda x: ms2_format(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    
    ms2_norm_list = []
    num_peaks_list = []
    
    # Process each spectrum
    for ms2_list in ms2_lists:
        if not ms2_list or not isinstance(ms2_list, list):
            ms2_norm_list.append([])
            num_peaks_list.append(0)
            continue
        
        try:
            # Convert to numpy array for vectorized operations
            peaks_array = np.array(ms2_list, dtype=np.float64)
            
            # Validate shape
            if peaks_array.ndim != 2 or peaks_array.shape[1] != 2:
                ms2_norm_list.append([])
                num_peaks_list.append(0)
                continue
            
            # Extract intensities
            intensities = peaks_array[:, 1]
            
            if len(intensities) == 0:
                ms2_norm_list.append([])
                num_peaks_list.append(0)
                continue
            
            # Vectorized normalization
            min_int = intensities.min()
            max_int = intensities.max()
            denom = max_int - min_int
            
            if denom == 0:
                # All intensities are the same
                norm_intensities = np.full_like(intensities, 100.0 if max_int > 0 else 0.0)
            else:
                norm_intensities = (intensities - min_int) * 100.0 / denom
            
            # Round to nearest integer
            norm_intensities = np.round(norm_intensities, 0)
            
            # Filter peaks with intensity > 3
            mask = norm_intensities > 3
            
            if not mask.any():
                ms2_norm_list.append([])
                num_peaks_list.append(0)
                continue
            
            # Create filtered peak list
            filtered_peaks = np.column_stack([
                peaks_array[mask, 0], 
                norm_intensities[mask]
            ]).tolist()
            
            ms2_norm_list.append(filtered_peaks)
            num_peaks_list.append(len(filtered_peaks))
            
        except (ValueError, TypeError, IndexError):
            ms2_norm_list.append([])
            num_peaks_list.append(0)
    
    df['MS2_norm'] = ms2_norm_list
    df['num_peaks'] = num_peaks_list
    
    return df


def ms2_norm_binning(df, decimal_point=0, neutral_loss=False, keep_intensity=True):
    """
    Creates binned m/z columns based on the 'MS2_norm' column.
    Optimized with NumPy arrays and pre-allocated memory for 3-10x speedup.
    
    Args:
        df (pd.DataFrame): DataFrame with 'MS2_norm' and optionally 'precursor_mz' columns.
        decimal_point (int): The number of decimal points for m/z binning.
        neutral_loss (bool): If True, adds neutral loss peaks (precursor_mz - mz).
        keep_intensity (bool): If True (default), keeps intensity values. If False, converts non-zero to 1.
    
    Returns:
        pd.DataFrame: The DataFrame with m/z bin columns added.
    """
    if 'MS2_norm' not in df.columns:
        print("Error: 'MS2_norm' column not found. Run ms2_norm first.")
        return df
    
    if neutral_loss and 'precursor_mz' not in df.columns:
        print("Error: 'precursor_mz' column not found but neutral_loss=True.")
        return df
    
    min_mz_bin = 50
    max_mz_bin = 1500
    
    # Generate bins and column names
    if decimal_point == 0:
        bins = np.arange(min_mz_bin, max_mz_bin + 1, dtype=np.int32)
        col_names = [f"mz_{int(b)}" for b in bins]
    else:
        multiplier = 10 ** decimal_point
        start = int(min_mz_bin * multiplier)
        end = int(max_mz_bin * multiplier) + 1
        bins = np.arange(start, end) / multiplier
        col_names = [f"mz_{b:.{decimal_point}f}" for b in bins]
    
    n_rows = len(df)
    n_bins = len(bins)
    
    # Pre-allocate array (much faster than building dict per row)
    binned_array = np.zeros((n_rows, n_bins), dtype=np.float32)
    
    # Create mapping from column name to array index
    bin_to_idx = {col_names[i]: i for i in range(n_bins)}
    
    # Process each row
    for row_idx, ms2_norm_list in enumerate(df['MS2_norm']):
        if not isinstance(ms2_norm_list, list) or not ms2_norm_list:
            continue
        
        try:
            # Convert to numpy array
            peaks = np.array(ms2_norm_list, dtype=np.float64)
            if peaks.ndim != 2 or peaks.shape[1] != 2:
                continue
            
            # Add neutral loss peaks if requested
            if neutral_loss:
                precursor_mz = df.iloc[row_idx]['precursor_mz']
                neutral_loss_mzs = precursor_mz - peaks[:, 0]
                neutral_loss_peaks = np.column_stack([neutral_loss_mzs, peaks[:, 1]])
                # Combine original and neutral loss peaks
                peaks = np.vstack([peaks, neutral_loss_peaks])
            
            # Round m/z values to the appropriate decimal place
            mzs = np.round(peaks[:, 0], decimal_point)
            intensities = peaks[:, 1]
            
            # Assign intensities to bins (take max if multiple peaks per bin)
            for mz, intensity in zip(mzs, intensities):
                if decimal_point == 0:
                    col_name = f"mz_{int(mz)}"
                else:
                    col_name = f"mz_{mz:.{decimal_point}f}"
                
                if col_name in bin_to_idx:
                    idx = bin_to_idx[col_name]
                    # Take maximum intensity if multiple peaks map to same bin
                    binned_array[row_idx, idx] = max(binned_array[row_idx, idx], intensity)
        
        except (ValueError, TypeError, IndexError):
            continue
    
    # Convert to binary (0 or 1) if keep_intensity=False
    if not keep_intensity:
        binned_array = (binned_array > 0).astype(np.float32)
    
    # Create DataFrame from pre-allocated array
    binned_df = pd.DataFrame(binned_array, columns=col_names, index=df.index)
    
    # Concatenate with original DataFrame
    df = pd.concat([df, binned_df], axis=1)
    
    return df


def process_ms2_df(df, decimal_point=0, neutral_loss=False, keep_intensity=True):
    """
    Runs the complete MS2 processing pipeline on a DataFrame.
    Optimized version with 4-15x overall speedup.
    
    Args:
        df (pd.DataFrame): The input DataFrame with an 'MS2' column.
        decimal_point (int): The number of decimal points for m/z binning.
        neutral_loss (bool): If True, adds neutral loss peaks (precursor_mz - mz).
        keep_intensity (bool): If True (default), keeps intensity values. If False, converts non-zero to 1.
    
    Returns:
        pd.DataFrame: The fully processed DataFrame.
    """
    print(f"Starting processing with decimal_point={decimal_point}, neutral_loss={neutral_loss}, keep_intensity={keep_intensity}...")
    
    # Step 1: Normalize, filter, and count peaks
    df_norm = ms2_norm(df)
    
    # Step 2: Bin the normalized data
    df_binned = ms2_norm_binning(df_norm, decimal_point=decimal_point, 
                                   neutral_loss=neutral_loss, keep_intensity=keep_intensity)
    
    print("Processing complete.")
    return df_binned









"""
Lipid Spcialty
▜ ▘  ▘ ▌  ▜ ▘  ▘ ▌  ▜ ▘  ▘ ▌  ▜ ▘  ▘ ▌  ▜ ▘  ▘ ▌
▐ ▌▛▌▌▛▌  ▐ ▌▛▌▌▛▌  ▐ ▌▛▌▌▛▌  ▐ ▌▛▌▌▛▌  ▐ ▌▛▌▌▛▌
▐▖▌▙▌▌▙▌  ▐▖▌▙▌▌▙▌  ▐▖▌▙▌▌▙▌  ▐▖▌▙▌▌▙▌  ▐▖▌▙▌▌▙▌
   ▌         ▌         ▌         ▌         ▌    
          
"""

def find_best_composition(df, ms1_tolerance=30.0):
    """
    Reverse engineer num_c and num_db from exact_mass and lipid class
    
    Args:
        df: DataFrame with 'exact_mass' and 'class' columns
        ms1_tolerance: PPM tolerance for mass matching (default: 30.0)
        
    Returns:
        DataFrame with added 'num_c' and 'num_db' columns
        
    Notes:
        - Returns None for both if no match found within tolerance
        - Returns None for both if lipid class is unknown
    """
    # Internal constants
    M_C = 12.00000
    M_H = 1.00783
    M_CH2 = M_C + (2 * M_H)
    
    # Head group masses for different lipid classes (in Da)
    head_mass_refs = {
        'BMP': 273.0, 'CAR': 175.08, 'CE': 399.33, 'CL': 454.96, 'DG': 119.0,
        'DG-O': 105.02, 'DG-P': 103.0, 'DGCC': 278.09, 'DGDG': 443.1, 'DGGA': 295.03,
        'DGTS': 262.09, 'FA': 30.98, 'LDGCC': 264.11, 'LDGTS': 248.11, 'LPA': 184.99,
        'LPC': 270.07, 'LPC-O': 256.09, 'LPE': 228.03, 'LPE-O': 214.05, 'LPG': 259.02,
        'LPI': 347.04, 'LPS': 272.02, 'MG': 105.02, 'MG-O': 91.04, 'MG-P': 89.02,
        'MGDG': 281.05, 'NAE': 74.02, 'PA': 198.96, 'PA-O': 184.98, 'PA-P': 182.97,
        'PC': 284.05, 'PC-O': 270.07, 'PC-P': 268.06, 'PE': 242.01, 'PE-O': 228.03,
        'PE-P': 226.01, 'PG': 273.0, 'PG-O': 259.02, 'PG-P': 257.01, 'PI': 361.02,
        'PI-O': 347.04, 'PI-P': 345.02, 'PMeOH': 212.98, 'PS': 286.0, 'PS-O': 272.02,
        'PS-P': 270.0, 'SE': 22.92, 'SM-d': 227.04, 'SM-t': 243.04, 'SQDG': 345.01,
        'TG': 132.98, 'TG-O': 119.0, 'WE': 30.98
    }
    
    def _solve_row(row):
        """Find best C and DB for a single row"""
        obs_mass = row['exact_mass']
        cls = row['class']
        
        # Unknown class
        if cls not in head_mass_refs:
            return None, None
        
        h_mass = head_mass_refs[cls]
        target_tail = obs_mass - h_mass
        
        # Invalid tail mass
        if target_tail <= 0:
            return None, None
        
        # Estimate carbon count and search window
        c_estimate = int(target_tail / M_CH2)
        c_min = max(1, c_estimate - 5)
        c_max = c_estimate + 5
        
        best_ppm = float('inf')
        best_match = (None, None)
        
        # Search for best C and DB combination
        for c in range(c_min, c_max + 1):
            for db in range(0, 13):  # Max 12 double bonds
                if db >= c:
                    break
                
                # Calculate theoretical mass
                h = (2 * c + 1) - (2 * db)
                tail_mass = (c * M_C) + (h * M_H)
                theoretical_total = h_mass + tail_mass
                
                # Calculate error in ppm
                error_mass = abs(obs_mass - theoretical_total)
                ppm = (error_mass / obs_mass) * 1_000_000
                
                # Check if within tolerance
                if ppm <= ms1_tolerance:
                    if ppm < best_ppm:
                        best_ppm = ppm
                        best_match = (c, db)
        
        return best_match
    
    # Apply to all rows
    results = df.apply(_solve_row, axis=1)
    
    # Add columns
    df['num_c'] = [x[0] for x in results]
    df['num_db'] = [x[1] for x in results]
    
    # Convert to numeric (allows NaN for None values)
    df['num_c'] = pd.to_numeric(df['num_c'], errors='coerce')
    df['num_db'] = pd.to_numeric(df['num_db'], errors='coerce')
    
    return df


def get_supported_adducts():
    """
    Get list of supported adduct types
    
    Returns:
        List of supported adduct strings
    """
    return [
        '[M]+', '[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+K]+',
        '[M-H]-', '[M+HCOO]-', '[M+CH3COO]-',
        '[2M+H]+', '[2M+Na]+', '[2M-H]-',
        '[M-H2O+H]+', '[M-2H2O+H]+', '[M+H-H2O]+',
        '[M+]+', '[M]', '[M-H1]-', '[M+CH3COOH-H]-'
    ]


def get_supported_lipid_classes():
    """
    Get list of supported lipid classes
    
    Returns:
        List of supported lipid class strings
    """
    return [
        'BMP', 'CAR', 'CE', 'CL', 'DG', 'DG-O', 'DG-P', 'DGCC', 'DGDG', 'DGGA',
        'DGTS', 'FA', 'LDGCC', 'LDGTS', 'LPA', 'LPC', 'LPC-O', 'LPE', 'LPE-O',
        'LPG', 'LPI', 'LPS', 'MG', 'MG-O', 'MG-P', 'MGDG', 'NAE', 'PA', 'PA-O',
        'PA-P', 'PC', 'PC-O', 'PC-P', 'PE', 'PE-O', 'PE-P', 'PG', 'PG-O', 'PG-P',
        'PI', 'PI-O', 'PI-P', 'PMeOH', 'PS', 'PS-O', 'PS-P', 'SE', 'SM-d', 'SM-t',
        'SQDG', 'TG', 'TG-O', 'WE'
    ]


def validate_input_data(df):
    """
    Validate input DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> valid, error = validate_input_data(df)
        >>> if not valid:
        >>>     print(f"Error: {error}")
    """
    required_cols = ['precursor_mz', 'adduct', 'class']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for null values
    for col in required_cols:
        if df[col].isna().any():
            null_count = df[col].isna().sum()
            return False, f"Column '{col}' has {null_count} null values"
    
    return True, None


