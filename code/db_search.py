import pandas as pd
import os
import argparse
import pandas as pd
import sqlite3
import numpy as np
import ms_entropy as me
import spectral_entropy #How to download: https://github.com/YuanyueLi/SpectralEntropy/blob/master/example.ipynb
from tqdm.auto import tqdm # Import tqdm for the progress bar
import json
import spectral_entropy
import ast
import os

# --- Placeholder for your custom functions ---
# It's assumed that these functions are defined in your environment
# or imported from another module.
def normalize_ms2(spectrum_str: str) -> dict:
    """
    Normalizes MS2 intensities, where the max intensity becomes 100,
    and filters out any peaks with an original intensity below 3.
    It now also returns the count of the peaks in the normalized spectrum.

    Args:
        spectrum_str (str): A string representation of a list of mz-intensity pairs.

    Returns:
        dict: A dictionary with two keys:
              - 'normalized_spectrum': The list of normalized and filtered mz-intensity pairs.
              - 'peak_count': The number of peaks in the normalized spectrum.
    """
    # Safely convert the string to a Python list
    try:
        spectrum = ast.literal_eval(spectrum_str)
    except (ValueError, SyntaxError):
        # Handle cases where the string isn't a valid list
        spectrum = []
    
    # Return a dictionary with empty values if the spectrum is empty
    if not spectrum:
        return {'normalized_spectrum': [], 'peak_count': 0}
    
    # Find the maximum intensity in the spectrum to use for normalization
    # The 'or [0]' handles cases of empty lists
    try:
        max_intensity = max(item[1] for item in spectrum)
    except (IndexError, TypeError):
        # Handle cases where items in the list are not in the expected format
        max_intensity = 0

    # Return a dictionary with empty values if max intensity is 0
    if max_intensity == 0:
        return {'normalized_spectrum': [], 'peak_count': 0}

    # Use a list comprehension to filter, normalize, and round in one step
    normalized_spectrum = [
        [mz, round((intensity / max_intensity) * 100, 1)]
        for mz, intensity in spectrum if intensity >= 3
    ]

    # Get the number of peaks from the length of the new list
    peak_count = len(normalized_spectrum)
    
    return {
        'normalized_spectrum': normalized_spectrum,
        'num_peaks': peak_count
    }
    
    

ADDUCT_MASSES = {
    # --- Positive Ion Mode ---
    '[M+H]+': 1.007276,          # Proton
    '[M+Li]+': 7.016004,         # Lithium
    '[M+NH4]+': 18.033823,       # Ammonium
    '[M+Na]+': 22.989218,        # Sodium
    '[M+K]+': 38.963158,         # Potassium
    '[M+CH3OH+H]+': 33.033489,   # Methanol adduct
    '[M+ACN+H]+': 42.033823,     # Acetonitrile adduct
    '[M+2Na-H]+': 44.97116,      # Double sodium adduct, single proton loss
    '[M+ACN+Na]+': 64.01576,     # Acetonitrile + Sodium adduct
    
    # --- Common Neutral Losses (appear as positive ions) ---
    '[M-H2O+H]+': -17.003289,    # Loss of water
    '[M+H-H2O]+': -17.003289,    # Loss of water (alternative notation)
    '[M-NH3+H]+': -16.018724,    # Loss of ammonia
    '[M-OH]+': -17.00328866,     # Loss of hydroxyl group (from your original list)

    # --- Negative Ion Mode ---
    '[M-H]-': -1.007276,         # Deprotonation
    '[M-2H]2-': -2.014552,       # Double deprotonation (note: m/z is (M-2.014)/2)
    '[M+Na-2H]-': 20.974666,     # Sodium adduct, double proton loss
    '[M+K-2H]-': 36.948608,      # Potassium adduct, double proton loss
    '[M+Cl]-': 34.969402,        # Chloride
    '[M+Br]-': 78.918337,        # Bromide (using 79Br isotope)
    '[M+HCOO]-': 44.998201,      # Formate
    '[M+CH3COO]-': 59.013851,    # Acetate
    '[M+TFA]-': 112.985586       # Trifluoroacetate
}


def ms_db_search(db_path: str,
                   precursor_mz: float,
                   adduct: str,
                   MS1_tol: float,
                   query_MS2: list,
                   MS2_tol: float,
                   is_ppm: bool = True,
                   method: str = 'weighted_dot_product') -> pd.DataFrame:
    """
    Searches a database by mass, calculates multiple spectral similarity scores,
    and sorts by a specified score.
    """
    if not os.path.exists(db_path):
        # Silently return, as this will be handled in the batch function
        return pd.DataFrame()

    # --- Step 1: Calculate query exact mass from precursor m/z ---
    adduct_mass = ADDUCT_MASSES.get(adduct)
    if adduct_mass is None:
        return pd.DataFrame()
    query_exact_mass = precursor_mz - adduct_mass

    # --- Step 2: Calculate mass tolerance range for DB query ---
    if is_ppm:
        mass_error = (MS1_tol / 1_000_000) * query_exact_mass
    else:
        mass_error = MS1_tol
    mass_min = query_exact_mass - mass_error
    mass_max = query_exact_mass + mass_error

    # --- Step 3: Connect to DB and perform initial search by mass ---
    try:
        conn = sqlite3.connect(db_path)
        db_query = "SELECT *, exact_mass as reference_mass FROM compounds WHERE exact_mass BETWEEN ? AND ?"
        results_df = pd.read_sql_query(db_query, conn, params=(mass_min, mass_max))
        conn.close()
    except Exception as e:
        # Silently return on DB error
        return pd.DataFrame()

    if results_df.empty:
        return pd.DataFrame()

    # --- Step 4: Add new columns and perform spectral matching ---
    results_df['mass_diff_ppm'] = ((results_df['reference_mass'] - query_exact_mass) / query_exact_mass) * 1_000_000
    
    def calculate_scores(row):
        ref_spectrum_str = row['MS2_norm']
        scores = {
            'dot_product': 0.0,
            'weighted_dot_product': 0.0,
            'unweighted_entropy_similarity': 0.0,
            'entropy_similarity': 0.0
        }
        
        if not ref_spectrum_str or not isinstance(ref_spectrum_str, str):
            return pd.Series(scores)

        try:
            ref_spectrum = json.loads(ref_spectrum_str)
            if not ref_spectrum or not query_MS2:
                return pd.Series(scores)
            
            scores['dot_product'] = spectral_entropy.similarity(query_MS2, ref_spectrum, method="dot_product", ms2_da=MS2_tol)
            scores['weighted_dot_product'] = spectral_entropy.similarity(query_MS2, ref_spectrum, method="weighted_dot_product", ms2_da=MS2_tol)
            scores['unweighted_entropy_similarity'] = me.calculate_unweighted_entropy_similarity(query_MS2, ref_spectrum, ms2_tolerance_in_da=MS2_tol)
            scores['entropy_similarity'] = me.calculate_entropy_similarity(query_MS2, ref_spectrum, ms2_tolerance_in_da=MS2_tol)
            
            return pd.Series(scores)
        except (json.JSONDecodeError, TypeError):
            return pd.Series(scores)

    score_columns = results_df.apply(calculate_scores, axis=1)
    results_df = pd.concat([results_df, score_columns], axis=1)

    # --- Step 5: Sort by the specified score and finalize DataFrame ---
    if method not in results_df.columns:
        return results_df

    final_df = results_df.sort_values(by=method, ascending=False).reset_index(drop=True)
    
    return final_df


def batch_search_and_annotate(input_df: pd.DataFrame,
                              db_path: str,
                              MS1_tol: float = 10,
                              MS2_tol: float = 0.1,
                              is_ppm: bool = True,
                              method: str = 'weighted_dot_product',
                              MS2_threshold: float = 0.8) -> pd.DataFrame:
    """
    Searches the MS database for each row in an input DataFrame and returns the
    original DataFrame annotated with the top-scoring hit that meets a threshold.
    """
    required_cols = ['precursor_mz', 'adduct', 'MS2_norm']
    if not all(col in input_df.columns for col in required_cols):
        print(f"Error: Input DataFrame must contain the columns: {required_cols}")
        return input_df.copy()

    all_top_hits = []
    
    result_cols = [
        'name', 'formula', 'chain_info', 'adduct', 'class', 'category', 'MS2_norm', 'source',
        'mass_diff_ppm', 'dot_product', 'weighted_dot_product',
        'entropy_similarity', 'unweighted_entropy_similarity'
    ]

    for original_index, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Database Searching"):
        if row[['precursor_mz', 'adduct', 'MS2_norm']].isnull().any():
            continue
        
        precursor_mz = row['precursor_mz']
        adduct = row['adduct']
        query_ms2_data = row['MS2_norm']
        
        query_MS2 = []
        try:
            if isinstance(query_ms2_data, str):
                query_MS2 = json.loads(query_ms2_data)
            elif isinstance(query_ms2_data, list):
                query_MS2 = query_ms2_data
        except (json.JSONDecodeError, TypeError):
            continue

        if not query_MS2:
            continue

        results_df = ms_db_search(
            db_path=db_path,
            precursor_mz=precursor_mz,
            adduct=adduct,
            MS1_tol=MS1_tol,
            query_MS2=query_MS2,
            MS2_tol=MS2_tol,
            is_ppm=is_ppm,
            method=method
        )

        if not results_df.empty:
            top_hit = results_df.iloc[0]
            # --- FIX ---
            # Only keep the hit if its score meets the specified threshold
            if top_hit[method] >= MS2_threshold:
                top_hit_copy = top_hit.copy()
                top_hit_copy['original_index'] = original_index
                all_top_hits.append(top_hit_copy)
    
    if not all_top_hits:
        annotated_df = input_df.copy()
        for col in result_cols:
            annotated_df[f"{col}_hit"] = pd.NA
        return annotated_df

    results_df = pd.DataFrame(all_top_hits)
    results_df = results_df.set_index('original_index')
    
    annotated_df = input_df.join(results_df[result_cols], rsuffix='_hit')
    
    return annotated_df



def main(args):
    """
    Main function to run the lipidomics data processing and annotation pipeline.
    """
    # 1. Validate input file path
    if not os.path.exists(args.csv_path):
        print(f"Error: The file '{args.csv_path}' was not found.")
        return

    print(f"Loading data from '{args.csv_path}'...")
    data = pd.read_csv(args.csv_path)
    
    if 'index' not in data.columns:
        data.insert(0, "index", [f"F{i:06d}" for i in range(1, len(data) + 1)])

    # 2. Normalize MS2 spectra
    print("Normalizing MS2 spectra...")
    results = data['MS2'].apply(normalize_ms2)
    data['MS2_norm'] = [item['normalized_spectrum'] for item in results]
    data['num_peaks'] = [item['num_peaks'] for item in results]
    print("Normalization complete.")

    # 3. Perform database search and annotation
    print("Starting database search...")
    annotated_data = batch_search_and_annotate(
        input_df=data,
        db_path=args.db_path,
        MS1_tol=args.MS1_tol,
        MS2_tol=args.MS2_tol,
        method=args.method,
        MS2_threshold=args.MS2_threshold,
        is_ppm=args.is_ppm
    )
    print("Database search finished.")

    # 4. Separate annotated lipids from "dark" lipids
    search_result = annotated_data[~annotated_data['name'].isna() & (annotated_data['name'] != '')].copy()
    annotated_index = search_result.index
    dark_lipid = data[~data.index.isin(annotated_index)].copy()

    # 5. Save the results
    # Create the result directory if it doesn't exist
    os.makedirs(args.result_path, exist_ok=True)
    print(f"Saving results to '{args.result_path}'...")

    annotated_output_path = os.path.join(args.result_path, 'db_matched_df.csv')
    dark_lipid_output_path = os.path.join(args.result_path, 'dark_lipid.csv')

    search_result.to_csv(annotated_output_path, index=False)
    dark_lipid.to_csv(dark_lipid_output_path, index=False)
    print(f"Annotated data saved to: {annotated_output_path}")
    print(f"Dark lipid data saved to: {dark_lipid_output_path}")

    # 6. Print summary statistics
    print("\n--- Summary ---")
    total_spectra = len(data)
    annotated_count = len(search_result)
    dark_count = len(dark_lipid)
    
    annotated_percent = (annotated_count / total_spectra * 100) if total_spectra > 0 else 0
    dark_percent = (dark_count / total_spectra * 100) if total_spectra > 0 else 0

    print(f"Total spectra: {total_spectra}")
    print(f"Annotated spectra: {annotated_count} ({annotated_percent:.2f}%)")
    print(f"Unknown spectra: {dark_count} ({dark_percent:.2f}%)")
    print("---------------\n")

def str2bool(v):
    """Helper function to convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process and annotate lipidomics MS2 data against a database.")

    # --- Required Argument ---
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file containing spectral data.')

    # --- Optional Arguments ---
    parser.add_argument('--result_path', type=str, default='.',
                        help='Path to the directory where results will be saved. Defaults to the current directory.')
    
    parser.add_argument('--db_path', type=str, default='lipid_plus.db',
                        help='Path to the lipid database file. Default: lipid_plus.db')
                        
    parser.add_argument('--MS1_tol', type=float, default=0.005,
                        help='MS1 tolerance for database search. Default: 0.005 Da')
                        
    parser.add_argument('--MS2_tol', type=float, default=0.01,
                        help='MS2 tolerance for database search. Default: 0.01 Da')
                        
    parser.add_argument('--method', type=str, default='weighted_dot_product',
                        help='Scoring method for spectral matching. Default: weighted_dot_product')
                        
    parser.add_argument('--MS2_threshold', type=float, default=0.7,
                        help='MS2 score threshold for a match to be considered valid. Default: 0.7')
                        
    parser.add_argument('--is_ppm', type=str2bool, default=False,
                        help='Specify if MS1 tolerance is in ppm (e.g., True or False). Default: False (Da).')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Run the main function
    main(args)
