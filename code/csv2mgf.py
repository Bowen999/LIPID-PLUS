import pandas as pd
import ast
import re
import argparse
import os
from tqdm import tqdm

def infer_charge_from_adduct(adduct_str: str) -> str:
    """
    Infers the charge from a standard adduct notation string.

    Args:
        adduct_str: The string containing the adduct information (e.g., '[M+H]+', '[M+2Na]2+', '[M-H]-').

    Returns:
        A string representing the charge in MGF format (e.g., '1+', '2+', '1-'), or an empty string if not found.
    """
    if not isinstance(adduct_str, str):
        return ""

    # This regex looks for a charge notation like 'z+' or 'z-' at the end of the string.
    # It handles cases with or without a number (e.g., '[M+H]+' or '[M+2H]2+') and
    # with or without closing brackets (e.g., '[M-H]-' or 'M-H-').
    match = re.search(r'(\d*)[+\-]](?:_precursor)?$|(\d*)[+\-]$', adduct_str.strip())
    
    if match:
        # Determine which capture group was successful
        charge_num_str = match.group(1) if match.group(1) else match.group(2)
        
        charge_num = int(charge_num_str) if charge_num_str else 1
        sign = '+' if '+' in match.group(0) else '-'
        return f"{abs(charge_num)}{sign}"
        
    return "" # Return empty string if no valid charge pattern is found

def convert_csv_to_mgf(input_csv_path: str, output_mgf_path: str):
    """
    Reads a CSV file with spectral data and converts it to an MGF file.

    Args:
        input_csv_path: The file path for the input CSV.
        output_mgf_path: The file path for the resulting MGF file.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_mgf_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Open the output file for writing
    with open(output_mgf_path, 'w') as mgf_file:
        # Iterate over each row in the DataFrame with a progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting spectra"):
            try:
                # --- 1. Extract required data from the row ---
                pepmass = row.get('precursor_mz')
                charge = row.get('charge')
                adduct = row.get('adduct')
                ms2_norm_str = row.get('MS2_norm')

                # --- 2. Validate essential data ---
                # Skip row if precursor m/z or MS2 peaks are missing
                if pd.isna(pepmass) or pd.isna(ms2_norm_str):
                    continue

                # --- 3. Determine the charge ---
                charge_str = ""
                # Use the 'charge' column if it's not empty
                if pd.notna(charge) and str(charge).strip() != '':
                    try:
                        # Format numeric charges correctly (e.g., 2 -> 2+, -1 -> 1-)
                        charge_val = int(float(charge))
                        charge_str = f"{abs(charge_val)}{'+' if charge_val >= 0 else '-'}"
                    except (ValueError, TypeError):
                        # If it's not a simple number, assume it's already formatted (e.g., "2+")
                        charge_str = str(charge)
                # If charge is empty, infer it from the 'adduct' column
                elif pd.notna(adduct):
                    charge_str = infer_charge_from_adduct(str(adduct))
                
                # If charge still couldn't be determined, skip this entry
                if not charge_str:
                    continue

                # --- 4. Parse the MS2 peak list ---
                # The MS2_norm is a string representation of a list of lists.
                # ast.literal_eval safely evaluates it.
                ms2_peaks = ast.literal_eval(ms2_norm_str)
                if not isinstance(ms2_peaks, list) or len(ms2_peaks) == 0:
                    continue

                # --- 5. Write the entry to the MGF file ---
                mgf_file.write("BEGIN IONS\n")
                mgf_file.write(f"PEPMASS={pepmass}\n")
                mgf_file.write(f"CHARGE={charge_str}\n")
                
                # Create a descriptive title from the 'index' column, falling back to the row number
                title = row.get('index', index)
                mgf_file.write(f"TITLE={title}\n")
                
                # Add other optional metadata if it exists
                if 'adduct' in row and pd.notna(row['adduct']):
                    mgf_file.write(f"ADDUCT={row['adduct']}\n")
                if 'instrument' in row and pd.notna(row['instrument']):
                    mgf_file.write(f"INSTRUMENT={row['instrument']}\n")
                if 'formula' in row and pd.notna(row['formula']):
                    mgf_file.write(f"SEQ={row['formula']}\n") # Using SEQ as a common field for formula

                # Write the m/z and intensity pairs for the peaks
                for mz, intensity in ms2_peaks:
                    mgf_file.write(f"{mz} {intensity}\n")

                mgf_file.write("END IONS\n\n")

            except (ValueError, SyntaxError) as e:
                # This catches errors from parsing MS2_norm or charge
                print(f"Skipping row {index} due to a data formatting error: {e}")
                continue
            except Exception as e:
                # Catch any other unexpected errors
                print(f"An unexpected error occurred at row {index}: {e}")
                continue

    print(f"\nConversion complete. File saved to '{output_mgf_path}'")
    
if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a CSV file containing spectral data to the MGF format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file."
    )
    
    args = parser.parse_args()

    # Define the fixed output path
    output_path = 'result/process_file/input.mgf'

    # Run the conversion function
    convert_csv_to_mgf(args.input_csv, output_path)