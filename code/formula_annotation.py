#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lipidomics Data Processing and Formula Annotation Script.

This script performs the following steps:
1. Converts a CSV file containing spectral data into an MGF file.
2. Uses Msbuddy to annotate molecular formulas for the spectra in the MGF file.
3. Processes and merges the Msbuddy results with the original data.
4. Applies class-specific rules to refine lipid formula assignments.
5. Saves the intermediate and final results to a specified directory.

Usage:
python formula_annotation.py /path/to/your/data.csv --result_path /path/to/results --ms1_tol 20 --ms2_tol 15
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from msbuddy import Msbuddy, MsbuddyConfig

# Assuming these are custom local modules
import csv2mgf
import class_rule as cr


def validate_paths(input_path, output_path):
    """
    Validate input and output paths before processing.
    
    :param input_path: Path to the input CSV file.
    :param output_path: Directory to save the output files.
    :return: True if validation passes, False otherwise.
    """
    # Check if input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return False
    
    if not input_file.is_file():
        print(f"Error: Input path is not a file: {input_path}")
        return False
    
    # Check if input file has .csv extension
    if input_file.suffix.lower() != '.csv':
        print(f"Warning: Input file does not have .csv extension: {input_path}")
    
    # Check/create output directory
    output_dir = Path(output_path)
    
    # If output path exists and is a file, not a directory
    if output_dir.exists() and not output_dir.is_dir():
        print(f"Error: Output path exists but is not a directory: {output_path}")
        return False
    
    # Try to create output directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory validated/created: {output_path}")
    except PermissionError:
        print(f"Error: Permission denied when creating output directory: {output_path}")
        return False
    except Exception as e:
        print(f"Error: Could not create output directory: {e}")
        return False
    
    return True


def process_chunk(chunk_data, chunk_id, result_path, ms1_tol, ms2_tol):
    """
    Process a single chunk of data through the complete Msbuddy pipeline.
    
    :param chunk_data: DataFrame containing a chunk of the original data.
    :param chunk_id: Identifier for this chunk.
    :param result_path: Directory to save temporary files.
    :param ms1_tol: MS1 tolerance in ppm.
    :param ms2_tol: MS2 tolerance in ppm.
    :return: DataFrame with Msbuddy results for this chunk.
    """
    try:
        print(f"[Chunk {chunk_id}] Starting processing with {len(chunk_data)} items...")
        
        # --- Filter Data based on Adduct ---
        # If adduct is [M-H]-, we skip annotation and use empty directly.
        if 'adduct' in chunk_data.columns:
            # We want to process rows where adduct is NOT [M-H]-
            # Normalize check just in case (strip whitespace)
            mask = chunk_data['adduct'].astype(str).str.strip() != '[M-H]-'
            data_to_process = chunk_data[mask].copy()
            skipped_data = chunk_data[~mask].copy()
            
            if len(skipped_data) > 0:
                print(f"[Chunk {chunk_id}] Skipping {len(skipped_data)} items with [M-H]- adduct.")
        else:
            data_to_process = chunk_data.copy()
            skipped_data = pd.DataFrame()
            
        processed_data = []

        # --- Process Rows that need Annotation ---
        if len(data_to_process) > 0:
            # Create temporary paths for this chunk
            chunk_csv_path = os.path.join(result_path, f"chunk_{chunk_id}.csv")
            chunk_mgf_path = os.path.join(result_path, f"chunk_{chunk_id}.mgf")
            
            # Save chunk to CSV
            data_to_process.to_csv(chunk_csv_path, index=False)
            
            # Convert chunk CSV to MGF
            print(f"[Chunk {chunk_id}] Converting to MGF format...")
            csv2mgf.convert_csv_to_mgf(chunk_csv_path, chunk_mgf_path)
            
            # Run Msbuddy on this chunk
            print(f"[Chunk {chunk_id}] Running Msbuddy annotation...")
            msb_config = MsbuddyConfig(
                ms_instr=None,
                ppm=True,
                ms1_tol=ms1_tol,
                ms2_tol=ms2_tol,
                halogen=False
            )
            
            msb_engine = Msbuddy(msb_config)
            msb_engine.load_mgf(chunk_mgf_path)
            msb_engine.annotate_formula()
            buddy_summary = msb_engine.get_summary()
            
            # Process results
            print(f"[Chunk {chunk_id}] Processing results...")
            for item in buddy_summary:
                processed_item = {
                    'index': str(item.get('identifier')),
                    'mz': item.get('mz')
                }
                for i in range(1, 6):
                    rank_key = f'formula_rank_{i}'
                    processed_item[rank_key] = item.get(rank_key)
                processed_data.append(processed_item)
            
            # Clean up temporary files
            if os.path.exists(chunk_csv_path):
                os.remove(chunk_csv_path)
            if os.path.exists(chunk_mgf_path):
                os.remove(chunk_mgf_path)
        
        # --- Handle Skipped Rows (Directly Assign Empty) ---
        if len(skipped_data) > 0:
            for _, row in skipped_data.iterrows():
                processed_item = {
                    'index': str(row['index']),
                    'mz': row.get('mz')
                }
                # Assign empty (None) for all formula ranks
                for i in range(1, 6):
                    processed_item[f'formula_rank_{i}'] = None
                processed_data.append(processed_item)

        print(f"[Chunk {chunk_id}] Completed successfully!")
        return pd.DataFrame(processed_data)
        
    except Exception as e:
        print(f"[Chunk {chunk_id}] Error during processing: {e}")
        return pd.DataFrame()


def process_lipid_data(csv_path, result_path, ms1_tol, ms2_tol, n_cpus=None):
    """
    Main function to run the lipid data processing pipeline with parallel processing.

    :param csv_path: Path to the input CSV file.
    :param result_path: Directory to save the output files.
    :param ms1_tol: MS1 tolerance in ppm.
    :param ms2_tol: MS2 tolerance in ppm.
    :param n_cpus: Number of CPUs to use for parallel processing. If None, uses all available CPUs.
    """
    # Determine number of CPUs to use
    if n_cpus is None:
        n_cpus = cpu_count()
    else:
        n_cpus = min(n_cpus, cpu_count())
    
    print(f"Using {n_cpus} CPU cores for parallel processing.")
    
    # --- 1. Setup Paths ---
    print(f"Setting up output directory at: {result_path}")
    os.makedirs(result_path, exist_ok=True)
    buddy_result_path = os.path.join(result_path, "buddy_result.csv")
    final_result_path = os.path.join(result_path, "formula_result.csv")

    # --- 2. Load and Split Data ---
    print(f"Loading data from '{csv_path}'...")
    try:
        original_data_df = pd.read_csv(csv_path)
        total_rows = len(original_data_df)
        print(f"Loaded {total_rows} rows from CSV file.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Split dataframe into chunks
    if total_rows < n_cpus:
        # If fewer rows than CPUs, use fewer chunks
        n_cpus = max(1, total_rows)
        print(f"Adjusting to {n_cpus} chunks (fewer rows than requested CPUs).")
    
    chunk_size = max(1, total_rows // n_cpus)
    chunks = []
    
    for i in range(n_cpus):
        start_idx = i * chunk_size
        if i == n_cpus - 1:
            # Last chunk gets all remaining rows
            end_idx = total_rows
        else:
            end_idx = start_idx + chunk_size
        
        chunk = original_data_df.iloc[start_idx:end_idx].copy()
        chunks.append((chunk, i, result_path, ms1_tol, ms2_tol))
    
    print(f"Split data into {len(chunks)} chunks for parallel processing.")
    for i, (chunk, _, _, _, _) in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} rows")

    # --- 3. Process Chunks in Parallel ---
    print("\nStarting parallel processing of chunks...")
    
    if n_cpus > 1:
        # Use multiprocessing for parallel execution
        with Pool(processes=n_cpus) as pool:
            results = pool.starmap(process_chunk, chunks)
    else:
        # Sequential processing if only 1 CPU
        print("Processing sequentially (single CPU)...")
        results = [process_chunk(*chunk_args) for chunk_args in chunks]
    
    # --- 4. Combine Results ---
    print("\nCombining results from all chunks...")
    buddy_result_df = pd.concat(results, ignore_index=True)
    
    if len(buddy_result_df) == 0:
        print("Error: No results to save. Processing may have failed.")
        return
    
    print(f"Combined results: {len(buddy_result_df)} rows")
    buddy_result_df.to_csv(buddy_result_path, index=False)
    print(f"Msbuddy results saved to '{buddy_result_path}'.")

    # --- 5. Merge, Final Processing, and Save ---
    print("Merging Msbuddy results with original data...")
    try:
        merged_df = original_data_df.merge(buddy_result_df, on='index', how='left')
        
        print("Applying custom lipid formula rules...")
        final_result_df = cr.process_lipid_formulas(merged_df)
        
        # Rename the 'formula_rank_1' column to 'formula' for the final output
        if 'formula_rank_1' in final_result_df.columns:
            print("Renaming 'formula_rank_1' column to 'predicted_formula'.")
            final_result_df.rename(columns={'formula_rank_1': 'predicted_formula'}, inplace=True)
        
        final_result_df.to_csv(final_result_path, index=False)
        print(f"Final processed data saved to '{final_result_path}'.")
        print("\n" + "="*60)
        print("Script finished successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"An error occurred during the final processing step: {e}")


def main():
    """
    Argument parser and script entry point.
    """
    parser = argparse.ArgumentParser(
        description="Lipidomics Data Processing and Formula Annotation Script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_path", type=str, help="Path to the input CSV file containing spectral data.")
    parser.add_argument("--output_path", type=str, default="result/process_file", help="Folder path to save all output files (MGF, intermediate, and final results).")
    parser.add_argument("--ms1_tol", type=float, default=15.0, help="MS1 tolerance in ppm for formula annotation.")
    parser.add_argument("--ms2_tol", type=float, default=20.0, help="MS2 tolerance in ppm for formula annotation.")
    parser.add_argument("--n_cpus", type=int, default=None, help="Number of CPU cores to use for parallel processing. If not specified, uses all available cores.")
    
    args = parser.parse_args()
    
    # Validate paths before processing
    print("=" * 60)
    print("Step 1: Validating input and output paths...")
    print("=" * 60)
    if not validate_paths(args.input_path, args.output_path):
        print("\nPath validation failed. Exiting...")
        sys.exit(1)
    
    print("\nPath validation successful. Starting processing...\n")
    
    # Process the data
    process_lipid_data(args.input_path, args.output_path, args.ms1_tol, args.ms2_tol, args.n_cpus)


if __name__ == "__main__":
    main()