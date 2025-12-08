"""
Lipidomics Annotation Pipeline - All-in-One Script

This pipeline performs comprehensive lipidomics annotation through the following steps:
0. MS2 Processing - Normalizes and unfolds MS2 data
1. Database Search - Annotate known lipids using spectral database matching
2. Adduct Prediction - Predict adduct types for unknown lipids (skipped if all features identified by DB)
3. Class Prediction - Predict lipid classes using hybrid rule-based + ML approach (skipped if all features identified by DB)
4. Chain Composition Prediction - Predict detailed fatty acid chain compositions (skipped if all features identified by DB)

Usage:
    python run.py input.csv --result_path results/ --db_path lipid_plus.db
    python run.py input.csv --result_path results/ --n_jobs 8  # Use 8 parallel workers for PLSF
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import pandas as pd


class LipidAnnotationPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, input_path, result_path, db_path, adduct_model, class_model, plsf_model, 
                 MS1_tol=0.005, MS2_tol=0.01, MS2_threshold=0.7, 
                 ms1_tol_ppm=10, ms2_tol_ppm=20, n_jobs=4):
        """
        Initialize the pipeline
        
        Args:
            n_jobs: Number of parallel workers for PLSF prediction (default: 4)
        """
        self.input_path = Path(input_path)
        self.result_path = Path(result_path)
        self.db_path = Path(db_path)
        self.adduct_model = Path(adduct_model)
        self.class_model = Path(class_model)
        self.plsf_model = Path(plsf_model)
        
        self.MS1_tol = MS1_tol
        self.MS2_tol = MS2_tol
        self.MS2_threshold = MS2_threshold
        self.ms1_tol_ppm = ms1_tol_ppm
        self.ms2_tol_ppm = ms2_tol_ppm
        self.n_jobs = n_jobs
        
        # Create result directory
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        # Define intermediate file paths (these all go into the main result_path initially)
        self.processed_input_path = self.result_path / "processed_feature_table.csv"
        self.annotated_path = self.result_path / "db_matched_df.csv"
        self.dark_lipid_path = self.result_path / "dark_lipid.csv"
        self.adduct_pred_path = self.result_path / "adduct_predictions.csv"
        self.class_pred_path = self.result_path / "class_predictions.csv"
        self.final_output_path = self.result_path / "final_annotations.csv"
        # The final, combined file name
        self.final_combined_name = "identification_result.csv"
        
        # Statistics tracking
        self.stats = {
            'total_features': 0,
            'db_matched': 0,
            'ml_tier1': 0,  # confidence > 0.8
            'ml_tier2': 0,  # confidence <= 0.8
            'unidentified': 0
        }
    
    def print_header(self, text):
        """Print formatted section header"""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)
    
    def run_command(self, cmd, step_name):
        """Run a command and stream output to terminal directly"""
        print(f"\n→ Running: {' '.join(cmd)}")
        try:
            # check=True raises CalledProcessError on non-zero exit code
            subprocess.run(cmd, check=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error in {step_name}:")
            print(f"  Process exited with error code {e.returncode}")
            return False
        except FileNotFoundError:
            print(f"✗ Error: Script not found. Make sure the script exists in code/ directory.")
            return False

    def step0_process_ms2(self):
        """Step 0: Pre-process MS2 data (Unfold and Normalize)"""
        self.print_header("STEP 0: PROCESS MS2 DATA")

        # Ensure the script assumes process_ms2.py is in the code/ folder
        cmd = [
            "python", "code/process_ms2.py",
            str(self.input_path),
            "--output_path", str(self.processed_input_path),
            "--neutral_loss",     # Enable neutral loss logic
            "--decimal_point", "0" # Default decimal point
        ]
        
        success = self.run_command(cmd, "MS2 Processing")
        
        if success and self.processed_input_path.exists():
            # Count total features
            df = pd.read_csv(self.processed_input_path)
            self.stats['total_features'] = len(df)
            print(f"\n✓ Step 0 complete: MS2 data processed")
            print(f"  Total features: {self.stats['total_features']}")
            print(f"  Saved to: {self.processed_input_path.name}")
            return True
        else:
            print("\n✗ Error: Processed file was not created.")
            return False
    
    def step1_database_search(self):
        """Step 1: Database search for known lipids"""
        self.print_header("STEP 1: DATABASE SEARCH")
        
        if not self.db_path.exists():
            print(f"⚠ Warning: Database file not found: {self.db_path}")
            print("  Skipping database search step...")
            return False
        
        # NOTE: Using processed_input_path instead of raw input_path
        cmd = [
            "python", "code/db_search.py",
            str(self.processed_input_path),
            "--result_path", str(self.result_path),
            "--db_path", str(self.db_path),
            "--MS1_tol", str(self.MS1_tol),
            "--MS2_tol", str(self.MS2_tol),
            "--MS2_threshold", str(self.MS2_threshold),
            "--is_ppm", "False"
        ]
        
        success = self.run_command(cmd, "Database Search")
        
        if success and self.dark_lipid_path.exists():
            # Check how many features were matched and how many remain unknown
            if self.annotated_path.exists():
                db_df = pd.read_csv(self.annotated_path)
                self.stats['db_matched'] = len(db_df)
            
            dark_df = pd.read_csv(self.dark_lipid_path)
            num_dark = len(dark_df)
            
            print(f"\n✓ Step 1 complete:")
            print(f"  Database matched: {self.stats['db_matched']} features")
            print(f"  Unknown lipids: {num_dark} features")
            
            # Check if all features are identified
            if num_dark == 0:
                print(f"\n🎉 All {self.stats['total_features']} features identified by database search!")
                print("  Skipping ML prediction steps (2-4)...")
                return "all_identified"
            
            return True
        else:
            print("\n⚠ Database search completed but no dark lipids file found")
            return False
    
    def step2_adduct_prediction(self):
        """Step 2: Predict adducts for unknown lipids"""
        self.print_header("STEP 2: ADDUCT PREDICTION")
        
        # Determine input file
        if self.dark_lipid_path.exists():
            input_file = self.dark_lipid_path
        else:
            # Fallback to processed file if DB search was skipped or yielded nothing
            input_file = self.processed_input_path
        
        if not self.adduct_model.exists():
            print(f"✗ Error: Adduct model not found: {self.adduct_model}")
            return False
        
        cmd = [
            "python", "code/adduct_predict.py",
            str(input_file),
            str(self.adduct_model),
            "--output_path", str(self.adduct_pred_path)
        ]
        
        success = self.run_command(cmd, "Adduct Prediction")
        
        if success:
            print("\n✓ Step 2 complete: Adducts predicted")
            return True
        return False
    
    def step3_class_prediction(self):
        """Step 3: Predict lipid classes"""
        self.print_header("STEP 3: CLASS PREDICTION")
        
        if not self.adduct_pred_path.exists():
            print(f"✗ Error: Adduct predictions file not found: {self.adduct_pred_path}")
            return False
        
        if not self.class_model.exists():
            print(f"✗ Error: Class model not found: {self.class_model}")
            return False
        
        cmd = [
            "python", "code/class_predict.py",
            str(self.adduct_pred_path),
            str(self.class_model),
            "--output_path", str(self.class_pred_path),
            "--ms1_tol", str(self.ms1_tol_ppm),
            "--ms2_tol", str(self.ms2_tol_ppm)
        ]
        
        success = self.run_command(cmd, "Class Prediction")
        
        if success:
            print("\n✓ Step 3 complete: Classes predicted")
            return True
        return False
    
    def step4_plsf_prediction(self):
        """Step 4: Predict lipid chain compositions"""
        self.print_header("STEP 4: CHAIN COMPOSITION PREDICTION (PLSF)")
        
        if not self.class_pred_path.exists():
            print(f"✗ Error: Class predictions file not found: {self.class_pred_path}")
            return False
        
        if not self.plsf_model.exists():
            print(f"✗ Error: PLSF model not found: {self.plsf_model}")
            return False
        
        cmd = [
            "python", "code/plsf_predict.py",
            str(self.class_pred_path),
            str(self.plsf_model),
            "--output_path", str(self.final_output_path),
            "--n_jobs", str(self.n_jobs)
        ]
        
        success = self.run_command(cmd, "PLSF Prediction")
        
        if success:
            print("\n✓ Step 4 complete: Chain compositions predicted")
            return True
        return False
    
    def post_process_results(self):
        """Post-process identification results according to user rules:
        1. Remove columns starting with 'mz_' and specific list of other columns.
        2. If 'name' starts with 'Adduct', clear 'name' and prediction columns.
        3. If 'pred_confidence' > 0.8 AND 'chain_info' is empty, use 'name' value.
        """
        self.print_header("POST-PROCESSING RESULTS")
        
        try:
            identification_result_path = self.result_path / self.final_combined_name
            
            # Check if file exists
            if not identification_result_path.exists():
                print("⚠ No identification result file found to post-process")
                return False, None
            
            # Load the results
            df = pd.read_csv(identification_result_path)
            original_count = len(df)
            print(f"Loaded {original_count} rows from {self.final_combined_name}")
            
            # Rule 1: Remove columns starting with 'mz_' and specific columns
            columns_to_remove = [col for col in df.columns if col.startswith('mz_')]
            columns_to_remove.extend([
                'formula_1', 'formula_2', 'formula',
                'adduct_1', 'adduct_2', 'adduct',
                'collision_energy', 'instrument_type', 'ion_mode', 'precursor_type'
            ])
            
            # Only remove columns that actually exist
            columns_to_remove = [col for col in columns_to_remove if col in df.columns]
            
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
                print(f"Removed {len(columns_to_remove)} columns (including mz_* columns)")
            
            # Rule 2: If 'name' starts with 'Adduct', clear name and prediction columns
            if 'name' in df.columns:
                adduct_mask = df['name'].astype(str).str.startswith('Adduct', na=False)
                adduct_count = adduct_mask.sum()
                
                if adduct_count > 0:
                    # Clear 'name' column
                    df.loc[adduct_mask, 'name'] = ''
                    
                    # Clear prediction columns if they exist
                    pred_cols = ['pred_adduct', 'pred_class', 'pred_confidence', 'chain_info']
                    for col in pred_cols:
                        if col in df.columns:
                            df.loc[adduct_mask, col] = ''
                    
                    print(f"Cleared {adduct_count} rows where name started with 'Adduct'")
            
            # Rule 3: If pred_confidence > 0.8 AND chain_info is empty, use 'name' value
            if all(col in df.columns for col in ['pred_confidence', 'chain_info', 'name']):
                # Create mask for rows where:
                # - pred_confidence > 0.8
                # - chain_info is empty (NaN or empty string)
                # - name is not empty
                
                high_conf_mask = df['pred_confidence'] > 0.8
                empty_chain_mask = df['chain_info'].isna() | (df['chain_info'].astype(str).str.strip() == '')
                has_name_mask = df['name'].notna() & (df['name'].astype(str).str.strip() != '')
                
                rule3_mask = high_conf_mask & empty_chain_mask & has_name_mask
                rule3_count = rule3_mask.sum()
                
                if rule3_count > 0:
                    # For these rows, keep the existing 'name' value (no change needed)
                    print(f"Preserved {rule3_count} high-confidence predictions (>0.8) with empty chain_info")
            
            # Save the post-processed results
            df.to_csv(identification_result_path, index=False)
            print(f"\n✓ Post-processing complete")
            print(f"  Final file: {identification_result_path}")
            print(f"  Total rows: {len(df)}")
            
            return True, df
            
        except Exception as e:
            print(f"✗ Error during post-processing: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def merge_with_annotated(self):
        """Merge ML predictions with database annotations"""
        self.print_header("MERGING RESULTS")
        
        # Check if we have both files
        has_db = self.annotated_path.exists()
        has_pred = self.final_output_path.exists()
        
        if not has_db and not has_pred:
            print("⚠ No results to merge")
            return False
        
        final_path = self.result_path / self.final_combined_name
        
        if has_db and not has_pred:
            # Only database results
            print("→ Only database matches found, copying results...")
            import shutil
            shutil.copy(self.annotated_path, final_path)
            print(f"✓ Results saved to {self.final_combined_name}")
            return True
            
        elif not has_db and has_pred:
            # Only predictions
            print("→ Only ML predictions found, copying results...")
            import shutil
            shutil.copy(self.final_output_path, final_path)
            print(f"✓ Results saved to {self.final_combined_name}")
            return True
            
        else:
            # Both exist - merge them
            print("→ Merging database matches and ML predictions...")
            
            db_df = pd.read_csv(self.annotated_path)
            pred_df = pd.read_csv(self.final_output_path)
            
            # Concatenate
            combined_df = pd.concat([db_df, pred_df], ignore_index=True)
            
            # Save
            combined_df.to_csv(final_path, index=False)
            print(f"✓ Merged {len(db_df)} database matches + {len(pred_df)} predictions")
            print(f"  Total: {len(combined_df)} features")
            print(f"  Saved to: {self.final_combined_name}")
            
            return True

    def cleanup_intermediate_files(self):
        """Clean up intermediate files, keeping only final results"""
        self.print_header("CLEANING UP INTERMEDIATE FILES")
        
        intermediate_files = [
            self.processed_input_path,
            self.dark_lipid_path,
            self.adduct_pred_path,
            self.class_pred_path,
            self.final_output_path
        ]
        
        removed_count = 0
        for file_path in intermediate_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_count += 1
                    print(f"  Removed: {file_path.name}")
                except Exception as e:
                    print(f"  ⚠ Could not remove {file_path.name}: {e}")
        
        if removed_count > 0:
            print(f"\n✓ Cleaned up {removed_count} intermediate files")
        else:
            print("  No intermediate files to clean up")

    def calculate_statistics(self, final_df):
        """Calculate detailed statistics from the final results"""
        
        # Count database matches (rows without pred_confidence)
        if 'pred_confidence' in final_df.columns:
            db_mask = final_df['pred_confidence'].isna()
            self.stats['db_matched'] = db_mask.sum()
            
            # ML predictions with confidence
            ml_mask = final_df['pred_confidence'].notna()
            ml_df = final_df[ml_mask]
            
            # Tier 1: confidence > 0.8
            tier1_mask = ml_df['pred_confidence'] > 0.8
            self.stats['ml_tier1'] = tier1_mask.sum()
            
            # Tier 2: confidence <= 0.8
            tier2_mask = ml_df['pred_confidence'] <= 0.8
            self.stats['ml_tier2'] = tier2_mask.sum()
        else:
            # All are database matches if no pred_confidence column
            self.stats['db_matched'] = len(final_df)
            self.stats['ml_tier1'] = 0
            self.stats['ml_tier2'] = 0
        
        # Count unidentified (empty or missing name)
        if 'name' in final_df.columns:
            unidentified_mask = final_df['name'].isna() | (final_df['name'].astype(str).str.strip() == '')
            self.stats['unidentified'] = unidentified_mask.sum()
        else:
            self.stats['unidentified'] = 0

    def print_comprehensive_stats(self, final_df):
        """Print comprehensive statistics about the annotation results"""
        self.print_header("ANNOTATION STATISTICS")
        
        # Calculate statistics
        self.calculate_statistics(final_df)
        
        print(f"\n📊 FEATURE SUMMARY:")
        print(f"   Total features: {self.stats['total_features']}")
        print(f"\n📋 IDENTIFICATION BREAKDOWN:")
        print(f"   Database matched: {self.stats['db_matched']} ({self.stats['db_matched']/self.stats['total_features']*100:.1f}%)")
        print(f"   ML predictions:")
        print(f"     • Tier 1 (confidence > 0.8):  {self.stats['ml_tier1']} ({self.stats['ml_tier1']/self.stats['total_features']*100:.1f}%)")
        print(f"     • Tier 2 (confidence ≤ 0.8):  {self.stats['ml_tier2']} ({self.stats['ml_tier2']/self.stats['total_features']*100:.1f}%)")
        print(f"   Unidentified: {self.stats['unidentified']} ({self.stats['unidentified']/self.stats['total_features']*100:.1f}%)")
        
        # Verification
        identified = self.stats['db_matched'] + self.stats['ml_tier1'] + self.stats['ml_tier2'] + self.stats['unidentified']
        print(f"\n✓ Total accounted: {identified}/{self.stats['total_features']}")
        
        # Additional details if available
        if 'name' in final_df.columns:
            named_count = (~final_df['name'].isna() & (final_df['name'].astype(str).str.strip() != '')).sum()
            print(f"\n📝 NAMING SUMMARY:")
            print(f"   Features with names: {named_count}")
        
        print("\n" + "=" * 80)

    def run(self):
        """Run the complete pipeline"""
        print("\n" + "=" * 80)
        print("  LIPIDOMICS ANNOTATION PIPELINE")
        print("=" * 80)
        print(f"\nInput file: {self.input_path}")
        print(f"Result directory: {self.result_path}")
        print(f"Database: {self.db_path}")
        print(f"Parallel workers (PLSF): {self.n_jobs}")
        
        # Step 0: Process MS2 (New Step)
        if not self.step0_process_ms2():
            print("\n✗ Pipeline failed at Step 0: MS2 Processing")
            return False

        # Step 1: Database search (optional)
        db_success = self.step1_database_search()
        
        # Check if all features were identified by database
        if db_success == "all_identified":
            # Skip ML prediction steps
            # Just copy the annotated results to final output
            import shutil
            final_path = self.result_path / self.final_combined_name
            shutil.copy(self.annotated_path, final_path)
            
            # Post-process results (apply rules)
            success, final_df = self.post_process_results()
            
            # Print comprehensive statistics
            if success and final_df is not None:
                self.print_comprehensive_stats(final_df)
            
            # Clean up intermediate files
            self.cleanup_intermediate_files()
            
            # Final summary
            self.print_header("PIPELINE COMPLETE")
            print(f"\nAll features identified by database search - ML prediction skipped")
            print(f"Results saved to: {self.result_path}")
            print(f"Final output: {self.result_path / self.final_combined_name}")
            print("\n✓ Success!\n")
            
            return True
        
        # If not all features identified by DB, proceed with ML prediction
        if not db_success:
            print("\n→ Proceeding with full prediction pipeline on all data")
        
        # Step 2: Adduct prediction
        if not self.step2_adduct_prediction():
            print("\n✗ Pipeline failed at Step 2: Adduct Prediction")
            return False
        
        # Step 3: Class prediction
        if not self.step3_class_prediction():
            print("\n✗ Pipeline failed at Step 3: Class Prediction")
            return False
        
        # Step 4: PLSF prediction
        if not self.step4_plsf_prediction():
            print("\n✗ Pipeline failed at Step 4: PLSF Prediction")
            return False
        
        # Merge results
        self.merge_with_annotated()
        
        # Post-process results (apply rules)
        success, final_df = self.post_process_results()
        
        # Print comprehensive statistics
        if success and final_df is not None:
            self.print_comprehensive_stats(final_df)
            
        # Clean up intermediate files (NEW STEP)
        self.cleanup_intermediate_files()
        
        # Final summary
        self.print_header("PIPELINE COMPLETE")
        print(f"\nAll results saved to: {self.result_path}")
        print(f"Final output: {self.result_path / self.final_combined_name}")
        print("\n✓ Success!\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete Lipidomics Annotation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python run.py feature_df.csv --result_path results/
  python run.py feature_df.csv --result_path results/ --n_jobs 8

Directory Structure Requirement:
  your_project/
  ├── run.py                  # This script
  ├── code/                   # Helper scripts
  │   ├── process_ms2.py      # <--- MUST BE HERE
  │   ├── db_search.py
  │   ├── adduct_predict.py
  │   ├── class_predict.py
  │   └── plsf_predict.py
  ├── model/                  # Models
  └── dataset/                # Database
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input CSV file'
    )
    
    # Model paths (with defaults)
    parser.add_argument(
        '--adduct_model',
        type=str,
        default='model/adduct.joblib',
        help='Path to trained adduct prediction model'
    )
    
    parser.add_argument(
        '--class_model',
        type=str,
        default='model/class.joblib',
        help='Path to trained class prediction model'
    )
    
    parser.add_argument(
        '--plsf_model',
        type=str,
        default='model/plsf.joblib',
        help='Path to trained PLSF model'
    )
    
    # Optional arguments
    parser.add_argument(
        '--result_path',
        type=str,
        default='results',
        help='Directory to save all results'
    )
    
    parser.add_argument(
        '--db_path',
        type=str,
        default='dataset/lipid_plus.db',
        help='Path to lipid database file'
    )
    
    # Database search parameters
    parser.add_argument(
        '--MS1_tol',
        type=float,
        default=0.005,
        help='MS1 tolerance for database search in Da'
    )
    
    parser.add_argument(
        '--MS2_tol',
        type=float,
        default=0.01,
        help='MS2 tolerance for database search in Da'
    )
    
    parser.add_argument(
        '--MS2_threshold',
        type=float,
        default=0.7,
        help='Minimum MS2 similarity score for database match'
    )
    
    # Class prediction parameters
    parser.add_argument(
        '--ms1_tol_ppm',
        type=float,
        default=10.0,
        help='MS1 tolerance for class prediction in ppm'
    )
    
    parser.add_argument(
        '--ms2_tol_ppm',
        type=float,
        default=20.0,
        help='MS2 tolerance for class prediction in ppm'
    )
    
    # Parallel processing parameter
    parser.add_argument(
        '-j', '--n_jobs',
        type=int,
        default=4,
        help='Number of parallel workers for PLSF prediction (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_path).exists():
        print(f"✗ Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    # Create and run pipeline
    try:
        pipeline = LipidAnnotationPipeline(
            input_path=args.input_path,
            result_path=args.result_path,
            db_path=args.db_path,
            adduct_model=args.adduct_model,
            class_model=args.class_model,
            plsf_model=args.plsf_model,
            MS1_tol=args.MS1_tol,
            MS2_tol=args.MS2_tol,
            MS2_threshold=args.MS2_threshold,
            ms1_tol_ppm=args.ms1_tol_ppm,
            ms2_tol_ppm=args.ms2_tol_ppm,
            n_jobs=args.n_jobs
        )
        
        success = pipeline.run()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n")
    print("  Welcome to LIPID+")
    print("")
    print(r"""
    ██╗     ██╗██████╗ ██╗██████╗    ██╗
    ██║     ██║██╔══██╗██║██╔══██╗   ██║
    ██║     ██║██████╔╝██║██║  ██║████████╗
    ██║     ██║██╔═══╝ ██║██║  ██║╚══██╔══╝
    ███████╗██║██║     ██║██████╔╝   ██║
    ╚══════╝╚═╝╚═╝     ╚═╝╚═════╝    ╚═╝
    """)
    print("")
    main()
