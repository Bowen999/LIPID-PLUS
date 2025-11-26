"""
Lipidomics Annotation Pipeline - All-in-One Script

This pipeline performs comprehensive lipidomics annotation through the following steps:
0. MS2 Processing - Normalizes and unfolds MS2 data
1. Database Search - Annotate known lipids using spectral database matching
2. Adduct Prediction - Predict adduct types for unknown lipids
3. Class Prediction - Predict lipid classes using hybrid rule-based + ML approach
4. Chain Composition Prediction - Predict detailed fatty acid chain compositions

Usage:
    python run.py input.csv --result_path results/ --db_path lipid_plus.db
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
                 ms1_tol_ppm=10, ms2_tol_ppm=20):
        """
        Initialize the pipeline
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
            print(f"\n✓ Step 0 complete: MS2 data processed and saved to {self.processed_input_path.name}")
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
            # Check how many dark lipids we have
            dark_df = pd.read_csv(self.dark_lipid_path)
            print(f"\n✓ Step 1 complete: {len(dark_df)} unknown lipids to process")
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
            "python", "code/predict_plsf.py",
            str(self.plsf_model),
            str(self.class_pred_path),
            "--output_path", str(self.final_output_path)
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
            print(f"✓ Loaded {len(df)} rows from {identification_result_path.name}")
            
            # --- Rule 1: Remove columns ---
            # 1a. Columns starting with 'mz_'
            cols_to_drop = [col for col in df.columns if col.lower().startswith('mz_')]
            
            # 1b. Specific columns to drop
            specific_drop_list = [
                'classes_mz', 
                'classes_ms2', 
                'predicted_adduct', 
                'adduct_confidence', 
                'prediction_source', 
                'predicted_class', 
                'class_confidence'
            ]
            
            # Add specific columns if they exist in df
            cols_to_drop.extend([col for col in specific_drop_list if col in df.columns])
            
            # Deduplicate list
            cols_to_drop = list(set(cols_to_drop))
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                print(f"✓ Rule 1 applied: Dropped {len(cols_to_drop)} columns (mz_*, classes_*, etc.)")

            # --- Rule 2: If 'name' starts with "Adduct", clear 'name' and predicted columns ---
            # Identify columns related to predictions to be cleared
            # (Note: many specific predicted columns were dropped above, but plsf_* and others might remain)
            prediction_columns = [col for col in df.columns if 'pred' in col.lower() or 'plsf' in col.lower()]
            
            if 'name' in df.columns:
                # Mask rows where 'name' starts with "Adduct"
                adduct_mask = df['name'].astype(str).str.startswith('Adduct', na=False)
                num_adduct_rows = adduct_mask.sum()
                
                if num_adduct_rows > 0:
                    # Set 'name' to empty string
                    df.loc[adduct_mask, 'name'] = ''
                    
                    # Set prediction columns to empty string
                    for col in prediction_columns:
                        if col in df.columns:
                            df.loc[adduct_mask, col] = ''
                    print(f"✓ Rule 2 applied: Cleared name/predictions for {num_adduct_rows} 'Adduct' rows")
            
            # --- Rule 3: If pred_confidence > 0.8 AND chain_info is empty, use 'name' value ---
            if 'pred_confidence' in df.columns and 'name' in df.columns:
                # Ensure chain_info column exists
                if 'chain_info' not in df.columns:
                    df['chain_info'] = ''
                
                # Check for empty chain_info (NaN or empty string)
                chain_empty_mask = df['chain_info'].isna() | (df['chain_info'].astype(str).str.strip() == '')
                
                # Combined mask: High confidence AND empty chain_info
                high_conf_update_mask = (df['pred_confidence'] > 0.8) & chain_empty_mask
                
                num_updated = high_conf_update_mask.sum()
                
                if num_updated > 0:
                    # Update chain_info with the 'name' value
                    df.loc[high_conf_update_mask, 'chain_info'] = df.loc[high_conf_update_mask, 'name']
                    print(f"✓ Rule 3 applied: Updated chain_info for {num_updated} rows (conf > 0.8 & empty chain_info)")
            
            # Save the post-processed results
            df.to_csv(identification_result_path, index=False)
            print(f"✓ Post-processed results saved to: {identification_result_path.name}")
            
            return True, df
            
        except Exception as e:
            print(f"✗ Error in post-processing: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    # def merge_with_annotated(self):
    #     """Merge final predictions with database-annotated lipids"""
    #     self.print_header("MERGING AND FINALIZING RESULTS")
        
    #     try:
    #         output_path = self.result_path / self.final_combined_name
            
    #         # Load final predictions
    #         if self.final_output_path.exists():
    #             predictions_df = pd.read_csv(self.final_output_path)
    #             print(f"✓ Loaded {len(predictions_df)} predicted lipids")
    #         else:
    #             predictions_df = pd.DataFrame()
    #             print("⚠ No predictions file found")
            
    #         # Load database annotations
    #         if self.annotated_path.exists():
    #             annotated_df = pd.read_csv(self.annotated_path)
    #             print(f"✓ Loaded {len(annotated_df)} database-annotated lipids")
    #         else:
    #             annotated_df = pd.DataFrame()
    #             print("⚠ No database annotations found")
            
    #         # Combine results
    #         if not predictions_df.empty or not annotated_df.empty:
    #             # Merge both dataframes
    #             combined_df = pd.concat([annotated_df, predictions_df], ignore_index=True)
                
                
                
                
                
    #             combined_df.to_csv(output_path, index=False)
    #             print(f"\n✓ Combined results saved to: {output_path.name}")
    #             print(f"  Total lipids: {len(combined_df)}")
    #             print(f"    - Database annotated: {len(annotated_df)}")
    #             print(f"    - Predicted: {len(predictions_df)}")
    #         else:
    #             print("\n⚠ No data to save in final result file.")
            
    #         return True
    #     except Exception as e:
    #         print(f"✗ Error merging results: {e}")
    #         return False
    
    def merge_with_annotated(self):
        """Merge final predictions with database-annotated lipids and post-process."""
        self.print_header("MERGING AND PROCESSING RESULTS")
        
        try:
            output_path = self.result_path / self.final_combined_name
            
            # 1. Load DataFrames
            if self.final_output_path.exists():
                predictions_df = pd.read_csv(self.final_output_path)
                print(f"✓ Loaded {len(predictions_df)} predicted lipids")
            else:
                predictions_df = pd.DataFrame()
                print("⚠ No predictions file found")
            
            if self.annotated_path.exists():
                annotated_df = pd.read_csv(self.annotated_path)
                print(f"✓ Loaded {len(annotated_df)} database-annotated lipids")
            else:
                annotated_df = pd.DataFrame()
                print("⚠ No database annotations found")
            
            if predictions_df.empty and annotated_df.empty:
                print("\n⚠ No data to save in final result file.")
                return False

            # 2. Merge
            combined_df = pd.concat([annotated_df, predictions_df], ignore_index=True)
            print(f"  Initial combined count: {len(combined_df)}")

            # 3. Post-Processing Logic
            
            # --- Logic A: Update chain_info based on high confidence ---
            # Condition: pred_confidence > 0.8 AND chain_info is empty AND plsf_rank1 is not empty
            if 'pred_confidence' in combined_df.columns and 'plsf_rank1' in combined_df.columns:
                if 'chain_info' not in combined_df.columns:
                    combined_df['chain_info'] = ''
                
                # Check for empty chain_info (NaN or empty string)
                chain_empty_mask = combined_df['chain_info'].isna() | (combined_df['chain_info'].astype(str).str.strip() == '')
                
                # Check for non-empty plsf_rank1
                plsf_not_empty_mask = combined_df['plsf_rank1'].notna() & (combined_df['plsf_rank1'].astype(str).str.strip() != '')

                # Combined mask: High confidence AND empty chain_info AND non-empty plsf_rank1
                high_conf_update_mask = (combined_df['pred_confidence'] > 0.8) & chain_empty_mask & plsf_not_empty_mask
                
                num_updated = high_conf_update_mask.sum()
                if num_updated > 0:
                    combined_df.loc[high_conf_update_mask, 'chain_info'] = combined_df.loc[high_conf_update_mask, 'plsf_rank1']
                    print(f"✓ Updated chain_info for {num_updated} rows (conf > 0.8, empty chain_info, valid plsf_rank1)")

            # --- Logic B: Clear specific columns if name starts with 'Adduct' ---
            if 'name' in combined_df.columns:
                adduct_mask = combined_df['name'].astype(str).str.startswith('Adduct', na=False)
                num_adduct = adduct_mask.sum()
                
                if num_adduct > 0:
                    cols_to_clear = [
                        "name", "class", "category", "MS2_norm_hit", "source", "mass_diff_ppm",
                        "dot_product", "weighted_dot_product", "entropy_similarity",
                        "unweighted_entropy_similarity", "predicted_adduct", "adduct_confidence",
                        "prediction_source", "predicted_class", "class_confidence", "classes_mz",
                        "classes_ms2", "num_chain", "plsf_rank1", "plsf_rank2", "plsf_rank3",
                        "plsf_confidence", "pred_confidence"
                    ]
                    
                    # Intersect with existing columns
                    existing_clear_cols = [c for c in cols_to_clear if c in combined_df.columns]
                    
                    if existing_clear_cols:
                        combined_df.loc[adduct_mask, existing_clear_cols] = ""
                        print(f"✓ Cleared {len(existing_clear_cols)} columns for {num_adduct} rows where name starts with 'Adduct'")

            # --- Logic C: Drop specific columns ---
            # C1. Columns to drop specifically requested
            specific_drop_cols = [
                'classes_mz', 'classes_ms2', 'predicted_adduct', 'adduct_confidence', 
                'prediction_source', 'predicted_class', 'class_confidence', 'adduct_hit'
            ]
            
            # C2. Columns starting with 'mz_'
            mz_cols = [col for col in combined_df.columns if col.lower().startswith('mz_')]
            
            # Combine and deduplicate
            all_drop_cols = list(set(specific_drop_cols + mz_cols))
            
            # Only drop if they exist
            final_drop_cols = [c for c in all_drop_cols if c in combined_df.columns]
            
            if final_drop_cols:
                combined_df = combined_df.drop(columns=final_drop_cols)
                print(f"✓ Dropped {len(final_drop_cols)} columns (mz_* and internal prediction cols)")

            # 4. Save Final Result
            combined_df.to_csv(output_path, index=False)
            print(f"\n✓ Processed results saved to: {output_path.name}")
            print(f"  Final shape: {combined_df.shape}")
            
            return True, combined_df
            
        except Exception as e:
            print(f"✗ Error merging results: {e}")
            import traceback
            traceback.print_exc()
            return False, None
        
        
    def cleanup_intermediate_files(self):
        """Moves all intermediate CSV files to a 'process_files' subdirectory, 
        leaving only identification_result.csv in the main results directory."""
        self.print_header("CLEANING UP INTERMEDIATE FILES")
        
        intermediate_dir = self.result_path / "process_files"
        intermediate_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        
        # Use glob to find all CSV files in the main result directory
        for file_path in self.result_path.glob("*.csv"):
            file_name = file_path.name
            
            # Skip the final result file
            if file_name == self.final_combined_name:
                continue
            
            try:
                new_path = intermediate_dir / file_name
                file_path.rename(new_path)
                print(f"→ Moved {file_name} to {intermediate_dir.name}/")
                moved_count += 1
            except Exception as e:
                # Catching error in case the file was opened elsewhere or permissions issue
                print(f"✗ Could not move {file_name}: {e}")

        if moved_count > 0:
            print(f"\n✓ Cleanup complete. {moved_count} files moved to {intermediate_dir.name}/.")
        else:
            print("⚠ No intermediate CSV files found to move.")

        return True

    
    def print_comprehensive_stats(self, final_df):
        """Print comprehensive statistics about the annotation results"""
        self.print_header("ANNOTATION STATISTICS")
        
        try:
            if final_df is None or final_df.empty:
                print("⚠ No data available for statistics")
                return
            
            # Total features
            total_features = len(final_df)
            print(f"\n📊 TOTAL FEATURES: {total_features}")
            
            # Database search results
            print(f"\n🔍 DATABASE SEARCH RESULTS:")
            if self.annotated_path.exists():
                annotated_df = pd.read_csv(self.annotated_path)
                db_results = len(annotated_df)
                print(f"   • Number of matches: {db_results}")
                print(f"   • Method: Spectral similarity matching")
                print(f"   • MS1 tolerance: {self.MS1_tol} Da")
                print(f"   • MS2 tolerance: {self.MS2_tol} Da")
                print(f"   • MS2 similarity threshold: {self.MS2_threshold}")
            else:
                print("   • No database search results available")
            
            # Tier 1 predictions (high confidence predictions)
            print(f"\n⭐ TIER 1 PREDICTIONS (High Confidence):")
            if 'pred_confidence' in final_df.columns:
                tier1_mask = final_df['pred_confidence'] > 0.8
                tier1_count = tier1_mask.sum()
                print(f"   • Number of features: {tier1_count}")
                print(f"   • Confidence threshold: > 0.8")
                
                # Show average confidence for tier 1
                if tier1_count > 0:
                    avg_conf = final_df.loc[tier1_mask, 'pred_confidence'].mean()
                    print(f"   • Average confidence: {avg_conf:.3f}")
            else:
                print("   • No prediction confidence data available")
            
            # Tier 2 annotations (rows with non-empty names)
            print(f"\n📝 TIER 2 ANNOTATIONS (All identified features):")
            if 'name' in final_df.columns:
                # Count non-empty names (excluding NaN and empty strings)
                tier2_mask = final_df['name'].notna() & (final_df['name'].astype(str).str.strip() != '')
                tier2_count = tier2_mask.sum()
                print(f"   • Number of features: {tier2_count}")
                print(f"   • Percentage of total: {tier2_count/total_features*100:.1f}%")
            else:
                print("   • No name data available")
            
            # Additional breakdown by source if possible
            print(f"\n📋 ANNOTATION SOURCE BREAKDOWN:")
            
            # Try to determine source of annotations
            has_db = self.annotated_path.exists()
            has_pred = 'pred_confidence' in final_df.columns
            
            if has_db:
                db_df = pd.read_csv(self.annotated_path)
                db_count = len(db_df)
                print(f"   • Database matched: {db_count}")
            
            if has_pred:
                pred_only_mask = final_df['pred_confidence'].notna()
                pred_count = pred_only_mask.sum()
                print(f"   • ML predicted: {pred_count}")
            
            # Unidentified features
            if 'name' in final_df.columns:
                unidentified_mask = final_df['name'].isna() | (final_df['name'].astype(str).str.strip() == '')
                unidentified_count = unidentified_mask.sum()
                print(f"   • Unidentified: {unidentified_count}")
            
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"✗ Error generating statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the complete pipeline"""
        print("\n" + "=" * 80)
        print("  LIPIDOMICS ANNOTATION PIPELINE")
        print("=" * 80)
        print(f"\nInput file: {self.input_path}")
        print(f"Result directory: {self.result_path}")
        print(f"Database: {self.db_path}")
        
        # Step 0: Process MS2 (New Step)
        if not self.step0_process_ms2():
            print("\n✗ Pipeline failed at Step 0: MS2 Processing")
            return False

        # Step 1: Database search (optional)
        db_success = self.step1_database_search()
        
        # If no database or database search failed, process all (processed) input
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

Directory Structure Requirement:
  your_project/
  ├── run.py                  # This script
  ├── code/                   # Helper scripts
  │   ├── process_ms2.py      # <--- MUST BE HERE
  │   ├── db_search.py
  │   ├── adduct_predict.py
  │   ├── class_predict.py
  │   └── predict_plsf.py
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
            ms2_tol_ppm=args.ms2_tol_ppm
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
    main()