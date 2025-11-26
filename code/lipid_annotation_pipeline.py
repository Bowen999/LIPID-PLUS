#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lipidomics Annotation Pipeline - All-in-One Script

This pipeline performs comprehensive lipidomics annotation through the following steps:
1. Database Search - Annotate known lipids using spectral database matching
2. Adduct Prediction - Predict adduct types for unknown lipids
3. Class Prediction - Predict lipid classes using hybrid rule-based + ML approach
4. Chain Composition Prediction - Predict detailed fatty acid chain compositions

Usage:
    python lipid_annotation_pipeline.py input.csv --result_path results/ --db_path lipid.db --adduct_model adduct.joblib --class_model class.joblib --plsf_model plsf.joblib
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
        
        Args:
            input_path: Path to input CSV file
            result_path: Directory to save all results
            db_path: Path to lipid database (.db file)
            adduct_model: Path to trained adduct prediction model (.joblib)
            class_model: Path to trained class prediction model (.joblib)
            plsf_model: Path to trained PLSF prediction model (.joblib)
            MS1_tol: MS1 tolerance for database search (Da)
            MS2_tol: MS2 tolerance for database search (Da)
            MS2_threshold: Minimum MS2 score for database match
            ms1_tol_ppm: MS1 tolerance for class prediction (ppm)
            ms2_tol_ppm: MS2 tolerance for class prediction (ppm)
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
        
        # Define intermediate file paths
        self.annotated_path = self.result_path / "annotated_df.csv"
        self.dark_lipid_path = self.result_path / "dark_lipid.csv"
        self.adduct_pred_path = self.result_path / "adduct_predictions.csv"
        self.class_pred_path = self.result_path / "class_predictions.csv"
        self.final_output_path = self.result_path / "final_annotations.csv"
    
    def print_header(self, text):
        """Print formatted section header"""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)
    
    def run_command(self, cmd, step_name):
        """Run a command and handle errors"""
        print(f"\n→ Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error in {step_name}:")
            print(e.stderr)
            return False
        except FileNotFoundError:
            print(f"✗ Error: Command not found. Make sure the script exists.")
            return False
    
    def step1_database_search(self):
        """Step 1: Database search for known lipids"""
        self.print_header("STEP 1: DATABASE SEARCH")
        
        if not self.db_path.exists():
            print(f"⚠ Warning: Database file not found: {self.db_path}")
            print("  Skipping database search step...")
            return False
        
        cmd = [
            "python", "db_search.py",
            str(self.input_path),
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
            input_file = self.input_path
        
        if not self.adduct_model.exists():
            print(f"✗ Error: Adduct model not found: {self.adduct_model}")
            return False
        
        cmd = [
            "python", "adduct_predict.py",
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
            "python", "class_predict.py",
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
            "python", "predict_plsf.py",
            str(self.plsf_model),
            str(self.class_pred_path),
            "--output_path", str(self.final_output_path)
        ]
        
        success = self.run_command(cmd, "PLSF Prediction")
        
        if success:
            print("\n✓ Step 4 complete: Chain compositions predicted")
            return True
        return False
    
    def merge_with_annotated(self):
        """Merge final predictions with database-annotated lipids"""
        self.print_header("FINALIZING RESULTS")
        
        try:
            # Load final predictions
            if self.final_output_path.exists():
                predictions_df = pd.read_csv(self.final_output_path)
                print(f"✓ Loaded {len(predictions_df)} predicted lipids")
            else:
                predictions_df = pd.DataFrame()
                print("⚠ No predictions file found")
            
            # Load database annotations
            if self.annotated_path.exists():
                annotated_df = pd.read_csv(self.annotated_path)
                print(f"✓ Loaded {len(annotated_df)} database-annotated lipids")
            else:
                annotated_df = pd.DataFrame()
                print("⚠ No database annotations found")
            
            # Combine results
            if not predictions_df.empty and not annotated_df.empty:
                # Merge both dataframes
                combined_df = pd.concat([annotated_df, predictions_df], ignore_index=True)
                combined_path = self.result_path / "all_annotations_combined.csv"
                combined_df.to_csv(combined_path, index=False)
                print(f"\n✓ Combined results saved to: {combined_path}")
                print(f"  Total lipids: {len(combined_df)}")
                print(f"    - Database annotated: {len(annotated_df)}")
                print(f"    - Predicted: {len(predictions_df)}")
            elif not predictions_df.empty:
                print(f"\n✓ Final predictions saved to: {self.final_output_path}")
            elif not annotated_df.empty:
                print(f"\n✓ Database annotations only: {self.annotated_path}")
            
            return True
        except Exception as e:
            print(f"✗ Error merging results: {e}")
            return False
    
    def run(self):
        """Run the complete pipeline"""
        print("\n" + "=" * 80)
        print("  LIPIDOMICS ANNOTATION PIPELINE")
        print("=" * 80)
        print(f"\nInput file: {self.input_path}")
        print(f"Result directory: {self.result_path}")
        print(f"Database: {self.db_path}")
        
        # Step 1: Database search (optional)
        db_success = self.step1_database_search()
        
        # If no database or database search failed, process all input
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
        
        # Final summary
        self.print_header("PIPELINE COMPLETE")
        print(f"\nAll results saved to: {self.result_path}")
        print(f"Final output: {self.final_output_path}")
        print("\n✓ Success!\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete Lipidomics Annotation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Minimal (using defaults):
  python lipid_annotation_pipeline.py feature_df.csv

  # With custom parameters:
  python lipid_annotation_pipeline.py feature_df.csv \\
      --result_path my_results/ \\
      --MS1_tol 0.01 \\
      --MS2_threshold 0.8

  # With different model paths:
  python lipid_annotation_pipeline.py feature_df.csv \\
      --adduct_model custom/adduct.joblib \\
      --class_model custom/class.joblib \\
      --plsf_model custom/plsf.joblib

Default Paths:
  Models: model/adduct.joblib, model/class.joblib, model/plsf.joblib
  Database: dataset/lipid.db
  Results: results/

Input Requirements:
  - CSV file with columns: precursor_mz, ion_mode, MS2, and mz_* features
  - MS2 should be a list of [mz, intensity] pairs

Output:
  - annotated_df.csv: Database-matched lipids (if database provided)
  - dark_lipid.csv: Unknown lipids requiring prediction
  - adduct_predictions.csv: Predicted adduct types
  - class_predictions.csv: Predicted lipid classes
  - final_annotations.csv: Complete annotations with chain compositions
  - all_annotations_combined.csv: Combined database + predicted results
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
        help='Path to trained adduct prediction model (default: model/adduct.joblib)'
    )
    
    parser.add_argument(
        '--class_model',
        type=str,
        default='model/class.joblib',
        help='Path to trained class prediction model (default: model/class.joblib)'
    )
    
    parser.add_argument(
        '--plsf_model',
        type=str,
        default='model/plsf.joblib',
        help='Path to trained PLSF model (default: model/plsf.joblib)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--result_path',
        type=str,
        default='results',
        help='Directory to save all results (default: results)'
    )
    
    parser.add_argument(
        '--db_path',
        type=str,
        default='dataset/lipid.db',
        help='Path to lipid database file (default: dataset/lipid.db)'
    )
    
    # Database search parameters
    parser.add_argument(
        '--MS1_tol',
        type=float,
        default=0.005,
        help='MS1 tolerance for database search in Da (default: 0.005)'
    )
    
    parser.add_argument(
        '--MS2_tol',
        type=float,
        default=0.01,
        help='MS2 tolerance for database search in Da (default: 0.01)'
    )
    
    parser.add_argument(
        '--MS2_threshold',
        type=float,
        default=0.7,
        help='Minimum MS2 similarity score for database match (default: 0.7)'
    )
    
    # Class prediction parameters
    parser.add_argument(
        '--ms1_tol_ppm',
        type=float,
        default=10.0,
        help='MS1 tolerance for class prediction in ppm (default: 10.0)'
    )
    
    parser.add_argument(
        '--ms2_tol_ppm',
        type=float,
        default=20.0,
        help='MS2 tolerance for class prediction in ppm (default: 20.0)'
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
