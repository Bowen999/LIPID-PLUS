# import pandas as pd
# import numpy as np
# import joblib
# import argparse
# import sys
# from pathlib import Path

# # Import rule-based classification functions
# try:
#     import class_rule
# except ImportError:
#     print("✗ Error: class_rule.py not found. Please ensure it's in the same directory or in your Python path.")
#     sys.exit(1)


# def load_model(model_path):
#     """
#     Load the trained model and encoders from joblib file
#     """
#     try:
#         model_package = joblib.load(model_path)
#         print(f"✓ Model loaded successfully from: {model_path}")
#         print(f"  Model type: {model_package['model_name']}")
#         return model_package
#     except Exception as e:
#         print(f"✗ Error loading model: {str(e)}")
#         sys.exit(1)


# def prepare_input_data(data, ion_mode_encoder, adduct_encoder):
#     """
#     Prepare input data for prediction
#     """
#     # Make a copy to avoid modifying original
#     data_processed = data.copy()
    
#     # Round precursor_mz first (IMPORTANT!)
#     data_processed['precursor_mz'] = data_processed['precursor_mz'].round(2)
    
#     # Get all mz columns
#     mz_cols = [col for col in data.columns if col.startswith('mz')]
    
#     # Encode ion_mode
#     try:
#         data_processed['ion_mode_encoded'] = ion_mode_encoder.transform(data_processed['ion_mode'])
#     except Exception as e:
#         print(f"✗ Error encoding ion_mode: {str(e)}")
#         print(f"  Valid ion_mode values: {list(ion_mode_encoder.classes_)}")
#         sys.exit(1)
    
#     # Check if 'adduct' column exists, otherwise use 'predicted_adduct'
#     if 'adduct' in data.columns:
#         adduct_col = 'adduct'
#         print(f"  Using 'adduct' column for prediction")
#     elif 'predicted_adduct' in data.columns:
#         adduct_col = 'predicted_adduct'
#         print(f"  Using 'predicted_adduct' column for prediction")
#     else:
#         print(f"✗ Error: Neither 'adduct' nor 'predicted_adduct' column found in data")
#         sys.exit(1)
    
#     # Encode adduct - handle unseen labels
#     try:
#         # Check for unseen labels
#         known_adducts = set(adduct_encoder.classes_)
#         input_adducts = set(data_processed[adduct_col].unique())
#         unseen_adducts = input_adducts - known_adducts
        
#         if unseen_adducts:
#             print(f"  ⚠ Warning: Found {len(unseen_adducts)} unseen adduct(s): {unseen_adducts}")
#             print(f"  These will be marked as 'unknown' and predictions may be less accurate")
            
#             # Replace unseen adducts with the first known adduct (as placeholder)
#             # We'll track these rows to mark predictions as uncertain
#             unseen_mask = data_processed[adduct_col].isin(unseen_adducts)
#             placeholder_adduct = adduct_encoder.classes_[0]
#             data_processed.loc[unseen_mask, adduct_col + '_original'] = data_processed.loc[unseen_mask, adduct_col]
#             data_processed.loc[unseen_mask, adduct_col] = placeholder_adduct
#         else:
#             unseen_mask = pd.Series([False] * len(data_processed))
        
#         data_processed['adduct_encoded'] = adduct_encoder.transform(data_processed[adduct_col])
        
#     except Exception as e:
#         print(f"✗ Error encoding {adduct_col}: {str(e)}")
#         print(f"  Valid adduct values: {list(adduct_encoder.classes_)}")
#         sys.exit(1)
    
#     # Define feature columns (must match training)
#     feature_cols = ['precursor_mz'] + mz_cols + ['ion_mode_encoded', 'adduct_encoded']
    
#     # Check if all required columns exist
#     missing_cols = [col for col in feature_cols if col not in data_processed.columns]
#     if missing_cols:
#         print(f"✗ Missing required columns: {missing_cols}")
#         sys.exit(1)
    
#     # Select features
#     X = data_processed[feature_cols]
    
#     return X, feature_cols, unseen_mask


# def apply_rule_based_classification(data, ms1_tolerance_ppm=10, ms2_tolerance_da=0.01):
#     """
#     Apply rule-based classification using class_rule functions
    
#     Note: ms2_tolerance_da is converted from ppm. For ~500 m/z:
#     20 ppm ≈ 0.01 Da, so default 0.02 Da ≈ 40 ppm at 500 m/z
#     Adjust based on your instrument accuracy.
#     """
#     print(f"\n--- Applying rule-based classification ---")
#     print(f"  MS1 tolerance: {ms1_tolerance_ppm} ppm")
#     print(f"  MS2 tolerance: {ms2_tolerance_da} Da")
    
#     # Make a copy for rule-based processing
#     df_rules = data.copy()
    
#     # Apply rule-based methods
#     print("  Step 1: Identifying classes by precursor mass...")
#     df_rules = class_rule.batch_identify_by_mass(df_rules, tolerance_ppm=ms1_tolerance_ppm)
    
#     print("  Step 2: Identifying classes by MS2 fragments...")
#     # Modify identify_lipid_class_from_ms2 to use ms2_tolerance_da
#     # We need to temporarily override the function or pass tolerance
#     original_func = class_rule.identify_lipid_class_from_ms2
    
#     def patched_identify(precursor_mz, adduct, ms2_spectrum):
#         return original_func(precursor_mz, adduct, ms2_spectrum, tolerance=ms2_tolerance_da)
    
#     class_rule.identify_lipid_class_from_ms2 = patched_identify
#     df_rules = class_rule.batch_identify_by_ms2(df_rules)
#     class_rule.identify_lipid_class_from_ms2 = original_func  # Restore original
    
#     print("  Step 3: Finding overlap for final class...")
#     df_rules = class_rule.find_class_overlap(df_rules)
    
#     print("  Step 4: Adding category...")
#     df_rules = class_rule.add_final_category(df_rules)
    
#     # Identify successfully classified rows
#     rule_classified_mask = (df_rules['class'] != '') & (df_rules['class'].notna())
    
#     print(f"✓ Rule-based classification complete")
#     print(f"  Successfully classified: {rule_classified_mask.sum()} / {len(data)} rows")
    
#     return df_rules, rule_classified_mask


# def predict_classes(model_package, input_path, output_path, ms1_tolerance_ppm=20, ms2_tolerance_ppm=20, use_rules=True):
#     """
#     Make predictions on input data and save results
#     Combines rule-based and ML-based approaches
#     """
#     # Load input data
#     print(f"\nLoading input data from: {input_path}")
#     try:
#         data = pd.read_csv(input_path)
#         print(f"✓ Data loaded successfully")
#         print(f"  Number of samples: {len(data)}")
#     except Exception as e:
#         print(f"✗ Error loading data: {str(e)}")
#         sys.exit(1)
    
#     # Initialize result DataFrame
#     result = data.copy()
#     result['prediction_source'] = ''
#     result['predicted_class'] = ''
#     result['confidence_class'] = np.nan
    
#     # Step 1: Apply rule-based classification if enabled
#     rule_classified_mask = pd.Series([False] * len(data))
    
#     if use_rules:
#         try:
#             # Convert MS2 tolerance from ppm to Da (approximate conversion at 500 m/z)
#             # For more accurate conversion, you could use actual precursor m/z
#             # Formula: tolerance_da = (tolerance_ppm * precursor_mz) / 1e6
#             # Using 500 as reference: 20 ppm at 500 m/z = 0.01 Da
#             ms2_tolerance_da = (ms2_tolerance_ppm * 500) / 1e6
            
#             df_rules, rule_classified_mask = apply_rule_based_classification(
#                 data, ms1_tolerance_ppm, ms2_tolerance_da
#             )
            
#             # Copy rule-based results
#             result.loc[rule_classified_mask, 'predicted_class'] = df_rules.loc[rule_classified_mask, 'class']
#             result.loc[rule_classified_mask, 'category'] = df_rules.loc[rule_classified_mask, 'category']
#             result.loc[rule_classified_mask, 'prediction_source'] = 'rule-based'
#             result.loc[rule_classified_mask, 'confidence_class'] = 1.0  # High confidence for rule-based
            
#             # Add rule-based metadata columns
#             result['classes_mz'] = df_rules['classes_mz']
#             result['classes_ms2'] = df_rules['classes_ms2']
            
#         except Exception as e:
#             print(f"⚠ Warning: Rule-based classification failed: {str(e)}")
#             print("  Falling back to ML model for all predictions")
    
#     # Step 2: Apply ML model for remaining rows
#     ml_predict_mask = ~rule_classified_mask
    
#     if ml_predict_mask.sum() > 0:
#         print(f"\n--- Applying ML model for {ml_predict_mask.sum()} remaining rows ---")
        
#         # Extract model and encoders
#         model = model_package['model']
        
#         # Handle different possible key names for encoders
#         class_encoder = model_package.get('class_encoder') or model_package.get('label_encoder')
#         ion_mode_encoder = model_package.get('ion_mode_encoder')
#         adduct_encoder = model_package.get('adduct_encoder')
        
#         if class_encoder is None:
#             print(f"✗ Error: class_encoder not found in model package")
#             print(f"  Available keys: {list(model_package.keys())}")
#             sys.exit(1)
        
#         if ion_mode_encoder is None:
#             print(f"✗ Error: ion_mode_encoder not found in model package")
#             sys.exit(1)
        
#         if adduct_encoder is None:
#             print(f"✗ Error: adduct_encoder not found in model package")
#             sys.exit(1)
        
#         # Prepare input features for ML prediction
#         print("\nPreparing features for ML model...")
#         ml_data = data[ml_predict_mask].copy()
#         X, feature_cols, unseen_mask = prepare_input_data(ml_data, ion_mode_encoder, adduct_encoder)
#         print(f"✓ Features prepared")
#         print(f"  Number of features: {X.shape[1]}")
        
#         # Make predictions
#         print("\nMaking ML predictions...")
#         try:
#             predictions_encoded = model.predict(X)
#             predictions = class_encoder.inverse_transform(predictions_encoded)
#             print(f"✓ ML predictions completed")
#         except Exception as e:
#             print(f"✗ Error making predictions: {str(e)}")
#             sys.exit(1)
        
#         # Store ML predictions
#         ml_indices = result[ml_predict_mask].index
#         result.loc[ml_indices, 'predicted_class'] = predictions
#         result.loc[ml_indices, 'prediction_source'] = 'model-based'
        
#         # Mark predictions with unseen adducts as "unknown"
#         if unseen_mask.any():
#             unseen_indices = ml_indices[unseen_mask]
#             result.loc[unseen_indices, 'predicted_class'] = 'unknown'
#             result.loc[unseen_indices, 'confidence_class'] = 0.0
#             print(f"  ⚠ {unseen_mask.sum()} predictions marked as 'unknown' due to unseen adducts")
        
#         # Get prediction probabilities if available
#         if hasattr(model, 'predict_proba'):
#             try:
#                 probabilities = model.predict_proba(X)
#                 max_probabilities = np.max(probabilities, axis=1)
                
#                 # Set confidence for valid predictions
#                 valid_ml_mask = ml_predict_mask.copy()
#                 valid_ml_mask[ml_indices[unseen_mask]] = False
#                 result.loc[ml_indices[~unseen_mask], 'confidence_class'] = max_probabilities[~unseen_mask]
                
#                 print(f"  Average ML confidence: {max_probabilities[~unseen_mask].mean():.4f} (excluding unknowns)")
#             except:
#                 pass
    
#     # Add num_chain column
#     result = class_rule.add_num_chain(result)
    
#     # Save results
#     print(f"\nSaving results to: {output_path}")
#     try:
#         # Create output directory if it doesn't exist
#         output_dir = Path(output_path).parent
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         result.to_csv(output_path, index=False)
#         print(f"✓ Results saved successfully")
#     except Exception as e:
#         print(f"✗ Error saving results: {str(e)}")
#         sys.exit(1)
    
#     # Display summary
#     print("\n" + "="*70)
#     print("PREDICTION SUMMARY")
#     print("="*70)
#     print(f"Total predictions: {len(result)}")
#     print(f"\nPrediction sources:")
#     print(result['prediction_source'].value_counts().to_string())
    
#     if rule_classified_mask.any():
#         print(f"\nRule-based predictions: {rule_classified_mask.sum()}")
#     if ml_predict_mask.any():
#         print(f"ML-based predictions: {ml_predict_mask.sum()}")
#         if unseen_mask.any():
#             print(f"  └─ Unknown (unseen adducts): {unseen_mask.sum()}")
    
#     print(f"\nPredicted class distribution:")
#     print(result['predicted_class'].value_counts().to_string())
    
#     if 'category' in result.columns:
#         print(f"\nCategory distribution:")
#         print(result['category'].value_counts().to_string())
    
#     print("="*70)
    
#     return result


# def main():
#     parser = argparse.ArgumentParser(
#         description='Predict lipid classes using rule-based + ML hybrid approach',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Example usage:
#   python class_predict.py test.csv model/class.joblib --output_path result/pred_class.csv
#   python class_predict.py data.csv class.joblib -o predictions.csv --ms1_tol 15 --ms2_tol 20
#   python class_predict.py data.csv class.joblib --no-rules  # ML only, skip rules
#         """
#     )
    
#     parser.add_argument(
#         'input_path',
#         type=str,
#         help='Path to input CSV file containing features'
#     )
    
#     parser.add_argument(
#         'model_path',
#         type=str,
#         help='Path to trained model (.joblib file)'
#     )
    
#     parser.add_argument(
#         '--output_path', '-o',
#         type=str,
#         default='class_pred.csv',
#         help='Path to save predictions (default: class_pred.csv)'
#     )
    
#     parser.add_argument(
#         '--ms1_tol',
#         type=float,
#         default=10.0,
#         help='MS1 mass tolerance in ppm for rule-based classification (default: 20.0)'
#     )
    
#     parser.add_argument(
#         '--ms2_tol',
#         type=float,
#         default=20.0,
#         help='MS2 mass tolerance in ppm for rule-based classification (default: 20.0)'
#     )
    
#     parser.add_argument(
#         '--no-rules',
#         action='store_true',
#         help='Skip rule-based classification and use only ML model'
#     )
    
#     args = parser.parse_args()
    
#     # Verify input file exists
#     if not Path(args.input_path).exists():
#         print(f"✗ Error: Input file not found: {args.input_path}")
#         sys.exit(1)
    
#     # Verify model file exists
#     if not Path(args.model_path).exists():
#         print(f"✗ Error: Model file not found: {args.model_path}")
#         sys.exit(1)
    
#     print("="*70)
#     print("CLASS PREDICTION (HYBRID: RULE-BASED + ML)")
#     print("="*70)
    
#     # Load model
#     model_package = load_model(args.model_path)
    
#     # Make predictions
#     use_rules = not args.no_rules
#     results = predict_classes(
#         model_package, 
#         args.input_path, 
#         args.output_path,
#         ms1_tolerance_ppm=args.ms1_tol,
#         ms2_tolerance_ppm=args.ms2_tol,
#         use_rules=use_rules
#     )
    
#     # --- Add Category and Chain Information ---
#     print("Processing metadata (Category, Chain info)...")
#     if 'predicted_class' in results.columns:
#         # class_rule functions typically expect 'Final_Class' as the column name
#         # We temporarily map predicted_class to Final_Class to use the shared rules
#         results['Final_Class'] = results['predicted_class']
        
    
#         # Add Number of Chains
#         if hasattr(class_rule, 'add_num_chain'):
#             results = class_rule.add_num_chain(results)
            
#         # Add Final Category
#         if hasattr(class_rule, 'add_final_category'):
#             results = class_rule.add_final_category(results)
                
#         results = results.drop('Final_Class', axis=1) 
#         print("✓ Category and chain information added")

#     # Save predictions
#     if hasattr(args, 'output_path'):
#         save_path = args.output_path
#     else:
#         # Fallback or default if the argument name differs
#         save_path = 'pred_class.csv'
        
#     results.to_csv(save_path, index=False)
#     print(f"✓ Predictions saved to: {save_path}")
#     print("="*70)


# if __name__ == "__main__":
#     main()





import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path

# Import rule-based classification functions
try:
    import class_rule
except ImportError:
    print("✗ Error: class_rule.py not found. Please ensure it's in the same directory or in your Python path.")
    sys.exit(1)


def load_model(model_path):
    """
    Load the trained model and encoders from joblib file
    """
    try:
        model_package = joblib.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
        print(f"  Model type: {model_package['model_name']}")
        return model_package
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        sys.exit(1)


def prepare_input_data(data, ion_mode_encoder, adduct_encoder):
    """
    Prepare input data for prediction
    """
    # Make a copy to avoid modifying original
    data_processed = data.copy()
    
    # Round precursor_mz first (IMPORTANT!)
    data_processed['precursor_mz'] = data_processed['precursor_mz'].round(2)
    
    # Get all mz columns
    mz_cols = [col for col in data.columns if col.startswith('mz')]
    
    # Encode ion_mode
    try:
        data_processed['ion_mode_encoded'] = ion_mode_encoder.transform(data_processed['ion_mode'])
    except Exception as e:
        print(f"✗ Error encoding ion_mode: {str(e)}")
        print(f"  Valid ion_mode values: {list(ion_mode_encoder.classes_)}")
        sys.exit(1)
    
    # Check if 'adduct' column exists, otherwise use 'predicted_adduct'
    if 'adduct' in data.columns:
        adduct_col = 'adduct'
        print(f"  Using 'adduct' column for prediction")
    elif 'predicted_adduct' in data.columns:
        adduct_col = 'predicted_adduct'
        print(f"  Using 'predicted_adduct' column for prediction")
    else:
        print(f"✗ Error: Neither 'adduct' nor 'predicted_adduct' column found in data")
        sys.exit(1)
    
    # Encode adduct - handle unseen labels
    try:
        # Check for unseen labels
        known_adducts = set(adduct_encoder.classes_)
        input_adducts = set(data_processed[adduct_col].unique())
        unseen_adducts = input_adducts - known_adducts
        
        if unseen_adducts:
            print(f"  ⚠ Warning: Found {len(unseen_adducts)} unseen adduct(s): {unseen_adducts}")
            print(f"  These will be marked as 'unknown' and predictions may be less accurate")
            
            # Replace unseen adducts with the first known adduct (as placeholder)
            # We'll track these rows to mark predictions as uncertain
            unseen_mask = data_processed[adduct_col].isin(unseen_adducts)
            placeholder_adduct = adduct_encoder.classes_[0]
            data_processed.loc[unseen_mask, adduct_col + '_original'] = data_processed.loc[unseen_mask, adduct_col]
            data_processed.loc[unseen_mask, adduct_col] = placeholder_adduct
        else:
            unseen_mask = pd.Series([False] * len(data_processed))
        
        data_processed['adduct_encoded'] = adduct_encoder.transform(data_processed[adduct_col])
        
    except Exception as e:
        print(f"✗ Error encoding {adduct_col}: {str(e)}")
        print(f"  Valid adduct values: {list(adduct_encoder.classes_)}")
        sys.exit(1)
    
    # Define feature columns (must match training)
    feature_cols = ['precursor_mz'] + mz_cols + ['ion_mode_encoded', 'adduct_encoded']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in data_processed.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Select features
    X = data_processed[feature_cols]
    
    return X, feature_cols, unseen_mask


def apply_rule_based_classification(data, ms1_tolerance_ppm=10, ms2_tolerance_da=0.01):
    """
    Apply rule-based classification using class_rule functions
    
    Note: ms2_tolerance_da is converted from ppm. For ~500 m/z:
    20 ppm ≈ 0.01 Da, so default 0.02 Da ≈ 40 ppm at 500 m/z
    Adjust based on your instrument accuracy.
    """
    print(f"\n--- Applying rule-based classification ---")
    print(f"  MS1 tolerance: {ms1_tolerance_ppm} ppm")
    print(f"  MS2 tolerance: {ms2_tolerance_da} Da")
    
    # Make a copy for rule-based processing
    df_rules = data.copy()
    
    # Apply rule-based methods
    print("  Step 1: Identifying classes by precursor mass...")
    df_rules = class_rule.batch_identify_by_mass(df_rules, tolerance_ppm=ms1_tolerance_ppm)
    
    print("  Step 2: Identifying classes by MS2 fragments...")
    # Modify identify_lipid_class_from_ms2 to use ms2_tolerance_da
    # We need to temporarily override the function or pass tolerance
    original_func = class_rule.identify_lipid_class_from_ms2
    
    def patched_identify(precursor_mz, adduct, ms2_spectrum):
        return original_func(precursor_mz, adduct, ms2_spectrum, tolerance=ms2_tolerance_da)
    
    class_rule.identify_lipid_class_from_ms2 = patched_identify
    df_rules = class_rule.batch_identify_by_ms2(df_rules)
    class_rule.identify_lipid_class_from_ms2 = original_func  # Restore original
    
    print("  Step 3: Finding overlap for final class...")
    df_rules = class_rule.find_class_overlap(df_rules)
    
    print("  Step 4: Adding category...")
    df_rules = class_rule.add_final_category(df_rules)
    
    # Identify successfully classified rows
    rule_classified_mask = (df_rules['class'] != '') & (df_rules['class'].notna())
    
    print(f"✓ Rule-based classification complete")
    print(f"  Successfully classified: {rule_classified_mask.sum()} / {len(data)} rows")
    
    return df_rules, rule_classified_mask


def predict_classes(model_package, input_path, output_path, ms1_tolerance_ppm=20, ms2_tolerance_ppm=20, use_rules=True):
    """
    Make predictions on input data and save results
    Combines rule-based and ML-based approaches
    """
    # Load input data
    print(f"\nLoading input data from: {input_path}")
    try:
        data = pd.read_csv(input_path)
        print(f"✓ Data loaded successfully")
        print(f"  Number of samples: {len(data)}")
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        sys.exit(1)
    
    # Initialize result DataFrame
    result = data.copy()
    result['prediction_source'] = ''
    result['predicted_class'] = ''
    result['class_confidence'] = np.nan
    
    # Step 1: Apply rule-based classification if enabled
    rule_classified_mask = pd.Series([False] * len(data))
    
    if use_rules:
        try:
            # Convert MS2 tolerance from ppm to Da (approximate conversion at 500 m/z)
            # For more accurate conversion, you could use actual precursor m/z
            # Formula: tolerance_da = (tolerance_ppm * precursor_mz) / 1e6
            # Using 500 as reference: 20 ppm at 500 m/z = 0.01 Da
            ms2_tolerance_da = (ms2_tolerance_ppm * 500) / 1e6
            
            df_rules, rule_classified_mask = apply_rule_based_classification(
                data, ms1_tolerance_ppm, ms2_tolerance_da
            )
            
            # Copy rule-based results
            result.loc[rule_classified_mask, 'predicted_class'] = df_rules.loc[rule_classified_mask, 'class']
            result.loc[rule_classified_mask, 'category'] = df_rules.loc[rule_classified_mask, 'category']
            result.loc[rule_classified_mask, 'prediction_source'] = 'rule-based'
            result.loc[rule_classified_mask, 'class_confidence'] = 0.8  # High confidence for rule-based
            
            # Add rule-based metadata columns
            result['classes_mz'] = df_rules['classes_mz']
            result['classes_ms2'] = df_rules['classes_ms2']
            
        except Exception as e:
            print(f"⚠ Warning: Rule-based classification failed: {str(e)}")
            print("  Falling back to ML model for all predictions")
    
    # Step 2: Apply ML model for remaining rows
    ml_predict_mask = ~rule_classified_mask
    
    if ml_predict_mask.sum() > 0:
        print(f"\n--- Applying ML model for {ml_predict_mask.sum()} remaining rows ---")
        
        # Extract model and encoders
        model = model_package['model']
        
        # Handle different possible key names for encoders
        class_encoder = model_package.get('class_encoder') or model_package.get('label_encoder')
        ion_mode_encoder = model_package.get('ion_mode_encoder')
        adduct_encoder = model_package.get('adduct_encoder')
        
        if class_encoder is None:
            print(f"✗ Error: class_encoder not found in model package")
            print(f"  Available keys: {list(model_package.keys())}")
            sys.exit(1)
        
        if ion_mode_encoder is None:
            print(f"✗ Error: ion_mode_encoder not found in model package")
            sys.exit(1)
        
        if adduct_encoder is None:
            print(f"✗ Error: adduct_encoder not found in model package")
            sys.exit(1)
        
        # Prepare input features for ML prediction
        print("\nPreparing features for ML model...")
        ml_data = data[ml_predict_mask].copy()
        X, feature_cols, unseen_mask = prepare_input_data(ml_data, ion_mode_encoder, adduct_encoder)
        print(f"✓ Features prepared")
        print(f"  Number of features: {X.shape[1]}")
        
        # Make predictions
        print("\nMaking ML predictions...")
        try:
            predictions_encoded = model.predict(X)
            predictions = class_encoder.inverse_transform(predictions_encoded)
            print(f"✓ ML predictions completed")
        except Exception as e:
            print(f"✗ Error making predictions: {str(e)}")
            sys.exit(1)
        
        # Store ML predictions
        ml_indices = result[ml_predict_mask].index
        result.loc[ml_indices, 'predicted_class'] = predictions
        result.loc[ml_indices, 'prediction_source'] = 'model-based'
        
        # Mark predictions with unseen adducts as "unknown"
        if unseen_mask.any():
            unseen_indices = ml_indices[unseen_mask]
            result.loc[unseen_indices, 'predicted_class'] = 'unknown'
            result.loc[unseen_indices, 'class_confidence'] = 0.0
            print(f"  ⚠ {unseen_mask.sum()} predictions marked as 'unknown' due to unseen adducts")
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X)
                max_probabilities = np.max(probabilities, axis=1)
                
                # Set confidence for valid predictions
                valid_ml_mask = ml_predict_mask.copy()
                valid_ml_mask[ml_indices[unseen_mask]] = False
                result.loc[ml_indices[~unseen_mask], 'class_confidence'] = max_probabilities[~unseen_mask]
                
                print(f"  Average ML confidence: {max_probabilities[~unseen_mask].mean():.4f} (excluding unknowns)")
            except:
                pass
    
    # --- Add Category and Chain Information ---
    # We perform this step here to ensure 'predicted_class' is fully populated from both rules and ML
    print("Processing metadata (Category, Chain info)...")
    
    # Check if 'class' column exists (Ground Truth), if not use 'predicted_class'
    class_source_col = 'predicted_class'
    if 'class' in result.columns:
        print("  Using existing 'class' column for metadata")
        class_source_col = 'class'
    else:
        print("  Using 'predicted_class' for metadata")
    
    # Handle the requirement for 'class' column in class_rule functions
    # If we are using predicted_class, we need to temporarily create 'class' if it doesn't exist
    # because class_rule.add_num_chain expects df['class']
    temp_class_created = False
    
    if class_source_col == 'predicted_class':
        if 'class' not in result.columns:
            result['class'] = result['predicted_class']
            temp_class_created = True
    
    try:
        # Add Number of Chains
        if hasattr(class_rule, 'add_num_chain'):
            result = class_rule.add_num_chain(result)
            
        # Add Final Category
        if hasattr(class_rule, 'add_final_category'):
            result = class_rule.add_final_category(result)
            
        print("✓ Category and chain information added")
            
    except Exception as e:
        print(f"⚠ Warning: Could not add metadata: {e}")
    
    # If we created a temporary 'class' column, remove it so we don't output false ground truth
    if temp_class_created:
        result = result.drop(columns=['class'])

    # Save results
    print(f"\nSaving results to: {output_path}")
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result.to_csv(output_path, index=False)
        print(f"✓ Results saved successfully")
    except Exception as e:
        print(f"✗ Error saving results: {str(e)}")
        sys.exit(1)
    
    # Display summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total predictions: {len(result)}")
    print(f"\nPrediction sources:")
    print(result['prediction_source'].value_counts().to_string())
    
    if rule_classified_mask.any():
        print(f"\nRule-based predictions: {rule_classified_mask.sum()}")
    if ml_predict_mask.any():
        print(f"ML-based predictions: {ml_predict_mask.sum()}")
        if unseen_mask.any():
            print(f"  └─ Unknown (unseen adducts): {unseen_mask.sum()}")
    
    print(f"\nPredicted class distribution:")
    print(result['predicted_class'].value_counts().to_string())
    
    if 'category' in result.columns:
        print(f"\nCategory distribution:")
        print(result['category'].value_counts().to_string())
    
    print("="*70)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Predict lipid classes using rule-based + ML hybrid approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python class_predict.py test.csv model/class.joblib --output_path result/pred_class.csv
  python class_predict.py data.csv class.joblib -o predictions.csv --ms1_tol 15 --ms2_tol 20
  python class_predict.py data.csv class.joblib --no-rules  # ML only, skip rules
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input CSV file containing features'
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to trained model (.joblib file)'
    )
    
    parser.add_argument(
        '--output_path', '-o',
        type=str,
        default='class_pred.csv',
        help='Path to save predictions (default: class_pred.csv)'
    )
    
    parser.add_argument(
        '--ms1_tol',
        type=float,
        default=10.0,
        help='MS1 mass tolerance in ppm for rule-based classification (default: 20.0)'
    )
    
    parser.add_argument(
        '--ms2_tol',
        type=float,
        default=20.0,
        help='MS2 mass tolerance in ppm for rule-based classification (default: 20.0)'
    )
    
    parser.add_argument(
        '--no-rules',
        action='store_true',
        help='Skip rule-based classification and use only ML model'
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.input_path).exists():
        print(f"✗ Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    # Verify model file exists
    if not Path(args.model_path).exists():
        print(f"✗ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    print("="*70)
    print("CLASS PREDICTION (HYBRID: RULE-BASED + ML)")
    print("="*70)
    
    # Load model
    model_package = load_model(args.model_path)
    
    # Make predictions
    use_rules = not args.no_rules
    results = predict_classes(
        model_package, 
        args.input_path, 
        args.output_path,
        ms1_tolerance_ppm=args.ms1_tol,
        ms2_tolerance_ppm=args.ms2_tol,
        use_rules=use_rules
    )
    
    # Note: Metadata (Chain/Category) is now handled inside predict_classes

if __name__ == "__main__":
    main()