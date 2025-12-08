

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
#             unseen_mask = pd.Series([False] * len(data_processed), index=data_processed.index)
        
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
    
#     print("✓ Rule-based classification complete")
    
#     return df_rules


# def compute_class_overlap(classes_mz, classes_ms2, all_classes=None):
#     """
#     Compute the overlap between mz-based and MS2-based class identifications.
    
#     NEW LOGIC:
#     - If classes_mz is empty/[], treat as "all possible" (don't restrict by mz)
#     - If classes_ms2 is empty/[], treat as "all possible" (don't restrict by ms2)
#     - Return the list of overlapping classes (not just the first one)
    
#     Args:
#         classes_mz: List of classes from MS1/mz identification
#         classes_ms2: List of classes from MS2 identification  
#         all_classes: List of all possible classes (optional, for when both are empty)
    
#     Returns:
#         list: List of overlapping classes (preserving order from classes_mz if available)
#     """
#     # Handle empty lists - treat as "all possible"
#     mz_empty = not classes_mz or classes_mz == []
#     ms2_empty = not classes_ms2 or classes_ms2 == []
    
#     if mz_empty and ms2_empty:
#         # Both empty - return all classes if provided, else empty list
#         return list(all_classes) if all_classes else []
    
#     if mz_empty:
#         # MS1 is empty (all possible), so use MS2 results
#         return list(classes_ms2)
    
#     if ms2_empty:
#         # MS2 is empty (all possible), so use MS1 results
#         return list(classes_mz)
    
#     # Both have values - find intersection while preserving MS1 order
#     set_ms2 = set(classes_ms2)
#     ordered_overlap = [cls for cls in classes_mz if cls in set_ms2]
    
#     return ordered_overlap


# def predict_classes(model_package, input_path, output_path, ms1_tolerance_ppm=20, ms2_tolerance_ppm=20, use_rules=True):
#     """
#     Make predictions on input data and save results
    
#     IMPROVED PIPELINE:
#     1. Apply rule-based classification to get classes_mz and classes_ms2
#     2. Compute overlap between classes_mz and classes_ms2
#        - If either is empty [], treat as "all possible classes"
#     3. If overlap has exactly 1 class: Use that class (rule-based)
#     4. If overlap has > 1 class: Use ML model to predict within overlap set (model-based)
#     5. If overlap is empty: Use ML model to predict from all classes (model-based)
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
#     result['class_confidence'] = np.nan
#     result['overlap_classes'] = None  # Store the overlap for reference
    
#     # Extract model and encoders
#     model = model_package['model']
#     class_encoder = model_package.get('class_encoder') or model_package.get('label_encoder')
#     ion_mode_encoder = model_package.get('ion_mode_encoder')
#     adduct_encoder = model_package.get('adduct_encoder')
    
#     if class_encoder is None:
#         print(f"✗ Error: class_encoder not found in model package")
#         print(f"  Available keys: {list(model_package.keys())}")
#         sys.exit(1)
    
#     if ion_mode_encoder is None:
#         print(f"✗ Error: ion_mode_encoder not found in model package")
#         sys.exit(1)
    
#     if adduct_encoder is None:
#         print(f"✗ Error: adduct_encoder not found in model package")
#         sys.exit(1)
    
#     # Get all possible classes from the encoder
#     all_classes = list(class_encoder.classes_)
#     print(f"  All possible classes ({len(all_classes)}): {all_classes}")
    
#     # Step 1: Apply rule-based classification if enabled
#     df_rules = None
#     if use_rules:
#         try:
#             # Convert MS2 tolerance from ppm to Da (approximate conversion at 500 m/z)
#             ms2_tolerance_da = (ms2_tolerance_ppm * 500) / 1e6
            
#             df_rules = apply_rule_based_classification(
#                 data, ms1_tolerance_ppm, ms2_tolerance_da
#             )
            
#             # Add rule-based metadata columns
#             result['classes_mz'] = df_rules['classes_mz']
#             result['classes_ms2'] = df_rules['classes_ms2']
            
#         except Exception as e:
#             print(f"⚠ Warning: Rule-based classification failed: {str(e)}")
#             print("  Falling back to ML model for all predictions")
#             result['classes_mz'] = [[] for _ in range(len(data))]
#             result['classes_ms2'] = [[] for _ in range(len(data))]
#     else:
#         result['classes_mz'] = [[] for _ in range(len(data))]
#         result['classes_ms2'] = [[] for _ in range(len(data))]
    
#     # Step 2: Compute overlap for each row
#     print("\n--- Computing class overlaps ---")
#     overlap_list = []
#     for idx in result.index:
#         classes_mz = result.loc[idx, 'classes_mz']
#         classes_ms2 = result.loc[idx, 'classes_ms2']
#         overlap = compute_class_overlap(classes_mz, classes_ms2, all_classes)
#         overlap_list.append(overlap)
    
#     result['overlap_classes'] = overlap_list
    
#     # Step 3: Categorize rows based on overlap size
#     single_overlap_mask = result['overlap_classes'].apply(lambda x: len(x) == 1)
#     multi_overlap_mask = result['overlap_classes'].apply(lambda x: len(x) > 1)
#     no_overlap_mask = result['overlap_classes'].apply(lambda x: len(x) == 0)
    
#     print(f"  Single overlap (rule-based): {single_overlap_mask.sum()} rows")
#     print(f"  Multiple overlap (model-based with constraint): {multi_overlap_mask.sum()} rows")
#     print(f"  No overlap (model-based, unconstrained): {no_overlap_mask.sum()} rows")
    
#     # Step 4: Handle single overlap cases (rule-based)
#     if single_overlap_mask.any():
#         print("\n--- Applying rule-based classification for single overlap ---")
#         single_indices = result[single_overlap_mask].index
#         for idx in single_indices:
#             result.loc[idx, 'predicted_class'] = result.loc[idx, 'overlap_classes'][0]
#             result.loc[idx, 'prediction_source'] = 'rule-based'
#             result.loc[idx, 'class_confidence'] = 1.0
#         print(f"✓ Rule-based classification complete: {len(single_indices)} rows")
    
#     # Step 5: Handle multi-overlap and no-overlap cases (model-based)
#     ml_predict_mask = multi_overlap_mask | no_overlap_mask
    
#     if ml_predict_mask.sum() > 0:
#         print(f"\n--- Applying ML model for {ml_predict_mask.sum()} remaining rows ---")
        
#         # Prepare input features for ML prediction
#         print("\nPreparing features for ML model...")
#         ml_data = data[ml_predict_mask].copy()
#         X, feature_cols, unseen_mask = prepare_input_data(ml_data, ion_mode_encoder, adduct_encoder)
#         print(f"✓ Features prepared")
#         print(f"  Number of features: {X.shape[1]}")
        
#         # Get prediction probabilities
#         print("\nMaking ML predictions...")
#         try:
#             if hasattr(model, 'predict_proba'):
#                 probabilities = model.predict_proba(X)
#                 class_names = class_encoder.classes_
                
#                 ml_indices = result[ml_predict_mask].index
                
#                 for i, idx in enumerate(ml_indices):
#                     overlap = result.loc[idx, 'overlap_classes']
                    
#                     if len(overlap) > 1:
#                         # Multi-overlap: Constrain prediction to overlap set
#                         # Get indices of overlap classes in the encoder
#                         overlap_indices = [np.where(class_names == cls)[0][0] 
#                                          for cls in overlap if cls in class_names]
                        
#                         if overlap_indices:
#                             # Get probabilities only for overlap classes
#                             overlap_probs = probabilities[i, overlap_indices]
#                             # Find the best class within the overlap
#                             best_overlap_idx = np.argmax(overlap_probs)
#                             predicted_class = class_names[overlap_indices[best_overlap_idx]]
#                             confidence = overlap_probs[best_overlap_idx]
#                         else:
#                             # Fallback if no overlap classes found in encoder
#                             predicted_idx = np.argmax(probabilities[i])
#                             predicted_class = class_names[predicted_idx]
#                             confidence = probabilities[i, predicted_idx]
#                     else:
#                         # No overlap: Unconstrained prediction
#                         predicted_idx = np.argmax(probabilities[i])
#                         predicted_class = class_names[predicted_idx]
#                         confidence = probabilities[i, predicted_idx]
                    
#                     result.loc[idx, 'predicted_class'] = predicted_class
#                     result.loc[idx, 'prediction_source'] = 'model-based'
#                     result.loc[idx, 'class_confidence'] = confidence
                
#                 print(f"✓ ML predictions completed")
                
#                 # Mark predictions with unseen adducts as "unknown"
#                 if unseen_mask.any():
#                     unseen_indices = ml_indices[unseen_mask.values]
#                     result.loc[unseen_indices, 'predicted_class'] = 'unknown'
#                     result.loc[unseen_indices, 'class_confidence'] = 0.0
#                     print(f"  ⚠ {unseen_mask.sum()} predictions marked as 'unknown' due to unseen adducts")
                    
#             else:
#                 # Model doesn't support predict_proba, use regular predict
#                 predictions_encoded = model.predict(X)
#                 predictions = class_encoder.inverse_transform(predictions_encoded)
                
#                 ml_indices = result[ml_predict_mask].index
                
#                 # For models without predict_proba, we can't constrain to overlap
#                 # Use regular predictions
#                 for i, idx in enumerate(ml_indices):
#                     overlap = result.loc[idx, 'overlap_classes']
#                     predicted_class = predictions[i]
                    
#                     # If multi-overlap and prediction not in overlap, use first overlap class
#                     if len(overlap) > 1 and predicted_class not in overlap:
#                         predicted_class = overlap[0]
                    
#                     result.loc[idx, 'predicted_class'] = predicted_class
#                     result.loc[idx, 'prediction_source'] = 'model-based'
                
#                 print(f"✓ ML predictions completed (no probabilities available)")
                
#         except Exception as e:
#             print(f"✗ Error making predictions: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             sys.exit(1)
    
#     # --- Add Category and Chain Information ---
#     print("\nProcessing metadata (Category, Chain info)...")
    
#     # Handle the requirement for 'class' column in class_rule functions
#     temp_class_created = False
#     if 'class' not in result.columns:
#         result['class'] = result['predicted_class']
#         temp_class_created = True
    
#     try:
#         # Add Number of Chains
#         if hasattr(class_rule, 'add_num_chain'):
#             result = class_rule.add_num_chain(result)
            
#         # Add Final Category
#         if hasattr(class_rule, 'add_final_category'):
#             result = class_rule.add_final_category(result)
            
#         print("✓ Category and chain information added")
            
#     except Exception as e:
#         print(f"⚠ Warning: Could not add metadata: {e}")
    
#     # If we created a temporary 'class' column, remove it so we don't output false ground truth
#     if temp_class_created:
#         result = result.drop(columns=['class'])
    
#     # Convert overlap_classes list to string for CSV output
#     result['overlap_classes'] = result['overlap_classes'].apply(lambda x: str(x) if x else '[]')
    
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
    
#     print(f"\nRule-based predictions (single overlap): {single_overlap_mask.sum()}")
#     print(f"Model-based predictions (multi/no overlap): {ml_predict_mask.sum()}")
#     if multi_overlap_mask.any():
#         print(f"  └─ Multi-overlap (constrained): {multi_overlap_mask.sum()}")
#     if no_overlap_mask.any():
#         print(f"  └─ No overlap (unconstrained): {no_overlap_mask.sum()}")
    
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
#   python class_predict_x.py test.csv model/class.joblib --output_path result/pred_class.csv
#   python class_predict_x.py data.csv class.joblib -o predictions.csv --ms1_tol 15 --ms2_tol 20
#   python class_predict_x.py data.csv class.joblib --no-rules  # ML only, skip rules
  
# IMPROVED PREDICTION LOGIC:
#   - If classes_mz or classes_ms2 is empty [], treated as "all possible classes"
#   - Single overlap → rule-based prediction
#   - Multiple overlap → model-based prediction (constrained to overlap)
#   - No overlap → model-based prediction (unconstrained)
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
#         help='MS1 mass tolerance in ppm for rule-based classification (default: 10.0)'
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


# if __name__ == "__main__":
#     main()


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
#     result['class_confidence'] = np.nan
    
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
#             result.loc[rule_classified_mask, 'class_confidence'] = 0.8  # High confidence for rule-based
            
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
#             result.loc[unseen_indices, 'class_confidence'] = 0.0
#             print(f"  ⚠ {unseen_mask.sum()} predictions marked as 'unknown' due to unseen adducts")
        
#         # Get prediction probabilities if available
#         if hasattr(model, 'predict_proba'):
#             try:
#                 probabilities = model.predict_proba(X)
#                 max_probabilities = np.max(probabilities, axis=1)
                
#                 # Set confidence for valid predictions
#                 valid_ml_mask = ml_predict_mask.copy()
#                 valid_ml_mask[ml_indices[unseen_mask]] = False
#                 result.loc[ml_indices[~unseen_mask], 'class_confidence'] = max_probabilities[~unseen_mask]
                
#                 print(f"  Average ML confidence: {max_probabilities[~unseen_mask].mean():.4f} (excluding unknowns)")
#             except:
#                 pass
    
#     # --- Add Category and Chain Information ---
#     # We perform this step here to ensure 'predicted_class' is fully populated from both rules and ML
#     print("Processing metadata (Category, Chain info)...")
    
#     # Check if 'class' column exists (Ground Truth), if not use 'predicted_class'
#     class_source_col = 'predicted_class'
#     if 'class' in result.columns:
#         print("  Using existing 'class' column for metadata")
#         class_source_col = 'class'
#     else:
#         print("  Using 'predicted_class' for metadata")
    
#     # Handle the requirement for 'class' column in class_rule functions
#     # If we are using predicted_class, we need to temporarily create 'class' if it doesn't exist
#     # because class_rule.add_num_chain expects df['class']
#     temp_class_created = False
    
#     if class_source_col == 'predicted_class':
#         if 'class' not in result.columns:
#             result['class'] = result['predicted_class']
#             temp_class_created = True
    
#     try:
#         # Add Number of Chains
#         if hasattr(class_rule, 'add_num_chain'):
#             result = class_rule.add_num_chain(result)
            
#         # Add Final Category
#         if hasattr(class_rule, 'add_final_category'):
#             result = class_rule.add_final_category(result)
            
#         print("✓ Category and chain information added")
            
#     except Exception as e:
#         print(f"⚠ Warning: Could not add metadata: {e}")
    
#     # If we created a temporary 'class' column, remove it so we don't output false ground truth
#     if temp_class_created:
#         result = result.drop(columns=['class'])

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
    
#     # Note: Metadata (Chain/Category) is now handled inside predict_classes

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
            unseen_mask = pd.Series([False] * len(data_processed), index=data_processed.index)
        
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
    
    print("✓ Rule-based classification complete")
    
    return df_rules


def compute_class_overlap(classes_mz, classes_ms2, all_classes=None):
    """
    Compute the overlap between mz-based and MS2-based class identifications.
    
    NEW LOGIC:
    - If classes_mz is empty/[], treat as "all possible" (don't restrict by mz)
    - If classes_ms2 is empty/[], treat as "all possible" (don't restrict by ms2)
    - Return the list of overlapping classes (not just the first one)
    
    Args:
        classes_mz: List of classes from MS1/mz identification
        classes_ms2: List of classes from MS2 identification  
        all_classes: List of all possible classes (optional, for when both are empty)
    
    Returns:
        list: List of overlapping classes (preserving order from classes_mz if available)
    """
    # Handle empty lists - treat as "all possible"
    mz_empty = not classes_mz or classes_mz == []
    ms2_empty = not classes_ms2 or classes_ms2 == []
    
    if mz_empty and ms2_empty:
        # Both empty - return all classes if provided, else empty list
        return list(all_classes) if all_classes else []
    
    if mz_empty:
        # MS1 is empty (all possible), so use MS2 results
        return list(classes_ms2)
    
    if ms2_empty:
        # MS2 is empty (all possible), so use MS1 results
        return list(classes_mz)
    
    # Both have values - find intersection while preserving MS1 order
    set_ms2 = set(classes_ms2)
    ordered_overlap = [cls for cls in classes_mz if cls in set_ms2]
    
    return ordered_overlap


def predict_classes(model_package, input_path, output_path, ms1_tolerance_ppm=20, ms2_tolerance_ppm=20, use_rules=True):
    """
    Make predictions on input data and save results
    
    IMPROVED PIPELINE:
    1. Apply rule-based classification to get classes_mz and classes_ms2
    2. Compute overlap between classes_mz and classes_ms2
       - If either is empty [], treat as "all possible classes"
    3. If overlap has exactly 1 class: Use that class (rule-based)
    4. If overlap has > 1 class: Use ML model to predict within overlap set (model-based)
    5. If overlap is empty: Use ML model to predict from all classes (model-based)
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
    
    # Check if 'adduct' column exists; if not, use 'predicted_adduct' for all processing
    if 'adduct' not in data.columns:
        if 'predicted_adduct' in data.columns:
            print(f"  ℹ No 'adduct' column found. Using 'predicted_adduct' for all processing.")
            data['adduct'] = data['predicted_adduct']
        else:
            print(f"✗ Error: Neither 'adduct' nor 'predicted_adduct' column found in data")
            sys.exit(1)
    
    # Initialize result DataFrame
    result = data.copy()
    result['prediction_source'] = ''
    result['predicted_class'] = ''
    result['class_confidence'] = np.nan
    result['overlap_classes'] = None  # Store the overlap for reference
    
    # Extract model and encoders
    model = model_package['model']
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
    
    # Get all possible classes from the encoder
    all_classes = list(class_encoder.classes_)
    print(f"  All possible classes ({len(all_classes)}): {all_classes}")
    
    # Step 1: Apply rule-based classification if enabled
    df_rules = None
    if use_rules:
        try:
            # Convert MS2 tolerance from ppm to Da (approximate conversion at 500 m/z)
            ms2_tolerance_da = (ms2_tolerance_ppm * 500) / 1e6
            
            df_rules = apply_rule_based_classification(
                data, ms1_tolerance_ppm, ms2_tolerance_da
            )
            
            # Add rule-based metadata columns
            result['classes_mz'] = df_rules['classes_mz']
            result['classes_ms2'] = df_rules['classes_ms2']
            
        except Exception as e:
            print(f"⚠ Warning: Rule-based classification failed: {str(e)}")
            print("  Falling back to ML model for all predictions")
            result['classes_mz'] = [[] for _ in range(len(data))]
            result['classes_ms2'] = [[] for _ in range(len(data))]
    else:
        result['classes_mz'] = [[] for _ in range(len(data))]
        result['classes_ms2'] = [[] for _ in range(len(data))]
    
    # Step 2: Compute overlap for each row
    print("\n--- Computing class overlaps ---")
    overlap_list = []
    for idx in result.index:
        classes_mz = result.loc[idx, 'classes_mz']
        classes_ms2 = result.loc[idx, 'classes_ms2']
        overlap = compute_class_overlap(classes_mz, classes_ms2, all_classes)
        overlap_list.append(overlap)
    
    result['overlap_classes'] = overlap_list
    
    # Step 3: Categorize rows based on overlap size
    single_overlap_mask = result['overlap_classes'].apply(lambda x: len(x) == 1)
    multi_overlap_mask = result['overlap_classes'].apply(lambda x: len(x) > 1)
    no_overlap_mask = result['overlap_classes'].apply(lambda x: len(x) == 0)
    
    print(f"  Single overlap (rule-based): {single_overlap_mask.sum()} rows")
    print(f"  Multiple overlap (model-based with constraint): {multi_overlap_mask.sum()} rows")
    print(f"  No overlap (model-based, unconstrained): {no_overlap_mask.sum()} rows")
    
    # Step 4: Handle single overlap cases (rule-based)
    if single_overlap_mask.any():
        print("\n--- Applying rule-based classification for single overlap ---")
        single_indices = result[single_overlap_mask].index
        for idx in single_indices:
            result.loc[idx, 'predicted_class'] = result.loc[idx, 'overlap_classes'][0]
            result.loc[idx, 'prediction_source'] = 'rule-based'
            result.loc[idx, 'class_confidence'] = 1.0
        print(f"✓ Rule-based classification complete: {len(single_indices)} rows")
    
    # Step 5: Handle multi-overlap and no-overlap cases (model-based)
    ml_predict_mask = multi_overlap_mask | no_overlap_mask
    
    if ml_predict_mask.sum() > 0:
        print(f"\n--- Applying ML model for {ml_predict_mask.sum()} remaining rows ---")
        
        # Prepare input features for ML prediction
        print("\nPreparing features for ML model...")
        ml_data = data[ml_predict_mask].copy()
        X, feature_cols, unseen_mask = prepare_input_data(ml_data, ion_mode_encoder, adduct_encoder)
        print(f"✓ Features prepared")
        print(f"  Number of features: {X.shape[1]}")
        
        # Get prediction probabilities
        print("\nMaking ML predictions...")
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                class_names = class_encoder.classes_
                
                ml_indices = result[ml_predict_mask].index
                
                for i, idx in enumerate(ml_indices):
                    overlap = result.loc[idx, 'overlap_classes']
                    
                    if len(overlap) > 1:
                        # Multi-overlap: Constrain prediction to overlap set
                        # Get indices of overlap classes in the encoder
                        overlap_indices = [np.where(class_names == cls)[0][0] 
                                         for cls in overlap if cls in class_names]
                        
                        if overlap_indices:
                            # Get probabilities only for overlap classes
                            overlap_probs = probabilities[i, overlap_indices]
                            # Find the best class within the overlap
                            best_overlap_idx = np.argmax(overlap_probs)
                            predicted_class = class_names[overlap_indices[best_overlap_idx]]
                            confidence = overlap_probs[best_overlap_idx]
                        else:
                            # Fallback if no overlap classes found in encoder
                            predicted_idx = np.argmax(probabilities[i])
                            predicted_class = class_names[predicted_idx]
                            confidence = probabilities[i, predicted_idx]
                    else:
                        # No overlap: Unconstrained prediction
                        predicted_idx = np.argmax(probabilities[i])
                        predicted_class = class_names[predicted_idx]
                        confidence = probabilities[i, predicted_idx]
                    
                    result.loc[idx, 'predicted_class'] = predicted_class
                    result.loc[idx, 'prediction_source'] = 'model-based'
                    result.loc[idx, 'class_confidence'] = confidence
                
                print(f"✓ ML predictions completed")
                
                # Mark predictions with unseen adducts as "unknown"
                if unseen_mask.any():
                    unseen_indices = ml_indices[unseen_mask.values]
                    result.loc[unseen_indices, 'predicted_class'] = 'unknown'
                    result.loc[unseen_indices, 'class_confidence'] = 0.0
                    print(f"  ⚠ {unseen_mask.sum()} predictions marked as 'unknown' due to unseen adducts")
                    
            else:
                # Model doesn't support predict_proba, use regular predict
                predictions_encoded = model.predict(X)
                predictions = class_encoder.inverse_transform(predictions_encoded)
                
                ml_indices = result[ml_predict_mask].index
                
                # For models without predict_proba, we can't constrain to overlap
                # Use regular predictions
                for i, idx in enumerate(ml_indices):
                    overlap = result.loc[idx, 'overlap_classes']
                    predicted_class = predictions[i]
                    
                    # If multi-overlap and prediction not in overlap, use first overlap class
                    if len(overlap) > 1 and predicted_class not in overlap:
                        predicted_class = overlap[0]
                    
                    result.loc[idx, 'predicted_class'] = predicted_class
                    result.loc[idx, 'prediction_source'] = 'model-based'
                
                print(f"✓ ML predictions completed (no probabilities available)")
                
        except Exception as e:
            print(f"✗ Error making predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # --- Add Category and Chain Information ---
    print("\nProcessing metadata (Category, Chain info)...")
    
    # Handle the requirement for 'class' column in class_rule functions
    temp_class_created = False
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
    
    # Convert overlap_classes list to string for CSV output
    result['overlap_classes'] = result['overlap_classes'].apply(lambda x: str(x) if x else '[]')
    
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
    
    print(f"\nRule-based predictions (single overlap): {single_overlap_mask.sum()}")
    print(f"Model-based predictions (multi/no overlap): {ml_predict_mask.sum()}")
    if multi_overlap_mask.any():
        print(f"  └─ Multi-overlap (constrained): {multi_overlap_mask.sum()}")
    if no_overlap_mask.any():
        print(f"  └─ No overlap (unconstrained): {no_overlap_mask.sum()}")
    
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
  python class_predict_x.py test.csv model/class.joblib --output_path result/pred_class.csv
  python class_predict_x.py data.csv class.joblib -o predictions.csv --ms1_tol 15 --ms2_tol 20
  python class_predict_x.py data.csv class.joblib --no-rules  # ML only, skip rules
  
IMPROVED PREDICTION LOGIC:
  - If classes_mz or classes_ms2 is empty [], treated as "all possible classes"
  - Single overlap → rule-based prediction
  - Multiple overlap → model-based prediction (constrained to overlap)
  - No overlap → model-based prediction (unconstrained)
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
        help='MS1 mass tolerance in ppm for rule-based classification (default: 10.0)'
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


if __name__ == "__main__":
    main()