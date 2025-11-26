# """
# Predict lipid chain compositions using trained PLSF model

# Usage:
#     python predict_plsf.py plsf_model.joblib input.csv
#     python predict_plsf.py plsf_model.joblib input.csv --output_path results/output.csv
#     python predict_plsf.py plsf_model.joblib input.csv -o results/output.csv

# The input CSV should contain the required columns:
#     - precursor_mz (will be rounded to 0.01)
#     - adduct (categorical) or predicted_adduct if adduct is missing
#     - class (categorical) or predicted_class if class is missing
#     - num_chain (1, 2, 3, or 4)
#     - mz_* columns (binary 0/1)

# Special handling:
#     - If adduct or class are missing, predicted_adduct/predicted_class will be used
#     - Single-chain lipids (num_chain=1) use direct mass calculation instead of model
#     - All predictions include a plsf_confidence score (0-1)

# Output CSV will contain:
#     - All input columns (with categorical values decoded)
#     - num_c_total, num_db_total (predicted totals)
#     - num_c_1, num_db_1, num_c_2, num_db_2, num_c_3, num_db_3, num_c_4, num_db_4
#     - plsf_confidence (confidence score for the prediction)
# """

# import pandas as pd
# import numpy as np
# import joblib
# import argparse
# import sys
# import os
# from pathlib import Path
# from tqdm import tqdm


# class PLSFPredictor:
#     """Predictor class for PLSF models with confidence scores and single-chain handling"""
    
#     # Adduct masses for single-chain calculation
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
    
#     def __init__(self, model_path):
#         """
#         Load the trained model
        
#         Args:
#             model_path: Path to .joblib model file
#         """
#         print(f"Loading model from {model_path}...")
#         try:
#             self.model_data = joblib.load(model_path)
#         except FileNotFoundError:
#             print(f"Error: Model file '{model_path}' not found!")
#             sys.exit(1)
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             sys.exit(1)
        
#         # Extract model components
#         self.model_name = self.model_data['model_name']
#         self.models = self.model_data['models']
#         self.label_encoders = self.model_data['label_encoders']
#         self.base_features = self.model_data['base_features']
#         self.target_pairs = self.model_data['target_pairs']
#         self.split_by_chain = self.model_data['split_by_chain']
        
#         print(f"✓ Loaded model: {self.model_name}")
#         print(f"  Mode: {'Split by Chain' if self.split_by_chain else 'Unified'}")
#         # print(f"  Test Accuracy: {self.model_data['test_accuracy']:.2f}%")
    
#     def _predict_with_inverse_map(self, model, X):
#         """Make prediction and apply inverse mapping if XGBoost"""
#         # Convert to numpy array to avoid feature name warnings
#         if isinstance(X, pd.DataFrame):
#             X = X.values
#         pred = model.predict(X).astype(int)
#         if hasattr(model, 'inverse_map'):
#             pred = pd.Series(pred).map(model.inverse_map).values
#         return pred
    
#     def _get_prediction_confidence(self, model, X):
#         """
#         Get confidence score for prediction
#         Returns probability of the predicted class
#         """
#         if isinstance(X, pd.DataFrame):
#             X = X.values
        
#         try:
#             if hasattr(model, 'predict_proba'):
#                 # Get probability for each class
#                 proba = model.predict_proba(X)
#                 # Get max probability (confidence in predicted class)
#                 confidence = np.max(proba, axis=1)
#                 return confidence
#             else:
#                 # If model doesn't support predict_proba, return 1.0
#                 return np.ones(len(X))
#         except Exception:
#             return np.ones(len(X))
    
#     def _get_top_k_predictions(self, model, X, k=5):
#         """
#         Get top-k predictions with probabilities
        
#         Returns:
#             top_k_classes: Array of shape (n_samples, k) with class predictions
#             top_k_probs: Array of shape (n_samples, k) with probabilities
#         """
#         if isinstance(X, pd.DataFrame):
#             X = X.values
        
#         # Get all class probabilities
#         proba = model.predict_proba(X)
        
#         # Get top k indices
#         top_k_indices = np.argsort(proba, axis=1)[:, -k:][:, ::-1]  # Descending order
        
#         # Get corresponding probabilities
#         top_k_probs = np.take_along_axis(proba, top_k_indices, axis=1)
        
#         # Map back to original labels if model has inverse_map
#         if hasattr(model, 'inverse_map'):
#             inverse_map = model.inverse_map
#             top_k_classes = np.vectorize(lambda x: inverse_map[x])(top_k_indices)
#         else:
#             top_k_classes = top_k_indices
        
#         return top_k_classes, top_k_probs
    
#     def _format_chains_as_list(self, vals, max_stage):
#         """
#         Format chain predictions as sorted list of 8 integers
        
#         Args:
#             vals: List of values [c1, db1, c2, db2, ...]
#             max_stage: Number of chains
            
#         Returns:
#             List of 8 integers [c1, db1, c2, db2, c3, db3, c4, db4]
#             sorted by c (descending), then db (descending)
#         """
#         # Extract chains as (c, db) tuples
#         chains = []
#         for i in range(max_stage):
#             c = int(vals[i*2])
#             db = int(vals[i*2+1])
#             if c > 0 or db > 0:  # Only include non-zero chains
#                 chains.append((c, db))
        
#         # Sort chains: by c descending, then by db descending
#         chains.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
#         # Pad with zeros to have 4 chains total
#         while len(chains) < 4:
#             chains.append((0, 0))
        
#         # Flatten to list of 8 integers
#         result = []
#         for c, db in chains:
#             result.extend([c, db])
        
#         return result
    
#     def _prepare_data(self, df):
#         """Prepare input data (same preprocessing as training)"""
#         df = df.copy()
        
#         # Round precursor_mz
#         df['precursor_mz'] = df['precursor_mz'].round(2)
        
#         # Handle missing adduct and class by using predicted values if available
#         if 'adduct' not in df.columns:
#             if 'predicted_adduct' in df.columns:
#                 df['adduct'] = df['predicted_adduct']
#                 print("  Note: Using 'predicted_adduct' for missing 'adduct' column")
#             else:
#                 raise ValueError("Required column 'adduct' or 'predicted_adduct' not found")
#         else:
#             # Fill missing adduct values with predicted_adduct if available
#             if 'predicted_adduct' in df.columns:
#                 mask = df['adduct'].isna() | (df['adduct'] == '')
#                 if mask.any():
#                     df.loc[mask, 'adduct'] = df.loc[mask, 'predicted_adduct']
#                     print(f"  Note: Filled {mask.sum()} missing adduct values with predicted_adduct")
        
#         if 'class' not in df.columns:
#             if 'predicted_class' in df.columns:
#                 df['class'] = df['predicted_class']
#                 print("  Note: Using 'predicted_class' for missing 'class' column")
#             else:
#                 raise ValueError("Required column 'class' or 'predicted_class' not found")
#         else:
#             # Fill missing class values with predicted_class if available
#             if 'predicted_class' in df.columns:
#                 mask = df['class'].isna() | (df['class'] == '')
#                 if mask.any():
#                     df.loc[mask, 'class'] = df.loc[mask, 'predicted_class']
#                     print(f"  Note: Filled {mask.sum()} missing class values with predicted_class")
        
#         # Encode categorical variables
#         for col in ['adduct', 'class']:
#             # Check for unknown categories
#             unknown = set(df[col].unique()) - set(self.label_encoders[col].classes_)
#             if unknown:
#                 print(f"  Warning: Unknown values in '{col}': {unknown}")
#                 print(f"  These will be mapped to the first known category")
#                 # Map unknown to first known category
#                 df[col] = df[col].apply(
#                     lambda x: self.label_encoders[col].classes_[0] if x not in self.label_encoders[col].classes_ else x
#                 )
            
#             df[col] = self.label_encoders[col].transform(df[col])
        
#         return df
    
#     def _predict_unified(self, df):
#         """Make predictions using unified model"""
#         df_pred = df.copy()
        
#         # Predict totals for all
#         X_base = df_pred[self.base_features]
#         df_pred['num_c_total'] = self._predict_with_inverse_map(
#             self.models['total']['c_model'], X_base
#         )
#         df_pred['num_db_total'] = self._predict_with_inverse_map(
#             self.models['total']['db_model'], X_base
#         )
        
#         # Get confidence for totals
#         conf_c_total = self._get_prediction_confidence(self.models['total']['c_model'], X_base)
#         conf_db_total = self._get_prediction_confidence(self.models['total']['db_model'], X_base)
        
#         # Predict individual chains
#         confidences = []
#         for stage_idx, (c_col, db_col) in enumerate(self.target_pairs):
#             # Get features for this stage
#             features = self.base_features.copy()
#             features.extend(['num_c_total', 'num_db_total'])
#             for i in range(stage_idx):
#                 features.extend([f'num_c_{i+1}', f'num_db_{i+1}'])
            
#             X = df_pred[features]
            
#             # Predict num_c
#             df_pred[c_col] = self._predict_with_inverse_map(
#                 self.models['chains'][stage_idx]['c_model'], X
#             )
#             conf_c = self._get_prediction_confidence(
#                 self.models['chains'][stage_idx]['c_model'], X
#             )
            
#             # Predict num_db
#             X_with_c = df_pred[features + [c_col]]
#             df_pred[db_col] = self._predict_with_inverse_map(
#                 self.models['chains'][stage_idx]['db_model'], X_with_c
#             )
#             conf_db = self._get_prediction_confidence(
#                 self.models['chains'][stage_idx]['db_model'], X_with_c
#             )
        
#         # Calculate overall confidence (average of all predictions)
#         df_pred['plsf_confidence'] = 0.0
#         for i in range(len(df_pred)):
#             all_confs = [conf_c_total[i], conf_db_total[i]]
#             overall_conf = np.mean(all_confs)
#             df_pred.iloc[i, df_pred.columns.get_loc('plsf_confidence')] = overall_conf
        
#         return df_pred
    
#     def _predict_split_by_chain(self, df):
#         """Make predictions using split-by-chain models with top-5 alternatives"""
#         df_pred = df.copy()
        
#         # Predict totals
#         X_base = df_pred[self.base_features]
#         c_total_top, _ = self._get_top_k_predictions(
#             self.models['total']['c_model'], X_base, k=1
#         )
#         df_pred['num_c_total'] = c_total_top[:, 0]
        
#         db_total_top, _ = self._get_top_k_predictions(
#             self.models['total']['db_model'], X_base, k=1
#         )
#         df_pred['num_db_total'] = db_total_top[:, 0]
        
#         # Get confidence for totals
#         conf_c_total = self._get_prediction_confidence(self.models['total']['c_model'], X_base)
#         conf_db_total = self._get_prediction_confidence(self.models['total']['db_model'], X_base)
        
#         # Initialize all chain predictions to 0
#         for c_col, db_col in self.target_pairs:
#             df_pred[c_col] = 0
#             df_pred[db_col] = 0
        
#         # Initialize top-5 rank columns
#         for rank in range(1, 6):
#             df_pred[f'plsf_rank{rank}'] = ''
        
#         # Initialize confidence
#         df_pred['plsf_confidence'] = 0.0
        
#         # Predict for each num_chain group
#         num_chain_groups = list(self.models['by_chain'].items())
#         for num_chain, chain_data in tqdm(num_chain_groups, desc="Predicting by chain", leave=False):
#             mask = df_pred['num_chain'] == num_chain
#             if not mask.any():
#                 continue
            
#             df_group = df_pred[mask].copy()
#             chain_models = chain_data['models']
#             max_stage = chain_data['max_stage']
            
#             # Store top-5 combinations for each sample
#             top5_combinations = [[] for _ in range(len(df_group))]
            
#             # For each sample, generate top-5 complete predictions
#             for idx, (sample_idx, row) in enumerate(df_group.iterrows()):
#                 # Generate all combinations by exploring top-k at each stage
#                 candidates = [{'values': [], 'prob': 1.0}]
                
#                 for stage_idx in range(max_stage):
#                     c_col, db_col = self.target_pairs[stage_idx]
                    
#                     # Get features
#                     features = self.base_features.copy()
#                     features.extend(['num_c_total', 'num_db_total'])
#                     for i in range(stage_idx):
#                         features.extend([f'num_c_{i+1}', f'num_db_{i+1}'])
                    
#                     new_candidates = []
                    
#                     for candidate in candidates[:5]:  # Only expand top 5
#                         # Create temp row with current candidate values
#                         temp_row = row.copy()
#                         for i, val in enumerate(candidate['values']):
#                             if i < len(candidate['values']):
#                                 temp_row[self.target_pairs[i//2][i%2]] = val
                        
#                         X = pd.DataFrame([temp_row[features]])
                        
#                         # Get top-3 for num_c
#                         c_top3, c_prob3 = self._get_top_k_predictions(
#                             chain_models[stage_idx]['c_model'], X, k=3
#                         )
                        
#                         for c_val, c_prob in zip(c_top3[0], c_prob3[0]):
#                             temp_row[c_col] = c_val
#                             X_with_c = pd.DataFrame([temp_row[features + [c_col]]])
                            
#                             # Get top-2 for num_db
#                             db_top2, db_prob2 = self._get_top_k_predictions(
#                                 chain_models[stage_idx]['db_model'], X_with_c, k=2
#                             )
                            
#                             for db_val, db_prob in zip(db_top2[0], db_prob2[0]):
#                                 new_candidates.append({
#                                     'values': candidate['values'] + [c_val, db_val],
#                                     'prob': candidate['prob'] * c_prob * db_prob
#                                 })
                    
#                     # Keep top 5 by probability
#                     candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:5]
                
#                 top5_combinations[idx] = candidates
            
#             # Assign predictions
#             stage_confidences = []
#             for idx, (sample_idx, row) in enumerate(df_group.iterrows()):
#                 candidates = top5_combinations[idx]
                
#                 # Top prediction
#                 if candidates:
#                     top_pred = candidates[0]['values']
#                     for i in range(max_stage):
#                         c_col, db_col = self.target_pairs[i]
#                         df_group.loc[sample_idx, c_col] = top_pred[i*2]
#                         df_group.loc[sample_idx, db_col] = top_pred[i*2 + 1]
                
#                 # Store all top-5 as sorted lists of 8 integers
#                 for rank, candidate in enumerate(candidates, 1):
#                     vals = candidate['values']
#                     # Format as sorted list: [c1, db1, c2, db2, c3, db3, c4, db4]
#                     rank_list = self._format_chains_as_list(vals, max_stage)
#                     df_group.loc[sample_idx, f'plsf_rank{rank}'] = str(rank_list)
                
#                 # Calculate confidence
#                 df_group.loc[sample_idx, 'plsf_confidence'] = candidates[0]['prob'] if candidates else 0.0
            
#             # Update main dataframe
#             for c_col, db_col in self.target_pairs[:max_stage]:
#                 df_pred.loc[mask, c_col] = df_group[c_col]
#                 df_pred.loc[mask, db_col] = df_group[db_col]
#             df_pred.loc[mask, 'plsf_confidence'] = df_group['plsf_confidence']
#             for rank in range(1, 6):
#                 df_pred.loc[mask, f'plsf_rank{rank}'] = df_group[f'plsf_rank{rank}']
        
#         return df_pred
    
#     def predict(self, input_data):
#         """
#         Make predictions on input data
        
#         Args:
#             input_data: DataFrame with required columns or path to CSV
            
#         Returns:
#             DataFrame with predictions
#         """
#         # Load data if path provided
#         if isinstance(input_data, (str, Path)):
#             print(f"Loading data from {input_data}...")
#             df = pd.read_csv(input_data)
#         else:
#             df = input_data.copy()
        
#         print(f"  Samples: {len(df)}")
        
#         # Verify required columns
#         required_cols = {'precursor_mz', 'adduct', 'class', 'num_chain'}
#         missing = required_cols - set(df.columns)
#         if missing:
#             raise ValueError(f"Missing required columns: {missing}")
        
#         # Check for mz_ columns
#         mz_cols = [col for col in df.columns if col.startswith('mz_')]
#         if not mz_cols:
#             raise ValueError("No 'mz_' columns found in input data")
        
#         print(f"  Features: {len(mz_cols)} mz_ columns")
        
#         # Prepare data
#         print("Preparing data...")
#         df_prepared = self._prepare_data(df)
        
#         # Make predictions
#         print("Making predictions...")
#         if self.split_by_chain:
#             df_pred = self._predict_split_by_chain(df_prepared)
#         else:
#             df_pred = self._predict_unified(df_prepared)
        
#         print("✓ Predictions complete")
        
#         return df_pred
    
#     def predict_and_save(self, input_path, output_path='result.csv'):
#         """
#         Make predictions and save to CSV
        
#         Args:
#             input_path: Path to input CSV
#             output_path: Path to output CSV
#         """
#         # Make predictions
#         df_pred = self.predict(input_path)
        
#         # Create output directory if needed
#         output_dir = os.path.dirname(output_path)
#         if output_dir and not os.path.exists(output_dir):
#             print(f"Creating directory: {output_dir}")
#             os.makedirs(output_dir, exist_ok=True)
        
#         # Decode categorical variables back to original values
#         df_output = df_pred.copy()
#         for col in ['adduct', 'class']:
#             df_output[col] = self.label_encoders[col].inverse_transform(df_output[col])
        
#         # Save predictions
#         print(f"\nSaving predictions to {output_path}...")
#         df_output.to_csv(output_path, index=False)
#         print(f"✓ Saved {len(df_output)} predictions")
        
#         # Print summary
#         print("\nPrediction Summary:")
#         print(f"  Total C range: {df_output['num_c_total'].min()}-{df_output['num_c_total'].max()}")
#         print(f"  Total DB range: {df_output['num_db_total'].min()}-{df_output['num_db_total'].max()}")
#         print(f"  Confidence: mean={df_output['plsf_confidence'].mean():.3f}, "
#               f"min={df_output['plsf_confidence'].min():.3f}, "
#               f"max={df_output['plsf_confidence'].max():.3f}")
#         print(f"  num_chain distribution:")
#         for nc, count in df_output['num_chain'].value_counts().sort_index().items():
#             avg_conf = df_output[df_output['num_chain'] == nc]['plsf_confidence'].mean()
#             print(f"    {nc}: {count} samples ({count/len(df_output)*100:.1f}%), "
#                   f"avg confidence: {avg_conf:.3f}")


# def main():
#     """Main function for command line usage"""
#     parser = argparse.ArgumentParser(
#         description='Predict lipid chain compositions using trained PLSF model',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python predict_plsf.py plsf_model.joblib input.csv
#   python predict_plsf.py plsf_model.joblib input.csv --output_path results/output.csv
#   python predict_plsf.py plsf_model.joblib input.csv -o predictions.csv

# Input CSV Requirements:
#   - precursor_mz: float (will be rounded to 0.01)
#   - adduct: string (e.g., '[M+H]+', '[M+Na]+')
#   - class: string (e.g., 'PC', 'PE', 'PS')
#   - num_chain: integer (1, 2, 3, or 4)
#   - mz_*: binary columns (0 or 1)

# Output CSV Contains:
#   - All input columns
#   - num_c_total, num_db_total (predicted totals)
#   - num_c_1, num_db_1, ..., num_c_4, num_db_4 (predicted chains)
#         """
#     )
    
#     parser.add_argument(
#         'model_path',
#         help='Path to trained model (.joblib file)'
#     )
    
#     parser.add_argument(
#         'input_path',
#         help='Path to input CSV file'
#     )
    
#     parser.add_argument(
#         '-o', '--output_path',
#         default='result.csv',
#         help='Path to output CSV file (default: result.csv)'
#     )
    
#     args = parser.parse_args()
    
#     # Check if files exist
#     if not os.path.exists(args.model_path):
#         print(f"Error: Model file '{args.model_path}' not found!")
#         sys.exit(1)
    
#     if not os.path.exists(args.input_path):
#         print(f"Error: Input file '{args.input_path}' not found!")
#         sys.exit(1)
    
#     # Create predictor and make predictions
#     try:
#         predictor = PLSFPredictor(args.model_path)
#         predictor.predict_and_save(args.input_path, args.output_path)
#         print("\n✓ Success!")
#     except Exception as e:
#         print(f"\n✗ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

"""
Predict lipid chain compositions using trained PLSF model

Usage:
    python predict_plsf.py plsf_model.joblib input.csv
    python predict_plsf.py plsf_model.joblib input.csv --output_path results/output.csv
    python predict_plsf.py plsf_model.joblib input.csv -o results/output.csv

The input CSV should contain the required columns:
    - precursor_mz (will be rounded to 0.01)
    - adduct (categorical) or predicted_adduct if adduct is missing
    - class (categorical) or predicted_class if class is missing
    - num_chain (1, 2, 3, or 4)
    - mz_* columns (binary 0/1)

Special handling:
    - If adduct or class are missing, predicted_adduct/predicted_class will be used
    - Unknown adduct or class values result in empty predictions
    - Single-chain lipids (num_chain=1) use direct mass calculation instead of model
    - All predictions include a plsf_confidence score (0-1)

Output CSV will contain:
    - All input columns (excluding mz_* features)
    - name: Combined formatted name (e.g., "PC 16:0_18:1")
    - pred_confidence: Mean confidence score
    - plsf_rank1, plsf_rank2, plsf_rank3 (top 3 predictions)
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm


class PLSFPredictor:
    """Predictor class for PLSF models with confidence scores and single-chain handling"""
    
    # Adduct masses for single-chain calculation
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
    
    def __init__(self, model_path):
        """
        Load the trained model
        
        Args:
            model_path: Path to .joblib model file
        """
        print(f"Loading model from {model_path}...")
        try:
            self.model_data = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found!")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Extract model components
        self.model_name = self.model_data['model_name']
        self.models = self.model_data['models']
        self.label_encoders = self.model_data['label_encoders']
        self.base_features = self.model_data['base_features']
        self.target_pairs = self.model_data['target_pairs']
        self.split_by_chain = self.model_data['split_by_chain']
        
        print(f"✓ Loaded model: {self.model_name}")
        print(f"  Mode: {'Split by Chain' if self.split_by_chain else 'Unified'}")
        # print(f"  Test Accuracy: {self.model_data['test_accuracy']:.2f}%")
    
    def _predict_with_inverse_map(self, model, X):
        """Make prediction and apply inverse mapping if XGBoost"""
        # Convert to numpy array to avoid feature name warnings
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = model.predict(X).astype(int)
        if hasattr(model, 'inverse_map'):
            pred = pd.Series(pred).map(model.inverse_map).values
        return pred
    
    def _get_prediction_confidence(self, model, X):
        """
        Get confidence score for prediction
        Returns probability of the predicted class
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        try:
            if hasattr(model, 'predict_proba'):
                # Get probability for each class
                proba = model.predict_proba(X)
                # Get max probability (confidence in predicted class)
                confidence = np.max(proba, axis=1)
                return confidence
            else:
                # If model doesn't support predict_proba, return 1.0
                return np.ones(len(X))
        except Exception:
            return np.ones(len(X))
    
    def _get_top_k_predictions(self, model, X, k=5):
        """
        Get top-k predictions with probabilities
        
        Returns:
            top_k_classes: Array of shape (n_samples, k) with class predictions
            top_k_probs: Array of shape (n_samples, k) with probabilities
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get all class probabilities
        proba = model.predict_proba(X)
        
        # Get top k indices
        top_k_indices = np.argsort(proba, axis=1)[:, -k:][:, ::-1]  # Descending order
        
        # Get corresponding probabilities
        top_k_probs = np.take_along_axis(proba, top_k_indices, axis=1)
        
        # Map back to original labels if model has inverse_map
        if hasattr(model, 'inverse_map'):
            inverse_map = model.inverse_map
            top_k_classes = np.vectorize(lambda x: inverse_map[x])(top_k_indices)
        else:
            top_k_classes = top_k_indices
        
        return top_k_classes, top_k_probs
    
    def _format_chains_as_list(self, vals, max_stage):
        """
        Format chain predictions as sorted list of 8 integers
        
        Args:
            vals: List of values [c1, db1, c2, db2, ...]
            max_stage: Number of chains
            
        Returns:
            List of 8 integers [c1, db1, c2, db2, c3, db3, c4, db4]
            sorted by c (descending), then db (descending)
        """
        # Extract chains as (c, db) tuples
        chains = []
        for i in range(max_stage):
            c = int(vals[i*2])
            db = int(vals[i*2+1])
            if c > 0 or db > 0:  # Only include non-zero chains
                chains.append((c, db))
        
        # Sort chains: by c descending, then by db descending
        chains.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Pad with zeros to have 4 chains total
        while len(chains) < 4:
            chains.append((0, 0))
        
        # Flatten to list of 8 integers
        result = []
        for c, db in chains:
            result.extend([c, db])
        
        return result
    
    def _prepare_data(self, df):
        """Prepare input data (same preprocessing as training)"""
        df = df.copy()
        
        # Round precursor_mz
        df['precursor_mz'] = df['precursor_mz'].round(2)
        
        # Handle missing adduct and class by using predicted values if available
        if 'adduct' not in df.columns:
            if 'predicted_adduct' in df.columns:
                df['adduct'] = df['predicted_adduct']
                print("  Note: Using 'predicted_adduct' for missing 'adduct' column")
            else:
                raise ValueError("Required column 'adduct' or 'predicted_adduct' not found")
        else:
            # Fill missing adduct values with predicted_adduct if available
            if 'predicted_adduct' in df.columns:
                mask = df['adduct'].isna() | (df['adduct'] == '')
                if mask.any():
                    df.loc[mask, 'adduct'] = df.loc[mask, 'predicted_adduct']
                    print(f"  Note: Filled {mask.sum()} missing adduct values with predicted_adduct")
        
        if 'class' not in df.columns:
            if 'predicted_class' in df.columns:
                df['class'] = df['predicted_class']
                print("  Note: Using 'predicted_class' for missing 'class' column")
            else:
                raise ValueError("Required column 'class' or 'predicted_class' not found")
        else:
            # Fill missing class values with predicted_class if available
            if 'predicted_class' in df.columns:
                mask = df['class'].isna() | (df['class'] == '')
                if mask.any():
                    df.loc[mask, 'class'] = df.loc[mask, 'predicted_class']
                    print(f"  Note: Filled {mask.sum()} missing class values with predicted_class")
        
        # Store original columns to restore later and for identification
        df['_original_adduct'] = df['adduct']
        df['_original_class'] = df['class']
        
        # Initialize validity flag
        df['_valid_row'] = True
        
        # Encode categorical variables
        for col in ['adduct', 'class']:
            # Check for unknown categories
            unknown = set(df[col].unique()) - set(self.label_encoders[col].classes_)
            if unknown:
                print(f"  Warning: Unknown values in '{col}': {unknown}")
                print(f"  Rows with these values will have empty predictions")
                
                # Mark rows as invalid
                mask = df[col].isin(unknown)
                df.loc[mask, '_valid_row'] = False
                
                # Map unknown to first known category so encoding doesn't crash
                # (predictions will be cleared later based on _valid_row)
                safe_value = self.label_encoders[col].classes_[0]
                df.loc[mask, col] = safe_value
            
            df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _predict_unified(self, df):
        """Make predictions using unified model"""
        df_pred = df.copy()
        
        # Predict totals for all
        X_base = df_pred[self.base_features]
        df_pred['num_c_total'] = self._predict_with_inverse_map(
            self.models['total']['c_model'], X_base
        )
        df_pred['num_db_total'] = self._predict_with_inverse_map(
            self.models['total']['db_model'], X_base
        )
        
        # Get confidence for totals
        conf_c_total = self._get_prediction_confidence(self.models['total']['c_model'], X_base)
        conf_db_total = self._get_prediction_confidence(self.models['total']['db_model'], X_base)
        
        # Predict individual chains
        confidences = []
        for stage_idx, (c_col, db_col) in enumerate(self.target_pairs):
            # Get features for this stage
            features = self.base_features.copy()
            features.extend(['num_c_total', 'num_db_total'])
            for i in range(stage_idx):
                features.extend([f'num_c_{i+1}', f'num_db_{i+1}'])
            
            X = df_pred[features]
            
            # Predict num_c
            df_pred[c_col] = self._predict_with_inverse_map(
                self.models['chains'][stage_idx]['c_model'], X
            )
            conf_c = self._get_prediction_confidence(
                self.models['chains'][stage_idx]['c_model'], X
            )
            
            # Predict num_db
            X_with_c = df_pred[features + [c_col]]
            df_pred[db_col] = self._predict_with_inverse_map(
                self.models['chains'][stage_idx]['db_model'], X_with_c
            )
            conf_db = self._get_prediction_confidence(
                self.models['chains'][stage_idx]['db_model'], X_with_c
            )
        
        # Calculate overall confidence (average of all predictions)
        df_pred['plsf_confidence'] = 0.0
        for i in range(len(df_pred)):
            all_confs = [conf_c_total[i], conf_db_total[i]]
            overall_conf = np.mean(all_confs)
            df_pred.iloc[i, df_pred.columns.get_loc('plsf_confidence')] = overall_conf
        
        return df_pred
    
    def _predict_split_by_chain(self, df):
        """Make predictions using split-by-chain models with top-3 alternatives"""
        df_pred = df.copy()
        
        # Predict totals
        X_base = df_pred[self.base_features]
        c_total_top, _ = self._get_top_k_predictions(
            self.models['total']['c_model'], X_base, k=1
        )
        df_pred['num_c_total'] = c_total_top[:, 0]
        
        db_total_top, _ = self._get_top_k_predictions(
            self.models['total']['db_model'], X_base, k=1
        )
        df_pred['num_db_total'] = db_total_top[:, 0]
        
        # Get confidence for totals
        conf_c_total = self._get_prediction_confidence(self.models['total']['c_model'], X_base)
        conf_db_total = self._get_prediction_confidence(self.models['total']['db_model'], X_base)
        
        # Initialize all chain predictions to 0
        for c_col, db_col in self.target_pairs:
            df_pred[c_col] = 0
            df_pred[db_col] = 0
        
        # Initialize top-3 rank columns
        for rank in range(1, 4):
            df_pred[f'plsf_rank{rank}'] = ''
        
        # Initialize confidence
        df_pred['plsf_confidence'] = 0.0
        
        # Predict for each num_chain group
        num_chain_groups = list(self.models['by_chain'].items())
        for num_chain, chain_data in tqdm(num_chain_groups, desc="Predicting by chain", leave=False):
            mask = df_pred['num_chain'] == num_chain
            if not mask.any():
                continue
            
            df_group = df_pred[mask].copy()
            chain_models = chain_data['models']
            max_stage = chain_data['max_stage']
            
            # Store top-3 combinations for each sample
            top3_combinations = [[] for _ in range(len(df_group))]
            
            # For each sample, generate top-3 complete predictions
            for idx, (sample_idx, row) in enumerate(df_group.iterrows()):
                # Generate all combinations by exploring top-k at each stage
                candidates = [{'values': [], 'prob': 1.0}]
                
                for stage_idx in range(max_stage):
                    c_col, db_col = self.target_pairs[stage_idx]
                    
                    # Get features
                    features = self.base_features.copy()
                    features.extend(['num_c_total', 'num_db_total'])
                    for i in range(stage_idx):
                        features.extend([f'num_c_{i+1}', f'num_db_{i+1}'])
                    
                    new_candidates = []
                    
                    for candidate in candidates[:3]:  # Only expand top 3
                        # Create temp row with current candidate values
                        temp_row = row.copy()
                        for i, val in enumerate(candidate['values']):
                            if i < len(candidate['values']):
                                temp_row[self.target_pairs[i//2][i%2]] = val
                        
                        X = pd.DataFrame([temp_row[features]])
                        
                        # Get top-3 for num_c
                        c_top3, c_prob3 = self._get_top_k_predictions(
                            chain_models[stage_idx]['c_model'], X, k=3
                        )
                        
                        for c_val, c_prob in zip(c_top3[0], c_prob3[0]):
                            temp_row[c_col] = c_val
                            X_with_c = pd.DataFrame([temp_row[features + [c_col]]])
                            
                            # Get top-2 for num_db
                            db_top2, db_prob2 = self._get_top_k_predictions(
                                chain_models[stage_idx]['db_model'], X_with_c, k=2
                            )
                            
                            for db_val, db_prob in zip(db_top2[0], db_prob2[0]):
                                new_candidates.append({
                                    'values': candidate['values'] + [c_val, db_val],
                                    'prob': candidate['prob'] * c_prob * db_prob
                                })
                    
                    # Keep top 3 by probability
                    candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:3]
                
                top3_combinations[idx] = candidates
            
            # Assign predictions
            stage_confidences = []
            for idx, (sample_idx, row) in enumerate(df_group.iterrows()):
                candidates = top3_combinations[idx]
                
                # Top prediction
                if candidates:
                    top_pred = candidates[0]['values']
                    for i in range(max_stage):
                        c_col, db_col = self.target_pairs[i]
                        df_group.loc[sample_idx, c_col] = top_pred[i*2]
                        df_group.loc[sample_idx, db_col] = top_pred[i*2 + 1]
                
                # Store all top-3 as sorted lists of 8 integers
                for rank, candidate in enumerate(candidates, 1):
                    vals = candidate['values']
                    # Format as sorted list: [c1, db1, c2, db2, c3, db3, c4, db4]
                    rank_list = self._format_chains_as_list(vals, max_stage)
                    df_group.loc[sample_idx, f'plsf_rank{rank}'] = str(rank_list)
                
                # Calculate confidence
                df_group.loc[sample_idx, 'plsf_confidence'] = candidates[0]['prob'] if candidates else 0.0
            
            # Update main dataframe
            for c_col, db_col in self.target_pairs[:max_stage]:
                df_pred.loc[mask, c_col] = df_group[c_col]
                df_pred.loc[mask, db_col] = df_group[db_col]
            df_pred.loc[mask, 'plsf_confidence'] = df_group['plsf_confidence']
            for rank in range(1, 4):
                df_pred.loc[mask, f'plsf_rank{rank}'] = df_group[f'plsf_rank{rank}']
        
        return df_pred
    
    def predict(self, input_data):
        """
        Make predictions on input data
        
        Args:
            input_data: DataFrame with required columns or path to CSV
            
        Returns:
            DataFrame with predictions
        """
        # Load data if path provided
        if isinstance(input_data, (str, Path)):
            print(f"Loading data from {input_data}...")
            df = pd.read_csv(input_data)
        else:
            df = input_data.copy()
        
        print(f"  Samples: {len(df)}")
        
        # Verify required columns
        # Check strictly required columns
        if 'precursor_mz' not in df.columns:
            raise ValueError("Missing required column: 'precursor_mz'")
        if 'num_chain' not in df.columns:
            raise ValueError("Missing required column: 'num_chain'")

        # Check conditionally required columns
        if 'adduct' not in df.columns and 'predicted_adduct' not in df.columns:
            raise ValueError("Missing required column: 'adduct' (or 'predicted_adduct')")
        
        if 'class' not in df.columns and 'predicted_class' not in df.columns:
            raise ValueError("Missing required column: 'class' (or 'predicted_class')")
        
        # Check for mz_ columns
        mz_cols = [col for col in df.columns if col.startswith('mz_')]
        if not mz_cols:
            raise ValueError("No 'mz_' columns found in input data")
        
        print(f"  Features: {len(mz_cols)} mz_ columns")
        
        # Prepare data
        print("Preparing data...")
        df_prepared = self._prepare_data(df)
        
        # Make predictions
        print("Making predictions...")
        if self.split_by_chain:
            df_pred = self._predict_split_by_chain(df_prepared)
        else:
            df_pred = self._predict_unified(df_prepared)
            
        # Clear predictions for invalid rows (unknown adduct or class)
        if '_valid_row' in df_pred.columns:
            invalid_mask = ~df_pred['_valid_row']
            if invalid_mask.any():
                count = invalid_mask.sum()
                print(f"  Note: Cleared predictions for {count} rows with unknown adduct/class")
                
                # Columns to clear
                cols_to_clear = ['num_c_total', 'num_db_total', 'plsf_confidence']
                
                # Add chain columns
                for c_col, db_col in self.target_pairs:
                    cols_to_clear.extend([c_col, db_col])
                
                # Add rank columns
                for rank in range(1, 4):
                    cols_to_clear.append(f'plsf_rank{rank}')
                
                # Clear existing columns
                for col in cols_to_clear:
                    if col in df_pred.columns:
                        df_pred.loc[invalid_mask, col] = np.nan
            
            # Drop the validity mask before returning (optional, but keeps it clean)
            # Actually, we keep _original_adduct/class but we can drop _valid_row
            df_pred = df_pred.drop(columns=['_valid_row'])
        
        print("✓ Predictions complete")
        
        return df_pred
    
    def predict_and_save(self, input_path, output_path='result.csv'):
        """
        Make predictions and save to CSV
        
        Args:
            input_path: Path to input CSV
            output_path: Path to output CSV
        """
        # Make predictions
        df_pred = self.predict(input_path)
        
        # Capture statistics before dropping columns
        c_total_min = df_pred['num_c_total'].min()
        c_total_max = df_pred['num_c_total'].max()
        db_total_min = df_pred['num_db_total'].min()
        db_total_max = df_pred['num_db_total'].max()
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Decode categorical variables back to original values
        df_output = df_pred.copy()
        
        # Restore original strings (handles unknown values) or decode known values
        for col in ['adduct', 'class']:
            orig_col = f'_original_{col}'
            if orig_col in df_output.columns:
                # If we have the original column (including unknown values), use it
                df_output[col] = df_output[orig_col]
                df_output.drop(columns=[orig_col], inplace=True)
            else:
                # Fallback for backward compatibility or if original missing
                df_output[col] = self.label_encoders[col].inverse_transform(df_output[col])
        
        print("\nFormatting output columns...")
        
        # 1. Generate 'name' column
        def get_lipid_name(row):
            # Determine class to use
            cls = row.get('class')
            # If class is empty/NaN, try predicted_class
            if pd.isna(cls) or str(cls).strip() == '':
                cls = row.get('predicted_class', '')
            
            # Construct chains string
            chains = []
            for i in range(1, 5): # Chains 1 to 4
                c_key = f'num_c_{i}'
                db_key = f'num_db_{i}'
                
                if c_key in row and db_key in row:
                    c_val = row[c_key]
                    db_val = row[db_key]
                    
                    # Only process if not NaN
                    if pd.notna(c_val) and pd.notna(db_val):
                        c = int(float(c_val))
                        db = int(float(db_val))
                        
                        # Filter out 0:0 blocks
                        if c == 0 and db == 0:
                            continue
                            
                        chains.append(f"{c}:{db}")
            
            chain_str = "_".join(chains)
            return f"{cls} {chain_str}".strip()

        df_output['name'] = df_output.apply(get_lipid_name, axis=1)
        
        # 2. Calculate pred_confidence
        # Mean of (plsf_confidence, class_confidence, adduct_confidence), skipping missing
        conf_cols = ['plsf_confidence', 'class_confidence', 'adduct_confidence']
        available_conf_cols = [c for c in conf_cols if c in df_output.columns]
        
        if available_conf_cols:
            # Mean ignores NaNs by default
            df_output['pred_confidence'] = df_output[available_conf_cols].mean(axis=1).round(2)
        
        # 3. Drop columns
        cols_to_drop = []
        
        # Drop mz_ columns (input features)
        cols_to_drop.extend([c for c in df_output.columns if c.startswith('mz_')])
        
        # Drop numeric prediction columns (total and individual chains), BUT keep num_chain (input)
        # num_c_1, num_c_total start with num_c_, whereas num_chain starts with num_c but no underscore
        cols_to_drop.extend([c for c in df_output.columns if c.startswith('num_c_') or c.startswith('num_db_')])
        
        df_output.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Save predictions
        print(f"Saving predictions to {output_path}...")
        df_output.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df_output)} predictions")
        
        # Print summary (using statistics captured from df_pred before columns were dropped)
        print("\nPrediction Summary:")
        print(f"  Total C range: {c_total_min}-{c_total_max}")
        print(f"  Total DB range: {db_total_min}-{db_total_max}")
        
        if 'pred_confidence' in df_output.columns:
            print(f"  Pred Confidence: mean={df_output['pred_confidence'].mean():.3f}, "
                  f"min={df_output['pred_confidence'].min():.3f}, "
                  f"max={df_output['pred_confidence'].max():.3f}")
        elif 'plsf_confidence' in df_output.columns:
             print(f"  PLSF Confidence: mean={df_output['plsf_confidence'].mean():.3f}, "
                  f"min={df_output['plsf_confidence'].min():.3f}, "
                  f"max={df_output['plsf_confidence'].max():.3f}")
        
        # Check if num_chain exists in output, if not use df_pred for summary
        summary_df = df_output if 'num_chain' in df_output.columns else df_pred
        
        print(f"  num_chain distribution:")
        if 'num_chain' in summary_df.columns:
            for nc, count in summary_df['num_chain'].value_counts().sort_index().items():
                print(f"    {nc}: {count} samples ({count/len(summary_df)*100:.1f}%)")
        else:
            print("    (num_chain column not available for summary)")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description='Predict lipid chain compositions using trained PLSF model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_plsf.py plsf_model.joblib input.csv
  python predict_plsf.py plsf_model.joblib input.csv --output_path results/output.csv
  python predict_plsf.py plsf_model.joblib input.csv -o predictions.csv

Input CSV Requirements:
  - precursor_mz: float (will be rounded to 0.01)
  - adduct: string (e.g., '[M+H]+', '[M+Na]+')
  - class: string (e.g., 'PC', 'PE', 'PS')
  - num_chain: integer (1, 2, 3, or 4)
  - mz_*: binary columns (0 or 1)

Output CSV Contains:
  - All input columns (excluding mz_* features)
  - name: Combined formatted name (e.g., "PC 16:0_18:1")
  - pred_confidence: Mean confidence score
  - plsf_rank1, plsf_rank2, plsf_rank3 (top 3 predictions)
        """
    )
    
    parser.add_argument(
        'model_path',
        help='Path to trained model (.joblib file)'
    )
    
    parser.add_argument(
        'input_path',
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '-o', '--output_path',
        default='result.csv',
        help='Path to output CSV file (default: result.csv)'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        sys.exit(1)
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found!")
        sys.exit(1)
    
    # Create predictor and make predictions
    try:
        predictor = PLSFPredictor(args.model_path)
        predictor.predict_and_save(args.input_path, args.output_path)
        print("\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()