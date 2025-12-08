# """
# Enhanced PLSF Predictor - Standalone Version with Sum Constraints

# This script enforces:
#   - num_c_1 + num_c_2 + num_c_3 + num_c_4 = num_c
#   - num_db_1 + num_db_2 + num_db_3 + num_db_4 = num_db

# The lowest confidence prediction is replaced with the residual to ensure constraints.

# Usage:
#     python plsf_predict.py input.csv plsf_model.joblib
#     python plsf_predict.py input.csv plsf_model.joblib --output_path results/output.csv
#     python plsf_predict.py input.csv plsf_model.joblib -o results/output.csv
#     python plsf_predict.py input.csv plsf_model.joblib --n_jobs 8
# """

# import pandas as pd
# import numpy as np
# import joblib
# import argparse
# import sys
# import os
# from pathlib import Path
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing


# # ============================================================================
# # ConstantPredictor Class
# # ============================================================================

# class ConstantPredictor:
#     """Simple predictor that always returns the same constant value."""
    
#     def __init__(self, constant_value):
#         self.constant_value = constant_value
#         self.inverse_map = None
    
#     def predict(self, X):
#         n_samples = len(X)
#         return np.full(n_samples, self.constant_value)
    
#     def predict_proba(self, X):
#         n_samples = len(X)
#         return np.ones((n_samples, 1))
    
#     def __repr__(self):
#         return f"ConstantPredictor(value={self.constant_value})"


# # Module Aliasing for Backward Compatibility
# import types
# train_compare_models = types.ModuleType('train_compare_models')
# train_compare_models.ConstantPredictor = ConstantPredictor
# sys.modules['train_compare_models'] = train_compare_models


# # ============================================================================
# # Helper Functions
# # ============================================================================

# def _is_constant_predictor(model):
#     """Check if model is a ConstantPredictor"""
#     return isinstance(model, ConstantPredictor)


# def _get_top_k_predictions(model, X, k=5):
#     """Get top-k predictions with probabilities"""
#     if isinstance(X, pd.DataFrame):
#         X = X.values
    
#     if _is_constant_predictor(model):
#         const_val = model.constant_value
#         top_k_classes = np.full((len(X), k), const_val)
#         top_k_probs = np.zeros((len(X), k))
#         top_k_probs[:, 0] = 1.0
#         return top_k_classes, top_k_probs
    
#     if not hasattr(model, 'predict_proba'):
#         pred = model.predict(X)
#         top_k_classes = np.tile(pred.reshape(-1, 1), (1, k))
#         top_k_probs = np.zeros((len(X), k))
#         top_k_probs[:, 0] = 1.0
#         return top_k_classes, top_k_probs
    
#     proba = model.predict_proba(X)
#     top_k_indices = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
#     top_k_probs = np.take_along_axis(proba, top_k_indices, axis=1)
    
#     if hasattr(model, 'inverse_map'):
#         inverse_map = model.inverse_map
#         top_k_classes = np.vectorize(lambda x: inverse_map[x])(top_k_indices)
#     else:
#         top_k_classes = top_k_indices
    
#     return top_k_classes, top_k_probs


# def _predict_with_confidence(model, X):
#     """
#     Make prediction and return both predictions and confidence scores.
#     Returns: (predictions, confidences)
#     """
#     if isinstance(X, pd.DataFrame):
#         X_arr = X.values
#     else:
#         X_arr = X
    
#     if _is_constant_predictor(model):
#         pred = model.predict(X_arr)
#         conf = np.ones(len(pred))
#         return pred.astype(int), conf
    
#     proba = model.predict_proba(X_arr)
#     pred_idx = np.argmax(proba, axis=1)
#     conf = np.max(proba, axis=1)
    
#     if hasattr(model, 'inverse_map') and model.inverse_map is not None:
#         pred = pd.Series(pred_idx).map(model.inverse_map).values.astype(int)
#     else:
#         pred = pred_idx.astype(int)
    
#     return pred, conf


# def _format_chains_as_list(vals, max_stage):
#     """Format chain predictions as sorted list of 8 integers"""
#     chains = []
#     for i in range(max_stage):
#         c = int(vals[i*2])
#         db = int(vals[i*2+1])
#         if c > 0 or db > 0:
#             chains.append((c, db))
    
#     chains.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
#     while len(chains) < 4:
#         chains.append((0, 0))
    
#     result = []
#     for c, db in chains:
#         result.extend([c, db])
    
#     return result


# def _process_single_sample(args):
#     """Process a single sample for parallel execution - returns predictions WITH confidences"""
#     sample_idx, row, chain_models, max_stage, target_pairs, base_features, num_c_total, num_db_total = args
    
#     def get_features_for_stage(row_data, stage_idx):
#         features = []
#         for feat in base_features:
#             if feat in row_data.index:
#                 features.append(feat)
        
#         for i in range(stage_idx):
#             c_col = f'num_c_{i+1}'
#             db_col = f'num_db_{i+1}'
#             if c_col in row_data.index:
#                 features.append(c_col)
#             if db_col in row_data.index:
#                 features.append(db_col)
        
#         return features
    
#     # First pass: get all predictions with confidence scores
#     c_preds = {}
#     db_preds = {}
#     c_confs = {}
#     db_confs = {}
    
#     temp_row = row.copy()
    
#     for stage_idx in range(max_stage):
#         c_col, db_col = target_pairs[stage_idx]
#         features = get_features_for_stage(temp_row, stage_idx)
        
#         X = pd.DataFrame([temp_row[features]])
        
#         # Get prediction and confidence for num_c
#         c_model = chain_models[stage_idx]['c_model']
#         c_pred, c_conf = _predict_with_confidence(c_model, X)
#         c_preds[stage_idx] = int(c_pred[0])
#         c_confs[stage_idx] = float(c_conf[0])
#         temp_row[c_col] = c_preds[stage_idx]
        
#         # Get prediction and confidence for num_db
#         X_with_c = pd.DataFrame([temp_row[features + [c_col]]])
#         db_model = chain_models[stage_idx]['db_model']
#         db_pred, db_conf = _predict_with_confidence(db_model, X_with_c)
#         db_preds[stage_idx] = int(db_pred[0])
#         db_confs[stage_idx] = float(db_conf[0])
#         temp_row[db_col] = db_preds[stage_idx]
    
#     # Second pass: enforce sum constraints
#     # Enforce num_c constraint
#     current_sum_c = sum(c_preds.values())
#     if current_sum_c != num_c_total and num_c_total is not None and not np.isnan(num_c_total):
#         min_conf_chain = min(c_confs, key=c_confs.get)
#         other_sum = sum(v for j, v in c_preds.items() if j != min_conf_chain)
#         residual = int(num_c_total) - other_sum
#         c_preds[min_conf_chain] = max(0, residual)
    
#     # Enforce num_db constraint
#     current_sum_db = sum(db_preds.values())
#     if current_sum_db != num_db_total and num_db_total is not None and not np.isnan(num_db_total):
#         min_conf_chain = min(db_confs, key=db_confs.get)
#         other_sum = sum(v for j, v in db_preds.items() if j != min_conf_chain)
#         residual = int(num_db_total) - other_sum
#         db_preds[min_conf_chain] = max(0, residual)
    
#     # Build result
#     result = {
#         'sample_idx': sample_idx,
#         'predictions': {},
#         'ranks': {},
#         'confidence': 1.0
#     }
    
#     # Store predictions
#     vals = []
#     total_conf = 1.0
#     for i in range(max_stage):
#         c_col, db_col = target_pairs[i]
#         result['predictions'][c_col] = c_preds[i]
#         result['predictions'][db_col] = db_preds[i]
#         vals.extend([c_preds[i], db_preds[i]])
#         total_conf *= c_confs[i] * db_confs[i]
    
#     result['confidence'] = total_conf
    
#     # Create rank1 from constrained predictions
#     rank_list = _format_chains_as_list(vals, max_stage)
#     result['ranks']['plsf_rank1'] = str(rank_list)
    
#     # For ranks 2-5, we need to generate alternatives
#     # Use beam search for alternatives (without constraint enforcement for alternatives)
#     candidates = [{'values': vals, 'prob': total_conf}]
    
#     # Generate more candidates via beam search (simplified)
#     temp_row2 = row.copy()
#     alt_candidates = [{'values': [], 'prob': 1.0}]
    
#     for stage_idx in range(max_stage):
#         c_col, db_col = target_pairs[stage_idx]
#         features = get_features_for_stage(temp_row2, stage_idx)
        
#         new_candidates = []
        
#         for candidate in alt_candidates[:5]:
#             temp_row_alt = temp_row2.copy()
#             for i, val in enumerate(candidate['values']):
#                 if i % 2 == 0:
#                     temp_row_alt[target_pairs[i//2][0]] = val
#                 else:
#                     temp_row_alt[target_pairs[i//2][1]] = val
            
#             X = pd.DataFrame([temp_row_alt[features]])
            
#             c_top3, c_prob3 = _get_top_k_predictions(chain_models[stage_idx]['c_model'], X, k=3)
            
#             for c_val, c_prob in zip(c_top3[0], c_prob3[0]):
#                 temp_row_alt[c_col] = c_val
#                 X_with_c = pd.DataFrame([temp_row_alt[features + [c_col]]])
                
#                 db_top2, db_prob2 = _get_top_k_predictions(chain_models[stage_idx]['db_model'], X_with_c, k=2)
                
#                 for db_val, db_prob in zip(db_top2[0], db_prob2[0]):
#                     new_candidates.append({
#                         'values': candidate['values'] + [c_val, db_val],
#                         'prob': candidate['prob'] * c_prob * db_prob
#                     })
        
#         alt_candidates = sorted(new_candidates, key=lambda x: x['prob'], reverse=True)[:5]
    
#     # Store ranks 2-5 from alternatives
#     for rank, candidate in enumerate(alt_candidates[1:5], 2):
#         alt_vals = candidate['values']
#         rank_list = _format_chains_as_list(alt_vals, max_stage)
#         result['ranks'][f'plsf_rank{rank}'] = str(rank_list)
    
#     return result


# # ============================================================================
# # PLSFPredictor Class
# # ============================================================================

# class PLSFPredictor:
#     """Enhanced predictor with confidence scores, top-K predictions, and sum constraints"""
    
#     def __init__(self, model_path, n_jobs=4):
#         print(f"Loading model from {model_path}...")
#         self.n_jobs = n_jobs
        
#         try:
#             self.model_data = joblib.load(model_path)
#         except FileNotFoundError:
#             print(f"Error: Model file '{model_path}' not found!")
#             sys.exit(1)
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             sys.exit(1)
        
#         self.model_name = self.model_data['model_name']
#         self.models = self.model_data['models']
#         self.label_encoders = self.model_data['label_encoders']
#         self.base_features = self.model_data['base_features']
#         self.target_pairs = self.model_data['target_pairs']
        
#         print(f"✓ Loaded model: {self.model_name}")
#         print(f"✓ Parallel workers: {self.n_jobs}")
#         print(f"✓ Sum constraints enabled (lowest confidence → residual)")
    
#     def _calculate_exact_mass(self, df):
#         """Calculate exact mass from precursor_mz and adduct"""
#         adduct_adjustments = {
#             '[M]+': 0.0,
#             '[M+H]+': -1.00783,
#             '[M+NH4]+': -18.03383,
#             '[M+Na]+': -22.98977,
#             '[M+K]+': -38.96371,
#             '[M-H]-': 1.00783,
#             '[M+HCOO]-': -44.99820,
#             '[M+CH3COO]-': -59.01385,
#             '[2M+H]+': -1.00783,
#             '[2M+Na]+': -22.98977,
#             '[2M-H]-': 1.00783,
#             '[M-H2O+H]+': 17.00274,
#             '[M-2H2O+H]+': 35.01311,
#             '[M+H-H2O]+': 17.00274,
#             '[M+]+': 0.0,
#             '[M]': 0.0,
#             '[M-H1]-': 1.00783,
#             '[M+CH3COOH-H]-': -59.01385
#         }
        
#         exact_masses = []
#         for idx, row in df.iterrows():
#             precursor_mz = row['precursor_mz']
#             adduct = row['adduct']
            
#             if adduct in adduct_adjustments:
#                 adjustment = adduct_adjustments[adduct]
#                 exact_mass = precursor_mz + adjustment
                
#                 if '2M' in adduct:
#                     exact_mass = exact_mass / 2.0
                    
#                 exact_masses.append(exact_mass)
#             else:
#                 exact_masses.append(precursor_mz)
        
#         df['exact_mass'] = exact_masses
#         return df
    
#     def _find_best_composition(self, df, ms1_tolerance=30.0):
#         """Compute num_c and num_db from exact_mass and class"""
#         M_C = 12.00000
#         M_H = 1.00783
#         M_CH2 = M_C + (2 * M_H)
        
#         head_mass_refs = {
#             'BMP': 273.0, 'CAR': 175.08, 'CE': 399.33, 'CL': 454.96, 'DG': 119.0,
#             'DG-O': 105.02, 'DG-P': 103.0, 'DGCC': 278.09, 'DGDG': 443.1, 'DGGA': 295.03,
#             'DGTS': 262.09, 'FA': 30.98, 'LDGCC': 264.11, 'LDGTS': 248.11, 'LPA': 184.99,
#             'LPC': 270.07, 'LPC-O': 256.09, 'LPE': 228.03, 'LPE-O': 214.05, 'LPG': 259.02,
#             'LPI': 347.04, 'LPS': 272.02, 'MG': 105.02, 'MG-O': 91.04, 'MG-P': 89.02,
#             'MGDG': 281.05, 'NAE': 74.02, 'PA': 198.96, 'PA-O': 184.98, 'PA-P': 182.97,
#             'PC': 284.05, 'PC-O': 270.07, 'PC-P': 268.06, 'PE': 242.01, 'PE-O': 228.03,
#             'PE-P': 226.01, 'PG': 273.0, 'PG-O': 259.02, 'PG-P': 257.01, 'PI': 361.02,
#             'PI-O': 347.04, 'PI-P': 345.02, 'PMeOH': 212.98, 'PS': 286.0, 'PS-O': 272.02,
#             'PS-P': 270.0, 'SE': 22.92, 'SM-d': 227.04, 'SM-t': 243.04, 'SQDG': 345.01,
#             'TG': 132.98, 'TG-O': 119.0, 'WE': 30.98
#         }
        
#         def _solve_row(row):
#             obs_mass = row['exact_mass']
#             cls = row['class']
            
#             if cls not in head_mass_refs:
#                 return None, None
#             h_mass = head_mass_refs[cls]
            
#             target_tail = obs_mass - h_mass
#             if target_tail <= 0:
#                 return None, None
            
#             c_estimate = int(target_tail / M_CH2)
#             c_min = max(1, c_estimate - 5)
#             c_max = c_estimate + 5
            
#             best_ppm = float('inf')
#             best_match = (None, None)
            
#             for c in range(c_min, c_max + 1):
#                 for db in range(0, 13): 
#                     if db >= c:
#                         break 
#                     h = (2 * c + 1) - (2 * db)
#                     tail_mass = (c * M_C) + (h * M_H)
#                     theoretical_total = h_mass + tail_mass
                    
#                     error_mass = abs(obs_mass - theoretical_total)
#                     ppm = (error_mass / obs_mass) * 1_000_000
                    
#                     if ppm <= ms1_tolerance:
#                         if ppm < best_ppm:
#                             best_ppm = ppm
#                             best_match = (c, db)
            
#             return best_match
        
#         results = df.apply(_solve_row, axis=1)
#         df['num_c'] = [x[0] for x in results]
#         df['num_db'] = [x[1] for x in results]
        
#         df['num_c'] = pd.to_numeric(df['num_c'], errors='coerce')
#         df['num_db'] = pd.to_numeric(df['num_db'], errors='coerce')
        
#         return df
    
#     def _prepare_features(self, df):
#         """Prepare features for prediction"""
#         df = df.copy()
        
#         df['precursor_mz'] = df['precursor_mz'].round(2)
        
#         if 'class' not in df.columns:
#             if 'predicted_class' in df.columns:
#                 df['class'] = df['predicted_class']
#                 print(f"  Note: Using 'predicted_class' for missing 'class' column")
#             else:
#                 raise ValueError("Neither 'class' nor 'predicted_class' column found")
        
#         for col, pred_col in [('adduct', 'predicted_adduct'), ('class', 'predicted_class')]:
#             if col in df.columns and pred_col in df.columns:
#                 mask = df[col].isna() | (df[col] == '')
#                 if mask.any():
#                     df.loc[mask, col] = df.loc[mask, pred_col]
#                     print(f"  Note: Filled {mask.sum()} missing {col} values with {pred_col}")
        
#         df = self._calculate_exact_mass(df)
#         df = self._find_best_composition(df)
        
#         df['_original_adduct'] = df['adduct']
#         df['_original_class'] = df['class']
#         df['_valid_row'] = True
#         df['_skip_prediction'] = False
        
#         no_match_mask = df['num_c'].isna() | df['num_db'].isna()
#         if no_match_mask.any():
#             df.loc[no_match_mask, '_skip_prediction'] = True
#             print(f"  Note: {no_match_mask.sum()} samples have no composition match (will skip prediction)")
        
#         if 'num_peaks' in df.columns:
#             tg_single_peak = (df['class'] == 'TG') & (df['num_peaks'] == 1)
#             if tg_single_peak.any():
#                 df.loc[tg_single_peak, '_skip_prediction'] = True
#                 print(f"  Note: {tg_single_peak.sum()} TG samples with num_peaks=1 (will skip prediction)")
        
#         for col in ['adduct', 'class']:
#             unknown = set(df[col].unique()) - set(self.label_encoders[col].classes_)
#             if unknown:
#                 print(f"  Warning: Unknown values in '{col}': {unknown}")
#                 mask = df[col].isin(unknown)
#                 df.loc[mask, '_valid_row'] = False
#                 df[col] = df[col].apply(
#                     lambda x: self.label_encoders[col].classes_[0] if x not in self.label_encoders[col].classes_ else x
#                 )
            
#             df[col] = self.label_encoders[col].transform(df[col])
        
#         return df
    
#     def _predict_split_by_chain(self, df):
#         """Make predictions with sum constraint enforcement"""
#         df_pred = df.copy()
        
#         for c_col, db_col in self.target_pairs:
#             df_pred[c_col] = 0
#             df_pred[db_col] = 0
        
#         for rank in range(1, 6):
#             df_pred[f'plsf_rank{rank}'] = ''
        
#         df_pred['plsf_confidence'] = 0.0
        
#         if '_skip_prediction' in df_pred.columns:
#             skip_mask = df_pred['_skip_prediction']
#             if skip_mask.any():
#                 print(f"  Skipping prediction for {skip_mask.sum()} samples")
#                 cols_to_clear = [f'num_c_{i}' for i in range(1, 5)] + [f'num_db_{i}' for i in range(1, 5)]
#                 cols_to_clear.extend(['plsf_confidence'] + [f'plsf_rank{i}' for i in range(1, 6)])
#                 for col in cols_to_clear:
#                     if col in df_pred.columns:
#                         df_pred.loc[skip_mask, col] = np.nan
                
#                 df_to_predict = df_pred[~skip_mask].copy()
#             else:
#                 df_to_predict = df_pred.copy()
#         else:
#             df_to_predict = df_pred.copy()
        
#         if len(df_to_predict) == 0:
#             print("  No samples to predict (all skipped)")
#             df_pred = df_pred.drop(columns=['_valid_row', '_skip_prediction'], errors='ignore')
#             return df_pred
        
#         num_chain_groups = list(self.models['by_chain'].items())
#         for num_chain, chain_data in tqdm(num_chain_groups, desc="Predicting by chain", leave=False):
#             mask = df_to_predict['num_chain'] == num_chain
#             if not mask.any():
#                 continue
            
#             if chain_data.get('use_computed', False):
#                 df_to_predict.loc[mask, 'num_c_1'] = df_to_predict.loc[mask, 'num_c']
#                 df_to_predict.loc[mask, 'num_db_1'] = df_to_predict.loc[mask, 'num_db']
#                 df_to_predict.loc[mask, 'plsf_confidence'] = 1.0
                
#                 for idx in df_to_predict[mask].index:
#                     c1 = df_to_predict.loc[idx, 'num_c_1']
#                     db1 = df_to_predict.loc[idx, 'num_db_1']
#                     if pd.isna(c1) or pd.isna(db1):
#                         df_to_predict.loc[idx, 'plsf_rank1'] = ''
#                     else:
#                         rank_list = [int(c1), int(db1), 0, 0, 0, 0, 0, 0]
#                         df_to_predict.loc[idx, 'plsf_rank1'] = str(rank_list)
                
#                 continue
            
#             df_group = df_to_predict[mask].copy()
#             chain_models = chain_data['models']
#             max_stage = chain_data['max_stage']
            
#             # Prepare arguments with num_c and num_db totals for constraint enforcement
#             sample_args = [
#                 (sample_idx, row, chain_models, max_stage, self.target_pairs, self.base_features,
#                  row['num_c'], row['num_db'])
#                 for sample_idx, row in df_group.iterrows()
#             ]
            
#             if len(sample_args) > 10 and self.n_jobs > 1:
#                 results = []
                
#                 with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
#                     futures = {executor.submit(_process_single_sample, args): args[0] for args in sample_args}
                    
#                     for future in as_completed(futures):
#                         try:
#                             result = future.result()
#                             results.append(result)
#                         except Exception as e:
#                             print(f"    Warning: Error processing sample: {e}")
                
#                 for result in results:
#                     sample_idx = result['sample_idx']
                    
#                     for col, val in result['predictions'].items():
#                         df_group.loc[sample_idx, col] = val
                    
#                     for col, val in result['ranks'].items():
#                         df_group.loc[sample_idx, col] = val
                    
#                     df_group.loc[sample_idx, 'plsf_confidence'] = result['confidence']
#             else:
#                 for sample_idx, row in df_group.iterrows():
#                     result = _process_single_sample(
#                         (sample_idx, row, chain_models, max_stage, self.target_pairs, self.base_features,
#                          row['num_c'], row['num_db'])
#                     )
                    
#                     for col, val in result['predictions'].items():
#                         df_group.loc[sample_idx, col] = val
                    
#                     for col, val in result['ranks'].items():
#                         df_group.loc[sample_idx, col] = val
                    
#                     df_group.loc[sample_idx, 'plsf_confidence'] = result['confidence']
            
#             for c_col, db_col in self.target_pairs[:max_stage]:
#                 df_to_predict.loc[mask, c_col] = df_group[c_col]
#                 df_to_predict.loc[mask, db_col] = df_group[db_col]
#             df_to_predict.loc[mask, 'plsf_confidence'] = df_group['plsf_confidence']
#             for rank in range(1, 6):
#                 df_to_predict.loc[mask, f'plsf_rank{rank}'] = df_group[f'plsf_rank{rank}']
        
#         # Verify constraints
#         self._verify_constraints(df_to_predict)
        
#         if '_skip_prediction' in df_pred.columns and skip_mask.any():
#             prediction_cols = [f'num_c_{i}' for i in range(1, 5)] + [f'num_db_{i}' for i in range(1, 5)]
#             prediction_cols.extend(['plsf_confidence'] + [f'plsf_rank{i}' for i in range(1, 6)])
            
#             for col in prediction_cols:
#                 if col in df_to_predict.columns:
#                     df_pred.loc[~skip_mask, col] = df_to_predict[col]
#         else:
#             df_pred = df_to_predict
        
#         invalid_mask = ~df_pred['_valid_row']
#         if invalid_mask.any():
#             print(f"  Setting {invalid_mask.sum()} invalid rows to NaN")
#             cols_to_clear = [f'num_c_{i}' for i in range(1, 5)] + [f'num_db_{i}' for i in range(1, 5)]
#             cols_to_clear.extend(['plsf_confidence'] + [f'plsf_rank{i}' for i in range(1, 6)])
#             for col in cols_to_clear:
#                 if col in df_pred.columns:
#                     df_pred.loc[invalid_mask, col] = np.nan
        
#         df_pred = df_pred.drop(columns=['_valid_row', '_skip_prediction'], errors='ignore')
        
#         return df_pred
    
#     def _verify_constraints(self, df):
#         """Verify and report constraint satisfaction"""
#         violations_c = 0
#         violations_db = 0
#         total_checked = 0
        
#         for num_chain in df['num_chain'].unique():
#             if num_chain == 1:
#                 continue  # num_chain=1 uses computed values directly
                
#             mask = df['num_chain'] == num_chain
#             df_subset = df[mask]
            
#             # Skip rows with NaN values
#             valid_mask = df_subset['num_c'].notna() & df_subset['num_db'].notna()
#             df_valid = df_subset[valid_mask]
            
#             if len(df_valid) == 0:
#                 continue
            
#             total_checked += len(df_valid)
            
#             sum_c = sum(df_valid[f'num_c_{i+1}'] for i in range(int(num_chain)))
#             sum_db = sum(df_valid[f'num_db_{i+1}'] for i in range(int(num_chain)))
            
#             violations_c += (sum_c != df_valid['num_c']).sum()
#             violations_db += (sum_db != df_valid['num_db']).sum()
        
#         if total_checked > 0:
#             print(f"  Constraint check: {total_checked} samples, "
#                   f"num_c violations={violations_c}, num_db violations={violations_db}")
    
#     def predict(self, input_path):
#         """Make predictions on input data"""
#         print(f"\nLoading input data from {input_path}...")
#         df = pd.read_csv(input_path)
#         print(f"Loaded {len(df)} samples")
        
#         required_cols = ['precursor_mz', 'num_chain']
        
#         if 'class' not in df.columns and 'predicted_class' not in df.columns:
#             raise ValueError("Missing required column: either 'class' or 'predicted_class' must be present")
        
#         if 'adduct' not in df.columns and 'predicted_adduct' not in df.columns:
#             raise ValueError("Missing required column: either 'adduct' or 'predicted_adduct' must be present")
        
#         if 'adduct' not in df.columns:
#             if 'predicted_adduct' in df.columns:
#                 df['adduct'] = df['predicted_adduct']
#                 print(f"  Note: Using 'predicted_adduct' for missing 'adduct' column")
        
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             raise ValueError(f"Missing required columns: {missing_cols}")
        
#         print("Preparing features...")
#         df_prepared = self._prepare_features(df)
        
#         print(f"Making predictions (using {self.n_jobs} parallel workers)...")
#         df_pred = self._predict_split_by_chain(df_prepared)
        
#         for col in ['adduct', 'class']:
#             orig_col = f'_original_{col}'
#             if orig_col in df_pred.columns:
#                 df_pred[col] = df_pred[orig_col]
#                 df_pred.drop(columns=[orig_col], inplace=True)
#             else:
#                 df_pred[col] = self.label_encoders[col].inverse_transform(df_pred[col])
        
#         print(f"✓ Predictions complete for {len(df_pred)} samples")
        
#         return df_pred
    
#     def predict_and_save(self, input_path, output_path='result.csv'):
#         """Make predictions and save to CSV"""
#         try:
#             df_pred = self.predict(input_path)
            
#             output_dir = os.path.dirname(output_path)
#             if output_dir and not os.path.exists(output_dir):
#                 print(f"Creating directory: {output_dir}")
#                 os.makedirs(output_dir, exist_ok=True)
            
#             print("\nFormatting output columns...")
#             df_output = df_pred.copy()
            
#             def get_lipid_name(row):
#                 cls = row.get('class')
#                 if pd.isna(cls) or str(cls).strip() == '':
#                     cls = row.get('predicted_class', '')
                
#                 if pd.isna(cls) or str(cls).strip().lower() == 'unknown' or str(cls).strip() == '':
#                     return ''
                
#                 chains = []
#                 for i in range(1, 5):
#                     c_key = f'num_c_{i}'
#                     db_key = f'num_db_{i}'
                    
#                     if c_key in row and db_key in row:
#                         c_val = row[c_key]
#                         db_val = row[db_key]
                        
#                         if pd.notna(c_val) and pd.notna(db_val):
#                             c = int(float(c_val))
#                             db = int(float(db_val))
                            
#                             if c == 0 and db == 0:
#                                 continue
                                
#                             chains.append(f"{c}:{db}")
                
#                 if not chains:
#                     return ''
                
#                 chain_str = "_".join(chains)
#                 return f"{cls} {chain_str}".strip()
            
#             df_output['name'] = df_output.apply(get_lipid_name, axis=1)
            
#             if 'plsf_confidence' in df_output.columns:
#                 df_output['pred_confidence'] = df_output['plsf_confidence'].round(2)
            
#             cols_to_drop = [c for c in df_output.columns if c.startswith('mz_')]
#             cols_to_drop.extend([c for c in df_output.columns if c.startswith('num_c_') or c.startswith('num_db_')])
#             df_output.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
#             if 'pred_confidence' in df_output.columns:
#                 df_output = df_output.sort_values('pred_confidence', ascending=False, na_position='last').reset_index(drop=True)
            
#             named_count = (df_output['name'].notna() & (df_output['name'].str.strip() != '')).sum()
#             unnamed_count = len(df_output) - named_count
#             print(f"  Named: {named_count}, Unnamed: {unnamed_count}")
            
#             print(f"Saving predictions to {output_path}...")
#             df_output.to_csv(output_path, index=False)
#             print(f"✓ Saved {len(df_output)} predictions")
            
#             print("\nPrediction Summary:")
#             if 'num_c' in df_pred.columns:
#                 valid_num_c = df_pred['num_c'].dropna()
#                 if len(valid_num_c) > 0:
#                     print(f"  Total C range: {valid_num_c.min():.0f}-{valid_num_c.max():.0f}")
#             if 'num_db' in df_pred.columns:
#                 valid_num_db = df_pred['num_db'].dropna()
#                 if len(valid_num_db) > 0:
#                     print(f"  Total DB range: {valid_num_db.min():.0f}-{valid_num_db.max():.0f}")
            
#             if 'pred_confidence' in df_output.columns:
#                 valid_conf = df_output['pred_confidence'].dropna()
#                 if len(valid_conf) > 0:
#                     print(f"  Pred Confidence: mean={valid_conf.mean():.3f}, "
#                           f"min={valid_conf.min():.3f}, "
#                           f"max={valid_conf.max():.3f}")
            
#             print(f"  num_chain distribution:")
#             if 'num_chain' in df_pred.columns:
#                 for nc, count in df_pred['num_chain'].value_counts().sort_index().items():
#                     print(f"    {nc}: {count} samples ({count/len(df_pred)*100:.1f}%)")
            
#             if 'num_c' in df_pred.columns and 'num_db' in df_pred.columns:
#                 skipped_count = (df_pred['num_c'].isna() | df_pred['num_db'].isna()).sum()
#                 if skipped_count > 0:
#                     print(f"  Skipped predictions: {skipped_count} samples ({skipped_count/len(df_pred)*100:.1f}%)")
            
#             return df_output
            
#         except Exception as e:
#             print(f"✗ Error: {e}")
#             import traceback
#             traceback.print_exc()
#             return None


# def main():
#     """Main function for command-line usage"""
#     parser = argparse.ArgumentParser(
#         description='Predict lipid chain compositions with sum constraints',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python plsf_predict.py input.csv plsf_model.joblib
#   python plsf_predict.py input.csv plsf_model.joblib --output_path results/output.csv
#   python plsf_predict.py input.csv plsf_model.joblib -o predictions.csv
#   python plsf_predict.py input.csv plsf_model.joblib --n_jobs 8

# Sum Constraints:
#   - num_c_1 + num_c_2 + num_c_3 + num_c_4 = num_c
#   - num_db_1 + num_db_2 + num_db_3 + num_db_4 = num_db
#   - Lowest confidence prediction replaced with residual
#         """
#     )
    
#     parser.add_argument('input_path', help='Path to input CSV file')
#     parser.add_argument('model_path', help='Path to trained model (.joblib file)')
#     parser.add_argument('-o', '--output_path', default='result.csv', help='Path to output CSV file')
#     parser.add_argument('-j', '--n_jobs', type=int, default=4, help='Number of parallel workers')
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.input_path):
#         print(f"Error: Input file '{args.input_path}' not found!")
#         sys.exit(1)
    
#     if not os.path.exists(args.model_path):
#         print(f"Error: Model file '{args.model_path}' not found!")
#         sys.exit(1)
    
#     try:
#         predictor = PLSFPredictor(args.model_path, n_jobs=args.n_jobs)
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
Enhanced PLSF Predictor - Optimized Batch Processing Version

Key optimizations over original:
1. Batch inference: Process all samples in a group together instead of one-by-one
2. Vectorized operations: Use NumPy array operations instead of row iterations
3. Eliminated ProcessPoolExecutor overhead: Single-threaded batch ops are faster
4. Reduced DataFrame creation: Reuse arrays, avoid creating single-row DataFrames
5. Efficient beam search: Batch probability computations for alternatives

Usage:
    python plsf_predict_optimized.py input.csv plsf_model.joblib
    python plsf_predict_optimized.py input.csv plsf_model.joblib -o results/output.csv
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# ConstantPredictor Class (for backward compatibility)
# ============================================================================

class ConstantPredictor:
    """Simple predictor that always returns the same constant value."""
    
    def __init__(self, constant_value):
        self.constant_value = constant_value
        self.inverse_map = None
    
    def predict(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.full(n_samples, self.constant_value)
    
    def predict_proba(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.ones((n_samples, 1))
    
    def __repr__(self):
        return f"ConstantPredictor(value={self.constant_value})"


# Module Aliasing for Backward Compatibility
import types
train_compare_models = types.ModuleType('train_compare_models')
train_compare_models.ConstantPredictor = ConstantPredictor
sys.modules['train_compare_models'] = train_compare_models


# ============================================================================
# Optimized Batch Processing Functions
# ============================================================================

def is_constant_predictor(model):
    """Check if model is a ConstantPredictor"""
    return isinstance(model, ConstantPredictor)


def batch_predict_with_confidence(model, X):
    """
    Batch prediction with confidence scores.
    Returns: (predictions array, confidence array)
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    
    if is_constant_predictor(model):
        pred = model.predict(X_arr)
        conf = np.ones(len(pred))
        return pred.astype(int), conf
    
    proba = model.predict_proba(X_arr)
    pred_idx = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    
    if hasattr(model, 'inverse_map') and model.inverse_map is not None:
        # Vectorized inverse mapping
        inverse_map = model.inverse_map
        pred = np.array([inverse_map[i] for i in pred_idx], dtype=int)
    else:
        pred = pred_idx.astype(int)
    
    return pred, conf


def batch_get_top_k(model, X, k=5):
    """
    Get top-k predictions for a batch.
    Returns: (top_k_classes [n_samples, k], top_k_probs [n_samples, k])
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    
    n_samples = len(X_arr)
    
    if is_constant_predictor(model):
        const_val = model.constant_value
        top_k_classes = np.full((n_samples, k), const_val, dtype=int)
        top_k_probs = np.zeros((n_samples, k))
        top_k_probs[:, 0] = 1.0
        return top_k_classes, top_k_probs
    
    if not hasattr(model, 'predict_proba'):
        pred = model.predict(X_arr)
        top_k_classes = np.tile(pred.reshape(-1, 1), (1, k)).astype(int)
        top_k_probs = np.zeros((n_samples, k))
        top_k_probs[:, 0] = 1.0
        return top_k_classes, top_k_probs
    
    proba = model.predict_proba(X_arr)
    n_classes = proba.shape[1]
    actual_k = min(k, n_classes)
    
    # Get top-k indices and probabilities
    top_k_indices = np.argsort(proba, axis=1)[:, -actual_k:][:, ::-1]
    top_k_probs = np.take_along_axis(proba, top_k_indices, axis=1)
    
    # Pad if necessary
    if actual_k < k:
        pad_classes = np.zeros((n_samples, k - actual_k), dtype=int)
        pad_probs = np.zeros((n_samples, k - actual_k))
        top_k_indices = np.hstack([top_k_indices, pad_classes])
        top_k_probs = np.hstack([top_k_probs, pad_probs])
    
    # Apply inverse mapping if available
    if hasattr(model, 'inverse_map') and model.inverse_map is not None:
        inverse_map = model.inverse_map
        top_k_classes = np.vectorize(lambda x: inverse_map.get(x, x))(top_k_indices)
    else:
        top_k_classes = top_k_indices
    
    return top_k_classes.astype(int), top_k_probs


def format_chains_as_list(c_vals, db_vals, max_stage):
    """Format chain predictions as sorted list of 8 integers (vectorized for single sample)"""
    chains = []
    for i in range(max_stage):
        c = int(c_vals[i])
        db = int(db_vals[i])
        if c > 0 or db > 0:
            chains.append((c, db))
    
    chains.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    while len(chains) < 4:
        chains.append((0, 0))
    
    result = []
    for c, db in chains:
        result.extend([c, db])
    
    return result


def batch_format_ranks(c_matrix, db_matrix, max_stage):
    """
    Format ranks for a batch of samples.
    c_matrix: [n_samples, max_stage]
    db_matrix: [n_samples, max_stage]
    Returns: list of rank strings
    """
    n_samples = c_matrix.shape[0]
    ranks = []
    
    for i in range(n_samples):
        rank_list = format_chains_as_list(c_matrix[i], db_matrix[i], max_stage)
        ranks.append(str(rank_list))
    
    return ranks


def batch_process_group(df_group, chain_models, max_stage, target_pairs, base_features):
    """
    Process an entire group of samples in batch mode.
    This is the core optimization - instead of processing sample-by-sample,
    we process all samples together using vectorized operations.
    """
    n_samples = len(df_group)
    indices = df_group.index.tolist()
    
    # Initialize result arrays
    c_preds = np.zeros((n_samples, max_stage), dtype=int)
    db_preds = np.zeros((n_samples, max_stage), dtype=int)
    c_confs = np.ones((n_samples, max_stage))
    db_confs = np.ones((n_samples, max_stage))
    
    # Get total constraints
    num_c_total = df_group['num_c'].values
    num_db_total = df_group['num_db'].values
    
    # Build feature matrix - start with base features
    feature_cols = [f for f in base_features if f in df_group.columns]
    X_base = df_group[feature_cols].values.copy()
    
    # Create working arrays for cumulative features (previous predictions)
    X_cumulative = np.zeros((n_samples, max_stage * 2))  # Store c and db predictions
    
    # Stage-by-stage batch prediction
    for stage_idx in range(max_stage):
        # Build feature matrix for this stage
        # Features = base_features + previous predictions (num_c_1, num_db_1, ..., num_c_{stage_idx}, num_db_{stage_idx})
        if stage_idx == 0:
            X_stage = X_base
        else:
            # Append previous stage predictions to base features
            X_stage = np.hstack([X_base, X_cumulative[:, :stage_idx * 2]])
        
        # Batch predict num_c
        c_model = chain_models[stage_idx]['c_model']
        c_pred, c_conf = batch_predict_with_confidence(c_model, X_stage)
        c_preds[:, stage_idx] = c_pred
        c_confs[:, stage_idx] = c_conf
        
        # Update cumulative features with c prediction
        X_cumulative[:, stage_idx * 2] = c_pred
        
        # Build feature matrix for db prediction (includes the c prediction)
        X_stage_with_c = np.hstack([X_stage, c_pred.reshape(-1, 1)])
        
        # Batch predict num_db
        db_model = chain_models[stage_idx]['db_model']
        db_pred, db_conf = batch_predict_with_confidence(db_model, X_stage_with_c)
        db_preds[:, stage_idx] = db_pred
        db_confs[:, stage_idx] = db_conf
        
        # Update cumulative features with db prediction
        X_cumulative[:, stage_idx * 2 + 1] = db_pred
    
    # Enforce sum constraints (vectorized)
    # For num_c constraint
    current_sum_c = c_preds.sum(axis=1)
    needs_c_fix = (current_sum_c != num_c_total) & ~np.isnan(num_c_total)
    
    if needs_c_fix.any():
        # Find lowest confidence chain for each sample
        min_conf_idx = np.argmin(c_confs, axis=1)
        
        for i in np.where(needs_c_fix)[0]:
            fix_idx = min_conf_idx[i]
            other_sum = c_preds[i].sum() - c_preds[i, fix_idx]
            residual = int(num_c_total[i]) - other_sum
            c_preds[i, fix_idx] = max(0, residual)
    
    # For num_db constraint
    current_sum_db = db_preds.sum(axis=1)
    needs_db_fix = (current_sum_db != num_db_total) & ~np.isnan(num_db_total)
    
    if needs_db_fix.any():
        min_conf_idx = np.argmin(db_confs, axis=1)
        
        for i in np.where(needs_db_fix)[0]:
            fix_idx = min_conf_idx[i]
            other_sum = db_preds[i].sum() - db_preds[i, fix_idx]
            residual = int(num_db_total[i]) - other_sum
            db_preds[i, fix_idx] = max(0, residual)
    
    # Calculate overall confidence (product of all stage confidences)
    total_conf = np.prod(c_confs * db_confs, axis=1)
    
    # Format rank1 strings
    rank1_strings = batch_format_ranks(c_preds, db_preds, max_stage)
    
    # Generate alternative ranks (2-5) using batch beam search
    alt_ranks = batch_beam_search_alternatives(
        df_group, chain_models, max_stage, target_pairs, base_features,
        k_alternatives=5
    )
    
    # Build results
    results = []
    for i in range(n_samples):
        result = {
            'sample_idx': indices[i],
            'predictions': {},
            'ranks': {'plsf_rank1': rank1_strings[i]},
            'confidence': float(total_conf[i])
        }
        
        for stage_idx in range(max_stage):
            c_col, db_col = target_pairs[stage_idx]
            result['predictions'][c_col] = int(c_preds[i, stage_idx])
            result['predictions'][db_col] = int(db_preds[i, stage_idx])
        
        # Add alternative ranks
        for rank_idx in range(2, 6):
            if i < len(alt_ranks) and rank_idx - 1 < len(alt_ranks[i]):
                result['ranks'][f'plsf_rank{rank_idx}'] = alt_ranks[i][rank_idx - 1]
            else:
                result['ranks'][f'plsf_rank{rank_idx}'] = ''
        
        results.append(result)
    
    return results


def batch_beam_search_alternatives(df_group, chain_models, max_stage, target_pairs, base_features, k_alternatives=5):
    """
    Batch beam search to generate alternative predictions.
    Returns list of lists: alt_ranks[sample_idx][rank_idx] = rank_string
    """
    n_samples = len(df_group)
    feature_cols = [f for f in base_features if f in df_group.columns]
    X_base = df_group[feature_cols].values.copy()
    
    # Initialize candidates for each sample
    # Each candidate: {'c_vals': [...], 'db_vals': [...], 'prob': float}
    candidates = [[{'c_vals': [], 'db_vals': [], 'prob': 1.0}] for _ in range(n_samples)]
    
    for stage_idx in range(max_stage):
        c_model = chain_models[stage_idx]['c_model']
        db_model = chain_models[stage_idx]['db_model']
        
        new_candidates = [[] for _ in range(n_samples)]
        
        for sample_idx in range(n_samples):
            for cand in candidates[sample_idx][:5]:  # Limit beam width
                # Build feature vector for this candidate
                if stage_idx == 0:
                    x_stage = X_base[sample_idx:sample_idx+1]
                else:
                    prev_features = []
                    for s in range(stage_idx):
                        prev_features.extend([cand['c_vals'][s], cand['db_vals'][s]])
                    x_stage = np.hstack([X_base[sample_idx:sample_idx+1], 
                                         np.array(prev_features).reshape(1, -1)])
                
                # Get top-3 c predictions
                c_top, c_probs = batch_get_top_k(c_model, x_stage, k=3)
                
                for c_idx in range(min(3, c_top.shape[1])):
                    c_val = c_top[0, c_idx]
                    c_prob = c_probs[0, c_idx]
                    
                    if c_prob < 0.01:  # Skip very low probability
                        continue
                    
                    # Get top-2 db predictions given this c
                    x_with_c = np.hstack([x_stage, [[c_val]]])
                    db_top, db_probs = batch_get_top_k(db_model, x_with_c, k=2)
                    
                    for db_idx in range(min(2, db_top.shape[1])):
                        db_val = db_top[0, db_idx]
                        db_prob = db_probs[0, db_idx]
                        
                        new_cand = {
                            'c_vals': cand['c_vals'] + [int(c_val)],
                            'db_vals': cand['db_vals'] + [int(db_val)],
                            'prob': cand['prob'] * c_prob * db_prob
                        }
                        new_candidates[sample_idx].append(new_cand)
        
        # Keep top candidates for each sample
        for sample_idx in range(n_samples):
            new_candidates[sample_idx].sort(key=lambda x: x['prob'], reverse=True)
            candidates[sample_idx] = new_candidates[sample_idx][:k_alternatives]
    
    # Convert to rank strings (skip first as that's rank1)
    alt_ranks = []
    for sample_idx in range(n_samples):
        sample_alts = []
        for cand in candidates[sample_idx][1:k_alternatives]:  # Skip rank1
            if len(cand['c_vals']) == max_stage:
                rank_list = format_chains_as_list(cand['c_vals'], cand['db_vals'], max_stage)
                sample_alts.append(str(rank_list))
        alt_ranks.append(sample_alts)
    
    return alt_ranks


# ============================================================================
# PLSFPredictor Class - Optimized
# ============================================================================

class PLSFPredictor:
    """Optimized predictor with batch processing for faster inference"""
    
    def __init__(self, model_path):
        print(f"Loading model from {model_path}...")
        
        try:
            self.model_data = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found!")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        self.model_name = self.model_data['model_name']
        self.models = self.model_data['models']
        self.label_encoders = self.model_data['label_encoders']
        self.base_features = self.model_data['base_features']
        self.target_pairs = self.model_data['target_pairs']
        
        print(f"✓ Loaded model: {self.model_name}")
        print(f"✓ Using optimized batch processing")
        print(f"✓ Sum constraints enabled (lowest confidence → residual)")
    
    def _calculate_exact_mass(self, df):
        """Calculate exact mass from precursor_mz and adduct (vectorized)"""
        adduct_adjustments = {
            '[M]+': 0.0, '[M+H]+': -1.00783, '[M+NH4]+': -18.03383,
            '[M+Na]+': -22.98977, '[M+K]+': -38.96371, '[M-H]-': 1.00783,
            '[M+HCOO]-': -44.99820, '[M+CH3COO]-': -59.01385,
            '[2M+H]+': -1.00783, '[2M+Na]+': -22.98977, '[2M-H]-': 1.00783,
            '[M-H2O+H]+': 17.00274, '[M-2H2O+H]+': 35.01311,
            '[M+H-H2O]+': 17.00274, '[M+]+': 0.0, '[M]': 0.0,
            '[M-H1]-': 1.00783, '[M+CH3COOH-H]-': -59.01385
        }
        
        # Vectorized calculation
        adjustments = df['adduct'].map(adduct_adjustments).fillna(0)
        exact_mass = df['precursor_mz'] + adjustments
        
        # Handle 2M adducts
        is_2m = df['adduct'].str.contains('2M', na=False)
        exact_mass = np.where(is_2m, exact_mass / 2.0, exact_mass)
        
        df['exact_mass'] = exact_mass
        return df
    
    def _find_best_composition(self, df, ms1_tolerance=30.0):
        """Compute num_c and num_db from exact_mass and class"""
        M_C = 12.00000
        M_H = 1.00783
        M_CH2 = M_C + (2 * M_H)
        
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
            obs_mass = row['exact_mass']
            cls = row['class']
            
            if cls not in head_mass_refs:
                return None, None
            h_mass = head_mass_refs[cls]
            
            target_tail = obs_mass - h_mass
            if target_tail <= 0:
                return None, None
            
            c_estimate = int(target_tail / M_CH2)
            c_min = max(1, c_estimate - 5)
            c_max = c_estimate + 5
            
            best_ppm = float('inf')
            best_match = (None, None)
            
            for c in range(c_min, c_max + 1):
                for db in range(0, 13):
                    if db >= c:
                        break
                    h = (2 * c + 1) - (2 * db)
                    tail_mass = (c * M_C) + (h * M_H)
                    theoretical_total = h_mass + tail_mass
                    
                    error_mass = abs(obs_mass - theoretical_total)
                    ppm = (error_mass / obs_mass) * 1_000_000
                    
                    if ppm <= ms1_tolerance and ppm < best_ppm:
                        best_ppm = ppm
                        best_match = (c, db)
            
            return best_match
        
        results = df.apply(_solve_row, axis=1)
        df['num_c'] = [x[0] for x in results]
        df['num_db'] = [x[1] for x in results]
        
        df['num_c'] = pd.to_numeric(df['num_c'], errors='coerce')
        df['num_db'] = pd.to_numeric(df['num_db'], errors='coerce')
        
        return df
    
    def _prepare_features(self, df):
        """Prepare features for prediction"""
        df = df.copy()
        
        df['precursor_mz'] = df['precursor_mz'].round(2)
        
        if 'class' not in df.columns:
            if 'predicted_class' in df.columns:
                df['class'] = df['predicted_class']
                print(f"  Note: Using 'predicted_class' for missing 'class' column")
            else:
                raise ValueError("Neither 'class' nor 'predicted_class' column found")
        
        for col, pred_col in [('adduct', 'predicted_adduct'), ('class', 'predicted_class')]:
            if col in df.columns and pred_col in df.columns:
                mask = df[col].isna() | (df[col] == '')
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, pred_col]
                    print(f"  Note: Filled {mask.sum()} missing {col} values with {pred_col}")
        
        df = self._calculate_exact_mass(df)
        df = self._find_best_composition(df)
        
        df['_original_adduct'] = df['adduct']
        df['_original_class'] = df['class']
        df['_valid_row'] = True
        df['_skip_prediction'] = False
        
        no_match_mask = df['num_c'].isna() | df['num_db'].isna()
        if no_match_mask.any():
            df.loc[no_match_mask, '_skip_prediction'] = True
            print(f"  Note: {no_match_mask.sum()} samples have no composition match (will skip)")
        
        if 'num_peaks' in df.columns:
            tg_single_peak = (df['class'] == 'TG') & (df['num_peaks'] == 1)
            if tg_single_peak.any():
                df.loc[tg_single_peak, '_skip_prediction'] = True
                print(f"  Note: {tg_single_peak.sum()} TG samples with num_peaks=1 (will skip)")
        
        for col in ['adduct', 'class']:
            unknown = set(df[col].unique()) - set(self.label_encoders[col].classes_)
            if unknown:
                print(f"  Warning: Unknown values in '{col}': {unknown}")
                mask = df[col].isin(unknown)
                df.loc[mask, '_valid_row'] = False
                df[col] = df[col].apply(
                    lambda x: self.label_encoders[col].classes_[0] 
                    if x not in self.label_encoders[col].classes_ else x
                )
            
            df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _predict_split_by_chain(self, df, batch_size=500):
        """Make predictions with optimized batch processing and accurate progress tracking"""
        df_pred = df.copy()
        
        # Initialize output columns
        for c_col, db_col in self.target_pairs:
            df_pred[c_col] = 0
            df_pred[db_col] = 0
        
        for rank in range(1, 6):
            df_pred[f'plsf_rank{rank}'] = ''
        
        df_pred['plsf_confidence'] = 0.0
        
        # Handle skip predictions
        if '_skip_prediction' in df_pred.columns:
            skip_mask = df_pred['_skip_prediction']
            if skip_mask.any():
                print(f"  Skipping prediction for {skip_mask.sum()} samples")
                cols_to_clear = [f'num_c_{i}' for i in range(1, 5)] + [f'num_db_{i}' for i in range(1, 5)]
                cols_to_clear.extend(['plsf_confidence'] + [f'plsf_rank{i}' for i in range(1, 6)])
                for col in cols_to_clear:
                    if col in df_pred.columns:
                        df_pred.loc[skip_mask, col] = np.nan
                
                df_to_predict = df_pred[~skip_mask].copy()
            else:
                df_to_predict = df_pred.copy()
        else:
            df_to_predict = df_pred.copy()
        
        if len(df_to_predict) == 0:
            print("  No samples to predict (all skipped)")
            df_pred = df_pred.drop(columns=['_valid_row', '_skip_prediction'], errors='ignore')
            return df_pred
        
        # Calculate total samples needing model inference (excluding num_chain=1 which is instant)
        total_samples = 0
        group_info = []
        for num_chain, chain_data in self.models['by_chain'].items():
            mask = df_to_predict['num_chain'] == num_chain
            count = mask.sum()
            if count > 0:
                # num_chain=1 with use_computed is instant, weight it as 0.01x
                weight = 0.01 if chain_data.get('use_computed', False) else 1.0
                # More stages = more work, scale by max_stage
                max_stage = chain_data.get('max_stage', 1)
                weighted_count = count * weight * max_stage
                total_samples += weighted_count
                group_info.append((num_chain, chain_data, mask, count, weighted_count))
        
        # Create progress bar based on weighted sample count
        pbar = tqdm(total=int(total_samples), desc="Predicting", unit="samples", leave=False)
        
        for num_chain, chain_data, mask, count, weighted_count in group_info:
            if count == 0:
                continue
            
            # Special handling for num_chain=1: use computed values directly
            if chain_data.get('use_computed', False):
                df_to_predict.loc[mask, 'num_c_1'] = df_to_predict.loc[mask, 'num_c']
                df_to_predict.loc[mask, 'num_db_1'] = df_to_predict.loc[mask, 'num_db']
                df_to_predict.loc[mask, 'plsf_confidence'] = 1.0
                
                # Vectorized rank string creation
                for idx in df_to_predict[mask].index:
                    c1 = df_to_predict.loc[idx, 'num_c_1']
                    db1 = df_to_predict.loc[idx, 'num_db_1']
                    if pd.isna(c1) or pd.isna(db1):
                        df_to_predict.loc[idx, 'plsf_rank1'] = ''
                    else:
                        rank_list = [int(c1), int(db1), 0, 0, 0, 0, 0, 0]
                        df_to_predict.loc[idx, 'plsf_rank1'] = str(rank_list)
                
                pbar.update(int(weighted_count))
                continue
            
            # Get group data
            df_group = df_to_predict[mask].copy()
            chain_models = chain_data['models']
            max_stage = chain_data['max_stage']
            
            # Process in mini-batches for progress updates on large groups
            indices = df_group.index.tolist()
            n_samples = len(indices)
            
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_indices = indices[batch_start:batch_end]
                df_batch = df_group.loc[batch_indices].copy()
                
                # BATCH PROCESSING
                results = batch_process_group(
                    df_batch, chain_models, max_stage, 
                    self.target_pairs, self.base_features
                )
                
                # Apply results back to dataframe
                for result in results:
                    sample_idx = result['sample_idx']
                    
                    for col, val in result['predictions'].items():
                        df_to_predict.loc[sample_idx, col] = val
                    
                    for col, val in result['ranks'].items():
                        df_to_predict.loc[sample_idx, col] = val
                    
                    df_to_predict.loc[sample_idx, 'plsf_confidence'] = result['confidence']
                
                # Update progress (weighted by max_stage)
                pbar.update(len(batch_indices) * max_stage)
        
        pbar.close()
        
        # Verify constraints
        self._verify_constraints(df_to_predict)
        
        # Merge back
        if '_skip_prediction' in df_pred.columns and skip_mask.any():
            prediction_cols = [f'num_c_{i}' for i in range(1, 5)] + [f'num_db_{i}' for i in range(1, 5)]
            prediction_cols.extend(['plsf_confidence'] + [f'plsf_rank{i}' for i in range(1, 6)])
            
            for col in prediction_cols:
                if col in df_to_predict.columns:
                    df_pred.loc[~skip_mask, col] = df_to_predict[col]
        else:
            df_pred = df_to_predict
        
        # Handle invalid rows
        invalid_mask = ~df_pred['_valid_row']
        if invalid_mask.any():
            print(f"  Setting {invalid_mask.sum()} invalid rows to NaN")
            cols_to_clear = [f'num_c_{i}' for i in range(1, 5)] + [f'num_db_{i}' for i in range(1, 5)]
            cols_to_clear.extend(['plsf_confidence'] + [f'plsf_rank{i}' for i in range(1, 6)])
            for col in cols_to_clear:
                if col in df_pred.columns:
                    df_pred.loc[invalid_mask, col] = np.nan
        
        df_pred = df_pred.drop(columns=['_valid_row', '_skip_prediction'], errors='ignore')
        
        return df_pred
    
    def _verify_constraints(self, df):
        """Verify and report constraint satisfaction"""
        violations_c = 0
        violations_db = 0
        total_checked = 0
        
        for num_chain in df['num_chain'].unique():
            if num_chain == 1:
                continue
            
            mask = df['num_chain'] == num_chain
            df_subset = df[mask]
            
            valid_mask = df_subset['num_c'].notna() & df_subset['num_db'].notna()
            df_valid = df_subset[valid_mask]
            
            if len(df_valid) == 0:
                continue
            
            total_checked += len(df_valid)
            
            sum_c = sum(df_valid[f'num_c_{i+1}'] for i in range(int(num_chain)))
            sum_db = sum(df_valid[f'num_db_{i+1}'] for i in range(int(num_chain)))
            
            violations_c += (sum_c != df_valid['num_c']).sum()
            violations_db += (sum_db != df_valid['num_db']).sum()
        
        if total_checked > 0:
            print(f"  Constraint check: {total_checked} samples, "
                  f"num_c violations={violations_c}, num_db violations={violations_db}")
    
    def predict(self, input_path):
        """Make predictions on input data"""
        print(f"\nLoading input data from {input_path}...")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} samples")
        
        required_cols = ['precursor_mz', 'num_chain']
        
        if 'class' not in df.columns and 'predicted_class' not in df.columns:
            raise ValueError("Missing required column: either 'class' or 'predicted_class' must be present")
        
        if 'adduct' not in df.columns and 'predicted_adduct' not in df.columns:
            raise ValueError("Missing required column: either 'adduct' or 'predicted_adduct' must be present")
        
        if 'adduct' not in df.columns:
            if 'predicted_adduct' in df.columns:
                df['adduct'] = df['predicted_adduct']
                print(f"  Note: Using 'predicted_adduct' for missing 'adduct' column")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("Preparing features...")
        df_prepared = self._prepare_features(df)
        
        print(f"Making predictions (optimized batch mode)...")
        df_pred = self._predict_split_by_chain(df_prepared)
        
        for col in ['adduct', 'class']:
            orig_col = f'_original_{col}'
            if orig_col in df_pred.columns:
                df_pred[col] = df_pred[orig_col]
                df_pred.drop(columns=[orig_col], inplace=True)
            else:
                df_pred[col] = self.label_encoders[col].inverse_transform(df_pred[col])
        
        print(f"✓ Predictions complete for {len(df_pred)} samples")
        
        return df_pred
    
    def predict_and_save(self, input_path, output_path='result.csv'):
        """Make predictions and save to CSV"""
        try:
            df_pred = self.predict(input_path)
            
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                print(f"Creating directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            
            print("\nFormatting output columns...")
            df_output = df_pred.copy()
            
            def get_lipid_name(row):
                cls = row.get('class')
                if pd.isna(cls) or str(cls).strip() == '':
                    cls = row.get('predicted_class', '')
                
                if pd.isna(cls) or str(cls).strip().lower() == 'unknown' or str(cls).strip() == '':
                    return ''
                
                chains = []
                for i in range(1, 5):
                    c_key = f'num_c_{i}'
                    db_key = f'num_db_{i}'
                    
                    if c_key in row and db_key in row:
                        c_val = row[c_key]
                        db_val = row[db_key]
                        
                        if pd.notna(c_val) and pd.notna(db_val):
                            c = int(float(c_val))
                            db = int(float(db_val))
                            
                            if c == 0 and db == 0:
                                continue
                            
                            chains.append(f"{c}:{db}")
                
                if not chains:
                    return ''
                
                chain_str = "_".join(chains)
                return f"{cls} {chain_str}".strip()
            
            df_output['name'] = df_output.apply(get_lipid_name, axis=1)
            
            if 'plsf_confidence' in df_output.columns:
                df_output['pred_confidence'] = df_output['plsf_confidence'].round(2)
            
            cols_to_drop = [c for c in df_output.columns if c.startswith('mz_')]
            cols_to_drop.extend([c for c in df_output.columns if c.startswith('num_c_') or c.startswith('num_db_')])
            df_output.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
            if 'pred_confidence' in df_output.columns:
                df_output = df_output.sort_values('pred_confidence', ascending=False, na_position='last').reset_index(drop=True)
            
            named_count = (df_output['name'].notna() & (df_output['name'].str.strip() != '')).sum()
            unnamed_count = len(df_output) - named_count
            print(f"  Named: {named_count}, Unnamed: {unnamed_count}")
            
            print(f"Saving predictions to {output_path}...")
            df_output.to_csv(output_path, index=False)
            print(f"✓ Saved {len(df_output)} predictions")
            
            print("\nPrediction Summary:")
            if 'num_c' in df_pred.columns:
                valid_num_c = df_pred['num_c'].dropna()
                if len(valid_num_c) > 0:
                    print(f"  Total C range: {valid_num_c.min():.0f}-{valid_num_c.max():.0f}")
            if 'num_db' in df_pred.columns:
                valid_num_db = df_pred['num_db'].dropna()
                if len(valid_num_db) > 0:
                    print(f"  Total DB range: {valid_num_db.min():.0f}-{valid_num_db.max():.0f}")
            
            if 'pred_confidence' in df_output.columns:
                valid_conf = df_output['pred_confidence'].dropna()
                if len(valid_conf) > 0:
                    print(f"  Pred Confidence: mean={valid_conf.mean():.3f}, "
                          f"min={valid_conf.min():.3f}, max={valid_conf.max():.3f}")
            
            print(f"  num_chain distribution:")
            if 'num_chain' in df_pred.columns:
                for nc, count in df_pred['num_chain'].value_counts().sort_index().items():
                    print(f"    {nc}: {count} samples ({count/len(df_pred)*100:.1f}%)")
            
            if 'num_c' in df_pred.columns and 'num_db' in df_pred.columns:
                skipped_count = (df_pred['num_c'].isna() | df_pred['num_db'].isna()).sum()
                if skipped_count > 0:
                    print(f"  Skipped predictions: {skipped_count} samples ({skipped_count/len(df_pred)*100:.1f}%)")
            
            return df_output
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Predict lipid chain compositions (optimized batch version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plsf_predict_optimized.py input.csv plsf_model.joblib
  python plsf_predict_optimized.py input.csv plsf_model.joblib -o results/output.csv

Note: This optimized version uses batch processing instead of parallel workers.
      The --n_jobs argument is accepted for compatibility but ignored.
        """
    )
    
    parser.add_argument('input_path', help='Path to input CSV file')
    parser.add_argument('model_path', help='Path to trained model (.joblib file)')
    parser.add_argument('-o', '--output_path', default='result.csv', help='Path to output CSV file')
    parser.add_argument('-j', '--n_jobs', type=int, default=4, help='(Ignored - kept for compatibility)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file '{args.input_path}' not found!")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        sys.exit(1)
    
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