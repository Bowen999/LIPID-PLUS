import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path

def load_model(model_path):
    """
    Load the trained model and encoders from joblib file
    """
    try:
        model_package = joblib.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
        print(f"  Model type: {model_package['model_name']}")
        print(f"  Validation Accuracy: {model_package['validation_accuracy']:.4f}")
        print(f"  Test Accuracy: {model_package['test_accuracy']:.4f}")
        return model_package
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        sys.exit(1)


def prepare_input_data(data, ion_mode_encoder):
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
    
    # Define feature columns (must match training)
    feature_cols = ['precursor_mz'] + mz_cols + ['ion_mode_encoded']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_cols if col not in data_processed.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Select features
    X = data_processed[feature_cols]
    
    return X, feature_cols


def predict_adducts(model_package, input_path, output_path):
    """
    Make predictions on input data and save results
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
    
    # Extract model and encoders
    model = model_package['model']
    adduct_encoder = model_package['adduct_encoder']
    ion_mode_encoder = model_package['ion_mode_encoder']
    
    # Prepare input features
    print("\nPreparing features...")
    X, feature_cols = prepare_input_data(data, ion_mode_encoder)
    print(f"✓ Features prepared")
    print(f"  Number of features: {X.shape[1]}")
    
    # Make predictions
    print("\nMaking predictions...")
    try:
        predictions_encoded = model.predict(X)
        predictions = adduct_encoder.inverse_transform(predictions_encoded)
        print(f"✓ Predictions completed")
    except Exception as e:
        print(f"✗ Error making predictions: {str(e)}")
        sys.exit(1)
    
    # Add predictions to original data
    result = data.copy()
    result['predicted_adduct'] = predictions
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(X)
            max_probabilities = np.max(probabilities, axis=1)
            result['adduct_confidence'] = max_probabilities
            print(f"  Average confidence: {max_probabilities.mean():.4f}")
        except:
            pass
    
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
    print(f"Total predictions: {len(predictions)}")
    print(f"\nPredicted adduct distribution:")
    print(result['predicted_adduct'].value_counts().to_string())
    print("="*70)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Predict adducts using trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python adduct_predict.py test.csv model/adduct.joblib --output_path result/pred_adduct.csv
  python adduct_predict.py data.csv adduct.joblib -o predictions.csv
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
        default='adduct_predict.csv',
        help='Path to save predictions (default: adduct_predict.csv)'
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
    print("ADDUCT PREDICTION")
    print("="*70)
    
    # Load model
    model_package = load_model(args.model_path)
    
    # Make predictions
    results = predict_adducts(model_package, args.input_path, args.output_path)
    
    print("\n✓ Prediction completed successfully!")


if __name__ == "__main__":
    main()
    
    