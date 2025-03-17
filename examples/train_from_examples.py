#!/usr/bin/env python3
"""
Train CWT models using the example JSON files.

This script:
1. Loads all example JSON files from the json_samples directory
2. Converts them to a training dataset format
3. Trains a model on this dataset
4. Saves the trained model

Usage:
    python examples/train_from_examples.py [--model-type rf] [--output-dir models]
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add project root to path so we can import from cwt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Try to import from cwt.py directly
try:
    from cwt import create_model, save_model_metadata, plot_feature_importance, plot_confusion_matrix
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    DIRECT_IMPORT = True
except ImportError:
    DIRECT_IMPORT = False
    print("Could not import from cwt.py directly, will use subprocess for training")

# Constants
EXAMPLES_DIR = Path(__file__).resolve().parent / "json_samples"
OUTPUT_DIR = Path("models")
MODEL_NAME = "Cognitive_State_Prediction_Model"
DEFAULT_MODEL_TYPE = "rf"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CWT models from example JSON files')
    parser.add_argument('--model-type', '-m', type=str, default=DEFAULT_MODEL_TYPE, 
                       help='Type of model to train (rf, svm, gb, mlp, knn, lr)')
    parser.add_argument('--output-dir', '-o', type=str, default=str(OUTPUT_DIR),
                       help='Directory to save the trained model')
    parser.add_argument('--test-size', '-t', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output')
    return parser.parse_args()

def load_json_examples(directory):
    """Load all JSON files from a directory."""
    json_files = list(directory.glob("*.json"))
    examples = []

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both individual samples and batch files
            if isinstance(data, list):
                print(f"Loaded batch file: {file_path.name} with {len(data)} samples")
                examples.extend(data)
            else:
                print(f"Loaded sample: {file_path.name}")
                examples.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Total examples loaded: {len(examples)}")
    return examples

def examples_to_dataframe(examples):
    """Convert JSON examples to a pandas DataFrame."""
    # Extract features and convert to DataFrame
    df = pd.DataFrame(examples)
    
    # Remove non-numeric columns before processing
    # Keep track of which columns are strings vs numeric
    string_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            string_columns.append(col)
    
    # Remove string columns temporarily
    numeric_df = df.drop(columns=string_columns)
    
    # Ensure all required columns exist
    required_features = [
        "pulse_rate", "blood_pressure_sys", "resp_rate", 
        "pupil_diameter_left", "pupil_diameter_right",
        "fixation_duration", "blink_rate", "workload_intensity", 
        "gaze_x", "gaze_y", "alpha_power", "theta_power"
    ]
    
    missing_columns = [col for col in required_features if col not in numeric_df.columns]
    if missing_columns:
        print(f"Warning: Missing required columns in examples: {missing_columns}")
        print("Adding these columns with default values")
        for col in missing_columns:
            numeric_df[col] = np.nan
    
    # Fill missing values with column means for numeric columns only
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    # Create cognitive_state column based on workload_intensity
    if "workload_intensity" in numeric_df.columns:
        # Create categorical cognitive state based on workload intensity
        numeric_df["cognitive_state"] = pd.qcut(
            numeric_df["workload_intensity"], 
            q=3, 
            labels=["Low", "Medium", "High"]
        )
        print("Created cognitive_state column from workload_intensity")
    else:
        print("Error: workload_intensity column is required but not found in examples")
        return None
    
    # Add back string columns if they exist
    for col in string_columns:
        if col != 'sample_id':  # Skip the sample_id column, we don't need it for training
            numeric_df[col] = df[col]
    
    print(f"Created DataFrame with shape: {numeric_df.shape}")
    return numeric_df

def train_model_subprocess(df, model_type, output_dir, test_size):
    """Train a model using the cwt.py script through subprocess."""
    # Save DataFrame to temporary CSV
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Save as CSV files in the format expected by cwt.py
    # Split the data into three files based on cognitive state
    physio_cols = ["subject_id", "timestamp", "pulse_rate", "blood_pressure_sys", 
                   "resp_rate", "skin_conductance", "cognitive_state"]
    eeg_cols = ["subject_id", "timestamp", "alpha_power", "theta_power", 
                "alpha_theta_ratio", "cognitive_state"]
    gaze_cols = ["subject_id", "timestamp", "pupil_diameter_left", "pupil_diameter_right",
                "fixation_duration", "blink_rate", "workload_intensity", 
                "gaze_x", "gaze_y", "cognitive_state"]
    
    # Add missing columns
    if "subject_id" not in df.columns:
        df["subject_id"] = [f"P{i:03d}" for i in range(len(df))]
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(start=datetime.now(), periods=len(df), freq="1min")
    if "skin_conductance" not in df.columns:
        df["skin_conductance"] = np.random.normal(8, 3, len(df))
    if "alpha_theta_ratio" not in df.columns:
        if "alpha_power" in df.columns and "theta_power" in df.columns:
            df["alpha_theta_ratio"] = df["alpha_power"] / df["theta_power"]
        else:
            df["alpha_theta_ratio"] = np.random.normal(1.0, 0.3, len(df))
    
    # Export all three files with only the columns relevant to each
    physio_path = temp_dir / "Enhanced_Workload_Clinical_Data.csv"
    eeg_path = temp_dir / "000_EEG_Cluster_ANOVA_Results.csv"
    gaze_path = temp_dir / "008_01.csv"
    
    # Save only columns that exist in the DataFrame
    df[sorted([col for col in physio_cols if col in df.columns])].to_csv(physio_path, index=False)
    df[sorted([col for col in eeg_cols if col in df.columns])].to_csv(eeg_path, index=False)
    df[sorted([col for col in gaze_cols if col in df.columns])].to_csv(gaze_path, index=False)
    
    print(f"Saved temporary data files in {temp_dir}")
    
    # Create temporary environment file
    env_path = temp_dir / ".env"
    with open(env_path, "w") as f:
        f.write(f"PHYSIO_DATA_PATH={physio_path}\n")
        f.write(f"EEG_DATA_PATH={eeg_path}\n")
        f.write(f"GAZE_DATA_PATH={gaze_path}\n")
        f.write(f"MODEL_OUTPUT_DIR={output_dir}\n")
        f.write(f"MODEL_NAME={MODEL_NAME}\n")
        f.write(f"TEST_SIZE={test_size}\n")
        f.write(f"DEFAULT_MODEL_TYPE={model_type}\n")
    
    print(f"Created temporary .env file at {env_path}")
    
    # Run the train command
    cmd = [sys.executable, "cwt.py", "train", "--model-type", model_type]
    print(f"Running: {' '.join(cmd)}")
    
    # Set environment variables to point to the temporary .env file
    env = os.environ.copy()
    env["DOTENV_PATH"] = str(env_path)
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error training model: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def train_model_direct(df, model_type, output_dir, test_size):
    """Train a model directly using imported functions from cwt.py."""
    # Prepare features and target
    features = ["pulse_rate", "blood_pressure_sys", "resp_rate", 
                "pupil_diameter_left", "pupil_diameter_right",
                "fixation_duration", "blink_rate", "workload_intensity", 
                "gaze_x", "gaze_y", "alpha_power", "theta_power"]
    
    # Ensure all features exist
    for feat in features:
        if feat not in df.columns:
            print(f"Feature {feat} not found, adding with default values")
            df[feat] = np.random.normal(df.mean().mean(), df.std().mean(), len(df))
    
    X = df[features]
    y = df["cognitive_state"]
    
    # Create model version
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_path = Path(output_dir) / f"{MODEL_NAME}_{model_version}.joblib"
    scaler_output_path = Path(output_dir) / f"scaler_{model_version}.joblib"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Check if we have enough samples for stratification
    n_classes = len(y.unique())
    n_samples = len(df)
    min_test_samples = max(2, n_classes)  # Need at least n_classes samples in test set
    
    # Calculate min test_size that would work with our data
    min_test_size = min_test_samples / n_samples
    
    # Use the larger of the provided test_size and the minimum required
    actual_test_size = max(test_size, min_test_size)
    if actual_test_size > test_size:
        print(f"Warning: Increasing test_size from {test_size:.2f} to {actual_test_size:.2f} due to small dataset size")
    
    # For very small datasets, just skip stratification
    if n_samples < n_classes * 2:
        print(f"Warning: Dataset too small for stratification ({n_samples} samples, {n_classes} classes)")
        print("Training on all data, with a 50% holdout for validation")
        
        # Just split 50/50 without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.5, random_state=42
        )
    else:
        # Split into train and test with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=actual_test_size, stratify=y, random_state=42
        )
    
    # Create and train model
    model = create_model(model_type)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save model and scaler
    joblib.dump(model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    
    # Create plots
    cm_path = Path(output_dir) / f"confusion_matrix_{model_version}.png"
    plot_confusion_matrix(y_test, y_pred, cm_path)
    
    if hasattr(model, 'feature_importances_'):
        fi_path = Path(output_dir) / f"feature_importance_{model_version}.png"
        plot_feature_importance(model, X.columns, fi_path)
    
    # Save metadata
    metadata_path = Path(output_dir) / f"metadata_{model_version}.json"
    save_model_metadata(model, model_type, list(X.columns), accuracy, class_report)
    
    print(f"Model saved to {model_output_path}")
    print(f"Scaler saved to {scaler_output_path}")
    
    return True

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("TRAINING CWT MODEL FROM EXAMPLE JSON FILES")
    print("=" * 80)
    
    # Load examples
    examples = load_json_examples(EXAMPLES_DIR)
    if not examples:
        print("No examples found. Please add JSON examples to the json_samples directory.")
        return
    
    # Convert to DataFrame
    df = examples_to_dataframe(examples)
    if df is None:
        return
    
    # Train model
    if DIRECT_IMPORT:
        print("Training model directly using imported functions")
        success = train_model_direct(df, args.model_type, args.output_dir, args.test_size)
    else:
        print("Training model using subprocess")
        success = train_model_subprocess(df, args.model_type, args.output_dir, args.test_size)
    
    if success:
        print("\n" + "=" * 80)
        print("Model training complete!")
        print("=" * 80)
        print("\nThe model has been trained on the example data and saved to the models directory.")
        print("You can now use it for predictions:")
        print(f"  python cwt.py predict --input examples/json_samples/low_workload.json")
    else:
        print("\n" + "=" * 80)
        print("Model training failed.")
        print("=" * 80)

if __name__ == "__main__":
    main() 