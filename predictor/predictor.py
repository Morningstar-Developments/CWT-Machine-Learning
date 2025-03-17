#!/usr/bin/env python3
"""
Cognitive Workload Assessment and Prediction Tool

This module implements a machine learning pipeline for predicting cognitive workload
based on physiological, EEG, and gaze data. It provides:
  - Data loading and basic preprocessing
  - Optional hyperparameter tuning for model training (RandomizedSearchCV)
  - Model saving (Random Forest + scaler)
  - Easy prediction with the latest or user-specified model

Usage:
    python predictor.py train
    python predictor.py predict --input [input_data.json]
    python predictor.py list-models
"""

import os
import sys
import json
import logging
from datetime import datetime
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------- ENVIRONMENT AND CONFIGURATION ---------------------- #
# Load environment variables if .env file exists
if os.path.exists('.env'):
    load_dotenv()

# Set up logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/cwt.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('cwt')

# Warn if certain environment variables are missing
if 'PHYSIO_DATA_PATH' not in os.environ:
    logger.warning("PHYSIO_DATA_PATH not set in the environment. Using default path.")

# Data paths configuration
DATA_FILES = {
    "physiological": os.getenv('PHYSIO_DATA_PATH', 'data/Enhanced_Workload_Clinical_Data.csv'),
    "eeg": os.getenv('EEG_DATA_PATH', 'data/000_EEG_Cluster_ANOVA_Results.csv'),
    "gaze": os.getenv('GAZE_DATA_PATH', 'data/008_01.csv')
}

# Model and output configuration
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', 'models')
MODEL_NAME = os.getenv('MODEL_NAME', 'Cognitive_State_Prediction_Model')
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.joblib")
SCALER_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"scaler_{MODEL_VERSION}.joblib")
METADATA_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"metadata_{MODEL_VERSION}.json")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Training parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# Random Forest hyperparameters
RF_N_ESTIMATORS = int(os.getenv('RF_N_ESTIMATORS', 100))
RF_MAX_DEPTH = os.getenv('RF_MAX_DEPTH', None)
RF_MAX_DEPTH = int(RF_MAX_DEPTH) if RF_MAX_DEPTH and RF_MAX_DEPTH.isdigit() else None
RF_N_JOBS = int(os.getenv('RF_N_JOBS', -1))

# Whether to use hyperparameter tuning via RandomizedSearchCV (set TUNE_MODEL=true in .env)
TUNE_MODEL = (os.getenv('TUNE_MODEL', 'false').lower() == 'true')


# ---------------------- HELPER FUNCTIONS ---------------------- #
def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists on disk, raising an error otherwise.
    """
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}. Please verify your data path or environment settings."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    return True


def safe_read_csv(file_path: str) -> pd.DataFrame:
    """
    Safely read a CSV file with error handling.
    """
    try:
        validate_file_exists(file_path)
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {str(e)}")
        raise


def save_model_metadata(
    model: Any, features: List[str], accuracy: float, class_report: Dict[str, Any]
) -> None:
    """
    Save metadata about the model for reproducibility.
    Includes accuracy, classification report, feature set, and hyperparameters.
    """
    metadata = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "classification_report": class_report,
        "features": features,
        "parameters": {
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "RF_N_ESTIMATORS": RF_N_ESTIMATORS,
            "RF_MAX_DEPTH": RF_MAX_DEPTH,
            "RF_N_JOBS": RF_N_JOBS,
            "tune_model": TUNE_MODEL
        }
    }
    try:
        with open(METADATA_OUTPUT_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model metadata saved to {METADATA_OUTPUT_PATH}")
    except Exception as e:
        logger.exception(f"Failed to save model metadata: {str(e)}")
        raise


def plot_feature_importance(model: Any, feature_names: List[str], output_path: str = None) -> None:
    """
    Plot feature importance from the trained model.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_, skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")
        plt.clf()
    else:
        plt.show()


def plot_confusion_matrix(y_true: Any, y_pred: Any, output_path: str = None) -> None:
    """
    Plot confusion matrix from predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_true)),
                yticklabels=sorted(set(y_true)))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Confusion matrix plot saved to {output_path}")
        plt.clf()
    else:
        plt.show()


# ---------------------- DATA LOADING AND PREPROCESSING ---------------------- #
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the physiological, EEG, and gaze data from CSV files.
    
    Returns:
        (df_physio, df_eeg, df_gaze)
    """
    logger.info("Loading data files...")
    try:
        df_physio = safe_read_csv(DATA_FILES["physiological"])
        df_eeg = safe_read_csv(DATA_FILES["eeg"])
        df_gaze = safe_read_csv(DATA_FILES["gaze"])
        
        logger.info(f"Loaded physiological data: {df_physio.shape}")
        logger.info(f"Loaded EEG data: {df_eeg.shape}")
        logger.info(f"Loaded gaze data: {df_gaze.shape}")
        
        return df_physio, df_eeg, df_gaze
    except Exception as e:
        logger.exception(f"Error loading data: {str(e)}")
        raise


def preprocess_data(
    df_physio: pd.DataFrame, df_eeg: pd.DataFrame, df_gaze: pd.DataFrame
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Preprocess and merge the physiological, EEG, and gaze data.
    Returns:
        (processed_df, fitted_scaler, feature_names)
    """
    logger.info("Preprocessing data...")
    try:
        # Ensure that each DataFrame has a 'timestamp' column and convert to datetime.
        for df, name in [(df_physio, "physiological"), (df_eeg, "eeg"), (df_gaze, "gaze")]:
            if "timestamp" not in df.columns:
                logger.error(f"'timestamp' column not found in {name} data")
                raise KeyError(f"'timestamp' column not found in {name} data")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Merge datasets based on nearest timestamp
        df_merged = pd.merge_asof(df_physio.sort_values("timestamp"), 
                                  df_eeg.sort_values("timestamp"), 
                                  on="timestamp")
        df_merged = pd.merge_asof(df_merged.sort_values("timestamp"), 
                                  df_gaze.sort_values("timestamp"), 
                                  on="timestamp")
        
        # Drop rows with missing values after merge
        missing_values = df_merged.isnull().sum()
        if missing_values.any():
            logger.warning(
                "Missing values detected after merge:\n"
                f"{missing_values[missing_values > 0]}"
            )
            logger.info("Dropping rows with missing values")
            df_merged = df_merged.dropna()
        
        # Define the expected features
        expected_features = [
            "pulse_rate", "blood_pressure_sys", "resp_rate",
            "pupil_diameter_left", "pupil_diameter_right",
            "fixation_duration", "blink_rate", 
            "workload_intensity", "gaze_x", "gaze_y",
            "alpha_power", "theta_power"
        ]
        
        # Validate features exist
        missing_feats = [f for f in expected_features if f not in df_merged.columns]
        if missing_feats:
            logger.error(f"Missing features in dataset: {missing_feats}")
            raise ValueError(f"Missing features in dataset: {missing_feats}")
        
        # Filter out just those columns
        df_selected = df_merged[expected_features].copy()
        
        # Standardize features
        scaler = StandardScaler()
        features_no_target = [f for f in expected_features if f != "workload_intensity"]
        df_selected[features_no_target] = scaler.fit_transform(df_selected[features_no_target])
        
        # Label Encoding for Workload Intensity (target variable),
        # converting it to "Low", "Medium", "High" bins
        df_selected["cognitive_state"] = pd.qcut(
            df_selected["workload_intensity"], q=3, labels=["Low", "Medium", "High"]
        )
        
        logger.info(f"Preprocessing complete. Final dataset shape: {df_selected.shape}")
        
        # Return (df, scaler, list_of_feature_cols_for_training)
        training_cols = [col for col in df_selected.columns if col not in ["workload_intensity", "cognitive_state"]]
        return df_selected, scaler, training_cols
    
    except Exception as e:
        logger.exception(f"Error during preprocessing: {str(e)}")
        raise


# ---------------------- MODEL TRAINING ---------------------- #
def train_model(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[Union[RandomForestClassifier, Any], float, pd.DataFrame, pd.Series, np.ndarray]:
    """
    Train a Random Forest classifier for cognitive state prediction.
    If TUNE_MODEL is True in the environment, performs a RandomizedSearchCV
    to find better hyperparameters for improved prediction quality.

    Returns:
        (trained_model, accuracy, X_test, y_test, predicted_labels)
    """
    logger.info("Starting model training...")
    try:
        # Prepare training data
        X = df[feature_cols]
        y = df["cognitive_state"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
        )
        
        if not TUNE_MODEL:
            # Quick train
            model = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                random_state=RANDOM_SEED,
                n_jobs=RF_N_JOBS
            )
            logger.info("Fitting Random Forest without hyperparameter tuning...")
            model.fit(X_train, y_train)
        else:
            # Randomized search for hyperparameter tuning
            logger.info("Performing RandomizedSearchCV for hyperparameter tuning...")
            param_dist = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_leaf": [1, 2, 4, 8],
                "bootstrap": [True, False]
            }
            rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=RF_N_JOBS)
            random_search = RandomizedSearchCV(
                rf, param_distributions=param_dist, n_iter=10,
                cv=3, verbose=1, random_state=RANDOM_SEED, n_jobs=RF_N_JOBS
            )
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            logger.info(f"Best hyperparameters found: {random_search.best_params_}")

        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log multiple metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Test Accuracy:    {accuracy:.3f}")
        logger.info(f"Test F1-score:    {f1:.3f}")
        logger.info(f"Test Precision:   {precision:.3f}")
        logger.info(f"Test Recall:      {recall:.3f}")
        
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Plot confusion matrix and feature importance
        cm_path = os.path.join(MODEL_OUTPUT_DIR, f"confusion_matrix_{MODEL_VERSION}.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        
        fi_path = os.path.join(MODEL_OUTPUT_DIR, f"feature_importance_{MODEL_VERSION}.png")
        plot_feature_importance(model, feature_cols, fi_path)
        
        # Save model
        joblib.dump(model, MODEL_OUTPUT_PATH)
        logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
        
        # Save metadata
        save_model_metadata(model, feature_cols, accuracy, class_report_dict)
        
        return model, accuracy, X_test, y_test, y_pred
    
    except Exception as e:
        logger.exception(f"Error during model training: {str(e)}")
        raise


# ---------------------- PREDICTION FUNCTION ---------------------- #
def validate_input_features(input_data: Dict[str, Any], required_features: List[str]) -> None:
    """
    Ensure that input data contains all required features.
    """
    missing = [feat for feat in required_features if feat not in input_data]
    if missing:
        error_msg = f"Missing input features: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def predict_new_data(
    model_path: str, scaler_path: str, new_data: Dict[str, Any]
) -> Tuple[str, Dict[str, float]]:
    """
    Make predictions on new data using the trained model.

    Returns:
        (predicted_cognitive_state, class_probabilities_dict)
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        required_features = (
            list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_") else list(new_data.keys())
        )
        validate_input_features(new_data, required_features)
        
        new_data_df = pd.DataFrame([new_data])
        new_data_scaled = scaler.transform(new_data_df[required_features])
        
        prediction = model.predict(new_data_scaled)
        prediction_proba = model.predict_proba(new_data_scaled)
        
        proba_dict = {
            cls: float(prob)
            for cls, prob in zip(model.classes_, prediction_proba[0])
        }
        
        logger.info(f"Prediction: {prediction[0]}")
        logger.debug(f"Class probabilities: {proba_dict}")
        return prediction[0], proba_dict
    
    except Exception as e:
        logger.exception(f"Error during prediction: {str(e)}")
        raise


# ---------------------- COMMAND LINE INTERFACE ---------------------- #
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Cognitive Workload Assessment Tool'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    subparsers.add_parser('train', help='Train a new model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--input', '-i', type=str, required=True, help='JSON file with input data')
    predict_parser.add_argument('--model', '-m', type=str, help='Path to model file')
    predict_parser.add_argument('--scaler', '-s', type=str, help='Path to scaler file')
    
    # List models command
    subparsers.add_parser('list-models', help='List available trained models')
    
    # Set default action when no command is provided
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit(1, "Error: No command specified. Use 'train', 'predict', or 'list-models'.\n")
    
    return args


def find_latest_model() -> Tuple[str, str]:
    """
    Find the latest trained model in the models directory, with robust version parsing.
    """
    models_path = Path(MODEL_OUTPUT_DIR)
    model_files = list(models_path.glob(f"{MODEL_NAME}_*.joblib"))

    if not model_files:
        logger.error("No trained models found")
        return None, None

    # Use max() with st_mtime for the newest model by modification time
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Example: Cognitive_State_Prediction_Model_20231012_103000.joblib
    parts = latest_model.stem.split('_')
    if len(parts) < 3:
        logger.warning(f"Unexpected model naming convention for {latest_model.name}. Using fallback timestamp.")
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        # Join everything after the first two parts as the version
        model_version = '_'.join(parts[2:])
    
    scaler_file = models_path / f"scaler_{model_version}.joblib"
    if not scaler_file.exists():
        logger.error(f"Scaler file not found: {scaler_file}")
        return None, None

    return str(latest_model), str(scaler_file)


def list_available_models() -> None:
    """
    List all available trained models in the models directory.
    """
    models_path = Path(MODEL_OUTPUT_DIR)
    model_files = list(models_path.glob(f"{MODEL_NAME}_*.joblib"))
    
    if not model_files:
        logger.info("No trained models found")
        return
    
    print("\nAvailable trained models:")
    print("-" * 80)
    print(f"{'Model Name':<50} {'Modified Date':<22} {'Size (KB)':<10}")
    print("-" * 80)
    
    for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        mod_time = datetime.fromtimestamp(model_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = model_file.stat().st_size / 1024
        print(f"{model_file.name:<50} {mod_time:<22} {size_kb:.2f}")


def main() -> None:
    """
    Main entry point for the application.
    """
    args = parse_args()
    
    if args.command == 'train':
        logger.info("Starting full model training pipeline.")
        try:
            # Load and preprocess data
            df_physio, df_eeg, df_gaze = load_data()
            df_processed, scaler, feature_cols = preprocess_data(df_physio, df_eeg, df_gaze)
            
            # Save scaler for later use
            joblib.dump(scaler, SCALER_OUTPUT_PATH)
            logger.info(f"Scaler saved to {SCALER_OUTPUT_PATH}")
            
            # Train the model
            model, accuracy, X_test, y_test, y_pred = train_model(df_processed, feature_cols)
            
            logger.info(f"Model training complete. Accuracy: {accuracy:.3f}")
            logger.info(f"Model saved at {MODEL_OUTPUT_PATH}")
            logger.info(f"Scaler saved at {SCALER_OUTPUT_PATH}")
        except Exception as e:
            logger.exception(f"Training pipeline failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'predict':
        model_path = args.model
        scaler_path = args.scaler
        
        # If model or scaler paths are not provided, try to find the latest ones
        if not model_path or not scaler_path:
            model_path, scaler_path = find_latest_model()
            if not model_path or not scaler_path:
                logger.error("No trained models found and no model path provided.")
                sys.exit(1)
            logger.info(f"Using latest model: {model_path}")
            logger.info(f"Using latest scaler: {scaler_path}")
        
        try:
            # Load input JSON file
            with open(args.input, 'r') as f:
                input_data = json.load(f)
            
            prediction, probabilities = predict_new_data(model_path, scaler_path, input_data)
            
            # Output prediction result
            print("\nPrediction Results:")
            print("-" * 50)
            print(f"Predicted Cognitive State: {prediction}")
            print("\nClass Probabilities:")
            for cls, prob in probabilities.items():
                print(f"  {cls}: {prob:.4f}")
            
        except Exception as e:
            logger.exception(f"Prediction failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'list-models':
        list_available_models()
    
    else:
        logger.error(f"Unknown command: {args.command}")
        logger.error("Use 'train', 'predict', or 'list-models'.")
        sys.exit(1)


if __name__ == "__main__":
    main()