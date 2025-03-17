#!/usr/bin/env python3
"""
Cognitive Workload Assessment and Prediction Tool

This module implements a machine learning pipeline for predicting cognitive workload
based on physiological, EEG, and gaze data. The tool preprocesses multi-modal data,
trains a Random Forest classifier, and provides prediction functionality.

Usage:
    python cwt.py train
    python cwt.py predict [input_data_path]
"""

import os
import sys
import json
import logging
from datetime import datetime
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables if .env file exists
if os.path.exists('.env'):
    load_dotenv()

# ---------------------- CONFIGURATION ---------------------- #
# Set up logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/cwt.log')

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('cwt')

# Data paths
DATA_FILES = {
    "physiological": os.getenv('PHYSIO_DATA_PATH', 'data/Enhanced_Workload_Clinical_Data.csv'),
    "eeg": os.getenv('EEG_DATA_PATH', 'data/000_EEG_Cluster_ANOVA_Results.csv'),
    "gaze": os.getenv('GAZE_DATA_PATH', 'data/008_01.csv')
}

# Model configuration
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', 'models')
MODEL_NAME = os.getenv('MODEL_NAME', 'Cognitive_State_Prediction_Model')
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.joblib")
SCALER_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"scaler_{MODEL_VERSION}.joblib")
METADATA_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"metadata_{MODEL_VERSION}.json")

# Training parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# Ensure model directory exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ---------------------- HELPER FUNCTIONS ---------------------- #
def validate_file_exists(file_path):
    """Validate that a file exists on disk."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def safe_read_csv(file_path):
    """Safely read a CSV file with error handling."""
    try:
        validate_file_exists(file_path)
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def save_model_metadata(model, features, accuracy, class_report):
    """Save metadata about the model for reproducibility."""
    metadata = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "classification_report": class_report,
        "features": features,
        "parameters": {
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED
        }
    }
    
    with open(METADATA_OUTPUT_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to {METADATA_OUTPUT_PATH}")

def plot_feature_importance(model, feature_names, output_path=None):
    """Plot feature importance from the trained model."""
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
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """Plot confusion matrix from predictions."""
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
    else:
        plt.show()

# ---------------------- LOAD DATA ---------------------- #
def load_data():
    """
    Load the physiological, EEG, and gaze data from CSV files.
    
    Returns:
        tuple: Tuple containing dataframes (df_physio, df_eeg, df_gaze)
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
        logger.error(f"Error loading data: {str(e)}")
        raise

# ---------------------- PREPROCESSING ---------------------- #
def preprocess_data(df_physio, df_eeg, df_gaze):
    """
    Preprocess and merge the physiological, EEG, and gaze data.
    
    Args:
        df_physio (DataFrame): Physiological data
        df_eeg (DataFrame): EEG data
        df_gaze (DataFrame): Gaze tracking data
        
    Returns:
        tuple: Processed dataframe and fitted scaler (df, scaler)
    """
    logger.info("Preprocessing data...")
    
    try:
        # Convert timestamps to datetime
        logger.debug("Converting timestamps to datetime format")
        for df, name in [(df_physio, "physiological"), (df_eeg, "eeg"), (df_gaze, "gaze")]:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except KeyError:
                logger.error(f"'timestamp' column not found in {name} data")
                raise
            except Exception as e:
                logger.error(f"Error converting timestamps in {name} data: {str(e)}")
                raise
        
        # Merge based on nearest timestamp
        logger.debug("Merging datasets based on timestamp")
        try:
            df = pd.merge_asof(df_physio.sort_values("timestamp"), 
                              df_eeg.sort_values("timestamp"), 
                              on="timestamp")
            df = pd.merge_asof(df.sort_values("timestamp"), 
                              df_gaze.sort_values("timestamp"), 
                              on="timestamp")
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected after merge: \n{missing_values[missing_values > 0]}")
            logger.info("Dropping rows with missing values")
            df = df.dropna()
        
        # Select relevant features
        features = ["pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
                   "pupil_diameter_right", "fixation_duration", "blink_rate", "workload_intensity",
                   "gaze_x", "gaze_y", "alpha_power", "theta_power"]
        
        # Validate features exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features in dataset: {missing_features}")
            raise ValueError(f"Missing features in dataset: {missing_features}")
        
        logger.debug(f"Selected {len(features)} features for analysis")
        df = df[features]
        
        # Standardization
        logger.debug("Standardizing features")
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        
        # Label Encoding for Workload Intensity (Target Variable)
        logger.debug("Encoding cognitive state labels")
        df["cognitive_state"] = pd.qcut(df["workload_intensity"], q=3, labels=["Low", "Medium", "High"])
        
        logger.info(f"Preprocessing complete. Final dataset shape: {df.shape}")
        return df, scaler, features
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

# ---------------------- MODEL TRAINING ---------------------- #
def train_model(df, features):
    """
    Train a Random Forest classifier for cognitive state prediction.
    
    Args:
        df (DataFrame): Preprocessed data with features and target
        features (list): List of feature names
        
    Returns:
        object: Trained RandomForestClassifier model
    """
    logger.info("Training model...")
    
    try:
        X = df.drop(columns=["cognitive_state", "workload_intensity"])
        y = df["cognitive_state"]
        
        # Split data
        logger.debug(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_SEED}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
        
        # Train model
        logger.debug("Training Random Forest Classifier")
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        
        # Perform cross-validation
        logger.debug("Performing 5-fold cross-validation")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Fit the model on the full training data
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model Accuracy: {accuracy:.3f}")
        logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")
        
        # Plot and save confusion matrix
        logger.debug("Generating confusion matrix")
        cm_path = os.path.join(MODEL_OUTPUT_DIR, f"confusion_matrix_{MODEL_VERSION}.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        
        # Plot and save feature importance
        logger.debug("Generating feature importance plot")
        fi_path = os.path.join(MODEL_OUTPUT_DIR, f"feature_importance_{MODEL_VERSION}.png")
        plot_feature_importance(model, X.columns, fi_path)
        
        # Save model, scaler, and metadata
        logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
        joblib.dump(model, MODEL_OUTPUT_PATH)
        
        # Save metadata
        save_model_metadata(model, list(X.columns), accuracy, class_report)
        
        return model, accuracy, X_test, y_test, y_pred
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

# ---------------------- PREDICTION FUNCTION ---------------------- #
def predict_new_data(model_path, scaler_path, new_data):
    """
    Make predictions on new data using the trained model.
    
    Args:
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        new_data (dict): Dictionary containing feature values
        
    Returns:
        str: Predicted cognitive state
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # Convert input to DataFrame
        logger.debug("Preparing input data for prediction")
        new_data_df = pd.DataFrame([new_data])
        
        # Scale the input data
        new_data_scaled = scaler.transform(new_data_df)
        
        # Make prediction
        logger.debug("Making prediction")
        prediction = model.predict(new_data_scaled)
        prediction_proba = model.predict_proba(new_data_scaled)
        
        # Get class probabilities
        classes = model.classes_
        proba_dict = {cls: float(prob) for cls, prob in zip(classes, prediction_proba[0])}
        
        logger.info(f"Prediction: {prediction[0]}")
        logger.debug(f"Class probabilities: {proba_dict}")
        
        return prediction[0], proba_dict
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

# ---------------------- COMMAND LINE INTERFACE ---------------------- #
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cognitive Workload Assessment Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--input', '-i', type=str, help='JSON file with input data')
    predict_parser.add_argument('--model', '-m', type=str, help='Path to model file')
    predict_parser.add_argument('--scaler', '-s', type=str, help='Path to scaler file')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available trained models')
    
    return parser.parse_args()

def find_latest_model():
    """Find the latest trained model in the models directory."""
    models_path = Path(MODEL_OUTPUT_DIR)
    model_files = list(models_path.glob(f"{MODEL_NAME}_*.joblib"))
    
    if not model_files:
        logger.error("No trained models found")
        return None, None
    
    # Sort by modification time (newest first)
    latest_model = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    model_version = latest_model.stem.split('_')[-2] + '_' + latest_model.stem.split('_')[-1]
    scaler_file = models_path / f"scaler_{model_version}.joblib"
    
    if not scaler_file.exists():
        logger.error(f"Scaler file not found for model {latest_model}")
        return None, None
    
    return str(latest_model), str(scaler_file)

def list_available_models():
    """List all available trained models."""
    models_path = Path(MODEL_OUTPUT_DIR)
    model_files = list(models_path.glob(f"{MODEL_NAME}_*.joblib"))
    
    if not model_files:
        logger.info("No trained models found")
        return
    
    print("\nAvailable trained models:")
    print("-" * 80)
    print(f"{'Model Name':<50} {'Creation Date':<20} {'Size (KB)':<10}")
    print("-" * 80)
    
    for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        creation_time = datetime.fromtimestamp(model_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = model_file.stat().st_size / 1024
        print(f"{model_file.name:<50} {creation_time:<20} {size_kb:.2f}")

# ---------------------- MAIN EXECUTION ---------------------- #
def main():
    """Main entry point for the application."""
    args = parse_args()
    
    if args.command == 'train':
        logger.info("Starting model training pipeline")
        try:
            # Load data
            df_physio, df_eeg, df_gaze = load_data()
            
            # Preprocess data
            df_processed, scaler, features = preprocess_data(df_physio, df_eeg, df_gaze)
            
            # Save scaler
            joblib.dump(scaler, SCALER_OUTPUT_PATH)
            logger.info(f"Scaler saved to {SCALER_OUTPUT_PATH}")
            
            # Train model
            model, accuracy, X_test, y_test, y_pred = train_model(df_processed, features)
            
            logger.info(f"Model training complete. Accuracy: {accuracy:.3f}")
            logger.info(f"Model saved at {MODEL_OUTPUT_PATH}")
            logger.info(f"Scaler saved at {SCALER_OUTPUT_PATH}")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'predict':
        if not args.input:
            logger.error("Input file is required for prediction")
            sys.exit(1)
        
        model_path = args.model
        scaler_path = args.scaler
        
        # If model path not provided, find the latest model
        if not model_path or not scaler_path:
            model_path, scaler_path = find_latest_model()
            if not model_path or not scaler_path:
                logger.error("No trained models found and no model path provided")
                sys.exit(1)
            logger.info(f"Using latest model: {model_path}")
            logger.info(f"Using latest scaler: {scaler_path}")
        
        try:
            # Load input data
            with open(args.input, 'r') as f:
                input_data = json.load(f)
            
            # Make prediction
            prediction, probabilities = predict_new_data(model_path, scaler_path, input_data)
            
            # Output prediction
            print("\nPrediction Results:")
            print("-" * 50)
            print(f"Predicted Cognitive State: {prediction}")
            print("\nClass Probabilities:")
            for cls, prob in probabilities.items():
                print(f"  {cls}: {prob:.4f}")
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'list-models':
        list_available_models()
    
    else:
        logger.error("No command specified. Use 'train', 'predict', or 'list-models'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
