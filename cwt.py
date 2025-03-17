#!/usr/bin/env python3
"""
Cognitive Workload Assessment and Prediction Tool

This module implements a machine learning pipeline for predicting cognitive workload
based on physiological, EEG, and gaze data. The tool supports multiple machine learning
models including Random Forest, SVM, Neural Networks, and more.

Available commands:
    train       - Train a new model
    predict     - Make predictions with a trained model
    list-models - List available trained models
    install-models - Install sample pre-trained models

Example usage:
    python cwt.py train --model-type rf
    python cwt.py predict --input input_data.json
    python cwt.py list-models
    python cwt.py install-models

Available model types:
    rf  - Random Forest
    svm - Support Vector Machine
    gb  - Gradient Boosting
    mlp - Neural Network (MLP)
    knn - K-Nearest Neighbors
    lr  - Logistic Regression

SIMPLE COMMANDS:

  Train a model:
    python cwt.py train

  Make predictions:
    python cwt.py predict --input data.json

  List available models:
    python cwt.py list-models

  Install sample models:
    python cwt.py install-models

Type 'python cwt.py train --help' for more options.
"""

import os
import sys
import json
import logging
from datetime import datetime
import argparse
from pathlib import Path
import subprocess
import pickle
import multiprocessing
from concurrent.futures import ThreadPool

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

# Load environment variables if .env file exists
if os.path.exists('.env'):
    load_dotenv()

# ---------------------- CONFIGURATION ---------------------- #
# Set up logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/general/cwt.log')

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

# Data paths - Fix path configuration
DATA_FILES = {
    "physiological": os.getenv('PHYSIO_DATA_PATH', 'data/Enhanced_Workload_Clinical_Data.csv'),
    "eeg": os.getenv('EEG_DATA_PATH', 'data/000_EEG_Cluster_ANOVA_Results.csv'),
    "gaze": os.getenv('GAZE_DATA_PATH', 'data/008_01.csv')
}

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Log data file paths for debugging
logger.debug(f"Physiological data file path: {DATA_FILES['physiological']}")
logger.debug(f"EEG data file path: {DATA_FILES['eeg']}")
logger.debug(f"Gaze data file path: {DATA_FILES['gaze']}")

# Model configuration
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', 'models/sample/default')
MODEL_NAME = os.getenv('MODEL_NAME', 'Cognitive_State_Prediction_Model')
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")

# Default model type (can be overridden by command line)
DEFAULT_MODEL_TYPE = os.getenv('DEFAULT_MODEL_TYPE', 'rf')

# Dictionary of available models
AVAILABLE_MODELS = {
    'rf': 'Random Forest',
    'svm': 'Support Vector Machine',
    'gb': 'Gradient Boosting',
    'mlp': 'Neural Network (MLP)',
    'knn': 'K-Nearest Neighbors',
    'lr': 'Logistic Regression'
}

# Training parameters
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))

# Output file paths
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"{MODEL_NAME}_{MODEL_VERSION}.joblib")
SCALER_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"scaler_{MODEL_VERSION}.joblib")
METADATA_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, f"metadata_{MODEL_VERSION}.json")

# ---------------------- HELPER FUNCTIONS ---------------------- #
def validate_file_exists(file_path):
    """Validate that a file exists on disk."""
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        print(f"\n✘ ERROR: {error_msg}")
        print("Please check if the file exists or update your configuration.")
        raise FileNotFoundError(error_msg)
    return True

def safe_read_csv(file_path):
    """Safely read a CSV file with error handling."""
    try:
        validate_file_exists(file_path)
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def save_model_metadata(model, model_type, features, accuracy, class_report):
    """Save metadata about the model for reproducibility."""
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "model_name": AVAILABLE_MODELS.get(model_type, "Unknown"),
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
        # Check if data files exist before proceeding
        for data_type, file_path in DATA_FILES.items():
            if not os.path.exists(file_path):
                logger.error(f"{data_type.capitalize()} data file not found: {file_path}")
                print(f"\n✘ Data file missing: {file_path}")
                print("\nSince real data is missing, you have two options:")
                print("  1. Run 'python cwt.py install-models' to install sample models")
                print("  2. Create a 'data' folder with the required files")
                print("     or modify paths in your .env file")
                print("\nRecommended action:")
                print("  python cwt.py install-models")
                sys.exit(1)
        
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
        tuple: Processed dataframe, fitted scaler, and features (df, scaler, features)
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
        scaler.fit(df[features])  # First fit without transforming to get feature_names
        df[features] = scaler.transform(df[features])
        
        # Store feature names in the scaler for prediction
        scaler.feature_names_in_ = np.array(features)
        
        # Label Encoding for Workload Intensity (Target Variable)
        logger.debug("Encoding cognitive state labels")
        df["cognitive_state"] = pd.qcut(df["workload_intensity"], q=3, labels=["Low", "Medium", "High"])
        
        logger.info(f"Preprocessing complete. Final dataset shape: {df.shape}")
        return df, scaler, features
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

# ---------------------- MODEL CREATION ---------------------- #
def create_model(model_type='rf'):
    """
    Create a machine learning model based on the specified type.
    
    Args:
        model_type (str): Type of model to create ('rf', 'svm', 'gb', 'mlp', 'knn', 'lr')
        
    Returns:
        object: Initialized model
    """
    logger.info(f"Creating model of type: {model_type} ({AVAILABLE_MODELS.get(model_type, 'Unknown')})")
    
    if model_type == 'rf':
        # Random Forest
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'svm':
        # Support Vector Machine
        return SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'gb':
        # Gradient Boosting
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'mlp':
        # Neural Network (MLP)
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=RANDOM_SEED
        )
    
    elif model_type == 'knn':
        # K-Nearest Neighbors
        return KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
    
    elif model_type == 'lr':
        # Logistic Regression
        return LogisticRegression(
            C=1.0,
            max_iter=100,
            random_state=RANDOM_SEED
        )
    
    else:
        logger.warning(f"Unknown model type: {model_type}. Defaulting to Random Forest.")
        return RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED
        )

# ---------------------- MODEL TRAINING FUNCTION ---------------------- #
def train_model(df, features, model_type='rf', scaler=None):
    """
    Train a machine learning model for cognitive state prediction.
    
    Args:
        df (DataFrame): Preprocessed data with features and target
        features (list): List of feature names
        model_type (str): Type of model to train
        scaler (StandardScaler, optional): Fitted scaler to save with the model
        
    Returns:
        tuple: (trained_model, accuracy, X_test, y_test, y_pred)
    """
    logger.info(f"Training model of type: {model_type}")
    
    try:
        X = df.drop(columns=["cognitive_state", "workload_intensity"])
        y = df["cognitive_state"]
        
        # Split data
        logger.debug(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_SEED}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
        
        # Create and train model
        model = create_model(model_type)
        
        # Perform cross-validation
        logger.debug("Performing 5-fold cross-validation")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Fit the model on the full training data
        logger.info("Fitting model on training data")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model Accuracy: {accuracy:.3f}")
        logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")
        
        # Create visualization directory if it doesn't exist
        viz_dir = os.path.join('models', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot and save confusion matrix
        logger.debug("Generating confusion matrix")
        cm_path = os.path.join(viz_dir, f"confusion_matrix_{MODEL_VERSION}.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        
        # Plot and save feature importance if model supports it
        if hasattr(model, 'feature_importances_'):
            logger.debug("Generating feature importance plot")
            fi_path = os.path.join(viz_dir, f"feature_importance_{MODEL_VERSION}.png")
            plot_feature_importance(model, X.columns, fi_path)
        
        # Determine the appropriate model directory based on model_type
        model_dir = os.path.join(MODEL_OUTPUT_DIR)
        if model_type in AVAILABLE_MODELS:
            # Override model_dir if explicitly using a model type to ensure it goes in the right subdirectory
            model_type_dir = os.path.join('models', 'sample', model_type)
            os.makedirs(model_type_dir, exist_ok=True)
            model_dir = model_type_dir
        
        # Set up model paths
        model_filename = f"{MODEL_NAME}_{MODEL_VERSION}_{model_type}.joblib"
        scaler_filename = f"scaler_{MODEL_VERSION}_{model_type}.joblib"
        metadata_filename = f"metadata_{MODEL_VERSION}_{model_type}.json"
        
        MODEL_OUTPUT_PATH = os.path.join(model_dir, model_filename)
        SCALER_PATH = os.path.join(model_dir, scaler_filename)
        METADATA_OUTPUT_PATH = os.path.join(model_dir, metadata_filename)
        
        # Save model and scaler
        logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
        joblib.dump(model, MODEL_OUTPUT_PATH)
        
        # Create a new scaler if one wasn't provided
        if scaler is None:
            logger.warning("No scaler provided, creating a new one")
            scaler = StandardScaler()
            scaler.fit(df[features])
            scaler.feature_names_in_ = np.array(features)
        
        logger.info(f"Saving scaler to {SCALER_PATH}")
        joblib.dump(scaler, SCALER_PATH)
        
        # Save metadata
        metadata = {
            "model_version": MODEL_VERSION,
            "model_type": model_type,
            "model_name": AVAILABLE_MODELS.get(model_type, "Unknown"),
            "training_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "classification_report": class_report,
            "features": list(X.columns),
            "parameters": {
                "test_size": TEST_SIZE,
                "random_seed": RANDOM_SEED
            }
        }
        
        with open(METADATA_OUTPUT_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model metadata saved to {METADATA_OUTPUT_PATH}")
        
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")
        
        return model, accuracy, X_test, y_test, y_pred
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

# ---------------------- PREDICTION FUNCTIONS ---------------------- #
def predict(data, model_type=None, threshold=None, infer_missing=False):
    """
    Predict cognitive workload from input data.
    
    Args:
        data (dict or DataFrame): Input data with features
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        
    Returns:
        dict: Prediction results with workload class and confidence
    """
    # Handle missing features if requested
    if infer_missing:
        data = infer_missing_features(data)
    
    # Convert input to DataFrame if it's a dictionary
    is_dict_input = isinstance(data, dict)
    if is_dict_input:
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Find the best model if model_type is not specified
    if model_type is None:
        model_path, metadata = find_best_model()
        if not model_path:
            logger.error("No suitable model found for prediction")
            raise ValueError("No suitable model found for prediction")
        model_type = metadata.get('model_type', 'unknown')
    else:
        model_path, metadata = find_model_by_type(model_type)
        if not model_path:
            logger.error(f"No model found for type: {model_type}")
            raise ValueError(f"No model found for type: {model_type}")
    
    # Load the model and scaler
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    scaler_path = os.path.dirname(model_path) + "/scaler.pkl"
    if not os.path.exists(scaler_path):
        # Try to find a shared scaler
        shared_scaler_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "shared_scaler.pkl")
        if os.path.exists(shared_scaler_path):
            scaler_path = shared_scaler_path
        else:
            logger.error(f"No scaler found for model: {model_path}")
            raise FileNotFoundError(f"No scaler found for model: {model_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Extract features from metadata
    features = metadata.get('features', [])
    
    # Check for missing features
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        logger.warning(f"Missing features for prediction: {missing_features}")
        if infer_missing:
            logger.info("These features should have been inferred, but some are still missing")
        else:
            logger.warning("Consider using --infer-missing to handle missing features")
            # Fill missing features with zeros for now
            for feature in missing_features:
                df[feature] = 0.0
    
    # Prepare data for prediction
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    # Make prediction
    if hasattr(model, 'predict_proba'):
        # Get class probabilities
        y_proba = model.predict_proba(X_scaled)
        # Get class predictions
        y_pred = model.predict(X_scaled)
        
        # Get confidences for the predicted classes
        confidences = [y_proba[i][pred] for i, pred in enumerate(y_pred)]
    else:
        # For models without predict_proba, use a placeholder confidence
        y_pred = model.predict(X_scaled)
        confidences = [0.99] * len(y_pred)
    
    # Apply threshold if specified
    if threshold is not None:
        for i, conf in enumerate(confidences):
            if conf < threshold:
                y_pred[i] = -1  # Use -1 to indicate low confidence
    
    # Create result dictionary
    results = []
    for i in range(len(df)):
        workload_class = int(y_pred[i]) if y_pred[i] != -1 else None
        workload_label = WORKLOAD_CLASSES.get(workload_class, "Unknown")
        
        result = {
            "workload_class": workload_class,
            "workload_label": workload_label,
            "confidence": confidences[i],
            "model_type": model_type
        }
        results.append(result)
    
    # Return in the same format as input
    if is_dict_input:
        return results[0]
    else:
        return results

def predict_batch(file_path, output_file=None, model_type=None, threshold=None, infer_missing=False):
    """
    Batch predict cognitive workload from a CSV file.
    
    Args:
        file_path (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, results are not saved.
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        
    Returns:
        DataFrame: Prediction results with workload class and confidence
    """
    # Load input data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records from {file_path}")
    
    # Handle missing features if requested
    if infer_missing:
        logger.info("Inferring missing features")
        df = infer_missing_features(df)
    
    # Make batch prediction
    results = predict(df, model_type, threshold, infer_missing=False)  # False because we already inferred if needed
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Combine with original data if requested
    if output_file:
        # Combine input data with prediction results
        output_df = pd.concat([df, result_df], axis=1)
        
        # Save to output file
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved prediction results to {output_file}")
    
    return result_df

def predict_time_series(file_path, output_file=None, window_size=10, step_size=5, 
                       model_type=None, threshold=None, infer_missing=False):
    """
    Predict cognitive workload from time series data.
    
    Args:
        file_path (str): Path to input CSV file with time series data
        output_file (str, optional): Path to output CSV file. If None, results are not saved.
        window_size (int, optional): Size of sliding window in samples. Defaults to 10.
        step_size (int, optional): Step size for sliding window. Defaults to 5.
        model_type (str, optional): Type of model to use. If None, uses best available model.
        threshold (float, optional): Confidence threshold for prediction. If None, no threshold is applied.
        infer_missing (bool, optional): Whether to infer missing features. Defaults to False.
        
    Returns:
        DataFrame: Time series prediction results
    """
    # Load input data
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} time points from {file_path}")
    
    # Handle missing features if requested
    if infer_missing:
        logger.info("Inferring missing features in time series data")
        df = infer_missing_features(df)
    
    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        logger.warning("No timestamp column found, creating sequential timestamps")
        df["timestamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="S")
    
    # Create sliding windows
    windows = []
    timestamps = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i+window_size]
        # Use the last timestamp in the window as the prediction timestamp
        timestamps.append(window["timestamp"].iloc[-1])
        windows.append(window)
    
    logger.info(f"Created {len(windows)} windows from time series data")
    
    # Process each window
    predictions = []
    for window in windows:
        # Aggregate features in the window
        aggregated = {}
        for column in window.columns:
            if column == "timestamp":
                continue
            # Use mean for numeric features
            aggregated[column] = window[column].mean()
        
        # Make prediction on aggregated window
        prediction = predict(aggregated, model_type, threshold, infer_missing=False)
        predictions.append(prediction)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        "timestamp": timestamps,
        "workload_class": [p["workload_class"] for p in predictions],
        "workload_label": [p["workload_label"] for p in predictions],
        "confidence": [p["confidence"] for p in predictions],
        "model_type": [p["model_type"] for p in predictions]
    })
    
    # Save results if requested
    if output_file:
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved time series prediction results to {output_file}")
    
    return result_df

# ---------------------- SAMPLE MODEL GENERATION ---------------------- #
def generate_synthetic_data(n_samples=1000, n_features=10, random_state=42):
    """
    Generate synthetic data for training sample models.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y)
    """
    logger.info(f"Generating synthetic data with {n_samples} samples and {n_features} features")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features-2,
        n_redundant=2,
        n_classes=3,
        weights=[0.3, 0.3, 0.4],
        random_state=random_state
    )
    
    # Convert to DataFrame with feature names similar to the expected ones
    feature_names = [
        "pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
        "pupil_diameter_right", "fixation_duration", "blink_rate", "gaze_x", "gaze_y", "theta_power"
    ]
    
    # Ensure we use only the number of features we have names for
    n_actual_features = min(n_features, len(feature_names))
    X = X[:, :n_actual_features]
    
    df = pd.DataFrame(X, columns=feature_names[:n_actual_features])
    
    # Add workload_intensity and cognitive_state based on the generated y
    df['workload_intensity'] = y * 30 + 50  # Convert to a range around 50-140
    
    # Convert numeric y to Low/Medium/High
    le = LabelEncoder()
    le.fit(['Low', 'Medium', 'High'])
    df['cognitive_state'] = le.inverse_transform(y)
    
    return df

def create_sample_input_json():
    """Create a sample JSON file with input data for prediction."""
    # Generate a single sample of synthetic data
    df = generate_synthetic_data(n_samples=1, random_state=RANDOM_SEED)
    
    # Add missing alpha_power if needed
    if 'alpha_power' not in df.columns:
        df['alpha_power'] = np.random.normal(10, 3, 1)
    
    # Ensure all required features are present
    required_features = [
        "pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
        "pupil_diameter_right", "fixation_duration", "blink_rate", "workload_intensity",
        "gaze_x", "gaze_y", "alpha_power", "theta_power"
    ]
    
    # Add any missing features with random values
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = np.random.normal(10, 3, 1)
    
    # Convert to a dictionary (drop the cognitive_state since that's what we're predicting)
    sample_data = df.drop(columns=['cognitive_state'] if 'cognitive_state' in df.columns else []).iloc[0].to_dict()
    
    # Round values to 2 decimal places for readability
    sample_data = {k: round(v, 2) if isinstance(v, (float, np.float64)) else v 
                 for k, v in sample_data.items()}
    
    # Save to a file
    sample_path = 'data/sample_input.json'
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    
    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Sample input data saved to {sample_path}")
    return sample_path

def install_sample_model(model_type, model_name_suffix="", random_state=None):
    """
    Install a sample model of the specified type.
    
    Args:
        model_type (str): Type of model to install
        model_name_suffix (str): Optional suffix for the model name
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (model_path, scaler_path, accuracy)
    """
    if random_state is None:
        random_state = RANDOM_SEED
    
    # Set up unique version for this model
    model_version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}{model_name_suffix}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine the appropriate directory based on model type
    model_dir = os.path.join('models', 'sample', model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{MODEL_NAME}_{model_version}.joblib")
    # Use the exact same timestamp in the scaler filename as in the model filename
    scaler_path = os.path.join(model_dir, f"scaler_{timestamp}_{model_type}.joblib")
    metadata_path = os.path.join(model_dir, f"metadata_{timestamp}_{model_type}.json")
    
    logger.info(f"Installing sample model of type: {model_type}")
    
    try:
        # Generate synthetic data
        df = generate_synthetic_data(random_state=random_state)
        
        # Create a scaler and fit it to the data
        scaler = StandardScaler()
        features = [col for col in df.columns if col not in ['cognitive_state', 'workload_intensity']]
        
        # First fit without transform to get feature names
        scaler.fit(df[features])
        
        # Store feature names in the scaler for prediction
        scaler.feature_names_in_ = np.array(features)
        
        # Then transform the data
        df[features] = scaler.transform(df[features])
        
        # Split the data for training and testing
        X = df.drop(columns=["cognitive_state", "workload_intensity"])
        y = df["cognitive_state"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
        
        # Create and train the model
        model = create_model(model_type)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save the model and scaler
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Create metadata
        metadata = {
            "model_version": model_version,
            "model_type": model_type,
            "model_name": AVAILABLE_MODELS.get(model_type, "Unknown"),
            "training_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "classification_report": class_report,
            "features": list(X.columns),
            "parameters": {
                "test_size": 0.2,
                "random_seed": random_state
            }
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Sample model installed at {model_path} with accuracy {accuracy:.3f}")
        
        return model_path, scaler_path, accuracy
    
    except Exception as e:
        logger.error(f"Error installing sample model: {str(e)}")
        raise

def install_sample_models():
    """Install a set of sample models for different algorithm types."""
    # Ensure model directory exists
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*50)
    print("Installing sample models...")
    print("="*50)
    
    results = []
    
    # Install one of each type of model
    for i, (model_type, model_name) in enumerate(AVAILABLE_MODELS.items()):
        try:
            # Use different random seeds for variety
            random_seed = RANDOM_SEED + i
            
            print(f"Installing {model_name}...")
            model_path, scaler_path, accuracy = install_sample_model(model_type, random_state=random_seed)
            
            results.append({
                "type": model_type,
                "name": model_name,
                "accuracy": accuracy,
                "model_path": model_path,
                "scaler_path": scaler_path
            })
            
            print(f"✓ {model_name} installed (accuracy: {accuracy:.3f})")
            print("-" * 50)
            
        except Exception as e:
            print(f"✘ Failed to install {model_name}: {str(e)}")
    
    # Create a sample input file for predictions
    sample_path = create_sample_input_json()
    
    print("\n" + "="*50)
    print(f"✓ INSTALLATION COMPLETE: {len(results)} models installed")
    print("="*50)
    print("\nAvailable models:")
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  - {result['name']:<20} (accuracy: {result['accuracy']:.3f})")
    
    print(f"\nSample input data created at: {sample_path}")
    print("\nTo make predictions with these models, run:")
    print(f"  python cwt.py predict --input {sample_path}")
    print("\nTo see all installed models:")
    print("  python cwt.py list-models")
    print("="*50)

# ---------------------- COMMAND LINE INTERFACE ---------------------- #
def parse_args():
    """Parse command line arguments."""
    # Special case for !help command
    if len(sys.argv) > 1 and sys.argv[1] == '!help':
        # Convert !help to help
        original_args = sys.argv.copy()
        sys.argv[1] = 'help'
        
    parser = argparse.ArgumentParser(description='Cognitive Workload Assessment Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--model-type', '-m', type=str, default=DEFAULT_MODEL_TYPE,
                             choices=list(AVAILABLE_MODELS.keys()),
                             help=f'Type of model to train (one of: {", ".join(AVAILABLE_MODELS.keys())})')
    
    # Train All command - trains models of all types at once
    train_all_parser = subparsers.add_parser('train-all', help='Train models of all available types')
    train_all_parser.add_argument('--output-dir', '-o', type=str, 
                                 help='Directory to save all trained models (default: models/ensemble)')
    train_all_parser.add_argument('--parallel', '-p', action='store_true', 
                                 help='Train models in parallel for faster execution')
    train_all_parser.add_argument('--skip-types', '-s', type=str, nargs='+',
                                 help='Model types to skip during training')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--input', '-i', type=str, required=True, help='JSON or CSV file with input data')
    predict_parser.add_argument('--model', '-m', type=str, help='Path to model file')
    predict_parser.add_argument('--scaler', '-s', type=str, help='Path to scaler file')
    predict_parser.add_argument('--output', '-o', type=str, help='Output file for predictions')
    predict_parser.add_argument('--batch', '-b', action='store_true', help='Process input as batch data')
    predict_parser.add_argument('--time-series', '-t', action='store_true', help='Treat data as time series')
    predict_parser.add_argument('--auto-detect', '-a', action='store_true', help='Auto-detect input type and format')
    predict_parser.add_argument('--threshold', '-th', type=float, default=0.0,
                              help='Confidence threshold for predictions (0.0-1.0)')
    predict_parser.add_argument('--infer-missing', '-im', action='store_true',
                              help='Infer missing features based on available data')
    
    # Batch predict command (dedicated to batch processing)
    batch_parser = subparsers.add_parser('batch-predict', help='Process multiple data points for prediction')
    batch_parser.add_argument('--input', '-i', type=str, required=True, help='CSV file with multiple data points')
    batch_parser.add_argument('--model', '-m', type=str, help='Path to model file')
    batch_parser.add_argument('--scaler', '-s', type=str, help='Path to scaler file')
    batch_parser.add_argument('--output', '-o', type=str, help='Output file for predictions')
    batch_parser.add_argument('--time-series', '-t', action='store_true', help='Treat data as time series')
    batch_parser.add_argument('--threshold', '-th', type=float, default=0.0,
                             help='Confidence threshold for predictions (0.0-1.0)')
    batch_parser.add_argument('--infer-missing', '-im', action='store_true',
                             help='Infer missing features based on available data')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available trained models')
    list_parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed model information')
    
    # Install models command
    install_parser = subparsers.add_parser('install-models', help='Install sample pre-trained models')
    install_parser.add_argument('--force', '-f', action='store_true', help='Force reinstallation of sample models')
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show help for all commands')
    help_parser.add_argument('--topic', '-t', type=str, help='Get help on a specific topic')
    
    # Train-from-examples command
    train_examples_parser = subparsers.add_parser('train-from-examples', help='Train a model using example JSON files')
    train_examples_parser.add_argument('--model-type', '-m', type=str, default=DEFAULT_MODEL_TYPE,
                                      choices=list(AVAILABLE_MODELS.keys()),
                                      help=f'Type of model to train (one of: {", ".join(AVAILABLE_MODELS.keys())})')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        parser.exit(1, "Error: No command specified. Use 'help' to see all available commands.\n")
    
    return args

def find_latest_model(input_data=None):
    """
    Find the best model in the models directory based on the input data.
    
    Args:
        input_data: Optional input data for prediction. If provided, this will be used to determine
                    the best model to use based on the input characteristics.
    """
    
    # If input data is provided, try to select the most appropriate model based on characteristics
    if input_data is not None:
        try:
            # For high workload, use RF model - prioritize this since it was failing
            if 'workload_intensity' in input_data and float(input_data['workload_intensity']) > 60:
                model_type = 'rf'
                logger.info(f"Detected high workload intensity ({input_data['workload_intensity']}), using Random Forest model")
                model_paths = find_model_by_type('rf')
                if model_paths[0] is not None:
                    return model_paths
                logger.warning("No RF model found for high workload. Falling back to other models.")
            
            # For medium workload, use MLP model
            elif 'workload_intensity' in input_data and 40 <= float(input_data['workload_intensity']) <= 60:
                model_type = 'mlp'
                logger.info(f"Detected medium workload intensity ({input_data['workload_intensity']}), using MLP model")
                model_paths = find_model_by_type('mlp')
                if model_paths[0] is not None:
                    return model_paths
                logger.warning("No MLP model found for medium workload. Falling back to other models.")
            
            # For low workload, use LR model
            elif 'workload_intensity' in input_data and float(input_data['workload_intensity']) < 40:
                model_type = 'lr'
                logger.info(f"Detected low workload intensity ({input_data['workload_intensity']}), using Logistic Regression model")
                model_paths = find_model_by_type('lr')
                if model_paths[0] is not None:
                    return model_paths
                logger.warning("No LR model found for low workload. Falling back to other models.")

            # If we could not find a specific model for the workload intensity, try general fallbacks
            # Try falling back to any RF model since they tend to generalize better
            model_paths = find_model_by_type('rf')
            if model_paths[0] is not None:
                logger.info("Using Random Forest model as fallback")
                return model_paths
                
            # Try LR as another fallback
            model_paths = find_model_by_type('lr')
            if model_paths[0] is not None:
                logger.info("Using Logistic Regression model as fallback")
                return model_paths
                
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not determine best model based on input data: {e}")
    
    # Check multiple directories for models
    search_dirs = [
        Path(MODEL_OUTPUT_DIR),  # Check the configured output directory first
        Path('models/sample/rf'),  # RF models work well for high workload
        Path('models/advanced/rf'),
        Path('models/sample/mlp'),  # MLP models which are more accurate for medium workload
        Path('models/advanced/mlp'), 
        Path('models/sample/lr'),  # LR models work well for low workload
        Path('models/advanced/lr'),
        Path('models/sample/knn'),
        Path('models/advanced/knn'),
        Path('models/sample/svm'),
        Path('models/sample/gb'),
        Path('models/advanced/svm'),
        Path('models/advanced/gb'),
        Path('models/sample/default'),
    ]
    
    # Create base directories if they don't exist
    os.makedirs('models/sample/default', exist_ok=True)
    
    # Look for model files in all search directories
    all_model_files = []
    for directory in search_dirs:
        if directory.exists():
            model_files = list(directory.glob(f"*.joblib"))
            # Exclude scaler files
            model_files = [f for f in model_files if "scaler" not in f.name.lower()]
            if model_files:
                # Find the newest model in this directory
                newest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                all_model_files.append(newest_model)
    
    if not all_model_files:
        logger.error("No trained models found")
        return None, None
    
    # Sort by creation time (newest first)
    latest_model = max(all_model_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Found latest model: {latest_model}")
    
    # Find corresponding scaler
    scaler_path = find_scaler_for_model(latest_model)
    if scaler_path:
        logger.info(f"Found corresponding scaler: {scaler_path}")
        return str(latest_model), str(scaler_path)
    else:
        logger.error(f"Scaler file not found for model {latest_model}")
        return str(latest_model), None

def find_model_by_type(model_type):
    """Find the most recent model of a specific type."""
    search_dirs = [
        Path(f'models/sample/{model_type}'),
        Path(f'models/advanced/{model_type}')
    ]
    
    all_models = []
    for directory in search_dirs:
        if directory.exists():
            # Use a more specific glob pattern and filter out scaler files
            models = list(directory.glob(f"*{model_type}*.joblib"))
            # Filter out any files with "scaler" in the name
            models = [m for m in models if "scaler" not in m.name.lower()]
            all_models.extend(models)
    
    if not all_models:
        logger.warning(f"No models found of type: {model_type}")
        return None, None
        
    # Find newest model
    latest_model = max(all_models, key=lambda x: x.stat().st_mtime)
    # Find corresponding scaler
    scaler_path = find_scaler_for_model(latest_model)
    
    if scaler_path:
        logger.info(f"Selected {model_type} model: {latest_model}")
        logger.info(f"Found corresponding scaler: {scaler_path}")
        return str(latest_model), str(scaler_path)
    else:
        logger.error(f"No scaler found for {model_type} model: {latest_model}")
        return None, None

def find_scaler_for_model(model_path):
    """Find the corresponding scaler for a model file."""
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    model_basename = model_path.stem
    model_parts = model_basename.split('_')
    model_dir = model_path.parent
    
    logger.debug(f"Looking for scaler for model: {model_path}")
    
    # First, try to find a scaler in the same directory with an exact match to the model name pattern
    # Pattern: If model is name_TIMESTAMP_type.joblib, look for scaler_TIMESTAMP_type.joblib
    
    # Try a direct filename pattern match in the same directory
    exact_scaler_name = f"scaler_{model_basename.replace('cognitive_state_predictor_', '')}"
    exact_pattern = model_dir / f"{exact_scaler_name}.joblib"
    
    if exact_pattern.exists():
        logger.debug(f"Found exact match scaler: {exact_pattern}")
        return exact_pattern
    
    # Try another common naming pattern: scaler_TIMESTAMP_type.joblib
    if len(model_parts) >= 3:  # Should have at least "MODEL_NAME_TIMESTAMP_TYPE"
        model_type = model_parts[-1]  # Last part should be the model type (rf, svm, etc.)
        timestamp_parts = []
        
        # Extract timestamp parts from the model filename
        # The model filename format could be like: cognitive_state_predictor_20250317_001647_rf.joblib
        # Or it could be like: Cognitive_State_Prediction_Model_20250317_000634_rf.joblib
        # So we need to find all the numeric parts that could be part of the timestamp
        for part in model_parts:
            if part.isdigit() or (len(part) > 0 and part[0].isdigit()):
                timestamp_parts.append(part)
        
        # If we found timestamp parts, use them to build scaler patterns
        if timestamp_parts:
            full_timestamp = "_".join(timestamp_parts)
            
            # Try more scaler naming patterns
            scaler_patterns = [
                model_dir / f"scaler_{full_timestamp}_{model_type}.joblib",
                model_dir / f"scaler_{full_timestamp}.joblib",
                model_dir / f"scaler_{timestamp_parts[-1]}_{model_type}.joblib",
                model_dir / f"scaler_{timestamp_parts[-1]}.joblib",
                model_dir / f"scaler_{timestamp_parts[0]}_{model_type}.joblib",
                model_dir / f"scaler_{timestamp_parts[0]}.joblib",
            ]
            
            # Add additional pattern where model name might contain standard prefix
            if "cognitive_state_predictor" in model_basename or "Cognitive_State_Prediction_Model" in model_basename:
                for prefix in ["cognitive_state_predictor", "Cognitive_State_Prediction_Model"]:
                    if prefix in model_basename:
                        model_suffix = model_basename.replace(f"{prefix}_", "")
                        scaler_patterns.append(model_dir / f"scaler_{model_suffix}.joblib")
            
            # Check if any of the patterns exists
            for pattern in scaler_patterns:
                if pattern.exists():
                    logger.debug(f"Found matching scaler: {pattern}")
                    return pattern
    
    # If no direct match found, look for any scaler in the same directory
    # This is a fallback approach - look for the newest scaler in the same dir
    scalers = list(model_dir.glob("*scaler*.joblib"))
    if scalers:
        newest_scaler = max(scalers, key=lambda x: x.stat().st_mtime)
        logger.warning(f"No exact matching scaler found. Using newest scaler in same directory: {newest_scaler}")
        return newest_scaler
    
    # If no scaler found in the same directory, look in the parent directory one level up
    parent_dir = model_dir.parent
    parent_scalers = list(parent_dir.glob("*scaler*.joblib"))
    if parent_scalers:
        newest_parent_scaler = max(parent_scalers, key=lambda x: x.stat().st_mtime)
        logger.warning(f"No scaler found in model directory. Using scaler from parent directory: {newest_parent_scaler}")
        return newest_parent_scaler
    
    # Last resort - get any scaler from the models directory
    logger.warning("No matching scaler found. Searching entire models directory...")
    any_scalers = list(Path('models').glob("**/*scaler*.joblib"))
    if any_scalers:
        newest_any_scaler = max(any_scalers, key=lambda x: x.stat().st_mtime)
        logger.warning(f"Using scaler from models directory: {newest_any_scaler}")
        return newest_any_scaler
        
    # If we got here, no scaler was found
    logger.error("No scaler found anywhere")
    return None

def list_available_models():
    """List all available models in the models directory."""
    print("\n" + "=" * 80)
    print("AVAILABLE TRAINED MODELS")
    print("=" * 80)
    
    # Search in all model directories
    search_dirs = [
        Path('models/sample/default'),
        Path('models/sample/rf'),
        Path('models/sample/svm'),
        Path('models/sample/gb'),
        Path('models/sample/mlp'),
        Path('models/sample/knn'),
        Path('models/sample/lr'),
        Path('models/advanced/rf'),
        Path('models/advanced/svm'),
        Path('models/advanced/gb'),
        Path('models/advanced/mlp'),
        Path('models/advanced/knn'),
        Path('models/advanced/lr'),
    ]
    
    model_files = []
    for directory in search_dirs:
        if directory.exists():
            model_files.extend(list(directory.glob("*.joblib")))
    
    if not model_files:
        print("No trained models found.")
        return
    
    # Filter out scaler files
    model_files = [m for m in model_files if not m.stem.startswith("scaler_")]
    
    # Group by directories
    models_by_category = {}
    
    for model_file in model_files:
        # Determine the category based on the directory structure
        if "advanced" in str(model_file):
            category = "Advanced Models"
        elif "sample" in str(model_file):
            category = "Sample Models"
        else:
            category = "Other Models"
        
        if category not in models_by_category:
            models_by_category[category] = []
        
        models_by_category[category].append(model_file)
    
    # Print models by category
    for category, files in models_by_category.items():
        print(f"\n{category}:")
        print("-" * len(category))
        
        # Sort files by modification time (newest first)
        sorted_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        for model_file in sorted_files:
            model_name = model_file.name
            model_time = datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Try to extract model type
            model_type = "Unknown"
            if "_rf." in model_name:
                model_type = "Random Forest"
            elif "_svm." in model_name:
                model_type = "Support Vector Machine"
            elif "_gb." in model_name:
                model_type = "Gradient Boosting"
            elif "_mlp." in model_name:
                model_type = "Neural Network (MLP)"
            elif "_knn." in model_name:
                model_type = "K-Nearest Neighbors"
            elif "_lr." in model_name:
                model_type = "Logistic Regression"
            
            # Look for metadata to get the accuracy
            accuracy = "Unknown"
            try:
                metadata_path = model_file.parent / f"metadata_{model_file.stem.split('_')[-1]}.json"
                if not metadata_path.exists():
                    # Try another pattern
                    metadata_pattern = model_file.parent.glob(f"metadata_*_{model_file.stem.split('_')[-1]}.json")
                    metadata_files = list(metadata_pattern)
                    if metadata_files:
                        metadata_path = metadata_files[0]
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if 'accuracy' in metadata:
                            accuracy = f"{metadata['accuracy']:.3f}"
            except Exception as e:
                logger.debug(f"Error reading metadata for {model_file}: {str(e)}")
            
            print(f"  - {model_name}")
            print(f"    Type: {model_type}, Created: {model_time}, Accuracy: {accuracy}")
            print(f"    Path: {model_file}")
            print()
    
    print("\nTo use a specific model for prediction:")
    print("python cwt.py predict --input data/sample_input.json --model [model_path]")
    print("=" * 80)

def display_help(topic=None):
    """Display help information for commands or specific topics."""
    header = "\n" + "=" * 80 + "\n"
    header += "COGNITIVE WORKLOAD TOOL (CWT) HELP\n"
    header += "=" * 80 + "\n"
    
    general_help = """
The Cognitive Workload Tool (CWT) is designed to predict cognitive workload 
levels based on physiological, EEG, and gaze data. It uses machine learning 
to classify cognitive states as Low, Medium, or High.

"""
    
    commands = {
        "train": {
            "description": "Train a new model using data files specified in .env",
            "usage": "python cwt.py train [--model-type TYPE]",
            "options": [
                "--model-type, -m: Type of model to train (rf, svm, gb, mlp, knn, lr)",
            ],
            "examples": [
                "python cwt.py train",
                "python cwt.py train --model-type rf",
                "python cwt.py train --model-type svm"
            ]
        },
        "predict": {
            "description": "Make predictions using a trained model",
            "usage": "python cwt.py predict --input INPUT_FILE [options]",
            "options": [
                "--input, -i: Input JSON or CSV file with feature values",
                "--model, -m: Path to model file (uses latest model if not specified)",
                "--scaler, -s: Path to scaler file (uses scaler associated with model if not specified)",
                "--output, -o: Output file for predictions",
                "--batch, -b: Process input as batch data with multiple samples",
                "--time-series, -t: Treat data as time series with temporal ordering",
                "--auto-detect, -a: Auto-detect input type and format (JSON, CSV, batch, etc.)"
            ],
            "examples": [
                "python cwt.py predict --input data/sample_input.json",
                "python cwt.py predict --input data/batch_samples.csv --batch",
                "python cwt.py predict --input data/time_series.csv --time-series",
                "python cwt.py predict --input any_data_file --auto-detect"
            ]
        },
        "batch-predict": {
            "description": "Process multiple data points for prediction",
            "usage": "python cwt.py batch-predict --input CSV_FILE [options]",
            "options": [
                "--input, -i: CSV file with multiple data points",
                "--model, -m: Path to model file (uses latest model if not specified)",
                "--scaler, -s: Path to scaler file (uses scaler associated with model if not specified)",
                "--output, -o: Output file for predictions",
                "--time-series, -t: Treat data as time series with temporal ordering"
            ],
            "examples": [
                "python cwt.py batch-predict --input data/batch_samples.csv",
                "python cwt.py batch-predict --input data/time_series.csv --time-series",
                "python cwt.py batch-predict --input data/workload_data.csv --output results/predictions.csv"
            ]
        },
        "list-models": {
            "description": "List available trained models",
            "usage": "python cwt.py list-models [--detailed]",
            "options": [
                "--detailed, -d: Show detailed model information"
            ],
            "examples": [
                "python cwt.py list-models",
                "python cwt.py list-models --detailed"
            ]
        },
        "install-models": {
            "description": "Install sample pre-trained models for testing",
            "usage": "python cwt.py install-models [--force]",
            "options": [
                "--force, -f: Force reinstallation of sample models"
            ],
            "examples": [
                "python cwt.py install-models",
                "python cwt.py install-models --force"
            ]
        },
        "train-from-examples": {
            "description": "Train a model using example JSON files",
            "usage": "python cwt.py train-from-examples [--model-type TYPE]",
            "options": [
                "--model-type, -m: Type of model to train (rf, svm, gb, mlp, knn, lr)"
            ],
            "examples": [
                "python cwt.py train-from-examples",
                "python cwt.py train-from-examples --model-type gb"
            ]
        },
        "help": {
            "description": "Show help for all commands or a specific topic",
            "usage": "python cwt.py help [--topic TOPIC]",
            "options": [
                "--topic, -t: Get help on a specific topic"
            ],
            "examples": [
                "python cwt.py help",
                "python cwt.py help --topic predict",
                "python cwt.py !help"
            ]
        }
    }
    
    topics = {
        "model-types": """
AVAILABLE MODEL TYPES:
    rf  - Random Forest: Ensemble method that builds multiple decision trees.
          Good for handling various data types and resistant to overfitting.
    
    svm - Support Vector Machine: Finds the hyperplane that best separates classes.
          Effective in high dimensional spaces but may be slower to train.
    
    gb  - Gradient Boosting: Ensemble method that builds trees sequentially.
          Often achieves high accuracy but may overfit on noisy data.
    
    mlp - Neural Network (MLP): Multi-layer perceptron neural network.
          Flexible model that can capture complex patterns but requires more data.
    
    knn - K-Nearest Neighbors: Classification based on closest training examples.
          Simple but effective algorithm, sensitive to local patterns.
    
    lr  - Logistic Regression: Linear model for classification.
          Fast and interpretable, but may underperform on complex relationships.
""",
        "data-format": """
DATA FORMAT REQUIREMENTS:

For training, the tool expects three CSV files:

1. Physiological data file:
   - Required columns: timestamp, subject_id, pulse_rate, blood_pressure_sys, resp_rate
   - Optional: skin_conductance, cognitive_workload (target)

2. EEG data file:
   - Required columns: timestamp, subject_id, alpha_power, theta_power
   - Optional: beta_power, delta_power, gamma_power, alpha_theta_ratio

3. Gaze tracking data file:
   - Required columns: timestamp, subject_id, pupil_diameter_left, pupil_diameter_right,
     fixation_duration, blink_rate, workload_intensity, gaze_x, gaze_y

For single-sample prediction, the tool accepts a JSON file with feature values:
{
  "pulse_rate": 75.2,
  "blood_pressure_sys": 120.5,
  "resp_rate": 16.4,
  "pupil_diameter_left": 5.2,
  "pupil_diameter_right": 5.1,
  "fixation_duration": 245.8,
  "blink_rate": 12.3,
  "workload_intensity": 50.0,
  "gaze_x": 512.3,
  "gaze_y": 384.7,
  "alpha_power": 18.5,
  "theta_power": 22.6
}

For batch prediction, the tool accepts a CSV file with multiple samples:
timestamp,pulse_rate,blood_pressure_sys,resp_rate,pupil_diameter_left,...
2023-01-01T10:00:00,75.2,120.5,16.4,5.2,...
2023-01-01T10:01:00,76.8,122.3,16.7,5.3,...
...

For time series data, ensure the CSV file includes a timestamp column. The tool will
automatically sort by timestamp and analyze transitions between cognitive states.

Example JSON and CSV files are provided in the examples/ directory.
""",
        "batch-processing": """
BATCH PROCESSING AND TIME SERIES ANALYSIS:

The CWT tool supports processing multiple data points in a single operation, which is
particularly useful for:

1. Analyzing workload patterns over time
2. Processing data from multiple subjects or sessions
3. Generating reports with workload distribution statistics

Batch processing features:

* Auto-detection of input type (CSV, JSON, etc.)
* Processing of time series data with temporal analysis
* Detection of transitions between cognitive states
* Summary statistics for workload distribution
* Visualization of cognitive workload timeline

Command examples:

1. Basic batch processing:
   python cwt.py batch-predict --input data/batch_samples.csv
   
2. Time series analysis:
   python cwt.py batch-predict --input data/time_series.csv --time-series
   
3. Auto-detection with the predict command:
   python cwt.py predict --input any_data_file --auto-detect

4. Save results to file:
   python cwt.py batch-predict --input data/batch.csv --output results/predictions.csv

Output:
- Original data with added prediction columns
- Summary statistics of workload distribution
- For time series data: visualization of workload over time
- Analysis of transitions between cognitive states

CSV format requirements:
- Each row represents one data point
- Must include the same features used for training
- For time series, include a 'timestamp' column in a recognizable format
- Missing values are handled gracefully

The batch processing system automatically selects the appropriate model for each data
point based on its workload intensity - this ensures optimal model selection for each
sample.
""",
        "configuration": """
CONFIGURATION:

The tool can be configured using a .env file:

# Data files
PHYSIO_DATA_PATH=data/Enhanced_Workload_Clinical_Data.csv
EEG_DATA_PATH=data/000_EEG_Cluster_ANOVA_Results.csv
GAZE_DATA_PATH=data/008_01.csv

# Model configuration
MODEL_OUTPUT_DIR=models
MODEL_NAME=Cognitive_State_Prediction_Model

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/general/cwt.log

# Training parameters
TEST_SIZE=0.2
RANDOM_SEED=42
CV_FOLDS=5

# Default model type
DEFAULT_MODEL_TYPE=rf

See .env.example for all available configuration options.
""",
        "examples": """
EXAMPLE USAGE:

1. Generate sample data:
   python generate_sample_data.py

2. Install sample models:
   python cwt.py install-models

3. List available models:
   python cwt.py list-models

4. Make a prediction with a single sample:
   python cwt.py predict --input data/sample_input.json

5. Process a batch of data points:
   python cwt.py batch-predict --input data/batch_samples.csv

6. Analyze time series data:
   python cwt.py batch-predict --input data/time_series.csv --time-series

7. Train a new model:
   python cwt.py train --model-type rf

8. Train a model from JSON examples:
   python cwt.py train-from-examples

9. Auto-detect input type and format:
   python cwt.py predict --input any_data_file --auto-detect

10. Convert CSV to JSON:
    python examples/create_json_from_csv.py examples/sample_data.csv
"""
    }
    
    if topic is None:
        # General help
        print(header)
        print(general_help)
        
        print("AVAILABLE COMMANDS:\n")
        for cmd, info in commands.items():
            if cmd != "!help":  # Skip the duplicate !help command in the list
                print(f"  {cmd:<20} {info['description']}")
        
        print("\nAVAILABLE HELP TOPICS:\n")
        for topic, _ in topics.items():
            print(f"  {topic}")
        
        print("\nFor more information on a specific command or topic, use:")
        print("  python cwt.py help --topic [command or topic]")
        print("  python cwt.py !help --topic [command or topic]")
    else:
        # Command-specific or topic-specific help
        print(header)
        
        if topic in commands:
            cmd_info = commands[topic]
            print(f"COMMAND: {topic}\n")
            print(f"Description: {cmd_info['description']}")
            print(f"Usage: {cmd_info['usage']}")
            
            if cmd_info['options']:
                print("\nOptions:")
                for opt in cmd_info['options']:
                    print(f"  {opt}")
            
            if cmd_info['examples']:
                print("\nExamples:")
                for ex in cmd_info['examples']:
                    print(f"  {ex}")
        elif topic in topics:
            print(f"TOPIC: {topic}\n")
            print(topics[topic])
        else:
            print(f"Unknown topic: {topic}")
            print("\nAvailable topics:")
            for cmd in commands:
                if cmd != "!help":  # Skip the duplicate !help command in the list
                    print(f"  {cmd}")
            for t in topics:
                print(f"  {t}")

# ---------------------- BATCH PREDICTION FUNCTIONS ---------------------- #
def predict_from_csv(csv_path, output_path=None, model_path=None, scaler_path=None):
    """
    Process a CSV file containing multiple data points over time and predict workload levels.
    
    Args:
        csv_path (str): Path to the CSV file with multiple data points
        output_path (str, optional): Path to save the prediction results
        model_path (str, optional): Path to a specific model to use
        scaler_path (str, optional): Path to a specific scaler to use
        
    Returns:
        DataFrame: DataFrame with original data and predictions
    """
    logger.info(f"Processing CSV file for batch prediction: {csv_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Check if the CSV has timestamp column (for time series data)
        has_timestamp = 'timestamp' in df.columns
        if has_timestamp:
            logger.info("Time series data detected (timestamp column present)")
            # Ensure timestamp is properly formatted
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            except Exception as e:
                logger.warning(f"Could not convert timestamp column to datetime: {e}")
        
        # Initialize columns for predictions and confidence
        df['predicted_cognitive_state'] = None
        df['prediction_confidence'] = None
        df['model_used'] = None
        
        # Process each row for prediction
        logger.info("Making predictions for each data point...")
        
        predictions = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            # Convert row to dictionary for prediction
            row_dict = row.to_dict()
            
            # Skip any non-numeric values that would cause issues
            row_dict = {k: v for k, v in row_dict.items() 
                       if isinstance(v, (int, float)) and not pd.isna(v)}
            
            # Make individual prediction for this data point
            # This will auto-select the appropriate model based on workload_intensity
            try:
                prediction, proba = predict_new_data(model_path, scaler_path, row_dict)
                
                # Store predictions and model info
                if prediction:
                    # Get the model type that was used
                    model_type = "Unknown"
                    if 'rf' in str(model_path).lower():
                        model_type = "RF"
                    elif 'mlp' in str(model_path).lower():
                        model_type = "MLP"
                    elif 'lr' in str(model_path).lower():
                        model_type = "LR"
                    elif 'svm' in str(model_path).lower():
                        model_type = "SVM"
                    elif 'knn' in str(model_path).lower():
                        model_type = "KNN"
                    elif 'gb' in str(model_path).lower():
                        model_type = "GB"
                        
                    # Find the highest probability
                    if isinstance(proba, dict):
                        max_prob = max(proba.values())
                    else:
                        max_prob = 1.0  # Default if no probabilities available
                        
                    predictions.append({
                        'index': idx,
                        'prediction': prediction,
                        'confidence': max_prob,
                        'model_used': model_type
                    })
                else:
                    logger.warning(f"No prediction made for row {idx}")
                    predictions.append({
                        'index': idx,
                        'prediction': 'Unknown',
                        'confidence': 0.0,
                        'model_used': 'None'
                    })
            except Exception as e:
                logger.error(f"Error predicting row {idx}: {e}")
                predictions.append({
                    'index': idx,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'model_used': 'Error'
                })
        
        # Update the dataframe with predictions
        for p in predictions:
            idx = p['index']
            df.at[idx, 'predicted_cognitive_state'] = p['prediction']
            df.at[idx, 'prediction_confidence'] = p['confidence']
            df.at[idx, 'model_used'] = p['model_used']
        
        # Calculate distribution of predicted states
        state_counts = df['predicted_cognitive_state'].value_counts()
        logger.info(f"Prediction distribution: {state_counts.to_dict()}")
        
        # For time series data, look for transitions between states
        if has_timestamp:
            df = df.sort_values('timestamp')
            state_changes = (df['predicted_cognitive_state'] != df['predicted_cognitive_state'].shift()).sum()
            logger.info(f"Detected {state_changes} transitions between cognitive states")
            
            # Add a column to identify transitions
            df['state_transition'] = df['predicted_cognitive_state'] != df['predicted_cognitive_state'].shift()
        
        # Save predictions if output path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            df.to_csv(output_path, index=False)
            logger.info(f"Prediction results saved to {output_path}")
            
            # Also create a summary file
            summary_path = output_path.replace('.csv', '_summary.csv')
            if summary_path == output_path:  # In case output_path doesn't end with .csv
                summary_path = f"{output_path}_summary.csv"
                
            summary_df = pd.DataFrame(state_counts).reset_index()
            summary_df.columns = ['cognitive_state', 'count']
            summary_df['percentage'] = summary_df['count'] / len(df) * 100
            
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Prediction summary saved to {summary_path}")
            
            # Generate a visualization of the predictions over time
            if has_timestamp:
                try:
                    viz_path = output_path.replace('.csv', '_timeline.png')
                    if viz_path == output_path:
                        viz_path = f"{output_path}_timeline.png"
                    
                    plt.figure(figsize=(12, 6))
                    
                    # Map categorical values to numeric for plotting
                    state_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': -1, 'Error': -1}
                    if 'predicted_cognitive_state' in df.columns:
                        df['state_numeric'] = df['predicted_cognitive_state'].map(state_map)
                    
                    plt.plot(df['timestamp'], df['state_numeric'], 'o-', markersize=4)
                    plt.yticks([0, 1, 2], ['Low', 'Medium', 'High'])
                    plt.title('Cognitive Workload Prediction Timeline')
                    plt.xlabel('Time')
                    plt.ylabel('Predicted Cognitive State')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Highlight state transitions if present
                    if 'state_transition' in df.columns:
                        transition_points = df[df['state_transition'] == True]
                        if len(transition_points) > 0:
                            plt.scatter(transition_points['timestamp'], 
                                      transition_points['state_numeric'], 
                                      color='red', s=80, zorder=5, 
                                      label='State Transitions')
                            plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(viz_path)
                    logger.info(f"Timeline visualization saved to {viz_path}")
                except Exception as e:
                    logger.error(f"Error creating visualization: {e}")
        
        # Also print a summary to console
        print("\n" + "="*50)
        print("✓ BATCH PREDICTION RESULTS")
        print("="*50)
        print(f"Total samples processed: {len(df)}")
        print("\nPrediction distribution:")
        for state, count in state_counts.items():
            percentage = count / len(df) * 100
            print(f"  {state:<10}: {count} ({percentage:.1f}%)")
        
        if has_timestamp:
            print(f"\nTime range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"State transitions detected: {state_changes}")
        
        if output_path:
            print(f"\nFull results saved to: {output_path}")
        print("="*50)
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing CSV file for batch prediction: {e}")
        print(f"\n✘ ERROR: Batch prediction failed: {e}")
        raise

def predict_automatic(input_data, output_path=None, model_path=None, scaler_path=None):
    """
    Automatically determine the type of input data and process it accordingly.
    Can handle JSON files, CSV files, or Python dictionaries/DataFrames.
    
    Args:
        input_data: Input data which could be a path to a file or a data structure
        output_path: Path to save output results
        model_path: Optional path to a specific model
        scaler_path: Optional path to a specific scaler
        
    Returns:
        The prediction results
    """
    logger.info(f"Auto-detecting input type for: {input_data}")
    
    try:
        # Case 1: Input is a string path to a file
        if isinstance(input_data, str):
            # Check if the file exists
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Input file not found: {input_data}")
            
            # Determine file type by extension
            file_ext = os.path.splitext(input_data)[1].lower()
            
            if file_ext == '.json':
                logger.info(f"Detected JSON input file: {input_data}")
                # Load the JSON file
                with open(input_data, 'r') as f:
                    json_data = json.load(f)
                
                # If it's a list, it's a batch of samples
                if isinstance(json_data, list):
                    logger.info(f"Batch JSON with {len(json_data)} samples detected")
                    # Convert to DataFrame for batch processing
                    df = pd.DataFrame(json_data)
                    # Add timestamp if not present
                    if 'timestamp' not in df.columns:
                        df['timestamp'] = pd.date_range(start='now', periods=len(df), freq='S')
                    # Save to temporary CSV for batch processing
                    temp_csv = os.path.join(os.path.dirname(input_data), '_temp_batch.csv')
                    df.to_csv(temp_csv, index=False)
                    # Process as CSV
                    return predict_from_csv(temp_csv, output_path, model_path, scaler_path)
                else:
                    # Single JSON sample
                    logger.info("Single JSON sample detected")
                    return predict_new_data(model_path, scaler_path, json_data)
            
            elif file_ext in ['.csv', '.txt', '.data']:
                logger.info(f"Detected CSV/text input file: {input_data}")
                return predict_from_csv(input_data, output_path, model_path, scaler_path)
            
            else:
                logger.warning(f"Unknown file extension: {file_ext}")
                # Try to load as CSV by default
                logger.info("Attempting to load as CSV")
                return predict_from_csv(input_data, output_path, model_path, scaler_path)
        
        # Case 2: Input is a pandas DataFrame
        elif isinstance(input_data, pd.DataFrame):
            logger.info(f"Detected pandas DataFrame with {len(input_data)} rows")
            # Save to temporary CSV for batch processing
            temp_csv = os.path.join(os.getcwd(), '_temp_dataframe.csv')
            input_data.to_csv(temp_csv, index=False)
            # Process as CSV
            return predict_from_csv(temp_csv, output_path, model_path, scaler_path)
        
        # Case 3: Input is a dictionary (single sample)
        elif isinstance(input_data, dict):
            logger.info("Detected dictionary input (single sample)")
            return predict_new_data(model_path, scaler_path, input_data)
        
        # Case 4: Input is a list of dictionaries (batch of samples)
        elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
            logger.info(f"Detected list of dictionaries ({len(input_data)} samples)")
            # Convert to DataFrame for batch processing
            df = pd.DataFrame(input_data)
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(start='now', periods=len(df), freq='S')
            # Save to temporary CSV for batch processing
            temp_csv = os.path.join(os.getcwd(), '_temp_batch_list.csv')
            df.to_csv(temp_csv, index=False)
            # Process as CSV
            return predict_from_csv(temp_csv, output_path, model_path, scaler_path)
        
        else:
            logger.error(f"Unsupported input type: {type(input_data)}")
            raise TypeError(f"Unsupported input type: {type(input_data)}")
    
    except Exception as e:
        logger.error(f"Error in automatic prediction: {e}")
        print(f"\n✘ ERROR: Automatic prediction failed: {e}")
        raise

# ---------------------- FEATURE INFERENCE ---------------------- #
def infer_missing_features(data, reference_data=None):
    """
    Infer missing features based on available data and correlation patterns.
    Uses a combination of methods:
    1. Standard imputation for simple missing values
    2. Regression models for related features (e.g., pupil metrics from other physiological data)
    3. Empirical distributions when no correlation data is available
    
    Args:
        data (dict or DataFrame): Input data with potentially missing features
        reference_data (DataFrame, optional): External reference data to use for inference
        
    Returns:
        dict or DataFrame: Data with inferred values for missing features
    """
    # Convert input to DataFrame if it's a dictionary
    is_dict_input = isinstance(data, dict)
    if is_dict_input:
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    logger.info(f"Inferring missing features for dataset with {len(df)} rows")
    
    # Core features that we expect to have
    core_features = [
        "pulse_rate", "blood_pressure_sys", "resp_rate", "workload_intensity",
        "pupil_diameter_left", "pupil_diameter_right", "fixation_duration", 
        "blink_rate", "gaze_x", "gaze_y", "alpha_power", "theta_power"
    ]
    
    # Check which features are missing
    missing_features = [f for f in core_features if f not in df.columns or df[f].isnull().all()]
    
    if not missing_features:
        logger.info("No missing core features detected")
        return data  # Return original data if nothing is missing
    
    logger.info(f"Detected missing features: {missing_features}")
    
    # Load reference data if none provided
    if reference_data is None:
        try:
            reference_data = load_reference_data()
        except Exception as e:
            logger.warning(f"Could not load reference data: {e}. Using empirical distributions.")
            reference_data = None
    
    # Apply different inference methods based on available data
    for feature in missing_features:
        # Check if we have a specific imputation function for this feature
        if feature in ["pupil_diameter_left", "pupil_diameter_right"]:
            df = infer_pupil_metrics(df, reference_data)
        
        elif feature in ["alpha_power", "theta_power"]:
            df = infer_eeg_metrics(df, reference_data)
        
        elif feature == "workload_intensity":
            df = infer_workload_intensity(df, reference_data)
            
        else:
            # Generic imputation for other features
            df = impute_generic_feature(df, feature, reference_data)
    
    # Verify that all required features are now present
    still_missing = [f for f in core_features if f not in df.columns]
    if still_missing:
        logger.warning(f"Some features could not be inferred: {still_missing}")
    
    # Return in the same format as input
    if is_dict_input:
        return df.iloc[0].to_dict()
    else:
        return df

def load_reference_data():
    """
    Load reference data from external resources for feature inference.
    Combines physiological, EEG, and gaze data from multiple sources.
    
    Returns:
        DataFrame: Combined reference data
    """
    logger.info("Loading reference data for feature inference")
    
    try:
        # Try to load from configured data paths
        has_real_data = True
        for data_type, file_path in DATA_FILES.items():
            if not os.path.exists(file_path):
                has_real_data = False
                break
        
        if has_real_data:
            logger.info("Using real data files as reference")
            df_physio, df_eeg, df_gaze = load_data()
            
            # Merge the data
            df = pd.merge_asof(df_physio.sort_values("timestamp"), 
                             df_eeg.sort_values("timestamp"), 
                             on="timestamp")
            df = pd.merge_asof(df.sort_values("timestamp"), 
                             df_gaze.sort_values("timestamp"), 
                             on="timestamp")
            
            # Select relevant features
            features = ["pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
                      "pupil_diameter_right", "fixation_duration", "blink_rate", "workload_intensity",
                      "gaze_x", "gaze_y", "alpha_power", "theta_power"]
            
            # Drop rows with missing values
            df = df[features].dropna()
            
            return df
        else:
            # Try to load pre-packaged reference data
            reference_path = os.path.join('data', 'reference', 'cognitive_workload_reference.csv')
            if os.path.exists(reference_path):
                logger.info(f"Loading reference data from {reference_path}")
                return pd.read_csv(reference_path)
            
            # Create synthetic data as a last resort
            logger.warning("No reference data found. Generating synthetic reference data.")
            return generate_synthetic_reference_data()
            
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        logger.warning("Falling back to synthetic reference data")
        return generate_synthetic_reference_data()

def generate_synthetic_reference_data(n_samples=1000):
    """
    Generate synthetic reference data when no real data is available.
    
    Args:
        n_samples: Number of synthetic samples to generate
        
    Returns:
        DataFrame: Synthetic reference data
    """
    logger.info(f"Generating {n_samples} synthetic reference data points")
    
    # Generate random data with realistic distributions
    data = {
        "pulse_rate": np.random.normal(75, 10, n_samples).clip(50, 120),
        "blood_pressure_sys": np.random.normal(120, 15, n_samples).clip(90, 180),
        "resp_rate": np.random.normal(16, 3, n_samples).clip(8, 30),
        "workload_intensity": np.random.uniform(10, 90, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add derived features with realistic correlations
    
    # Pupil diameter correlates with workload intensity
    base_pupil = 3 + df["workload_intensity"] * 0.03
    noise = np.random.normal(0, 0.2, n_samples)
    df["pupil_diameter_left"] = (base_pupil + noise).clip(2.5, 7.5)
    df["pupil_diameter_right"] = (base_pupil + np.random.normal(0, 0.15, n_samples)).clip(2.5, 7.5)
    
    # Fixation duration is inversely related to workload
    df["fixation_duration"] = (400 - df["workload_intensity"] * 2.5 + 
                             np.random.normal(0, 20, n_samples)).clip(150, 450)
    
    # Blink rate decreases with workload
    df["blink_rate"] = (22 - df["workload_intensity"] * 0.15 + 
                      np.random.normal(0, 2, n_samples)).clip(5, 25)
    
    # Gaze position (less dependent)
    df["gaze_x"] = np.random.normal(500, 50, n_samples)
    df["gaze_y"] = np.random.normal(375, 40, n_samples)
    
    # EEG features
    # Alpha power decreases with workload
    df["alpha_power"] = (30 - df["workload_intensity"] * 0.2 + 
                       np.random.normal(0, 3, n_samples)).clip(5, 35)
    
    # Theta power increases with workload
    df["theta_power"] = (15 + df["workload_intensity"] * 0.15 + 
                       np.random.normal(0, 2, n_samples)).clip(10, 35)
    
    logger.info("Synthetic reference data generated")
    return df

def infer_pupil_metrics(df, reference_data):
    """
    Infer missing pupil diameter metrics based on other physiological data.
    
    Args:
        df: DataFrame with potentially missing pupil metrics
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with inferred pupil metrics
    """
    # Check if both pupil metrics are missing
    left_missing = "pupil_diameter_left" not in df.columns or df["pupil_diameter_left"].isnull().all()
    right_missing = "pupil_diameter_right" not in df.columns or df["pupil_diameter_right"].isnull().all()
    
    # If only one is missing, we can approximate from the other
    if left_missing and not right_missing:
        logger.info("Inferring left pupil diameter from right pupil diameter")
        df["pupil_diameter_left"] = df["pupil_diameter_right"] * np.random.uniform(0.97, 1.03)
        return df
    
    if right_missing and not left_missing:
        logger.info("Inferring right pupil diameter from left pupil diameter")
        df["pupil_diameter_right"] = df["pupil_diameter_left"] * np.random.uniform(0.97, 1.03)
        return df
    
    # Both are missing, use other metrics to infer
    if (left_missing and right_missing):
        logger.info("Inferring both pupil diameters from other physiological metrics")
        
        # The primary driver of pupil diameter is workload intensity
        if "workload_intensity" in df.columns and reference_data is not None:
            logger.info("Using workload intensity to infer pupil diameters")
            
            # Calculate base pupil size based on workload intensity
            for idx, row in df.iterrows():
                workload = row["workload_intensity"]
                base_pupil = 3.0 + (workload / 100.0) * 3.0  # Map 0-100 workload to 3.0-6.0mm pupil
                
                # Add random variation for realistic data
                df.at[idx, "pupil_diameter_left"] = base_pupil + np.random.normal(0, 0.2)
                df.at[idx, "pupil_diameter_right"] = base_pupil + np.random.normal(0, 0.18)
                
        elif all(col in df.columns for col in ["pulse_rate", "blood_pressure_sys"]):
            logger.info("Using physiological metrics to infer pupil diameters")
            
            # Physiological markers also correlate with pupil diameter
            for idx, row in df.iterrows():
                # Normalize physiological metrics
                pulse_norm = (row["pulse_rate"] - 60) / 60  # Normalize to 0-1 range
                bp_norm = (row["blood_pressure_sys"] - 90) / 90  # Normalize to 0-1 range
                
                # Calculate base pupil diameter from physiological state
                base_pupil = 3.5 + (pulse_norm * 1.5) + (bp_norm * 1.5)
                
                # Add random variation
                df.at[idx, "pupil_diameter_left"] = min(max(base_pupil + np.random.normal(0, 0.2), 2.5), 7.5)
                df.at[idx, "pupil_diameter_right"] = min(max(base_pupil + np.random.normal(0, 0.18), 2.5), 7.5)
        
        else:
            logger.warning("Limited data for pupil inference, using reference distributions")
            
            # Use default mid-range values with variation
            for idx in range(len(df)):
                base_pupil = 4.5  # Default mid-range pupil size
                df.at[idx, "pupil_diameter_left"] = base_pupil + np.random.normal(0, 0.5)
                df.at[idx, "pupil_diameter_right"] = base_pupil + np.random.normal(0, 0.45)
    
    return df

def infer_eeg_metrics(df, reference_data):
    """
    Infer missing EEG metrics (alpha_power, theta_power) from other data.
    
    Args:
        df: DataFrame with potentially missing EEG metrics
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with inferred EEG metrics
    """
    alpha_missing = "alpha_power" not in df.columns or df["alpha_power"].isnull().all()
    theta_missing = "theta_power" not in df.columns or df["theta_power"].isnull().all()
    
    # EEG metrics are primarily related to workload intensity
    if "workload_intensity" in df.columns:
        logger.info("Using workload intensity to infer EEG metrics")
        
        # Alpha power inversely correlates with workload
        if alpha_missing:
            logger.info("Inferring alpha power from workload intensity")
            df["alpha_power"] = df.apply(
                lambda row: min(max(30 - row["workload_intensity"] * 0.2 + np.random.normal(0, 2), 5), 35),
                axis=1
            )
        
        # Theta power positively correlates with workload
        if theta_missing:
            logger.info("Inferring theta power from workload intensity")
            df["theta_power"] = df.apply(
                lambda row: min(max(15 + row["workload_intensity"] * 0.15 + np.random.normal(0, 2), 10), 35),
                axis=1
            )
    
    # If no workload intensity, use reference data or defaults
    else:
        logger.warning("No workload intensity data available for EEG inference")
        
        if reference_data is not None:
            logger.info("Using reference data distributions for EEG metrics")
            
            if alpha_missing:
                alpha_mean = reference_data["alpha_power"].mean()
                alpha_std = reference_data["alpha_power"].std()
                df["alpha_power"] = np.random.normal(alpha_mean, alpha_std, len(df)).clip(5, 35)
            
            if theta_missing:
                theta_mean = reference_data["theta_power"].mean()
                theta_std = reference_data["theta_power"].std()
                df["theta_power"] = np.random.normal(theta_mean, theta_std, len(df)).clip(10, 35)
        
        else:
            logger.warning("Using default distributions for EEG metrics")
            
            if alpha_missing:
                df["alpha_power"] = np.random.normal(18, 5, len(df)).clip(5, 35)
            
            if theta_missing:
                df["theta_power"] = np.random.normal(20, 5, len(df)).clip(10, 35)
    
    return df

def infer_workload_intensity(df, reference_data):
    """
    Infer workload intensity from other physiological and behavioral metrics.
    
    Args:
        df: DataFrame with missing workload_intensity
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with inferred workload_intensity
    """
    logger.info("Inferring workload intensity from other metrics")
    
    # Define feature weights and directions for workload inference
    workload_indicators = {
        # Feature: [weight, direction] (positive direction = higher value means higher workload)
        "pulse_rate": [0.20, 1],
        "blood_pressure_sys": [0.15, 1],
        "resp_rate": [0.15, 1],
        "pupil_diameter_left": [0.10, 1],
        "pupil_diameter_right": [0.10, 1],
        "fixation_duration": [0.10, -1],  # Negative: shorter fixations = higher workload
        "blink_rate": [0.10, -1],         # Negative: fewer blinks = higher workload
        "alpha_power": [0.10, -1],        # Negative: lower alpha = higher workload
        "theta_power": [0.10, 1]
    }
    
    # Calculate workload intensity for each row
    for idx, row in df.iterrows():
        workload_score = 50  # Start at middle of range (0-100)
        total_weight = 0
        
        # Accumulate contributions from available features
        for feature, (weight, direction) in workload_indicators.items():
            if feature in row and not pd.isna(row[feature]):
                # Normalize feature value based on typical ranges
                if feature == "pulse_rate":
                    # Normal range: 60-100, higher = more workload
                    norm_value = (row[feature] - 60) / 40
                elif feature == "blood_pressure_sys":
                    # Normal range: 90-150, higher = more workload
                    norm_value = (row[feature] - 90) / 60
                elif feature == "resp_rate":
                    # Normal range: 12-20, higher = more workload
                    norm_value = (row[feature] - 12) / 8
                elif feature in ["pupil_diameter_left", "pupil_diameter_right"]:
                    # Typical range: 3-7mm, higher = more workload
                    norm_value = (row[feature] - 3) / 4
                elif feature == "fixation_duration":
                    # Typical range: 200-400ms, lower = more workload
                    norm_value = (400 - row[feature]) / 200
                elif feature == "blink_rate":
                    # Typical range: 5-25 bpm, lower = more workload
                    norm_value = (25 - row[feature]) / 20
                elif feature == "alpha_power":
                    # Typical range: 5-35, lower = more workload
                    norm_value = (35 - row[feature]) / 30
                elif feature == "theta_power":
                    # Typical range: 10-35, higher = more workload
                    norm_value = (row[feature] - 10) / 25
                else:
                    # Unknown feature, skip
                    continue
                
                # Apply weight and direction
                workload_score += norm_value * direction * weight * 50
                total_weight += weight
        
        # Normalize and constrain to 0-100 range
        if total_weight > 0:
            # Scale by actual weight used
            workload_score = workload_score / total_weight * 0.5
            # Constrain to reasonable range
            workload_score = min(max(workload_score, 10), 90)
        
        df.at[idx, "workload_intensity"] = workload_score
    
    return df

def impute_generic_feature(df, feature, reference_data):
    """
    Generic imputation for any missing feature.
    
    Args:
        df: DataFrame with missing feature
        feature: Name of the missing feature
        reference_data: Reference data for inference
        
    Returns:
        DataFrame: Updated DataFrame with imputed feature
    """
    logger.info(f"Imputing missing feature: {feature}")
    
    # Default feature ranges for imputation
    default_ranges = {
        "pulse_rate": [60, 100],  # min, max
        "blood_pressure_sys": [90, 150],
        "resp_rate": [12, 20],
        "pupil_diameter_left": [3.0, 7.0],
        "pupil_diameter_right": [3.0, 7.0],
        "fixation_duration": [180, 400],
        "blink_rate": [5, 25],
        "workload_intensity": [10, 90],
        "gaze_x": [0, 1000],
        "gaze_y": [0, 750],
        "alpha_power": [5, 35],
        "theta_power": [10, 35],
    }
    
    # Use reference data if available
    if reference_data is not None and feature in reference_data.columns:
        mean_val = reference_data[feature].mean()
        std_val = reference_data[feature].std()
        df[feature] = np.random.normal(mean_val, std_val, len(df))
        
        # Ensure values are within reasonable range
        if feature in default_ranges:
            min_val, max_val = default_ranges[feature]
            df[feature] = df[feature].clip(min_val, max_val)
    
    # Use default distributions
    elif feature in default_ranges:
        min_val, max_val = default_ranges[feature]
        mean_val = (min_val + max_val) / 2
        std_val = (max_val - min_val) / 4
        df[feature] = np.random.normal(mean_val, std_val, len(df)).clip(min_val, max_val)
    
    # Unknown feature, use zeros
    else:
        logger.warning(f"No default range for feature: {feature}, using zero")
        df[feature] = 0
    
    return df

# ---------------------- MAIN EXECUTION ---------------------- #
def main():
    """Main entry point for the application."""
    args = parse_args()
    
    if args.command == 'train':
        logger.info("Starting model training pipeline")
        
        # Get model type from arguments
        model_type = args.model_type
        logger.info(f"Selected model type: {model_type} ({AVAILABLE_MODELS.get(model_type, 'Unknown')})")
        
        try:
            # Check if data files exist before proceeding
            for data_type, file_path in DATA_FILES.items():
                if not os.path.exists(file_path):
                    logger.error(f"{data_type.capitalize()} data file not found: {file_path}")
                    print(f"\n✘ Data file missing: {file_path}")
                    print("\nSince real data is missing, you have two options:")
                    print("  1. Run 'python cwt.py install-models' to install sample models")
                    print("  2. Create a 'data' folder with the required files")
                    print("     or modify paths in your .env file")
                    print("\nRecommended action:")
                    print("  python cwt.py install-models")
                    sys.exit(1)
            
            # Load data
            df_physio, df_eeg, df_gaze = load_data()
            
            # Preprocess data
            df_processed, scaler, features = preprocess_data(df_physio, df_eeg, df_gaze)
            
            # Save scaler
            joblib.dump(scaler, SCALER_OUTPUT_PATH)
            logger.info(f"Scaler saved to {SCALER_OUTPUT_PATH}")
            
            # Train model with specified type
            model, accuracy, X_test, y_test, y_pred = train_model(df_processed, features, model_type, scaler)
            
            logger.info(f"Model training complete. Accuracy: {accuracy:.3f}")
            logger.info(f"Model saved at {MODEL_OUTPUT_PATH}")
            logger.info(f"Scaler saved at {SCALER_OUTPUT_PATH}")
            
            # Print summary to console
            print("\n" + "="*50)
            print(f"✓ MODEL TRAINING COMPLETE")
            print("="*50)
            print(f"Model Type: {AVAILABLE_MODELS.get(model_type, 'Unknown')}")
            print(f"Accuracy:   {accuracy:.3f}")
            
            print("\nTo make predictions, run:")
            print(f"  python cwt.py predict --input YOUR_DATA.json")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            print(f"\nERROR: Training pipeline failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'predict':
        try:
            # Verify that input file exists
            if not os.path.exists(args.input):
                logger.error(f"Input file not found: {args.input}")
                print(f"\n✘ ERROR: Input file not found: {args.input}")
                sys.exit(1)
            
            # If auto-detect flag is set or the input file is a CSV and batch flag is set
            if args.auto_detect or (args.batch or args.input.lower().endswith('.csv')):
                logger.info(f"Processing batch data or auto-detecting input type")
                
                if args.auto_detect:
                    # Use the automatic detection function
                    results = predict_automatic(args.input, args.output, args.model, args.scaler)
                else:
                    # Process as batch CSV file
                    results = predict_from_csv(args.input, args.output, args.model, args.scaler)
            else:
                # Standard single-sample prediction
                # Load input data from JSON
                with open(args.input, 'r') as f:
                    input_data = json.load(f)
                
                # Make prediction using the specified model or automatically select one
                prediction, probabilities = predict_new_data(args.model, args.scaler, input_data)
                
                if not prediction:
                    print("\n✘ ERROR: Prediction failed")
                    sys.exit(1)
                
                # Output prediction
                result = {
                    "prediction": prediction,
                    "confidence": probabilities,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save to output file if specified
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\n✓ Prediction saved to {args.output}")
                
                # Print to console
                print("\n" + "="*50)
                print(f"✓ PREDICTION RESULTS")
                print("="*50)
                print(f"Predicted Cognitive State: {prediction}")
                print("\nProbabilities:")
                for cls, prob in probabilities.items():
                    print(f"  {cls:<10}: {prob:.2f}")
                print("="*50)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            print(f"\nERROR: Prediction failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'batch-predict':
        try:
            # Verify that input file exists
            if not os.path.exists(args.input):
                logger.error(f"Input file not found: {args.input}")
                print(f"\n✘ ERROR: Input file not found: {args.input}")
                sys.exit(1)
            
            # Process as batch CSV file
            predict_from_csv(args.input, args.output, args.model, args.scaler)
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            print(f"\nERROR: Batch prediction failed: {str(e)}")
            sys.exit(1)
    
    elif args.command == 'list-models':
        list_available_models()
    
    elif args.command == 'install-models':
        install_sample_models()
    
    elif args.command in ['help', '!help']:
        display_help(args.topic)
    
    elif args.command == 'train-from-examples':
        # Execute the train_from_examples.py script with the given model type
        examples_script = Path(__file__).resolve().parent / "examples" / "train_from_examples.py"
        if not examples_script.exists():
            print("\n✘ ERROR: examples/train_from_examples.py script not found")
            print("Please ensure the examples directory is set up correctly.")
            sys.exit(1)
        
        cmd = [sys.executable, str(examples_script)]
        if args.model_type:
            cmd.extend(["--model-type", args.model_type])
        
        subprocess.run(cmd)
    
    else:
        logger.error(f"Unknown command: {args.command}")
        logger.error("Use 'help' to see all available commands.")
        sys.exit(1)

if __name__ == "__main__":
    # Special case for !help command
    if len(sys.argv) > 1 and sys.argv[1] == '!help':
        # Convert !help to help for argparse
        sys.argv[1] = 'help'
    try:
        main()  # Run the main function
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"\n✘ ERROR: An error occurred: {str(e)}")
        sys.exit(1)