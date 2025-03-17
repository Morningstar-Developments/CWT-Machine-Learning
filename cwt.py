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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
MODEL_OUTPUT_DIR = os.getenv('MODEL_OUTPUT_DIR', 'models')
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

# ---------------------- MODEL TRAINING ---------------------- #
def train_model(df, features, model_type='rf'):
    """
    Train a machine learning model for cognitive state prediction.
    
    Args:
        df (DataFrame): Preprocessed data with features and target
        features (list): List of feature names
        model_type (str): Type of model to train
        
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
        
        # Plot and save confusion matrix
        logger.debug("Generating confusion matrix")
        cm_path = os.path.join(MODEL_OUTPUT_DIR, f"confusion_matrix_{MODEL_VERSION}.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        
        # Plot and save feature importance if model supports it
        if hasattr(model, 'feature_importances_'):
            logger.debug("Generating feature importance plot")
            fi_path = os.path.join(MODEL_OUTPUT_DIR, f"feature_importance_{MODEL_VERSION}.png")
            plot_feature_importance(model, X.columns, fi_path)
        
        # Save model, scaler, and metadata
        logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
        joblib.dump(model, MODEL_OUTPUT_PATH)
        
        # Save metadata
        save_model_metadata(model, model_type, list(X.columns), accuracy, class_report)
        
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
        tuple: (predicted_cognitive_state, class_probabilities)
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # Try to load metadata to determine model type
        try:
            metadata_path = model_path.replace('.joblib', '.metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_type = metadata.get('model_type', 'unknown')
                logger.info(f"Model type from metadata: {model_type}")
            else:
                model_type = "unknown"
                logger.info("No metadata found, model type unknown")
        except Exception as e:
            logger.warning(f"Could not determine model type from metadata: {str(e)}")
            model_type = "unknown"
        
        # Convert input to DataFrame
        logger.debug("Preparing input data for prediction")
        new_data_df = pd.DataFrame([new_data])
        
        # Scale the input data
        new_data_scaled = scaler.transform(new_data_df)
        
        # Make prediction
        logger.debug("Making prediction")
        prediction = model.predict(new_data_scaled)
        
        # Get class probabilities if model supports it
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(new_data_scaled)
            classes = model.classes_
            proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, prediction_proba[0])}
        else:
            # For models that don't support probability
            proba_dict = {"prediction_confidence": 1.0}
        
        logger.info(f"Prediction: {prediction[0]}")
        logger.debug(f"Class probabilities: {proba_dict}")
        
        return prediction[0], proba_dict
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

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
    
    # Convert to a dictionary (drop the cognitive_state since that's what we're predicting)
    sample_data = df.drop(columns=['cognitive_state']).iloc[0].to_dict()
    
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
    Train and install a sample model of the specified type.
    
    Args:
        model_type (str): Type of model to create
        model_name_suffix (str): Suffix to add to the model name
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model_path, accuracy)
    """
    if random_state is None:
        random_state = RANDOM_SEED
    
    # Set up unique version for this model
    model_version = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}{model_name_suffix}"
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"{MODEL_NAME}_{model_version}.joblib")
    scaler_path = os.path.join(MODEL_OUTPUT_DIR, f"scaler_{model_version}.joblib")
    metadata_path = os.path.join(MODEL_OUTPUT_DIR, f"metadata_{model_version}.json")
    
    logger.info(f"Installing sample model of type: {model_type}")
    
    try:
        # Generate synthetic data
        df = generate_synthetic_data(random_state=random_state)
        
        # Create a scaler and fit it to the data
        scaler = StandardScaler()
        features = [col for col in df.columns if col not in ['cognitive_state', 'workload_intensity']]
        df[features] = scaler.fit_transform(df[features])
        
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
        joblib.dump(scaler, scaler_path)
        
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
            },
            "note": "This is a sample model trained on synthetic data for demonstration purposes."
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Sample model installed at {model_path} with accuracy {accuracy:.3f}")
        return model_path, accuracy
        
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
            model_path, accuracy = install_sample_model(model_type, random_state=random_seed)
            
            results.append({
                "type": model_type,
                "name": model_name,
                "accuracy": accuracy,
                "path": model_path
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
    parser = argparse.ArgumentParser(description='Cognitive Workload Assessment Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--model-type', '-m', type=str, default=DEFAULT_MODEL_TYPE,
                             choices=list(AVAILABLE_MODELS.keys()),
                             help=f'Type of model to train (one of: {", ".join(AVAILABLE_MODELS.keys())})')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--input', '-i', type=str, required=True, help='JSON file with input data')
    predict_parser.add_argument('--model', '-m', type=str, help='Path to model file')
    predict_parser.add_argument('--scaler', '-s', type=str, help='Path to scaler file')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available trained models')
    
    # Install models command
    install_parser = subparsers.add_parser('install-models', help='Install sample pre-trained models')
    
    # Handle case where no command is provided
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit(1, "Error: No command specified. Use 'train', 'predict', 'list-models', or 'install-models'.\n")
    
    return args

def find_latest_model():
    """Find the latest trained model in the models directory."""
    models_path = Path(MODEL_OUTPUT_DIR)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Look for model files
    model_files = list(models_path.glob(f"{MODEL_NAME}_*.joblib"))
    
    if not model_files:
        logger.error("No trained models found")
        return None, None
    
    # Sort by modification time (newest first)
    latest_model = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    # Extract version information safely
    try:
        # Get the timestamp part of the filename
        model_stem = latest_model.stem  # e.g., "Cognitive_State_Prediction_Model_20230101_120000"
        parts = model_stem.split('_')
        
        # The version should be the date and time parts at the end
        if len(parts) >= 5:  # Ensure there are enough parts
            # Join the last two parts which should be date and time
            model_version = f"{parts[-2]}_{parts[-1]}"
        else:
            # Fallback if naming convention is unexpected
            logger.warning(f"Unexpected model file naming convention: {latest_model.name}")
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    except Exception as e:
        logger.warning(f"Error extracting model version: {str(e)}")
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Look for corresponding scaler file
    scaler_file = models_path / f"scaler_{model_version}.joblib"
    
    if not scaler_file.exists():
        logger.error(f"Scaler file not found for model {latest_model}")
        return None, None
    
    logger.info(f"Found latest model: {latest_model}")
    logger.info(f"Found corresponding scaler: {scaler_file}")
    
    return str(latest_model), str(scaler_file)

def list_available_models():
    """List all available trained models."""
    models_path = Path(MODEL_OUTPUT_DIR)
    
    # Ensure models directory exists
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Find all model files
    model_files = list(models_path.glob(f"{MODEL_NAME}_*.joblib"))
    
    if not model_files:
        print("\n✘ No trained models found")
        print("\nTo train a new model, run:")
        print("  python cwt.py train")
        print("  python cwt.py train --model-type rf")
        logger.info("No trained models found")
        return
    
    print("\n✓ Available trained models:")
    print("-" * 80)
    print(f"{'Model Name':<50} {'Model Type':<15} {'Accuracy':<10}")
    print("-" * 80)
    
    for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        # Try to get model type and accuracy from metadata
        model_type = "Unknown"
        accuracy = "N/A"
        
        try:
            # Extract version from filename
            parts = model_file.stem.split('_')
            if len(parts) >= 5:
                model_version = f"{parts[-2]}_{parts[-1]}"
                metadata_file = models_path / f"metadata_{model_version}.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    model_type = metadata.get('model_name', "Unknown")
                    accuracy = f"{float(metadata.get('accuracy', 0)):.3f}"
        except Exception as e:
            logger.warning(f"Error reading metadata for {model_file.name}: {str(e)}")
        
        print(f"{model_file.name:<50} {model_type:<15} {accuracy:<10}")
    
    print("\nTo make predictions, run:")
    print("  python cwt.py predict --input YOUR_DATA.json")
    
    print("\nTo train a new model, run one of:")
    for i, (model_key, model_name) in enumerate(AVAILABLE_MODELS.items()):
        if i < 3:  # Only show a few examples to keep it simple
            print(f"  python cwt.py train --model-type {model_key}")
    print("  (see all options with: python cwt.py train --help)")

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
            model, accuracy, X_test, y_test, y_pred = train_model(df_processed, features, model_type)
            
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
        model_path = args.model
        scaler_path = args.scaler
        
        # If model or scaler paths are not provided, try to find the latest ones
        if not model_path or not scaler_path:
            model_path, scaler_path = find_latest_model()
            if not model_path or not scaler_path:
                logger.error("No trained models found and no model path provided")
                print("\n✘ No trained models found")
                print("\nPlease train a model first:")
                print("  python cwt.py train")
                sys.exit(1)
            logger.info(f"Using latest model: {model_path}")
            logger.info(f"Using latest scaler: {scaler_path}")
        
        try:
            # Verify that input file exists
            if not os.path.exists(args.input):
                logger.error(f"Input file not found: {args.input}")
                print(f"\n✘ ERROR: Input file not found: {args.input}")
                sys.exit(1)
            
            # Load input data
            with open(args.input, 'r') as f:
                input_data = json.load(f)
            
            # Make prediction
            prediction, probabilities = predict_new_data(model_path, scaler_path, input_data)
            
            # Output prediction
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
    
    elif args.command == 'list-models':
        list_available_models()
    
    elif args.command == 'install-models':
        install_sample_models()
    
    else:
        logger.error(f"Unknown command: {args.command}")
        logger.error("Use 'train', 'predict', 'list-models', or 'install-models'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
