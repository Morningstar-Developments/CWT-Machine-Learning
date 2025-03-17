#!/usr/bin/env python3
"""
Script to organize models and logs directories for the Cognitive Workload Tool.

This script:
1. Organizes model files by type (rf, svm, gb, mlp, knn, lr)
2. Separates advanced models from sample models
3. Organizes logs by operation type

Usage:
    python organize_outputs.py
"""

import os
import sys
import shutil
import re
import json
from pathlib import Path
from datetime import datetime

def ensure_dir_exists(directory):
    """Ensure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def organize_models():
    """Organize model files into the correct directories."""
    print("Organizing models...")
    
    # Create directory structure if it doesn't exist
    base_dirs = [
        'models/sample/default',
        'models/sample/rf',
        'models/sample/svm', 
        'models/sample/gb',
        'models/sample/mlp',
        'models/sample/knn',
        'models/sample/lr',
        'models/advanced/rf',
        'models/advanced/svm',
        'models/advanced/gb',
        'models/advanced/mlp',
        'models/advanced/knn',
        'models/advanced/lr',
        'models/visualizations'
    ]
    
    for directory in base_dirs:
        ensure_dir_exists(directory)
    
    # Pattern to identify advanced models
    advanced_pattern = re.compile(r"Advanced_([a-z]+)_")
    
    # Pattern to identify sample models with type suffix
    sample_pattern = re.compile(r".*_([a-z]+)\.joblib$")
    
    # Pattern for visualization files
    viz_pattern = re.compile(r".*\.(png|jpg|jpeg|svg|pdf)$")
    
    # Get all files in the models directory
    model_files = []
    for root, _, files in os.walk('models'):
        if any(segment in root for segment in ['sample', 'advanced', 'visualizations']):
            continue  # Skip our organized directories
        
        for file in files:
            if file == 'README.md':
                continue  # Skip README file
            file_path = os.path.join(root, file)
            model_files.append(file_path)
    
    # Track moved files
    moved_files = []
    
    # Organize files
    for file_path in model_files:
        file_name = os.path.basename(file_path)
        
        # Check if it's a visualization file
        if viz_pattern.match(file_name):
            dest_dir = 'models/visualizations'
            dest_path = os.path.join(dest_dir, file_name)
            if os.path.exists(file_path) and not os.path.exists(dest_path):
                shutil.copy2(file_path, dest_path)
                moved_files.append((file_path, dest_path))
            continue
        
        # Check if it's an advanced model
        advanced_match = advanced_pattern.match(file_name)
        if advanced_match:
            model_type = advanced_match.group(1)
            if model_type in ['rf', 'svm', 'gb', 'mlp', 'knn', 'lr']:
                dest_dir = f'models/advanced/{model_type}'
                dest_path = os.path.join(dest_dir, file_name)
                if os.path.exists(file_path) and not os.path.exists(dest_path):
                    shutil.copy2(file_path, dest_path)
                    moved_files.append((file_path, dest_path))
            continue
        
        # Check if it's a sample model with type suffix
        sample_match = sample_pattern.match(file_name)
        if sample_match:
            model_type = sample_match.group(1)
            if model_type in ['rf', 'svm', 'gb', 'mlp', 'knn', 'lr']:
                dest_dir = f'models/sample/{model_type}'
                dest_path = os.path.join(dest_dir, file_name)
                if os.path.exists(file_path) and not os.path.exists(dest_path):
                    shutil.copy2(file_path, dest_path)
                    moved_files.append((file_path, dest_path))
                
                # Also move associated metadata and scaler files
                base_name = file_name.replace('.joblib', '')
                
                # Check for metadata file
                meta_file = f"metadata_{base_name.split('_')[-1]}.json"
                meta_path = os.path.join(os.path.dirname(file_path), meta_file)
                if os.path.exists(meta_path):
                    meta_dest = os.path.join(dest_dir, meta_file)
                    if not os.path.exists(meta_dest):
                        shutil.copy2(meta_path, meta_dest)
                        moved_files.append((meta_path, meta_dest))
                
                # Check for scaler file
                scaler_file = f"scaler_{base_name.split('_')[-1]}.joblib"
                scaler_path = os.path.join(os.path.dirname(file_path), scaler_file)
                if os.path.exists(scaler_path):
                    scaler_dest = os.path.join(dest_dir, scaler_file)
                    if not os.path.exists(scaler_dest):
                        shutil.copy2(scaler_path, scaler_dest)
                        moved_files.append((scaler_path, scaler_dest))
            continue
        
        # If it's not caught by any pattern, it's a default model
        dest_dir = 'models/sample/default'
        dest_path = os.path.join(dest_dir, file_name)
        if os.path.exists(file_path) and not os.path.exists(dest_path):
            shutil.copy2(file_path, dest_path)
            moved_files.append((file_path, dest_path))
    
    # Report results
    print(f"Organized {len(moved_files)} model files.")

def organize_logs():
    """Organize log files into the correct directories."""
    print("Organizing logs...")
    
    # Create directory structure if it doesn't exist
    log_dirs = [
        'logs/training',
        'logs/prediction',
        'logs/installation',
        'logs/general'
    ]
    
    for directory in log_dirs:
        ensure_dir_exists(directory)
    
    # Copy the main log file to the general logs directory
    main_log = 'logs/cwt.log'
    if os.path.exists(main_log):
        dest_path = os.path.join('logs/general', os.path.basename(main_log))
        if not os.path.exists(dest_path):
            shutil.copy2(main_log, dest_path)
            print(f"Copied main log to {dest_path}")
    
    # Create a new log file organization structure
    log_types = {
        'training': ['train', 'cv', 'cross-validation', 'model'],
        'prediction': ['predict', 'inference', 'confidence'],
        'installation': ['install', 'download', 'setup']
    }
    
    # Create specialized log files (only if they don't exist)
    timestamp = datetime.now().strftime('%Y%m%d')
    
    for log_type, keywords in log_types.items():
        log_file = f'logs/{log_type}/{log_type}_{timestamp}.log'
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write(f"# {log_type.capitalize()} log created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# This log file was created by the organize_outputs.py script\n\n")
    
    print("Log organization complete.")

def create_metadata():
    """Create metadata files for the directory organization."""
    print("Creating README files...")
    
    # Create README files for each directory
    readme_content = {
        'models/README.md': """# Models Directory

This directory contains all the trained models for the Cognitive Workload Tool.

## Directory Structure

- `sample/`: Models trained on synthetic or sample data
  - `default/`: Default models not tied to a specific algorithm
  - `rf/`: Random Forest models
  - `svm/`: Support Vector Machine models
  - `gb/`: Gradient Boosting models
  - `mlp/`: Neural Network (MLP) models
  - `knn/`: K-Nearest Neighbors models
  - `lr/`: Logistic Regression models

- `advanced/`: Pre-trained advanced models downloaded from public repositories
  - `rf/`: Advanced Random Forest models
  - `svm/`: Advanced Support Vector Machine models
  - `gb/`: Advanced Gradient Boosting models
  - `mlp/`: Advanced Neural Network (MLP) models
  - `knn/`: Advanced K-Nearest Neighbors models
  - `lr/`: Advanced Logistic Regression models

- `visualizations/`: Plots and visualizations related to model performance

## Files in Each Model Directory

Each model directory typically contains:
1. Model files (*.joblib)
2. Scaler files (scaler_*.joblib)
3. Metadata files (metadata_*.json)
""",
        'logs/README.md': """# Logs Directory

This directory contains all logs for the Cognitive Workload Tool.

## Directory Structure

- `general/`: General logs for all operations
- `training/`: Logs specific to model training
- `prediction/`: Logs specific to making predictions
- `installation/`: Logs specific to installation operations

## Log Format

Each log file follows the format: `[operation]_YYYYMMDD.log`
"""
    }
    
    for file_path, content in readme_content.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(content)
    
    print("README files created.")

def main():
    """Main entry point for the script."""
    print("=" * 50)
    print("ORGANIZING MODELS AND LOGS")
    print("=" * 50)
    
    organize_models()
    organize_logs()
    create_metadata()
    
    print("\n" + "=" * 50)
    print("ORGANIZATION COMPLETE")
    print("=" * 50)
    print("\nNew directory structure:")
    print("- models/")
    print("  ├── sample/")
    print("  │   ├── default/")
    print("  │   ├── rf/")
    print("  │   ├── svm/")
    print("  │   ├── gb/")
    print("  │   ├── mlp/")
    print("  │   ├── knn/")
    print("  │   └── lr/")
    print("  ├── advanced/")
    print("  │   ├── rf/")
    print("  │   ├── svm/")
    print("  │   ├── gb/")
    print("  │   ├── mlp/")
    print("  │   ├── knn/")
    print("  │   └── lr/")
    print("  └── visualizations/")
    print("- logs/")
    print("  ├── general/")
    print("  ├── training/")
    print("  ├── prediction/")
    print("  └── installation/")
    
    print("\nTo modify the default paths, update the .env file:")
    print("- MODEL_OUTPUT_DIR=models/sample/default")
    print("- LOG_FILE=logs/general/cwt.log")
    
    print("\nTo use a specific model type, use the --model-type argument:")
    print("  python cwt.py train --model-type rf")
    print("=" * 50)

if __name__ == "__main__":
    main() 