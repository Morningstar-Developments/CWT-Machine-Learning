#!/usr/bin/env python3
"""
Utility script to check and verify model and scaler files for the CWT tool.

This script helps diagnose issues with model and scaler compatibility by:
1. Listing all available models and their corresponding scalers
2. Verifying that each model has a valid scaler file
3. Providing suggestions for fixing scaler issues

Usage:
    python check_models.py [--fix]
"""

import os
import sys
import argparse
from pathlib import Path
import joblib
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('check_models')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check and verify model and scaler files')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix scaler issues')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing model files')
    return parser.parse_args()

def get_all_models(model_dir):
    """Find all model files in the specified directory."""
    models = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.joblib') and not file.startswith('scaler_'):
                model_path = os.path.join(root, file)
                models.append(model_path)
    return models

def get_corresponding_scaler(model_path):
    """
    Find the corresponding scaler for a model file.
    Uses similar logic to the find_latest_model function in cwt.py.
    """
    model_file = Path(model_path)
    model_dir = model_file.parent
    model_basename = model_file.stem
    model_parts = model_basename.split('_')
    
    # Get model type (last part)
    if len(model_parts) >= 3:
        model_type = model_parts[-1]
        
        # Extract timestamp parts
        timestamp_parts = []
        for part in model_parts:
            if part.isdigit() or (len(part) > 0 and part[0].isdigit()):
                timestamp_parts.append(part)
        
        if timestamp_parts:
            # Try different scaler naming patterns
            full_timestamp = "_".join(timestamp_parts)
            date_part = timestamp_parts[0] if timestamp_parts else ""
            
            # Check various scaler naming patterns
            scaler_patterns = [
                model_dir / f"scaler_{full_timestamp}_{model_type}.joblib",
                model_dir / f"scaler_{date_part}_{model_type}.joblib",
                model_dir / f"scaler_{model_type}.joblib",
                Path(model_dir.parent) / f"scaler_{full_timestamp}_{model_type}.joblib",
                Path(model_dir.parent) / f"scaler_{date_part}_{model_type}.joblib"
            ]
            
            for scaler_path in scaler_patterns:
                if scaler_path.exists():
                    return str(scaler_path)
    
    return None

def create_new_scaler(model_path, fix=False):
    """
    Create a new scaler for a model if one doesn't exist.
    Only actually creates it if fix=True, otherwise just reports.
    """
    try:
        model_file = Path(model_path)
        model_dir = model_file.parent
        model_basename = model_file.stem
        model_parts = model_basename.split('_')
        
        if len(model_parts) >= 3:
            model_type = model_parts[-1]
            
            # Extract timestamp parts
            timestamp_parts = []
            for part in model_parts:
                if part.isdigit() or (len(part) > 0 and part[0].isdigit()):
                    timestamp_parts.append(part)
            
            if timestamp_parts:
                full_timestamp = "_".join(timestamp_parts)
                new_scaler_path = model_dir / f"scaler_{full_timestamp}_{model_type}.joblib"
                
                if fix:
                    # Create a simple StandardScaler
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    
                    # Try to extract feature names from model if possible
                    try:
                        model = joblib.load(model_path)
                        if hasattr(model, 'feature_names_in_'):
                            scaler.feature_names_in_ = model.feature_names_in_
                    except Exception as e:
                        logger.warning(f"Could not extract feature names from model: {e}")
                    
                    # Save the scaler
                    joblib.dump(scaler, new_scaler_path)
                    logger.info(f"Created new scaler at {new_scaler_path}")
                    return str(new_scaler_path)
                else:
                    logger.info(f"Would create new scaler at {new_scaler_path} (use --fix to actually create it)")
                    return None
    except Exception as e:
        logger.error(f"Error creating scaler for {model_path}: {e}")
        return None

def check_model_scaler_compatibility(model_path, scaler_path):
    """Check if a model and scaler are compatible."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        model_features = getattr(model, 'feature_names_in_', None)
        scaler_features = getattr(scaler, 'feature_names_in_', None)
        
        if model_features is not None and scaler_features is not None:
            # Check if feature sets are the same
            model_set = set(model_features)
            scaler_set = set(scaler_features)
            
            if model_set == scaler_set:
                return True, "Model and scaler have matching features"
            else:
                missing_in_scaler = model_set - scaler_set
                missing_in_model = scaler_set - model_set
                
                issues = []
                if missing_in_scaler:
                    issues.append(f"Features in model but not in scaler: {missing_in_scaler}")
                if missing_in_model:
                    issues.append(f"Features in scaler but not in model: {missing_in_model}")
                
                return False, "; ".join(issues)
        else:
            if model_features is None:
                return None, "Model doesn't have feature_names_in_ attribute"
            else:
                return None, "Scaler doesn't have feature_names_in_ attribute"
    except Exception as e:
        return False, f"Error checking compatibility: {str(e)}"

def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("CHECKING MODELS AND SCALERS")
    print("=" * 80)
    
    # Get all models
    models = get_all_models(args.model_dir)
    if not models:
        print("\nNo model files found.")
        return
    
    print(f"\nFound {len(models)} model files.")
    
    issues_found = 0
    fixed_issues = 0
    
    for model_path in sorted(models):
        print(f"\nModel: {model_path}")
        
        # Find corresponding scaler
        scaler_path = get_corresponding_scaler(model_path)
        
        if scaler_path:
            print(f"  Scaler: {scaler_path}")
            
            # Check compatibility
            compatible, message = check_model_scaler_compatibility(model_path, scaler_path)
            if compatible is True:
                print(f"  Status: ✓ Compatible - {message}")
            elif compatible is False:
                print(f"  Status: ✗ Incompatible - {message}")
                issues_found += 1
            else:
                print(f"  Status: ? Unknown compatibility - {message}")
        else:
            print("  Scaler: ✗ Not found")
            issues_found += 1
            
            # Try to fix by creating a new scaler
            if args.fix:
                new_scaler = create_new_scaler(model_path, fix=True)
                if new_scaler:
                    print(f"  Fixed: ✓ Created new scaler at {new_scaler}")
                    fixed_issues += 1
                else:
                    print("  Fixed: ✗ Could not create scaler")
            else:
                create_new_scaler(model_path, fix=False)
    
    print("\n" + "=" * 80)
    if issues_found == 0:
        print("✓ All models have compatible scalers!")
    else:
        if args.fix:
            print(f"Found {issues_found} issues, fixed {fixed_issues}.")
            if issues_found > fixed_issues:
                print(f"There are still {issues_found - fixed_issues} issues to resolve.")
        else:
            print(f"Found {issues_found} issues. Run with --fix to attempt automatic fixes.")
    print("=" * 80)

if __name__ == "__main__":
    main() 