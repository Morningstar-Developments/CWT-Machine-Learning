#!/usr/bin/env python3
"""
Cognitive Workload Training (CWT) Tool - Commands Reference

This module provides a comprehensive dictionary of all available commands
and their usage examples for the CWT tool.
"""

COMMANDS = {
    "setup": {
        "description": "Set up the CWT environment and create necessary directories",
        "examples": [
            "python cwt.py setup"
        ],
        "options": {}
    },
    
    "train": {
        "description": "Train a specific cognitive workload model",
        "examples": [
            "python cwt.py train --model-type svm",
            "python cwt.py train --model-type random_forest --output-dir models/my_models"
        ],
        "options": {
            "--model-type": "Type of model to train (svm, random_forest, knn, mlp, gradient_boosting, decision_tree)",
            "--output-dir": "Directory to save the trained model"
        }
    },
    
    "train-all": {
        "description": "Train all available cognitive workload models",
        "examples": [
            "python cwt.py train-all",
            "python cwt.py train-all --parallel --output-dir models/ensemble",
            "python cwt.py train-all --skip-types svm,knn"
        ],
        "options": {
            "--output-dir": "Directory to save trained models (default: models/)",
            "--parallel": "Train models in parallel for faster execution",
            "--skip-types": "Comma-separated list of model types to skip"
        }
    },
    
    "predict": {
        "description": "Predict cognitive workload from input data",
        "examples": [
            "python cwt.py predict --input-json data/sample.json",
            "python cwt.py predict --input-values 'pulse_rate=75' 'blood_pressure_sys=120'",
            "python cwt.py predict --model-type mlp --infer-missing"
        ],
        "options": {
            "--input-json": "JSON file with input data",
            "--input-values": "Input values in key=value format",
            "--output-json": "JSON file to save prediction results",
            "--model-type": "Type of model to use for prediction",
            "--threshold": "Confidence threshold for prediction",
            "--infer-missing": "Infer missing features from available data"
        }
    },
    
    "batch-predict": {
        "description": "Batch predict cognitive workload from a CSV file",
        "examples": [
            "python cwt.py batch-predict --input-file data/batch_samples.csv",
            "python cwt.py batch-predict --input-file data/samples.csv --output-file results.csv --infer-missing"
        ],
        "options": {
            "--input-file": "Input CSV file with feature values",
            "--output-file": "Output CSV file to save prediction results",
            "--model-type": "Type of model to use for prediction",
            "--threshold": "Confidence threshold for prediction",
            "--infer-missing": "Infer missing features from available data"
        }
    },
    
    "time-series-predict": {
        "description": "Predict cognitive workload from time series data",
        "examples": [
            "python cwt.py time-series-predict --input-file data/time_series.csv",
            "python cwt.py time-series-predict --input-file data/series.csv --window-size 15 --step-size 5 --visualize"
        ],
        "options": {
            "--input-file": "Input CSV file with time series data",
            "--output-file": "Output CSV file to save prediction results",
            "--window-size": "Size of sliding window in samples (default: 10)",
            "--step-size": "Step size for sliding window (default: 5)",
            "--model-type": "Type of model to use for prediction",
            "--threshold": "Confidence threshold for prediction",
            "--visualize": "Visualize time series prediction results",
            "--infer-missing": "Infer missing features from available data"
        }
    }
}

def print_command_help(command=None):
    """
    Print help information for a specific command or all commands.
    
    Args:
        command (str, optional): Command to show help for. If None, shows all commands.
    """
    if command and command in COMMANDS:
        cmd_info = COMMANDS[command]
        print(f"\n=== {command} ===")
        print(f"{cmd_info['description']}")
        
        print("\nUsage examples:")
        for example in cmd_info['examples']:
            print(f"  {example}")
        
        if cmd_info['options']:
            print("\nOptions:")
            for option, desc in cmd_info['options'].items():
                print(f"  {option:<20} {desc}")
        
    else:
        print("\n=== CWT Tool Commands ===")
        for cmd, info in COMMANDS.items():
            print(f"{cmd:<20} {info['description']}")
        print("\nUse 'python commands.py COMMAND' to get detailed help for a specific command.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_command_help(sys.argv[1])
    else:
        print_command_help() 