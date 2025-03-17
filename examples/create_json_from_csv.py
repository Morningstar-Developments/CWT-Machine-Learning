#!/usr/bin/env python3
"""
Utility script to convert CSV data to JSON format for use with the CWT tool.

This script takes a CSV file containing physiological, EEG, and gaze data
and converts it to JSON files ready for prediction with the CWT tool.

Usage:
    python create_json_from_csv.py input.csv [--output output_dir] [--batch]
"""

import argparse
import csv
import json
import os
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Convert CSV data to JSON format for CWT predictions')
    parser.add_argument('input_csv', help='Input CSV file with physiological, EEG, and gaze data')
    parser.add_argument('--output', '-o', default='output_json', help='Output directory for JSON files')
    parser.add_argument('--batch', '-b', action='store_true', help='Create a single batch JSON file instead of individual files')
    parser.add_argument('--sample-id-col', default=None, help='Column name to use as sample ID')
    return parser.parse_args()

def csv_to_json(csv_file, output_dir, create_batch=False, id_column=None):
    """
    Convert CSV data to JSON format for CWT predictions.
    
    Args:
        csv_file: Path to input CSV file
        output_dir: Directory to save JSON files
        create_batch: Whether to create a single batch file or individual files
        id_column: Column to use as sample ID
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("Error: No data found in CSV file")
        return
    
    print(f"Read {len(rows)} rows from {csv_file}")
    
    # Required features for the model
    required_features = [
        "pulse_rate", "blood_pressure_sys", "resp_rate", 
        "pupil_diameter_left", "pupil_diameter_right",
        "fixation_duration", "blink_rate", "workload_intensity", 
        "gaze_x", "gaze_y", "alpha_power", "theta_power"
    ]
    
    # Optional features
    optional_features = ["alpha_theta_ratio", "skin_conductance"]
    
    # Check for required columns
    missing_features = [f for f in required_features if f not in rows[0]]
    if missing_features:
        print(f"Warning: Missing required features in CSV: {missing_features}")
        print("The model may not perform optimally without these features.")
    
    # Process rows
    json_rows = []
    for i, row in enumerate(rows):
        # Create a clean data dictionary with numeric values
        data = {}
        
        # Add sample ID if specified
        if id_column and id_column in row:
            data["sample_id"] = row[id_column]
        else:
            data["sample_id"] = f"S{i+1:03d}"
        
        # Add all features that exist in the CSV
        for feature in required_features + optional_features:
            if feature in row and row[feature]:
                try:
                    # Convert to appropriate numeric type
                    if '.' in row[feature]:
                        data[feature] = float(row[feature])
                    else:
                        data[feature] = int(row[feature])
                except ValueError:
                    print(f"Warning: Could not convert {feature}='{row[feature]}' to number, skipping")
        
        json_rows.append(data)
    
    # Create output files
    if create_batch:
        # Save as a single batch file
        batch_file = os.path.join(output_dir, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(batch_file, 'w') as f:
            json.dump(json_rows, f, indent=2)
        print(f"Created batch file with {len(json_rows)} samples: {batch_file}")
    else:
        # Save as individual files
        for data in json_rows:
            sample_id = data.get("sample_id", f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            file_name = os.path.join(output_dir, f"{sample_id}.json")
            with open(file_name, 'w') as f:
                json.dump(data, f, indent=2)
        print(f"Created {len(json_rows)} individual JSON files in {output_dir}")

def main():
    args = parse_args()
    csv_to_json(args.input_csv, args.output, args.batch, args.sample_id_col)
    
    print("\nTo use these files for prediction, run:")
    if args.batch:
        batch_file = Path(args.output) / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if batch_file.exists():
            print(f"python examples/predict_examples.py {batch_file}")
    else:
        print(f"python cwt.py predict --input {args.output}/[filename].json")

if __name__ == "__main__":
    main() 