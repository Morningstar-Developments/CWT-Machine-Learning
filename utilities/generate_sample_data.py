#!/usr/bin/env python
"""
Generate sample data files for the cognitive workload training pipeline.
This script creates synthetic data in the expected format for the
physiological, EEG, and gaze data files.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Generate physiological data
def generate_physiological_data(num_samples=500):
    """Generate synthetic physiological data."""
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=1)
    timestamps = [start_time + timedelta(seconds=i*5) for i in range(num_samples)]
    
    # Generate synthetic data
    data = {
        'timestamp': timestamps,
        'subject_id': np.random.randint(1, 10, num_samples),
        'heart_rate': np.random.normal(75, 15, num_samples),
        'blood_pressure_systolic': np.random.normal(120, 10, num_samples),
        'blood_pressure_diastolic': np.random.normal(80, 8, num_samples),
        'skin_conductance': np.random.normal(5, 2, num_samples),
        'respiration_rate': np.random.normal(16, 3, num_samples),
        'cognitive_load': np.random.choice(['Low', 'Medium', 'High'], num_samples),
        'pulse_rate': np.random.normal(75, 15, num_samples),
        'blood_pressure_sys': np.random.normal(120, 10, num_samples),
        'resp_rate': np.random.normal(16, 3, num_samples),
        'workload_intensity': np.random.normal(50, 20, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save the data
    file_path = 'data/Enhanced_Workload_Clinical_Data.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated physiological data: {file_path}")
    return df

# Generate EEG data
def generate_eeg_data(num_samples=500):
    """Generate synthetic EEG data."""
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=1)
    timestamps = [start_time + timedelta(seconds=i*5) for i in range(num_samples)]
    
    # Generate synthetic data
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    data = {
        'timestamp': timestamps,
        'subject_id': np.random.randint(1, 10, num_samples),
        'alpha_power': np.random.normal(10, 3, num_samples),
        'theta_power': np.random.normal(15, 4, num_samples),
    }
    
    # Add EEG channel data
    for channel in channels:
        data[f'{channel}_alpha'] = np.random.normal(10, 3, num_samples)
        data[f'{channel}_beta'] = np.random.normal(15, 4, num_samples)
        data[f'{channel}_gamma'] = np.random.normal(20, 5, num_samples)
        data[f'{channel}_delta'] = np.random.normal(25, 6, num_samples)
        data[f'{channel}_theta'] = np.random.normal(30, 7, num_samples)
    
    # Add cognitive load
    data['cognitive_load'] = np.random.choice(['Low', 'Medium', 'High'], num_samples)
    
    df = pd.DataFrame(data)
    
    # Save the data
    file_path = 'data/000_EEG_Cluster_ANOVA_Results.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated EEG data: {file_path}")
    return df

# Generate gaze data
def generate_gaze_data(num_samples=500):
    """Generate synthetic gaze data."""
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=1)
    timestamps = [start_time + timedelta(seconds=i*5) for i in range(num_samples)]
    
    # Generate synthetic data
    data = {
        'timestamp': timestamps,
        'subject_id': np.random.randint(1, 10, num_samples),
        'pupil_left': np.random.normal(3, 0.5, num_samples),
        'pupil_right': np.random.normal(3, 0.5, num_samples),
        'gaze_x': np.random.normal(0.5, 0.2, num_samples),
        'gaze_y': np.random.normal(0.5, 0.2, num_samples),
        'fixation_duration': np.random.normal(200, 50, num_samples),
        'saccade_length': np.random.normal(5, 1, num_samples),
        'blink_rate': np.random.normal(15, 5, num_samples),
        'cognitive_load': np.random.choice(['Low', 'Medium', 'High'], num_samples),
        'pupil_diameter_left': np.random.normal(3, 0.5, num_samples),
        'pupil_diameter_right': np.random.normal(3, 0.5, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Save the data
    file_path = 'data/008_01.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated gaze data: {file_path}")
    return df

if __name__ == "__main__":
    print("\n==================================================")
    print("Generating sample data for cognitive workload training...")
    print("==================================================")
    
    # Generate data
    physio_df = generate_physiological_data()
    eeg_df = generate_eeg_data()
    gaze_df = generate_gaze_data()
    
    print("\n==================================================")
    print("✓ Sample data generation complete")
    print("==================================================")
    
    print(f"Physiological data: {physio_df.shape[0]} samples with {physio_df.shape[1]} features")
    print(f"EEG data: {eeg_df.shape[0]} samples with {eeg_df.shape[1]} features")
    print(f"Gaze data: {gaze_df.shape[0]} samples with {gaze_df.shape[1]} features")
    
    print("\nTo train using this sample data, run:")
    print("  python cwt.py train")
    print("==================================================") 