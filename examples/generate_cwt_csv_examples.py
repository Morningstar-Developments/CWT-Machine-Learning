#!/usr/bin/env python3
"""
Generate CSV examples with different cognitive workload levels over time.

This script creates several CSV files that demonstrate different patterns
of cognitive workload (low, medium, high, transitions, fluctuations) over time.
These files can be used for testing, demonstration, and training.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# Create examples directory if it doesn't exist
os.makedirs('examples/csv_samples', exist_ok=True)

def generate_base_timestamps(duration_minutes=60, interval_seconds=15):
    """Generate a series of timestamps with given interval over a time period."""
    start_time = datetime(2025, 1, 1, 8, 0, 0)  # Start at 8:00 AM
    num_samples = int((duration_minutes * 60) / interval_seconds)
    timestamps = [start_time + timedelta(seconds=i*interval_seconds) for i in range(num_samples)]
    return timestamps

def generate_low_workload_data(timestamps):
    """Generate data representing consistently low cognitive workload."""
    num_samples = len(timestamps)
    
    data = {
        'timestamp': timestamps,
        'pulse_rate': np.random.normal(65, 3, num_samples),
        'blood_pressure_sys': np.random.normal(112, 4, num_samples),
        'resp_rate': np.random.normal(14, 0.8, num_samples),
        'pupil_diameter_left': np.random.normal(3.4, 0.2, num_samples),
        'pupil_diameter_right': np.random.normal(3.3, 0.2, num_samples),
        'fixation_duration': np.random.normal(320, 15, num_samples),
        'blink_rate': np.random.normal(18, 1.5, num_samples),
        'workload_intensity': np.random.normal(25, 5, num_samples),
        'gaze_x': np.random.normal(500, 20, num_samples),
        'gaze_y': np.random.normal(380, 15, num_samples),
        'alpha_power': np.random.normal(23, 2, num_samples),
        'theta_power': np.random.normal(15, 1.5, num_samples)
    }
    
    df = pd.DataFrame(data)
    df['alpha_theta_ratio'] = df['alpha_power'] / df['theta_power']
    df['skin_conductance'] = np.random.normal(5, 1, num_samples)
    
    return df

def generate_medium_workload_data(timestamps):
    """Generate data representing consistently medium cognitive workload."""
    num_samples = len(timestamps)
    
    data = {
        'timestamp': timestamps,
        'pulse_rate': np.random.normal(75, 3.5, num_samples),
        'blood_pressure_sys': np.random.normal(124, 4.5, num_samples),
        'resp_rate': np.random.normal(16.5, 1, num_samples),
        'pupil_diameter_left': np.random.normal(4.5, 0.2, num_samples),
        'pupil_diameter_right': np.random.normal(4.4, 0.2, num_samples),
        'fixation_duration': np.random.normal(250, 18, num_samples),
        'blink_rate': np.random.normal(14, 1.2, num_samples),
        'workload_intensity': np.random.normal(50, 8, num_samples),
        'gaze_x': np.random.normal(510, 25, num_samples),
        'gaze_y': np.random.normal(385, 20, num_samples),
        'alpha_power': np.random.normal(18, 1.8, num_samples),
        'theta_power': np.random.normal(20, 2, num_samples)
    }
    
    df = pd.DataFrame(data)
    df['alpha_theta_ratio'] = df['alpha_power'] / df['theta_power']
    df['skin_conductance'] = np.random.normal(8.5, 1.2, num_samples)
    
    return df

def generate_high_workload_data(timestamps):
    """Generate data representing consistently high cognitive workload."""
    num_samples = len(timestamps)
    
    data = {
        'timestamp': timestamps,
        'pulse_rate': np.random.normal(90, 4, num_samples),
        'blood_pressure_sys': np.random.normal(135, 5, num_samples),
        'resp_rate': np.random.normal(19, 1.2, num_samples),
        'pupil_diameter_left': np.random.normal(5.5, 0.3, num_samples),
        'pupil_diameter_right': np.random.normal(5.4, 0.3, num_samples),
        'fixation_duration': np.random.normal(185, 15, num_samples),
        'blink_rate': np.random.normal(10, 1, num_samples),
        'workload_intensity': np.random.normal(85, 7, num_samples),
        'gaze_x': np.random.normal(460, 30, num_samples),
        'gaze_y': np.random.normal(350, 25, num_samples),
        'alpha_power': np.random.normal(12, 1.5, num_samples),
        'theta_power': np.random.normal(26, 2.5, num_samples)
    }
    
    df = pd.DataFrame(data)
    df['alpha_theta_ratio'] = df['alpha_power'] / df['theta_power']
    df['skin_conductance'] = np.random.normal(13, 1.5, num_samples)
    
    return df

def generate_transition_data(timestamps):
    """Generate data representing a transition from low to high cognitive workload."""
    num_samples = len(timestamps)
    progress = np.linspace(0, 1, num_samples)  # Linear progression from 0 to 1
    
    # Create transition parameters for each metric
    def transition_values(low_mean, high_mean, low_std, high_std, progress):
        means = low_mean + (high_mean - low_mean) * progress
        stds = low_std + (high_std - low_std) * progress
        return np.array([np.random.normal(mean, std) for mean, std in zip(means, stds)])
    
    data = {
        'timestamp': timestamps,
        'pulse_rate': transition_values(65, 90, 3, 4, progress),
        'blood_pressure_sys': transition_values(112, 135, 4, 5, progress),
        'resp_rate': transition_values(14, 19, 0.8, 1.2, progress),
        'pupil_diameter_left': transition_values(3.4, 5.5, 0.2, 0.3, progress),
        'pupil_diameter_right': transition_values(3.3, 5.4, 0.2, 0.3, progress),
        'fixation_duration': transition_values(320, 185, 15, 15, progress),
        'blink_rate': transition_values(18, 10, 1.5, 1, progress),
        'workload_intensity': transition_values(25, 85, 5, 7, progress),
        'gaze_x': transition_values(500, 460, 20, 30, progress),
        'gaze_y': transition_values(380, 350, 15, 25, progress),
        'alpha_power': transition_values(23, 12, 2, 1.5, progress),
        'theta_power': transition_values(15, 26, 1.5, 2.5, progress)
    }
    
    df = pd.DataFrame(data)
    df['alpha_theta_ratio'] = df['alpha_power'] / df['theta_power']
    df['skin_conductance'] = transition_values(5, 13, 1, 1.5, progress)
    
    return df

def generate_fluctuating_data(timestamps):
    """Generate data representing fluctuating cognitive workload levels."""
    num_samples = len(timestamps)
    
    # Create a sine wave pattern to simulate fluctuations (0 to 1)
    fluctuation = 0.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, num_samples))
    
    # Create fluctuating parameters for each metric
    def fluctuate_values(low_mean, high_mean, low_std, high_std, fluctuation):
        means = low_mean + (high_mean - low_mean) * fluctuation
        stds = low_std + (high_std - low_std) * fluctuation
        return np.array([np.random.normal(mean, std) for mean, std in zip(means, stds)])
    
    data = {
        'timestamp': timestamps,
        'pulse_rate': fluctuate_values(65, 90, 3, 4, fluctuation),
        'blood_pressure_sys': fluctuate_values(112, 135, 4, 5, fluctuation),
        'resp_rate': fluctuate_values(14, 19, 0.8, 1.2, fluctuation),
        'pupil_diameter_left': fluctuate_values(3.4, 5.5, 0.2, 0.3, fluctuation),
        'pupil_diameter_right': fluctuate_values(3.3, 5.4, 0.2, 0.3, fluctuation),
        'fixation_duration': fluctuate_values(320, 185, 15, 15, fluctuation),
        'blink_rate': fluctuate_values(18, 10, 1.5, 1, fluctuation),
        'workload_intensity': fluctuate_values(25, 85, 5, 7, fluctuation),
        'gaze_x': fluctuate_values(500, 460, 20, 30, fluctuation),
        'gaze_y': fluctuate_values(380, 350, 15, 25, fluctuation),
        'alpha_power': fluctuate_values(23, 12, 2, 1.5, fluctuation),
        'theta_power': fluctuate_values(15, 26, 1.5, 2.5, fluctuation)
    }
    
    df = pd.DataFrame(data)
    df['alpha_theta_ratio'] = df['alpha_power'] / df['theta_power']
    df['skin_conductance'] = fluctuate_values(5, 13, 1, 1.5, fluctuation)
    
    return df

def generate_attention_pattern_data(timestamps):
    """Generate data representing periods of attention and distraction."""
    num_samples = len(timestamps)
    
    # Create a pattern with periods of high attention and distraction
    # Use a step function with some noise
    steps = np.zeros(num_samples)
    step_length = num_samples // 6  # 6 segments: high, low, high, low, high, low
    
    for i in range(6):
        start_idx = i * step_length
        end_idx = (i + 1) * step_length if i < 5 else num_samples
        steps[start_idx:end_idx] = 1 if i % 2 == 0 else 0
    
    # Add some noise to the transitions to make them more realistic
    noise = np.random.normal(0, 0.05, num_samples)
    pattern = steps + noise
    pattern = np.clip(pattern, 0, 1)  # Clip to 0-1 range
    
    # Create fluctuating parameters for each metric
    def pattern_values(low_mean, high_mean, low_std, high_std, pattern):
        means = low_mean + (high_mean - low_mean) * pattern
        stds = low_std + (high_std - low_std) * pattern
        return np.array([np.random.normal(mean, std) for mean, std in zip(means, stds)])
    
    data = {
        'timestamp': timestamps,
        'pulse_rate': pattern_values(70, 85, 3, 4, pattern),
        'blood_pressure_sys': pattern_values(115, 130, 4, 5, pattern),
        'resp_rate': pattern_values(15, 18, 0.8, 1.2, pattern),
        'pupil_diameter_left': pattern_values(3.8, 5.0, 0.2, 0.3, pattern),
        'pupil_diameter_right': pattern_values(3.7, 4.9, 0.2, 0.3, pattern),
        'fixation_duration': pattern_values(290, 210, 15, 15, pattern),
        'blink_rate': pattern_values(16, 12, 1.5, 1, pattern),
        'workload_intensity': pattern_values(35, 70, 5, 7, pattern),
        'gaze_x': pattern_values(490, 470, 20, 30, pattern),
        'gaze_y': pattern_values(375, 360, 15, 25, pattern),
        'alpha_power': pattern_values(20, 14, 2, 1.5, pattern),
        'theta_power': pattern_values(17, 23, 1.5, 2.5, pattern)
    }
    
    df = pd.DataFrame(data)
    df['alpha_theta_ratio'] = df['alpha_power'] / df['theta_power']
    df['skin_conductance'] = pattern_values(6, 11, 1, 1.5, pattern)
    
    return df

def main():
    """Generate and save all example CSV files."""
    print("Generating CWT CSV example files...")
    
    # Generate timestamps for a 1-hour period with readings every 15 seconds
    timestamps = generate_base_timestamps(duration_minutes=60, interval_seconds=15)
    
    # Generate datasets
    low_df = generate_low_workload_data(timestamps)
    med_df = generate_medium_workload_data(timestamps)
    high_df = generate_high_workload_data(timestamps)
    trans_df = generate_transition_data(timestamps)
    fluctuating_df = generate_fluctuating_data(timestamps)
    attention_df = generate_attention_pattern_data(timestamps)
    
    # Save datasets
    low_df.to_csv('examples/csv_samples/low_workload_time_series.csv', index=False)
    med_df.to_csv('examples/csv_samples/medium_workload_time_series.csv', index=False)
    high_df.to_csv('examples/csv_samples/high_workload_time_series.csv', index=False)
    trans_df.to_csv('examples/csv_samples/low_to_high_transition.csv', index=False)
    fluctuating_df.to_csv('examples/csv_samples/fluctuating_workload.csv', index=False)
    attention_df.to_csv('examples/csv_samples/attention_pattern.csv', index=False)
    
    print("CSV files generated successfully in examples/csv_samples/")
    print("Generated files:")
    print("  - low_workload_time_series.csv (Stable low workload)")
    print("  - medium_workload_time_series.csv (Stable medium workload)")
    print("  - high_workload_time_series.csv (Stable high workload)")
    print("  - low_to_high_transition.csv (Gradual transition from low to high workload)")
    print("  - fluctuating_workload.csv (Sinusoidal pattern of workload changes)")
    print("  - attention_pattern.csv (Step-like changes between focused and distracted states)")

if __name__ == "__main__":
    main() 