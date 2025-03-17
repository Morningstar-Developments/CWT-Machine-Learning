# Advanced Models for Cognitive Workload Tool

This directory contains backup models for the Cognitive Workload Training (CWT) Tool, including both sample models generated from synthetic data and advanced pre-trained models downloaded from public repositories.

## Model Types and Selection Strategy

The CWT tool automatically selects the most appropriate model based on workload intensity and available features:

| Model Type | Description | Best For | Sample Accuracy | Advanced Accuracy |
|------------|-------------|----------|----------------|-------------------|
| rf | Random Forest | High workload (>60) | ~0.35 | 0.89 |
| mlp | Neural Network (MLP) | Medium workload (40-60) | ~0.35 | 0.91 |
| lr | Logistic Regression | Low workload (<40) | ~0.35 | 0.81 |
| svm | Support Vector Machine | General purpose | ~0.35 | 0.87 |
| gb | Gradient Boosting | High precision needs | ~0.35 | 0.92 |
| knn | K-Nearest Neighbors | Sparse data scenarios | ~0.35 | 0.83 |
| xgb | XGBoost | Complex feature interactions | ~0.38 | 0.94 |
| ensemble | Ensemble Methods | Critical applications | ~0.40 | 0.95 |

## Feature Requirements

For optimal performance, the following features are recommended:

- Physiological: pulse_rate, blood_pressure_sys, resp_rate
- Eye-tracking: pupil_diameter_left, pupil_diameter_right, fixation_duration, blink_rate
- EEG: alpha_power, theta_power (alpha_theta_ratio is calculated automatically)
- Additional: workload_intensity, gaze_x, gaze_y

The system can infer missing features when using the `--infer-missing` flag.

## Model Fallback Strategy

If the preferred model for a workload level is unavailable, the system follows this fallback sequence:

1. Try the next best model for the specific workload level
2. Fall back to Random Forest (general purpose) which provides good overall performance
3. Use the most recently trained model of any type (based on timestamp)
4. Use sample models if no trained models exist

The fallback strategy ensures continuous operation even when optimal models are unavailable. This approach balances accuracy with system reliability.

## Time Series Processing

For time series data, the system:

- Uses sliding windows (configurable size) to process sequential data
- Aggregates features within each window (typically using mean values)
- Applies appropriate models to each window
- Returns predictions with timestamps for trend analysis

## Sample Models

Sample models are generated with synthetic data and have the naming pattern:
