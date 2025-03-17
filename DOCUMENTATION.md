# Cognitive Workload Tool - API Documentation

This document provides detailed information about the API and usage of the Cognitive Workload Tool (CWT).

## Command Line Interface

The tool provides a command line interface with several commands for different operations.

### Training Models

```bash
python cwt.py train [OPTIONS]
```

**Options:**

- `--model-type TEXT`: Type of model to train. Options: rf (Random Forest), svm (Support Vector Machine), gb (Gradient Boosting), nn (Neural Network), knn (K-Nearest Neighbors), lr (Logistic Regression). Default: rf
- `--output-dir TEXT`: Directory to save model files. Default: models
- `--test-size FLOAT`: Proportion of data to use for testing. Default: 0.2
- `--random-state INTEGER`: Random seed for reproducibility. Default: 42
- `--cv INTEGER`: Number of cross-validation folds. Default: 5
- `--verbose / --no-verbose`: Print detailed information. Default: --verbose

**Example:**

```bash
# Train a gradient boosting model with 10-fold cross-validation
python cwt.py train --model-type gb --cv 10
```

### Making Predictions

```bash
python cwt.py predict [OPTIONS]
```

**Options:**

- `--input TEXT`: Path to input JSON file with feature values. [required]
- `--model TEXT`: Path to model file. Default: latest trained model
- `--scaler TEXT`: Path to scaler file. Default: scaler associated with the model
- `--output TEXT`: Path to save prediction results. Default: prediction_results.json

**Example:**

```bash
# Make prediction with specific model and output file
python cwt.py predict --input data/new_sample.json --model models/gb_model_20250615.joblib --output results/prediction_result.json
```

### Installing Sample Models

```bash
python cwt.py install-models [OPTIONS]
```

**Options:**

- `--force / --no-force`: Force reinstallation of sample models. Default: --no-force

**Example:**

```bash
# Force reinstallation of sample models
python cwt.py install-models --force
```

### Listing Available Models

```bash
python cwt.py list-models [OPTIONS]
```

**Options:**

- `--detailed / --no-detailed`: Show detailed information about each model. Default: --no-detailed

**Example:**

```bash
# List models with detailed information
python cwt.py list-models --detailed
```

## Python API

The tool can also be used as a Python library:

```python
from cwt import CognitiveWorkloadTool

# Initialize the tool
cwt = CognitiveWorkloadTool()

# Train a model
cwt.train(model_type='rf', test_size=0.2, random_state=42, cv=5)

# Make predictions
input_data = {
    "pulse_rate": 75.2,
    "blood_pressure_sys": 120.5,
    "resp_rate": 16.4,
    "pupil_diameter_left": 5.2,
    "pupil_diameter_right": 5.1,
    "fixation_duration": 245.8,
    "blink_rate": 12.3,
    "workload_intensity": 0.0,
    "gaze_x": 512.3,
    "gaze_y": 384.7,
    "alpha_power": 18.5,
    "theta_power": 22.6
}
prediction = cwt.predict(input_data)
print(f"Predicted cognitive workload: {prediction}")

# Get model information
model_info = cwt.get_model_info()
print(model_info)
```

## Data Format Reference

### Input Data Format for Training

The tool expects three CSV files with the following features:

#### Physiological Data

- `timestamp`: Time of measurement
- `subject_id`: Participant identifier
- `pulse_rate`: Heart rate in BPM
- `blood_pressure_sys`: Systolic blood pressure
- `blood_pressure_dia`: Diastolic blood pressure
- `resp_rate`: Respiratory rate
- `skin_conductance`: Galvanic skin response
- `cognitive_workload`: Target variable (Low, Medium, High)

#### EEG Data

- `timestamp`: Time of measurement
- `subject_id`: Participant identifier
- `alpha_power`: Alpha wave power
- `beta_power`: Beta wave power
- `theta_power`: Theta wave power
- `delta_power`: Delta wave power
- `gamma_power`: Gamma wave power
- `alpha_theta_ratio`: Ratio of alpha to theta power

#### Gaze Tracking Data

- `timestamp`: Time of measurement
- `subject_id`: Participant identifier
- `pupil_diameter_left`: Left pupil diameter in mm
- `pupil_diameter_right`: Right pupil diameter in mm
- `fixation_duration`: Average fixation duration in ms
- `saccade_amplitude`: Average saccade amplitude in degrees
- `blink_rate`: Blinks per minute
- `workload_intensity`: Task intensity measure
- `gaze_x`: X-coordinate of gaze
- `gaze_y`: Y-coordinate of gaze

### Input Data Format for Prediction

For prediction, the tool accepts a JSON file with the following structure:

```json
{
  "pulse_rate": 75.2,
  "blood_pressure_sys": 120.5,
  "resp_rate": 16.4,
  "pupil_diameter_left": 5.2,
  "pupil_diameter_right": 5.1,
  "fixation_duration": 245.8,
  "blink_rate": 12.3,
  "workload_intensity": 0.0,
  "gaze_x": 512.3,
  "gaze_y": 384.7,
  "alpha_power": 18.5,
  "theta_power": 22.6,
  "alpha_theta_ratio": 1.2,
  "skin_conductance": 8.7
}
```

Not all features are required for prediction. The model will use the available features and make a best effort prediction.

### Output Data Format

The prediction output is a JSON file with the following structure:

```json
{
  "prediction": "Medium",
  "confidence": {
    "Low": 0.15,
    "Medium": 0.75,
    "High": 0.10
  },
  "model_info": {
    "model_type": "RandomForest",
    "accuracy": 0.82,
    "trained_date": "2025-06-15 14:23:45",
    "features_used": ["pulse_rate", "blood_pressure_sys", "resp_rate", ...]
  },
  "timestamp": "2025-06-16 09:12:34"
}
```

## Error Handling

The tool provides detailed error messages for common issues:

- **Missing data files**: Suggests checking the .env file configuration
- **Invalid data format**: Provides information about the expected format
- **Missing features**: Lists required features that are missing
- **Model not found**: Suggests installing sample models

## Advanced Configuration

The tool's behavior can be customized through the `.env` file:

```bash
# Feature engineering options
ENABLE_FEATURE_SCALING=true
CREATE_INTERACTION_FEATURES=true
DROP_CORRELATED_FEATURES=true
CORRELATION_THRESHOLD=0.9

# Model hyperparameters
# Random Forest
RF_N_ESTIMATORS=100
RF_MAX_DEPTH=None
RF_MIN_SAMPLES_SPLIT=2

# Support Vector Machine
SVM_C=1.0
SVM_KERNEL=rbf
SVM_GAMMA=scale

# Gradient Boosting
GB_N_ESTIMATORS=100
GB_LEARNING_RATE=0.1
GB_MAX_DEPTH=3

# Neural Network
NN_HIDDEN_LAYER_SIZES=100,50
NN_ACTIVATION=relu
NN_SOLVER=adam
NN_MAX_ITER=500

# K-Nearest Neighbors
KNN_N_NEIGHBORS=5
KNN_WEIGHTS=uniform
KNN_ALGORITHM=auto

# Logistic Regression
LR_C=1.0
LR_PENALTY=l2
LR_SOLVER=lbfgs
LR_MAX_ITER=100

# Model hyperparameters
# Random Forest
RF_N_ESTIMATORS=100
RF_MAX_DEPTH=None
RF_MIN_SAMPLES_SPLIT=2

# Support Vector Machine
SVM_C=1.0
SVM_KERNEL=rbf
SVM_GAMMA=scale

# Gradient Boosting
GB_N_ESTIMATORS=100
GB_LEARNING_RATE=0.1
GB_MAX_DEPTH=3

# Neural Network
NN_HIDDEN_LAYER_SIZES=100,50
NN_ACTIVATION=relu
NN_SOLVER=adam
NN_MAX_ITER=500

# K-Nearest Neighbors
KNN_N_NEIGHBORS=5
KNN_WEIGHTS=uniform
KNN_ALGORITHM=auto

# Logistic Regression
LR_C=1.0
LR_PENALTY=l2
LR_SOLVER=lbfgs
LR_MAX_ITER=100
```

## Troubleshooting Common Issues

### Missing Features

If your prediction input is missing some features, the tool will try to make a prediction using available features, but accuracy may be reduced.

### Data Synchronization

When training, ensure that timestamps in different data sources are aligned. The tool will attempt to synchronize data based on timestamps, but manual verification is recommended.

### Out of Memory Errors

If you encounter memory issues with large datasets, try:

1. Reducing the number of features
2. Using a subset of the data
3. Setting `CREATE_INTERACTION_FEATURES=false` in the .env file

### Performance Optimization

For faster training and prediction:

1. Use simpler models (like Random Forest with fewer estimators)
2. Reduce the number of features by setting `DROP_CORRELATED_FEATURES=true`
3. Use a smaller test size and fewer CV folds
