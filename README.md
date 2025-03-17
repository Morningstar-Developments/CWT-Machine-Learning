# Cognitive Workload Tracking (CWT) Learning Model

This project implements a machine learning pipeline for predicting cognitive workload states based on physiological, EEG, and gaze tracking data. The tool can be used to classify cognitive states as Low, Medium, or High.

## Features

- Multi-modal data integration (physiological, EEG, and gaze tracking)
- Robust data preprocessing and feature engineering
- Random Forest-based classification model
- Cross-validation for model reliability assessment
- Feature importance visualization
- Comprehensive logging
- Model versioning and metadata tracking
- Command-line interface for training and predictions

## Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/CWT-Learning_Model.git
cd CWT-Learning_Model
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Set up configuration:

```
cp .env.example .env
```

Edit the `.env` file to configure data paths and other settings.

## Project Structure

```
CWT-Learning_Model/
├── data/                  # Input data files
├── models/                # Saved models and metadata
├── logs/                  # Log files
├── cwt.py                 # Main application code
├── requirements.txt       # Project dependencies
├── .env.example           # Environment config template
└── README.md              # This file
```

## Usage

### Train a New Model

```
python cwt.py train
```

This will:

1. Load the data files specified in your config
2. Preprocess the data
3. Train a Random Forest classifier
4. Generate performance metrics and visualizations
5. Save the trained model and metadata

### Make Predictions

```
python cwt.py predict --input data/example_input.json
```

You can specify a specific model to use:

```
python cwt.py predict --input data/example_input.json --model models/your_model.joblib --scaler models/your_scaler.joblib
```

If no model is specified, the most recent model will be used.

### List Available Models

```
python cwt.py list-models
```

## Input Data Format

The prediction function expects a JSON file with the following structure:

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
  "theta_power": 22.6
}
```

## Data Requirements

The pipeline expects three CSV files:

- Physiological data with heart rate, blood pressure, etc.
- EEG data with brain wave metrics
- Gaze tracking data with eye movements and pupil measurements

All files must have a 'timestamp' column for synchronization.

## Customization

You can modify the `.env` file to customize:

- Data file paths
- Model output directory
- Logging configuration
- Training parameters

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
