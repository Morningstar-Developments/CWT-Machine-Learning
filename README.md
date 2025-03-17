# Cognitive Workload Assessment Tool (CWT)

This tool implements a machine learning pipeline for predicting cognitive workload states based on physiological, EEG, and gaze tracking data. The tool can be used to classify cognitive states as Low, Medium, or High.

## Features

- Multi-modal data integration (physiological, EEG, and gaze tracking)
- Support for multiple machine learning algorithms:
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
  - Neural Network (MLP)
  - K-Nearest Neighbors
  - Logistic Regression
- Automatic data preprocessing and feature standardization
- Model persistence and metadata tracking
- Visualization of model performance
- Pre-trained sample models for immediate use

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/CWT-Learning_Model.git
cd CWT-Learning_Model
```

2. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Install Sample Models

To quickly get started with pre-trained models (no real data needed):

```bash
python cwt.py install-models
```

This will create sample models trained on synthetic data and a sample input file for testing.

### Make Predictions

After installing sample models, you can make predictions:

```bash
python cwt.py predict --input data/sample_input.json
```

### List Available Models

To see all available trained models:

```bash
python cwt.py list-models
```

### Train a New Model

To train a new model with your own data:

```bash
python cwt.py train
```

Or specify a model type:

```bash
python cwt.py train --model-type rf  # Random Forest
```

## Data Format

The tool expects data in the following format:

1. Physiological data with heart rate, blood pressure, etc.
2. EEG data with brain wave measurements
3. Gaze tracking data with eye movement metrics

Each data file should include a timestamp column for synchronization.

Sample data can be generated using:

```bash
python generate_sample_data.py
```

## Configuration

You can customize the tool by modifying the `.env` file:

```
# Data files
PHYSIO_DATA_PATH=data/Enhanced_Workload_Clinical_Data.csv
EEG_DATA_PATH=data/000_EEG_Cluster_ANOVA_Results.csv
GAZE_DATA_PATH=data/008_01.csv

# Model configuration
MODEL_OUTPUT_DIR=models
MODEL_NAME=Cognitive_State_Prediction_Model

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/cwt.log

# Training parameters
TEST_SIZE=0.2
RANDOM_SEED=42

# Default model type
DEFAULT_MODEL_TYPE=rf
```

## Troubleshooting

If you encounter issues with data paths:

1. Ensure your data files are in the locations specified in `.env`
2. Or run `python cwt.py install-models` to use sample data and models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite it as:

```
@software{CognitiveWorkloadTool,
  author = {Your Name},
  title = {Cognitive Workload Assessment Tool},
  year = {2025},
  url = {https://github.com/yourusername/CWT-Learning_Model}
}
```

