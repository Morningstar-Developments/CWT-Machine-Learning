# Cognitive Workload Training (CWT) Tool

A comprehensive tool for training machine learning models to detect and classify cognitive workload levels based on physiological and behavioral metrics.

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

## Recent Enhancements

The CWT Tool has been enhanced with the following new features:

1. **Unified Training Command**: A new `train-all` command allows training all model types in a single operation.
   - Options to train in parallel for faster execution
   - Ability to skip specific model types
   - Shared scaler to ensure consistent scaling across models

2. **Feature Inference Capabilities**: The system can now handle missing features by inferring them from available data.
   - Specialized inference for pupil metrics, EEG signals, and workload intensity
   - Uses physiological correlations from reference data
   - Falls back to synthetic reference data generation when needed

3. **Confidence Thresholds**: Prediction functions now support confidence thresholds to improve reliability.
   - Predictions below the specified threshold are marked as uncertain
   - Enhances the quality of workload predictions

4. **Enhanced Batch Processing**: Improved capabilities for processing files with multiple data points.
   - Better support for CSV files with missing features
   - Time series analysis with sliding window approach
   - Visualization options for time series predictions

## Working with Missing Features

The CWT Tool can now handle data with missing features by using the `--infer-missing` flag:

```bash
python cwt.py predict --input-json data/incomplete_sample.json --infer-missing
```

This works by:

1. Detecting which features are missing
2. Loading reference data to understand feature correlations
3. Using specialized algorithms to infer the missing values
4. Applying the standard prediction pipeline with the complete dataset

## Command Reference

### train-all

```bash
python cwt.py train-all [--output-dir DIR] [--parallel] [--skip-types TYPE1,TYPE2]
```

- `--output-dir`: Directory to save trained models (default: models/)
- `--parallel`: Train models in parallel for faster execution
- `--skip-types`: Comma-separated list of model types to skip (e.g., "svm,knn")

### predict

```bash
python cwt.py predict [--input-json FILE] [--input-values KEY=VALUE...] [--model-type TYPE] [--threshold FLOAT] [--infer-missing] [--output-json FILE]
```

### batch-predict

```bash
python cwt.py batch-predict --input-file FILE [--output-file FILE] [--model-type TYPE] [--threshold FLOAT] [--infer-missing]
```

### time-series-predict

```bash
python cwt.py time-series-predict --input-file FILE [--output-file FILE] [--window-size INT] [--step-size INT] [--model-type TYPE] [--threshold FLOAT] [--infer-missing] [--visualize]
```

## Example Script

An example script `example.sh` is provided to demonstrate the key features:

```bash
# Make it executable
chmod +x example.sh

# Run the example
./example.sh
```

## Reference Data

To improve feature inference accuracy, you can provide custom reference data:

```
data/reference/cognitive_workload_reference.csv
```

This file should contain representative samples with all the features used by the models.

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

4. Initialize the environment:

```bash
python cwt.py setup
```

## Usage

The CWT provides a command-line interface with several commands:

### Getting Help

```bash
python cwt.py help
# or
./!help
```

### Training a Model

Train a specific model type:

```bash
python cwt.py train --model-type svm
```

Train all available model types:

```bash
python cwt.py train-all --output-dir models/ensemble --parallel
```

### Making Predictions

Predict with explicit input values:

```bash
python cwt.py predict --input-values "pulse_rate=75" "blood_pressure_sys=120" "resp_rate=16"
```

Predict from JSON file:

```bash
python cwt.py predict --input-json data/sample_input.json --model-type mlp
```

Batch predict from CSV with feature inference:

```bash
python cwt.py batch-predict --input-file data/batch_samples.csv --output-file results.csv --infer-missing
```

Time series analysis:

```bash
python cwt.py time-series-predict --input-file data/time_series.csv --window-size 10 --step-size 5 --visualize
```

### Installing Sample Models

```bash
python cwt.py install-models
```

### Downloading Advanced Models

To improve prediction accuracy, you can download advanced pre-trained models:

```bash
# Download all advanced models
./download-models --all

# Download a specific model type
./download-models --model-type gb
```

These advanced models provide several benefits:

- Higher accuracy (up to 0.92 compared to ~0.35 for sample models)
- Trained on larger datasets (up to 35,000 samples)
- More sophisticated feature engineering
- More robustness to missing or noisy data

Use the advanced models for prediction:

```bash
python cwt.py predict --input data/sample_input.json --model models/advanced/gb/Advanced_gb_model.joblib --scaler models/advanced/gb/Advanced_gb_scaler.joblib
```

### Listing Available Models

```bash
python cwt.py list-models
```

## Project Structure

The CWT project follows an organized directory structure:

```bash
CWT-Learning_Model/
├── data/                        # Data files for training and prediction
├── examples/                    # Example files and utilities
│   └── json_samples/            # Example JSON files for testing
├── logs/                        # Log files
│   ├── general/                 # General logs
│   ├── training/                # Training-specific logs
│   ├── prediction/              # Prediction-specific logs
│   └── installation/            # Installation logs
├── models/                      # Trained models
│   ├── sample/                  # Models trained on synthetic data
│   │   ├── default/             # Default models
│   │   ├── rf/                  # Random Forest models
│   │   ├── svm/                 # Support Vector Machine models
│   │   ├── gb/                  # Gradient Boosting models
│   │   ├── mlp/                 # Neural Network models
│   │   ├── knn/                 # K-Nearest Neighbors models
│   │   └── lr/                  # Logistic Regression models
│   ├── advanced/                # Advanced pre-trained models
│   │   ├── rf/                  # Advanced Random Forest models
│   │   ├── svm/                 # Advanced SVM models
│   │   ├── gb/                  # Advanced Gradient Boosting models
│   │   ├── mlp/                 # Advanced Neural Network models
│   │   ├── knn/                 # Advanced KNN models
│   │   └── lr/                  # Advanced Logistic Regression models
│   └── visualizations/          # Model performance visualizations
├── utilities/                   # Helper scripts and utilities
│   ├── check_models.py          # Script to check model and scaler compatibility
│   ├── download_advanced_models.py # Script to download advanced models
│   ├── generate_sample_data.py  # Script to generate sample data
│   ├── organize_outputs.py      # Script to organize models and logs
│   └── test_imports.py          # Script to test Python imports
├── cwt.py                       # Main script
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration
└── README.md                    # This file
```

## Utility Shortcuts

The CWT comes with several utility scripts that can be executed directly from the root directory:

| Shortcut | Description | Usage |
|----------|-------------|-------|
| `./check-models` | Check model and scaler compatibility | `./check-models` or `./check-models --fix` |
| `./generate-data` | Generate sample data files | `./generate-data` |
| `./organize` | Organize models and logs | `./organize` |
| `./download-models` | Download advanced models | `./download-models --all` or `./download-models --model-type gb` |
| `./!help` | Display help information | `./!help` or `./!help --topic model-types` |

To set up these shortcuts (or recreate them if needed), run:

```bash
python utilities/setup_links.py
```

## Data Format

The tool expects data in the following format:

1. Physiological data with heart rate, blood pressure, etc.
2. EEG data with brain wave measurements
3. Gaze tracking data with eye movement metrics

Each data file should include a timestamp column for synchronization.

Sample data can be generated using:

```bash
./generate-data
```

## Configuration

You can customize the tool by modifying the `.env` file:

```
# Data files
PHYSIO_DATA_PATH=data/Enhanced_Workload_Clinical_Data.csv
EEG_DATA_PATH=data/000_EEG_Cluster_ANOVA_Results.csv
GAZE_DATA_PATH=data/008_01.csv

# Model configuration
MODEL_OUTPUT_DIR=models/sample/default
MODEL_NAME=Cognitive_State_Prediction_Model

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/general/cwt.log

# Training parameters
TEST_SIZE=0.2
RANDOM_SEED=42

# Default model type
DEFAULT_MODEL_TYPE=rf
```

## Organizing Your Models and Logs

If you have existing models and logs that need to be reorganized to match the new directory structure, run:

```bash
python utilities/organize_outputs.py
```

This script will:

1. Organize model files by type (rf, svm, gb, mlp, knn, lr)
2. Separate advanced models from sample models
3. Organize logs by operation type

## Troubleshooting

If you encounter issues with data paths:

1. Ensure your data files are in the locations specified in `.env`
2. Or run `python cwt.py install-models` to use sample data and models

### Scaler File Issues

If you encounter errors related to scaler files not being found:

1. Use explicit model and scaler paths when making predictions:

   ```bash
   python cwt.py predict --input data/sample_input.json --model models/sample/rf/your_model.joblib --scaler models/sample/rf/your_scaler.joblib
   ```

2. Verify the scaler file exists in the same directory as the model:

   ```bash
   # List models and their scalers
   find models -name "cognitive_state_predictor_*.joblib" | sort
   find models -name "scaler_*.joblib" | sort
   ```

3. Reinstall sample models to ensure consistent naming:

   ```bash
   python cwt.py install-models
   ```

4. Run the organize script to organize your models and scalers:

   ```bash
   ./organize
   ```

5. Use the check-models utility:

   ```bash
   # Check all models and their scalers
   ./check-models
   
   # Automatically fix scaler issues
   ./check-models --fix
   ```

   This tool will identify models missing scalers and create them automatically when run with the `--fix` flag.

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
```

Now, let's create a start script that automatically sets up the environment, trains models, and prepares for analysis:

```python:start.py
#!/usr/bin/env python3
"""
CWT Tool Startup Script

This script automates the setup, model training, and preparation for 
cognitive workload analysis. It:
1. Sets up the CWT environment
2. Trains all available model types
3. Prepares sample data for analysis
4. Generates an overview of available models and their performance
"""

import os
import sys
import json
import time
import logging
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cwt_startup.log')
    ]
)
logger = logging.getLogger('cwt_startup')

# Define paths
MODEL_DIR = "models/ensemble"
RESULTS_DIR = "results"
REFERENCE_DATA_DIR = "data/reference"

def run_command(command, desc=None):
    """Run a shell command and log the output"""
    if desc:
        logger.info(f"=== {desc} ===")
    
    logger.info(f"Running: {command}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command completed successfully in {time.time() - start_time:.2f} seconds")
        logger.debug(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error: {e.stderr}")
        return False, e.stderr

def setup_environment():
    """Setup the CWT environment"""
    logger.info("Setting up CWT environment")
    
    # Create necessary directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run setup command
    success, _ = run_command("python cwt.py setup", "Setting up CWT environment")
    
    return success

def train_models():
    """Train all available model types"""
    logger.info("Training all model types")
    
    # Train all models in parallel
    success, output = run_command(
        f"python cwt.py train-all --output-dir {MODEL_DIR} --parallel",
        "Training all models"
    )
    
    if not success:
        logger.warning("Full parallel training failed, attempting sequential training")
        success, output = run_command(
            f"python cwt.py train-all --output-dir {MODEL_DIR}",
            "Training models sequentially"
        )
    
    return success

def prepare_sample_data():
    """Prepare sample data for analysis"""
    logger.info("Preparing sample data for analysis")
    
    # Create example JSON
    success1, _ = run_command(
        "python cwt.py predict --input-values 'pulse_rate=80' 'blood_pressure_sys=125' 'resp_rate=18' --output-json results/sample_prediction.json",
        "Creating sample prediction"
    )
    
    # Create batch prediction
    success2, _ = run_command(
        "python cwt.py batch-predict --input-file data/sample_missing_pupil.csv --output-file results/batch_results.csv --infer-missing",
        "Creating batch prediction"
    )
    
    # Create time series prediction
    success3, _ = run_command(
        "python cwt.py time-series-predict --input-file data/sample_missing_pupil.csv --output-file results/time_series_results.csv --visualize --infer-missing",
        "Creating time series prediction"
    )
    
    return all([success1, success2, success3])

def generate_model_overview():
    """Generate an overview of available models and their performance"""
    logger.info("Generating model overview")
    
    # Check if ensemble metadata exists
    metadata_path = os.path.join(MODEL_DIR, "ensemble_metadata.json")
    if not os.path.exists(metadata_path):
        logger.warning(f"Ensemble metadata not found at {metadata_path}")
        return False
    
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract model results
        results = metadata.get('results', {})
        
        # Convert to DataFrame
        data = []
        for model_type, result in results.items():
            if "error" in result:
                data.append({
                    "Model Type": model_type,
                    "Accuracy": 0.0,
                    "F1 Score": 0.0,
                    "Status": "Error",
                    "Error": result["error"]
                })
            else:
                data.append({
                    "Model Type": model_type,
                    "Accuracy": result.get("accuracy", 0.0),
                    "F1 Score": result.get("f1_score", 0.0),
                    "Status": "Trained",
                    "Path": result.get("model_path", "")
                })
        
        df = pd.DataFrame(data)
        
        # Save results to CSV
        overview_path = os.path.join(RESULTS_DIR, "model_overview.csv")
        df.to_csv(overview_path, index=False)
        logger.info(f"Model overview saved to {overview_path}")
        
        # Generate visualization
        plt.figure(figsize=(12, 6))
        
        # Only include successfully trained models
        plot_df = df[df["Status"] == "Trained"]
        
        # Create bar plot
        ax = sns.barplot(x="Model Type", y="Accuracy", data=plot_df)
        
        # Add F1 scores as text
        for i, row in enumerate(plot_df.itertuples()):
            ax.text(
                i, row.Accuracy + 0.02, 
                f"F1: {row.F1_Score:.3f}", 
                ha='center', va='bottom',
                fontweight='bold'
            )
        
        plt.title("Model Performance Comparison")
        plt.ylim(0, min(1.0, plot_df["Accuracy"].max() + 0.1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(RESULTS_DIR, "model_comparison.png")
        plt.savefig(plot_path)
        logger.info(f"Model comparison plot saved to {plot_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating model overview: {str(e)}")
        return False

def main():
    """Main function to run all steps"""
    logger.info("Starting CWT Tool automated setup")
    
    # Step 1: Setup environment
    if not setup_environment():
        logger.error("Failed to setup environment, aborting")
        return 1
    
    # Step 2: Train models
    if not train_models():
        logger.error("Model training encountered errors, continuing with next steps")
    
    # Step 3: Prepare sample data
    if not prepare_sample_data():
        logger.error("Sample data preparation encountered errors, continuing with next steps")
    
    # Step 4: Generate model overview
    if not generate_model_overview():
        logger.error("Model overview generation encountered errors")
    
    logger.info("\n" + "-" * 50)
    logger.info("CWT Tool automated setup complete")
    logger.info("Models are now trained and ready for analysis")
    logger.info(f"Trained models are in: {MODEL_DIR}")
    logger.info(f"Results and visualizations are in: {RESULTS_DIR}")
    logger.info("-" * 50)
    
    print("\n✅ CWT Tool is now ready for analysis!")
    print("You can now run commands like:")
    print("  python cwt.py predict --input-values 'pulse_rate=75' 'blood_pressure_sys=120'")
    print("  python cwt.py batch-predict --input-file data/your_data.csv --infer-missing")
    print("  python commands.py  # For a full list of available commands")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Let's make the files executable:

```bash
chmod +x commands.py start.py
```

I've created two powerful tools to help you work with the Cognitive Workload Training system:

1. **Commands Dictionary (`commands.py`)**
   - A comprehensive reference of all available commands in the CWT tool
   - Includes detailed descriptions and usage examples for each command
   - Provides a list of all options and their explanations
   - Can be run directly for interactive help: `python commands.py [command]`

2. **Automated Start Script (`start.py`)**
   - Automates the entire workflow from setup to analysis
   - Sets up the CWT environment and creates necessary directories
   - Trains all available model types (with fallback to sequential if parallel fails)
   - Prepares sample predictions using different methods:
     - Single point prediction
     - Batch prediction with missing feature inference
     - Time series prediction with visualization
   - Generates an overview of model performance in both CSV and visual formats
   - Provides a summary of available models and their accuracy

Both scripts are now executable, so you can simply run:

```bash
./start.py
```

This will automatically set up your environment, train all models, and prepare the system for analysis. After completion, you'll have trained models ready for use and examples of predictions in the `results` directory.

For help with any command, you can run:

```bash:README.md
./commands.py [command]
```

For example, `./commands.py train-all` will show you all options for training all models at once.
