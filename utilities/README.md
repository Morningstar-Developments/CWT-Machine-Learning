# CWT Utility Scripts

This directory contains utility scripts for the Cognitive Workload Tool (CWT). These scripts provide functionality for managing models, generating data, and maintaining the repository.

## Available Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `check_models.py` | Check model and scaler compatibility | `./check-models` or `python utilities/check_models.py --fix` |
| `generate_sample_data.py` | Generate sample training data | `./generate-data` or `python utilities/generate_sample_data.py` |
| `organize_outputs.py` | Organize models and logs | `./organize` or `python utilities/organize_outputs.py` |
| `download_advanced_models.py` | Download pre-trained models | `./download-models` or `python utilities/download_advanced_models.py --all` |
| `setup_links.py` | Create utility shortcuts | `python utilities/setup_links.py` |
| `test_imports.py` | Test Python imports | `python utilities/test_imports.py` |
| `cwt-help.sh` | Shell script for help system | `./utilities/cwt-help.sh` |
| `banghelp.py` | Python help script | `python utilities/banghelp.py` |

## Using the Shortcuts

For convenience, shortcuts for these scripts are available in the root directory. To set up or re-create these shortcuts, run:

```bash
python utilities/setup_links.py
```

This will create executable scripts in the project root for each utility.

## Script Details

### check_models.py

Checks all models in the models directory and verifies that each model has a corresponding scaler file. It can also fix issues by creating new scalers for models that are missing them.

Options:

- `--fix`: Automatically fix scaler issues
- `--model-dir MODEL_DIR`: Specify the directory containing models

### generate_sample_data.py

Generates synthetic data files for training and testing:

- Physiological data (heart rate, blood pressure, etc.)
- EEG data (brain wave measurements)
- Gaze tracking data (eye movement metrics)

The data is saved to the `data/` directory.

### organize_outputs.py

Organizes model and log files into the proper directory structure:

- Models are organized by type (rf, svm, gb, mlp, knn, lr)
- Logs are organized by operation type (general, training, prediction, installation)

### download_advanced_models.py

Downloads advanced pre-trained models for improved prediction accuracy.

Options:

- `--all`: Download all available advanced models
- `--model-type TYPE`: Download a specific model type (rf, svm, gb, mlp, knn, lr)

### setup_links.py

Creates executable shortcuts in the project root for commonly used utilities.

### test_imports.py

Tests Python imports to ensure all required packages are properly installed.
