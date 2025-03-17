# CWT Examples and Utilities Index

This document provides an overview of all the example files and utilities provided for the Cognitive Workload Tool (CWT).

## Example JSON Files

Located in the `json_samples` directory:

| Filename | Purpose | Description |
|----------|---------|-------------|
| `low_workload.json` | Prediction example | Data representing a subject with low cognitive workload |
| `medium_workload.json` | Prediction example | Data representing a subject with medium cognitive workload |
| `high_workload.json` | Prediction example | Data representing a subject with high cognitive workload |
| `incomplete_data.json` | Robustness testing | Sample with some missing features to test model robustness |
| `borderline_case.json` | Edge case testing | Data representing a borderline case between medium and high workload |
| `extreme_values.json` | Limit testing | Sample with extreme feature values to test model boundaries |
| `batch_samples.json` | Batch processing | Multiple samples in a single file for batch processing |

## Utility Scripts

| Filename | Purpose | Usage |
|----------|---------|-------|
| `predict_examples.py` | Demo script | `python examples/predict_examples.py` - Demonstrates running predictions on all example JSON files |
| `create_json_from_csv.py` | Data conversion | `python examples/create_json_from_csv.py examples/sample_data.csv` - Converts CSV data to JSON format |

## Example Data Files

| Filename | Purpose | Description |
|----------|---------|-------------|
| `sample_data.csv` | Conversion example | Sample CSV data with 10 subjects showing varying cognitive workload levels |

## Usage Examples

### Basic Prediction

```bash
# Predict with low workload example
python cwt.py predict --input examples/json_samples/low_workload.json

# Predict with high workload example
python cwt.py predict --input examples/json_samples/high_workload.json
```

### Converting CSV to JSON

```bash
# Convert CSV to individual JSON files
python examples/create_json_from_csv.py examples/sample_data.csv --output my_json_files

# Convert CSV to a single batch JSON file
python examples/create_json_from_csv.py examples/sample_data.csv --output my_json_files --batch
```

### Running All Examples

```bash
# Run all example predictions
python examples/predict_examples.py
```

## Feature Value Ranges

For reference, here are typical value ranges for the main features:

| Feature | Low Workload | Medium Workload | High Workload |
|---------|--------------|----------------|---------------|
| pulse_rate | 60-70 | 70-85 | 85-100+ |
| blood_pressure_sys | 100-120 | 120-130 | 130-150+ |
| resp_rate | 12-15 | 15-18 | 18-25 |
| pupil_diameter_left | 3-4 mm | 4-5 mm | 5-7 mm |
| fixation_duration | 300-400 ms | 220-300 ms | 100-220 ms |
| blink_rate | 16-22 bpm | 12-16 bpm | 5-12 bpm |
| alpha_power | 20-30 μV² | 15-20 μV² | 5-15 μV² |
| theta_power | 12-18 μV² | 18-23 μV² | 23-35 μV² |
| alpha_theta_ratio | 1.2-2.0 | 0.7-1.2 | 0.2-0.7 |

## Expected Prediction Outcomes

When running predictions on the example files:

- `low_workload.json` should predict "Low" cognitive workload
- `medium_workload.json` should predict "Medium" cognitive workload
- `high_workload.json` should predict "High" cognitive workload
- `borderline_case.json` may predict either "Medium" or "High" with close confidence values
- `extreme_values.json` should predict "High" with high confidence
- `incomplete_data.json` demonstrates the model's ability to handle missing features

## Next Steps

After exploring these examples, you can:

1. Create your own JSON files based on these templates
2. Convert your own CSV data using the provided utility
3. Integrate the CWT tool into your own applications
4. Train new models with your own datasets
