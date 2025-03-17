# Cognitive Workload Tool - Example Files

This directory contains example files for the Cognitive Workload Tool (CWT), demonstrating how to format input data and use the tool for predictions.

## JSON Sample Files

The `json_samples` directory contains example JSON files representing different cognitive workload states:

- **low_workload.json**: Example of a subject with low cognitive workload
- **medium_workload.json**: Example of a subject with medium cognitive workload
- **high_workload.json**: Example of a subject with high cognitive workload
- **incomplete_data.json**: Example with some missing features to demonstrate robustness
- **borderline_case.json**: Example of a borderline case between medium and high workload
- **extreme_values.json**: Example with extreme values to test model robustness
- **batch_samples.json**: Example of multiple samples for batch processing

## Feature Descriptions

Each JSON file contains physiological, EEG, and gaze-tracking features:

| Feature | Description | Typical Range | Unit |
|---------|-------------|---------------|------|
| pulse_rate | Heart rate | 60-100 | beats per minute |
| blood_pressure_sys | Systolic blood pressure | 90-140 | mmHg |
| resp_rate | Respiratory rate | 12-20 | breaths per minute |
| pupil_diameter_left | Left pupil diameter | 2-8 | mm |
| pupil_diameter_right | Right pupil diameter | 2-8 | mm |
| fixation_duration | Average fixation duration | 150-500 | ms |
| blink_rate | Eye blink frequency | 8-21 | blinks per minute |
| workload_intensity | Task difficulty metric | 0-100 | arbitrary units |
| gaze_x | Horizontal gaze position | varies | pixels |
| gaze_y | Vertical gaze position | varies | pixels |
| alpha_power | Alpha wave power in EEG | 5-25 | μV² |
| theta_power | Theta wave power in EEG | 5-30 | μV² |
| alpha_theta_ratio | Ratio of alpha to theta power | 0.1-2.0 | ratio |
| skin_conductance | Galvanic skin response | 2-20 | μS |

## How to Use These Examples

### Running Predictions

You can use these examples to test the CWT prediction functionality:

```bash
# Predict with a specific example
python cwt.py predict --input examples/json_samples/low_workload.json

# Predict with the incomplete data example
python cwt.py predict --input examples/json_samples/incomplete_data.json
```

### Running All Examples

A helper script is provided to run predictions on all examples:

```bash
# Make sure you have sample models installed
python cwt.py install-models

# Run all examples
python examples/predict_examples.py
```

This will:
1. Process each example JSON file
2. Run predictions using the CWT tool
3. Display the results

### Understanding the Results

When you run predictions, the tool will output:
- The predicted cognitive workload state (Low, Medium, High)
- Confidence scores for each possible state
- The model used for prediction

### Batch Processing

For batch processing, you can use the batch_samples.json file structure as a template. The predict_examples.py script demonstrates how to process batches of samples.

## Creating Your Own Input Files

To create your own input files:

1. Use these examples as templates
2. Include at minimum the core features (pulse_rate, blood_pressure_sys, pupil_diameter_left, etc.)
3. Save as JSON format
4. Run predictions with `python cwt.py predict --input your_file.json`

## Feature Importance

The most important features for prediction, based on the default Random Forest model, are typically:
- workload_intensity
- theta_power
- alpha_power
- pupil_diameter_left
- pupil_diameter_right

Including accurate values for these features is particularly important for accurate predictions. 