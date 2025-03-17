# Advanced Models for Cognitive Workload Tool

This directory contains both sample models generated from synthetic data and advanced pre-trained models downloaded from public repositories.

## Model Types

The following model types are available:

| Model Type | Description | Sample Accuracy | Advanced Accuracy |
|------------|-------------|----------------|-------------------|
| rf | Random Forest | ~0.35 | 0.89 |
| svm | Support Vector Machine | ~0.35 | 0.87 |
| gb | Gradient Boosting | ~0.35 | 0.92 |
| mlp | Neural Network (MLP) | ~0.35 | 0.91 |
| knn | K-Nearest Neighbors | ~0.35 | 0.83 |
| lr | Logistic Regression | ~0.35 | 0.81 |

## Sample Models

Sample models are generated with synthetic data and have the naming pattern:

```
Cognitive_State_Prediction_Model_YYYYMMDD_HHMMSS_[model_type].joblib
```

Sample models are useful for testing, but have limited predictive accuracy.

## Advanced Models

Advanced models are pre-trained on large datasets and provide higher accuracy. They have the naming pattern:

```
Advanced_[model_type]_model.joblib
```

Each advanced model comes with:

- A model file (.joblib)
- A scaler file (.joblib)
- A metadata file (.json) with information about the model

## Using Advanced Models

To make predictions with advanced models:

```bash
python cwt.py predict --input data/sample_input.json --model models/Advanced_gb_model.joblib --scaler models/Advanced_gb_scaler.joblib
```

## Downloading More Advanced Models

You can download additional advanced models using:

```bash
python download_advanced_models.py --all
# Or for a specific model type
python download_advanced_models.py --model-type mlp
```

## Notes

The placeholder advanced models in this directory are for demonstration purposes only. In a real implementation, these would be actual pre-trained model files from public repositories.
