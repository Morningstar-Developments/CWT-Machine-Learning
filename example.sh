#!/bin/bash

# Example script to demonstrate the Cognitive Workload Training (CWT) Tool

echo "=== Cognitive Workload Training (CWT) Tool Examples ==="
echo

# 1. Train all model types
echo "=== 1. Training all model types ==="
python cwt.py train-all --output-dir models/ensemble

# 2. Predict with missing pupil data
echo
echo "=== 2. Predicting cognitive workload with missing pupil data ==="
echo "    a) Without feature inference"
python cwt.py batch-predict --input-file data/sample_missing_pupil.csv --output-file results_without_inference.csv

echo
echo "    b) With feature inference"
python cwt.py batch-predict --input-file data/sample_missing_pupil.csv --output-file results_with_inference.csv --infer-missing

# 3. Time series prediction
echo
echo "=== 3. Time series prediction ==="
python cwt.py time-series-predict --input-file data/sample_missing_pupil.csv --output-file time_series_results.csv --window-size 5 --step-size 1 --infer-missing --visualize

echo
echo "=== Examples completed ==="
echo "Results are saved in:"
echo "  - results_without_inference.csv"
echo "  - results_with_inference.csv"
echo "  - time_series_results.csv"
echo "  - time_series_results.png (visualization)" 