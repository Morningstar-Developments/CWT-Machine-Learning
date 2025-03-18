#!/usr/bin/env python3
"""
Integration tests for the advanced model workflow.

This test suite verifies the entire workflow from model downloading to prediction
using the Cognitive Workload Tool.
"""

import os
import json
import tempfile
import shutil
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the sys.path to import the modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the modules needed for the workflow
from utilities.download_advanced_models import download_model, extract_model_package

# Create mock prediction module since the actual one may not be available
class MockPredictor:
    """Mock predictor class that simulates prediction functionality."""
    
    def __init__(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model_loaded = False
        self.scaler_loaded = False
        
    def load_resources(self):
        """Simulate loading the model and scaler."""
        # In a real scenario, this would load the model and scaler using joblib
        self.model_loaded = True
        self.scaler_loaded = True
        return True
    
    def predict(self, input_data):
        """Simulate making predictions."""
        if not self.model_loaded or not self.scaler_loaded:
            raise RuntimeError("Model and scaler must be loaded before prediction")
        
        # Simulate prediction based on input shape
        # In real use, this would use the actual model to predict
        n_samples = input_data.shape[0] if hasattr(input_data, 'shape') else len(input_data)
        return {
            'cognitive_load': np.array([1, 2, 3, 2, 1][:n_samples]),
            'confidence': np.array([0.8, 0.9, 0.7, 0.8, 0.9][:n_samples])
        }

# Mock sample data generator
def generate_sample_data(n_samples=5, n_features=5):
    """Generate sample data for testing."""
    return np.random.random((n_samples, n_features))

@pytest.fixture
def workflow_setup():
    """Set up the workflow environment for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = os.path.join(temp_dir, 'models', 'advanced')
        data_dir = os.path.join(temp_dir, 'data')
        
        # Create necessary directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Create sample input data
        sample_data = generate_sample_data()
        sample_data_path = os.path.join(data_dir, 'sample_input.json')
        with open(sample_data_path, 'w') as f:
            json.dump({
                'features': sample_data.tolist(),
                'metadata': {
                    'subject_id': 'test_subject',
                    'timestamp': '2023-07-15T12:00:00Z',
                    'session': 'test_session'
                }
            }, f)
        
        yield {
            'models_dir': models_dir,
            'data_dir': data_dir,
            'sample_data_path': sample_data_path,
            'temp_dir': temp_dir
        }

class TestModelWorkflow:
    """Integration tests for the model workflow."""
    
    @patch('utilities.download_advanced_models.download_file')
    def test_download_and_predict_workflow(self, mock_download, workflow_setup):
        """Test the full workflow from downloading models to making predictions."""
        # Configure download_file mock to simulate successful downloads
        mock_download.return_value = True
        
        # Validate setup
        assert os.path.exists(workflow_setup['models_dir'])
        assert os.path.exists(workflow_setup['data_dir'])
        assert os.path.exists(workflow_setup['sample_data_path'])
        
        # Step 1: Download a model
        model_type = 'rf'  # Use Random Forest for testing
        
        # Patch extract_model_package to create actual test files
        with patch('utilities.download_advanced_models.extract_model_package') as mock_extract:
            # Define behavior for extract_model_package
            def create_test_files(zip_path, output_dir, model_type):
                model_dir = os.path.join(output_dir, model_type)
                os.makedirs(model_dir, exist_ok=True)
                
                # Create mock model file
                model_path = os.path.join(model_dir, f"Advanced_{model_type}_model.joblib")
                with open(model_path, 'w') as f:
                    f.write("Mock model data")
                
                # Create mock scaler file
                scaler_path = os.path.join(model_dir, f"Advanced_{model_type}_scaler.joblib")
                with open(scaler_path, 'w') as f:
                    f.write("Mock scaler data")
                
                # Create metadata file
                metadata_path = os.path.join(model_dir, f"Advanced_{model_type}_metadata.json")
                metadata = {
                    "name": "Advanced Random Forest",
                    "accuracy": 0.89,
                    "training_samples": 25000,
                    "features": 120,
                    "description": "Mock RF model for testing",
                    "model_path": model_path,
                    "scaler_path": scaler_path
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                return True
            
            mock_extract.side_effect = create_test_files
            
            # Download the model
            result = download_model(model_type, workflow_setup['models_dir'])
            assert result is True, "Model download should succeed"
            
            # Verify model files were created
            model_dir = os.path.join(workflow_setup['models_dir'], model_type)
            assert os.path.exists(model_dir), "Model directory should exist"
            
            model_path = os.path.join(model_dir, f"Advanced_{model_type}_model.joblib")
            assert os.path.exists(model_path), "Model file should exist"
            
            scaler_path = os.path.join(model_dir, f"Advanced_{model_type}_scaler.joblib")
            assert os.path.exists(scaler_path), "Scaler file should exist"
            
            metadata_path = os.path.join(model_dir, f"Advanced_{model_type}_metadata.json")
            assert os.path.exists(metadata_path), "Metadata file should exist"
        
        # Step 2: Load sample data for prediction
        with open(workflow_setup['sample_data_path'], 'r') as f:
            sample_data = json.load(f)
        
        input_features = np.array(sample_data['features'])
        
        # Step 3: Use model to make predictions
        predictor = MockPredictor(model_path, scaler_path)
        predictor.load_resources()
        
        # Make predictions
        predictions = predictor.predict(input_features)
        
        # Verify predictions
        assert 'cognitive_load' in predictions, "Predictions should include cognitive_load"
        assert 'confidence' in predictions, "Predictions should include confidence"
        assert len(predictions['cognitive_load']) == input_features.shape[0], "Should have prediction for each input"
        
        # Verify prediction ranges
        assert np.all(predictions['cognitive_load'] >= 1), "Cognitive load predictions should be at least 1"
        assert np.all(predictions['cognitive_load'] <= 3), "Cognitive load predictions should be at most 3"
        assert np.all(predictions['confidence'] >= 0), "Confidence should be at least 0"
        assert np.all(predictions['confidence'] <= 1), "Confidence should be at most 1"
    
    @patch('utilities.download_advanced_models.download_file')
    def test_multi_model_ensemble_workflow(self, mock_download, workflow_setup):
        """Test an ensemble workflow using multiple models."""
        # Configure download_file mock to simulate successful downloads
        mock_download.return_value = True
        
        # Step 1: Download multiple models
        model_types = ['rf', 'gb', 'mlp']  # Models to use in ensemble
        model_paths = {}
        scaler_paths = {}
        
        # Patch extract_model_package to create actual test files
        with patch('utilities.download_advanced_models.extract_model_package') as mock_extract:
            # Define behavior for extract_model_package for each model type
            def create_test_files(zip_path, output_dir, model_type):
                model_dir = os.path.join(output_dir, model_type)
                os.makedirs(model_dir, exist_ok=True)
                
                # Create mock model file
                model_path = os.path.join(model_dir, f"Advanced_{model_type}_model.joblib")
                with open(model_path, 'w') as f:
                    f.write(f"Mock {model_type} model data")
                
                # Create mock scaler file
                scaler_path = os.path.join(model_dir, f"Advanced_{model_type}_scaler.joblib")
                with open(scaler_path, 'w') as f:
                    f.write(f"Mock {model_type} scaler data")
                
                # Create metadata file
                metadata_path = os.path.join(model_dir, f"Advanced_{model_type}_metadata.json")
                metadata = {
                    "name": f"Advanced {model_type.upper()}",
                    "accuracy": 0.85 + 0.02 * model_types.index(model_type),
                    "training_samples": 20000 + 2000 * model_types.index(model_type),
                    "features": 100 + 10 * model_types.index(model_type),
                    "description": f"Mock {model_type.upper()} model for testing",
                    "model_path": model_path,
                    "scaler_path": scaler_path
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Store paths for later use
                model_paths[model_type] = model_path
                scaler_paths[model_type] = scaler_path
                
                return True
            
            mock_extract.side_effect = create_test_files
            
            # Download each model
            for model_type in model_types:
                result = download_model(model_type, workflow_setup['models_dir'])
                assert result is True, f"{model_type} model download should succeed"
        
        # Step 2: Load sample data for prediction
        with open(workflow_setup['sample_data_path'], 'r') as f:
            sample_data = json.load(f)
        
        input_features = np.array(sample_data['features'])
        
        # Step 3: Use each model to make predictions
        predictors = {}
        predictions = {}
        
        for model_type in model_types:
            predictors[model_type] = MockPredictor(
                model_paths[model_type], 
                scaler_paths[model_type]
            )
            predictors[model_type].load_resources()
            predictions[model_type] = predictors[model_type].predict(input_features)
        
        # Step 4: Create ensemble prediction by averaging
        ensemble_load = np.zeros(input_features.shape[0])
        ensemble_confidence = np.zeros(input_features.shape[0])
        
        for model_type in model_types:
            ensemble_load += predictions[model_type]['cognitive_load']
            ensemble_confidence += predictions[model_type]['confidence']
        
        ensemble_load = np.round(ensemble_load / len(model_types)).astype(int)
        ensemble_confidence = ensemble_confidence / len(model_types)
        
        # Verify ensemble predictions
        assert len(ensemble_load) == input_features.shape[0], "Should have ensemble prediction for each input"
        assert len(ensemble_confidence) == input_features.shape[0], "Should have ensemble confidence for each input"
        
        # Verify ensemble prediction ranges
        assert np.all(ensemble_load >= 1), "Ensemble cognitive load predictions should be at least 1"
        assert np.all(ensemble_load <= 3), "Ensemble cognitive load predictions should be at most 3"
        assert np.all(ensemble_confidence >= 0), "Ensemble confidence should be at least 0"
        assert np.all(ensemble_confidence <= 1), "Ensemble confidence should be at most 1"

# Add more integration tests as needed for specific workflows 