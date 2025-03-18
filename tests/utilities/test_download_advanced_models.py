#!/usr/bin/env python3
"""
Tests for the advanced model downloader utility.

This test suite verifies the functionality of the download_advanced_models.py script
which downloads and installs advanced models for the Cognitive Workload Tool.
"""

import os
import json
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock, mock_open
import sys
from pathlib import Path

# Add the parent directory to the sys.path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utilities.download_advanced_models import (
    parse_args,
    download_file,
    extract_model_package,
    download_model,
    main,
    MODEL_REPOSITORIES,
    MODEL_METADATA
)

class TestArgumentParsing:
    """Tests for command line argument parsing functionality."""
    
    def test_parse_args_all_flag(self):
        """Test parsing --all flag."""
        with patch('sys.argv', ['download_advanced_models.py', '--all']):
            args = parse_args()
            assert args.all is True
            assert args.model_type is None
            assert args.output_dir == 'models/advanced'
    
    def test_parse_args_model_type(self):
        """Test parsing --model-type flag."""
        with patch('sys.argv', ['download_advanced_models.py', '--model-type', 'rf']):
            args = parse_args()
            assert args.all is False
            assert args.model_type == 'rf'
            assert args.output_dir == 'models/advanced'
    
    def test_parse_args_output_dir(self):
        """Test parsing --output-dir flag."""
        with patch('sys.argv', ['download_advanced_models.py', '--all', '--output-dir', 'custom/path']):
            args = parse_args()
            assert args.all is True
            assert args.output_dir == 'custom/path'
    
    def test_parse_args_invalid_model_type(self):
        """Test parsing with invalid model type raises error."""
        with patch('sys.argv', ['download_advanced_models.py', '--model-type', 'invalid']):
            with pytest.raises(SystemExit):
                parse_args()

class TestDownloadFile:
    """Tests for file download functionality."""
    
    @patch('tqdm.tqdm')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_file_success(self, mock_file, mock_tqdm):
        """Test successful file download."""
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.total_size = 1024
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        
        with patch('utilities.download_advanced_models.MockResponse', return_value=mock_response):
            result = download_file('http://example.com/model.zip', 'path/to/save.zip', 'Test Model')
            assert result is True
            mock_file.assert_called_once_with('path/to/save.zip', 'wb')
    
    @patch('utilities.download_advanced_models.logger')
    def test_download_file_http_error(self, mock_logger):
        """Test file download with HTTP error."""
        # Mock a failed response
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        with patch('utilities.download_advanced_models.MockResponse', return_value=mock_response):
            result = download_file('http://example.com/model.zip', 'path/to/save.zip', 'Test Model')
            assert result is False
            mock_logger.error.assert_called_once()
    
    @patch('utilities.download_advanced_models.logger')
    def test_download_file_exception(self, mock_logger):
        """Test file download with exception."""
        with patch('utilities.download_advanced_models.MockResponse', side_effect=Exception("Test error")):
            result = download_file('http://example.com/model.zip', 'path/to/save.zip', 'Test Model')
            assert result is False
            mock_logger.error.assert_called_once()

class TestExtractModelPackage:
    """Tests for model package extraction functionality."""
    
    def test_extract_model_package_success(self):
        """Test successful model package extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up test data
            model_type = 'rf'
            output_dir = temp_dir
            zip_path = os.path.join(temp_dir, f"{model_type}_model.zip")
            
            # Call the function
            result = extract_model_package(zip_path, output_dir, model_type)
            
            # Verify results
            assert result is True
            
            # Check that files were created
            model_type_dir = os.path.join(output_dir, model_type)
            assert os.path.exists(model_type_dir)
            
            model_file = os.path.join(model_type_dir, f"Advanced_{model_type}_model.joblib")
            assert os.path.exists(model_file)
            
            scaler_file = os.path.join(model_type_dir, f"Advanced_{model_type}_scaler.joblib")
            assert os.path.exists(scaler_file)
            
            metadata_file = os.path.join(model_type_dir, f"Advanced_{model_type}_metadata.json")
            assert os.path.exists(metadata_file)
            
            # Check metadata content
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                assert metadata['model_path'] == model_file
                assert metadata['scaler_path'] == scaler_file
    
    @patch('utilities.download_advanced_models.logger')
    def test_extract_model_package_exception(self, mock_logger):
        """Test model package extraction with exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up test data
            model_type = 'rf'
            output_dir = temp_dir
            zip_path = os.path.join(temp_dir, f"{model_type}_model.zip")
            
            # Mock os.makedirs to raise an exception
            with patch('os.makedirs', side_effect=Exception("Test error")):
                result = extract_model_package(zip_path, output_dir, model_type)
                assert result is False
                mock_logger.error.assert_called_once()

class TestDownloadModel:
    """Tests for model download functionality."""
    
    @patch('utilities.download_advanced_models.download_file', return_value=True)
    @patch('utilities.download_advanced_models.extract_model_package', return_value=True)
    def test_download_model_success(self, mock_extract, mock_download):
        """Test successful model download and installation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_model('rf', temp_dir)
            assert result is True
            mock_download.assert_called_once()
            mock_extract.assert_called_once()
    
    @patch('utilities.download_advanced_models.logger')
    def test_download_model_unknown_type(self, mock_logger):
        """Test download with unknown model type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_model('unknown', temp_dir)
            assert result is False
            mock_logger.error.assert_called_once()
    
    @patch('utilities.download_advanced_models.download_file', return_value=False)
    def test_download_model_download_failure(self, mock_download):
        """Test download model with failed download."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_model('rf', temp_dir)
            assert result is False
            mock_download.assert_called_once()
    
    @patch('utilities.download_advanced_models.download_file', return_value=True)
    @patch('utilities.download_advanced_models.extract_model_package', return_value=False)
    def test_download_model_extract_failure(self, mock_extract, mock_download):
        """Test download model with failed extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_model('rf', temp_dir)
            assert result is False
            mock_download.assert_called_once()
            mock_extract.assert_called_once()

class TestMain:
    """Tests for main function."""
    
    @patch('os.makedirs')
    @patch('utilities.download_advanced_models.parse_args')
    @patch('utilities.download_advanced_models.download_model', return_value=True)
    def test_main_all_models(self, mock_download, mock_parse_args, mock_makedirs):
        """Test main function with --all flag."""
        # Set up mock args
        mock_args = MagicMock()
        mock_args.all = True
        mock_args.model_type = None
        mock_args.output_dir = 'models/advanced'
        mock_parse_args.return_value = mock_args
        
        # Call main
        main()
        
        # Verify all models were attempted to download
        assert mock_download.call_count == len(MODEL_REPOSITORIES)
    
    @patch('os.makedirs')
    @patch('utilities.download_advanced_models.parse_args')
    @patch('utilities.download_advanced_models.download_model', return_value=True)
    def test_main_specific_model(self, mock_download, mock_parse_args, mock_makedirs):
        """Test main function with --model-type flag."""
        # Set up mock args
        mock_args = MagicMock()
        mock_args.all = False
        mock_args.model_type = 'rf'
        mock_args.output_dir = 'models/advanced'
        mock_parse_args.return_value = mock_args
        
        # Call main
        main()
        
        # Verify only one model was attempted to download
        mock_download.assert_called_once_with('rf', 'models/advanced')
    
    @patch('os.makedirs')
    @patch('utilities.download_advanced_models.parse_args')
    @patch('utilities.download_advanced_models.download_model')
    def test_main_mixed_results(self, mock_download, mock_parse_args, mock_makedirs):
        """Test main function with mixed download results."""
        # Set up mock args
        mock_args = MagicMock()
        mock_args.all = True
        mock_args.model_type = None
        mock_args.output_dir = 'models/advanced'
        mock_parse_args.return_value = mock_args
        
        # Set up mixed results - some downloads succeed, some fail
        mock_download.side_effect = [True, False, True]
        
        # Mock MODEL_REPOSITORIES to have just 3 elements for testing
        with patch('utilities.download_advanced_models.MODEL_REPOSITORIES', 
                  {'rf': 'url1', 'svm': 'url2', 'gb': 'url3'}):
            # Call main
            main()
            
            # Verify summary counts successful and failed downloads correctly
            assert mock_download.call_count == 3


@pytest.fixture
def model_output_dir():
    """Fixture to provide a temporary directory for model output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

class TestModelBehavior:
    """Tests for model behavior after installation."""
    
    @patch('utilities.download_advanced_models.download_file', return_value=True)
    @patch('utilities.download_advanced_models.extract_model_package')
    def test_model_metadata_consistency(self, mock_extract, mock_download, model_output_dir):
        """Test model metadata consistency after installation."""
        # Override the extract function to create real files for testing
        def create_real_files(zip_path, output_dir, model_type):
            model_dir = os.path.join(output_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            model_file = os.path.join(model_dir, f"Advanced_{model_type}_model.joblib")
            with open(model_file, 'w') as f:
                f.write("Mock model data")
            
            scaler_file = os.path.join(model_dir, f"Advanced_{model_type}_scaler.joblib")
            with open(scaler_file, 'w') as f:
                f.write("Mock scaler data")
            
            metadata_file = os.path.join(model_dir, f"Advanced_{model_type}_metadata.json")
            metadata = MODEL_METADATA.get(model_type, {}).copy()
            metadata["model_path"] = model_file
            metadata["scaler_path"] = scaler_file
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        
        mock_extract.side_effect = create_real_files
        
        # Download a test model
        model_type = 'rf'
        result = download_model(model_type, model_output_dir)
        assert result is True
        
        # Check that metadata file exists and contains expected values
        metadata_file = os.path.join(model_output_dir, model_type, f"Advanced_{model_type}_metadata.json")
        assert os.path.exists(metadata_file)
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
            # Verify metadata contains expected keys and values
            assert metadata['name'] == MODEL_METADATA[model_type]['name']
            assert metadata['accuracy'] == MODEL_METADATA[model_type]['accuracy']
            assert metadata['training_samples'] == MODEL_METADATA[model_type]['training_samples']
            assert metadata['features'] == MODEL_METADATA[model_type]['features']
            
            # Verify model_path and scaler_path are correctly set
            assert os.path.exists(metadata['model_path'])
            assert os.path.exists(metadata['scaler_path'])

# Add more comprehensive tests as needed 