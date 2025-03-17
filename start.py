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