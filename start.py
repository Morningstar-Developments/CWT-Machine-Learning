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

# Define available commands for display
AVAILABLE_COMMANDS = {
    "setup": "Set up the CWT environment and create necessary directories",
    "train": "Train a specific cognitive workload model",
    "train-all": "Train all available cognitive workload models",
    "predict": "Predict cognitive workload from input data",
    "batch-predict": "Batch predict cognitive workload from a CSV file",
    "time-series-predict": "Predict cognitive workload from time series data"
}

def display_welcome_banner():
    """Display a welcome banner with CWT tool information"""
    banner = r"""
   ______ _    _ _______   _______          _ 
  / _____) |  | (_______)_(_______)        | |
 | /     | |  | |_     _(_)  _  | |     ___| |
 | |     | |  | | |   | | | | | | |    /___) |
 | \_____| |__| | |   | | | |_| | |___|___ | |
  \______)____/  |___|  |_|_____|_____|___/|_|
                                                
   Cognitive Workload Training Tool
   -------------------------------
   A comprehensive platform for cognitive workload detection and analysis
    """
    print(banner)
    print("Starting CWT Tool setup and model training process...\n")
    print("This will:")
    print("  1. Set up the CWT environment")
    print("  2. Train all available model types")
    print("  3. Prepare sample data and predictions")
    print("  4. Generate model performance visualizations\n")
    print("=" * 70)
    print()

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

def display_command_reference():
    """Display a comprehensive command reference"""
    print("\n" + "=" * 70)
    print(" " * 25 + "COMMAND REFERENCE")
    print("=" * 70)
    
    # Import commands module if available
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import commands
        commands.print_command_help()
    except ImportError:
        # Fallback if commands module is not available
        print("\nAvailable Commands:")
        for cmd, desc in AVAILABLE_COMMANDS.items():
            print(f"  {cmd:<20} {desc}")
    
    # Common command examples
    print("\nCommon Usage Examples:")
    print("  python cwt.py setup")
    print("  python cwt.py train --model-type svm")
    print("  python cwt.py train-all --parallel")
    print("  python cwt.py predict --input-values 'pulse_rate=75' 'blood_pressure_sys=120'")
    print("  python cwt.py batch-predict --input-file data/samples.csv --infer-missing")
    print("  python cwt.py time-series-predict --input-file data/time_series.csv --visualize")
    
    print("\nFor detailed help on any command:")
    print("  ./commands.py [command]")
    print("  Example: ./commands.py train-all")
    
    print("\nUtility Scripts:")
    print("  ./start.py            # Run this setup script again")
    print("  ./check-models        # Verify model integrity")
    print("  ./organize            # Organize models and logs")
    print("  ./generate-data       # Generate sample data")
    print("  ./download-models     # Download advanced pre-trained models")
    
    print("\nDocumentation:")
    print("  See README.md for complete documentation")
    print("=" * 70)

def main():
    """Main function to run all steps"""
    # Display welcome banner
    display_welcome_banner()
    
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
    
    # Success message
    print("\n" + "=" * 70)
    print(" " * 25 + "SETUP COMPLETE")
    print("=" * 70)
    print("\n✅ CWT Tool is now ready for analysis!")
    print(f"• Trained models are in: {MODEL_DIR}")
    print(f"• Results and visualizations are in: {RESULTS_DIR}")
    
    # Display command reference
    display_command_reference()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 