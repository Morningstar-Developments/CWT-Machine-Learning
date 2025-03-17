#!/usr/bin/env python3
"""
Setup script for creating executable links and shortcuts for CWT tools.

This script creates:
1. Executable shortcuts for commonly used utility scripts
2. Symbolic links with convenient names

Usage:
    python utilities/setup_links.py
"""

import os
import sys
import stat
import platform
from pathlib import Path


def create_executable_script(script_path, target_script, description=""):
    """Create a small wrapper script that calls another script."""
    with open(script_path, 'w') as f:
        if platform.system() == 'Windows':
            # Windows batch file
            f.write('@echo off\n')
            if description:
                f.write(':: {}\n'.format(description))
            f.write('python {} %*\n'.format(target_script))
        else:
            # Unix shell script
            f.write('#!/bin/bash\n')
            if description:
                f.write('# {}\n'.format(description))
            f.write('python {} "$@"\n'.format(target_script))

    # Make the script executable
    current_mode = os.stat(script_path).st_mode
    os.chmod(script_path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Created executable script: {script_path}")


def setup_links():
    """Set up all necessary links and shortcuts."""
    print("\n=== Setting up CWT utility shortcuts ===\n")
    
    # Define shortcuts to create
    shortcuts = [
        {
            'name': 'check-models',
            'target': 'utilities/check_models.py',
            'description': 'Utility to check and fix model/scaler compatibility'
        },
        {
            'name': 'generate-data',
            'target': 'utilities/generate_sample_data.py',
            'description': 'Generate sample data files for training and testing'
        },
        {
            'name': 'organize',
            'target': 'utilities/organize_outputs.py',
            'description': 'Organize models and logs into proper directories'
        },
        {
            'name': 'download-models',
            'target': 'utilities/download_advanced_models.py',
            'description': 'Download advanced pre-trained models'
        }
    ]
    
    # Create each shortcut
    for shortcut in shortcuts:
        script_path = shortcut['name']
        if platform.system() == 'Windows':
            script_path += '.bat'
        
        create_executable_script(
            script_path,
            shortcut['target'],
            shortcut['description']
        )
    
    # Create !help shortcut if on Unix
    if platform.system() != 'Windows':
        with open('!help', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('# Shortcut to display CWT help\n')
            f.write('./utilities/cwt-help.sh "$@"\n')
        
        os.chmod('!help', 
                os.stat('!help').st_mode | 
                stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print("Created !help shortcut")
    
    print("\nSetup complete. You can now use these shortcuts from the root directory.")
    print("\nExample usage:")
    print("  ./check-models         # Check model and scaler compatibility")
    print("  ./check-models --fix   # Fix model and scaler issues")
    print("  ./generate-data        # Generate sample data")
    print("  ./organize             # Organize models and logs")
    print("  ./download-models --all  # Download all advanced models")


if __name__ == "__main__":
    # Ensure we're running from the project root
    if not Path('cwt.py').exists():
        print("Error: This script must be run from the project root directory.")
        print("Usage: python utilities/setup_links.py")
        sys.exit(1)
    
    setup_links() 