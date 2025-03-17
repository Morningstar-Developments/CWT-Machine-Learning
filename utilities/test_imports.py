#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly.
This can help the IDE recognize the packages.
"""

print("Testing imports...")

# Test all imports used in cwt.py
import os
import sys
import json
import logging
from datetime import datetime
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

print("All imports successful!")
print(f"dotenv imported from: {load_dotenv.__module__}")

# Test loading a .env file
if os.path.exists('.env'):
    load_dotenv()
    print(".env file loaded successfully")
else:
    print("No .env file found, but import works") 