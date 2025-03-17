import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# ---------------------- CONFIGURATION ---------------------- #
DATA_FILES = {
    "physiological": "Enhanced_Workload_Clinical_Data.csv",
    "eeg": "000_EEG_Cluster_ANOVA_Results.csv",
    "gaze": "008_01.csv"
}
MODEL_OUTPUT_PATH = "Cognitive_State_Prediction_Model.joblib"

# ---------------------- LOAD DATA ---------------------- #
def load_data():
    df_physio = pd.read_csv(DATA_FILES["physiological"])
    df_eeg = pd.read_csv(DATA_FILES["eeg"])
    df_gaze = pd.read_csv(DATA_FILES["gaze"])
    return df_physio, df_eeg, df_gaze

# ---------------------- PREPROCESSING ---------------------- #
def preprocess_data(df_physio, df_eeg, df_gaze):
    # Merge datasets on timestamp or closest match
    df_physio["timestamp"] = pd.to_datetime(df_physio["timestamp"])
    df_eeg["timestamp"] = pd.to_datetime(df_eeg["timestamp"])
    df_gaze["timestamp"] = pd.to_datetime(df_gaze["timestamp"])
    
    # Merge based on nearest timestamp
    df = pd.merge_asof(df_physio.sort_values("timestamp"), df_eeg.sort_values("timestamp"), on="timestamp")
    df = pd.merge_asof(df.sort_values("timestamp"), df_gaze.sort_values("timestamp"), on="timestamp")
    
    # Select relevant features
    features = ["pulse_rate", "blood_pressure_sys", "resp_rate", "pupil_diameter_left",
                "pupil_diameter_right", "fixation_duration", "blink_rate", "workload_intensity",
                "gaze_x", "gaze_y", "alpha_power", "theta_power"]
    df = df[features]
    
    # Standardization
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Label Encoding for Workload Intensity (Target Variable)
    df["cognitive_state"] = pd.qcut(df["workload_intensity"], q=3, labels=["Low", "Medium", "High"])
    
    return df, scaler

# ---------------------- MODEL TRAINING ---------------------- #
def train_model(df):
    X = df.drop(columns=["cognitive_state", "workload_intensity"])
    y = df["cognitive_state"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    print("Classification Report:\n", class_report)
    
    # Save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved at {MODEL_OUTPUT_PATH}")
    
    return model

# ---------------------- PREDICTION FUNCTION ---------------------- #
def predict_new_data(model, scaler, new_data):
    new_data = pd.DataFrame([new_data])
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction[0]

# ---------------------- MAIN EXECUTION ---------------------- #
if __name__ == "__main__":
    print("Loading Data...")
    df_physio, df_eeg, df_gaze = load_data()
    
    print("Preprocessing Data...")
    df_processed, scaler = preprocess_data(df_physio, df_eeg, df_gaze)
    
    print("Training Model...")
    model = train_model(df_processed)
    
    print("Model training complete. Ready for inference.")
