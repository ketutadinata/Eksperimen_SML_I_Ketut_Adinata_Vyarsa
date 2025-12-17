# ============================================================
# modelling.py
# Training model BASIC dengan MLflow AUTOLOG (Tanpa Tuning)
# Target: BASIC (2 pts)
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACT_DIR, "train_preprocessed.csv")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# MLflow Experiment Name
EXPERIMENT_NAME = "Personality_Classification_Basic"

# ------------------------------------------------------------
# SET MLFLOW EXPERIMENT
# ------------------------------------------------------------
print("Setting up MLflow experiment...")
mlflow.set_experiment(EXPERIMENT_NAME)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop("Personality", axis=1)
y = df["Personality"]

# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ------------------------------------------------------------
# SCALING
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan scaler
scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)

# ------------------------------------------------------------
# TRAINING dengan MLFLOW AUTOLOG (BASIC REQUIREMENT)
# ------------------------------------------------------------
print("\nStarting model training with MLflow autolog...")

# Aktifkan autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Basic"):
    
    # Model tanpa hyperparameter tuning
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    # Training (autolog akan otomatis mencatat semua metrics)
    model.fit(X_train_scaled, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Simpan model ke local artifacts
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(model, model_path)
    
    print(f"\n✅ Model trained with autolog!")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"✅ Model saved to: {model_path}")

print("\n" + "="*50)
print("Training completed successfully!")
print("="*50)