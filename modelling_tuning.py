#!/usr/bin/env python3
# ============================================================
# modelling_tuning.py
# Training model dengan MLflow tracking + Hyperparameter Tuning
# Target: SKILLED (3 pts)
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ============================================================
# KONFIGURASI
# ============================================================
ARTIFACT_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACT_DIR, "train_preprocessed.csv")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

EXPERIMENT_NAME = "Personality_Classification_Tuning"

print("="*60)
print("MLflow Model Training with Hyperparameter Tuning")
print("="*60)

# ============================================================
# SET MLFLOW EXPERIMENT
# ============================================================
print("\n[1/7] Setting up MLflow experiment...")
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"✓ Experiment: {EXPERIMENT_NAME}")

# ============================================================
# LOAD DATASET
# ============================================================
print("\n[2/7] Loading dataset...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop("Personality", axis=1)
y = df["Personality"]

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
print("\n[3/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================================
# SCALING
# ============================================================
print("\n[4/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved: {scaler_path}")

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
print("\n[5/7] Starting hyperparameter tuning...")
print("This may take a few minutes...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best CV score: {grid_search.best_score_:.4f}")

# ============================================================
# EVALUASI MODEL
# ============================================================
print("\n[6/7] Evaluating model...")
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("="*60)

# ============================================================
# MLFLOW MANUAL LOGGING (SKILLED)
# ============================================================
print("\n[7/7] Logging to MLflow...")

with mlflow.start_run(run_name="RandomForest_Tuned"):
    
    # Log parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Log metrics (sama dengan autolog)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_score_weighted", f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Log model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="RandomForest_Personality_Classifier"
    )
    
    # Log scaler
    mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
    
    # Save model locally
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    
    print("✓ Model logged to MLflow")
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)