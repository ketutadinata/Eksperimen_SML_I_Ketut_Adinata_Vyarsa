#!/usr/bin/env python3
# ============================================================
# modelling_tuning.py
# Training model dengan MLflow tracking + Hyperparameter Tuning
# Target: SKILLED (Lengkap dengan URI Localhost & Confusion Matrix)
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
    confusion_matrix
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
# [CRITICAL] SET MLFLOW TRACKING URI
# ============================================================
# Ini adalah bagian revisi utama agar tersimpan di Localhost Server
print("\n[1/8] Setting up MLflow experiment...")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"✓ Tracking URI : http://127.0.0.1:5000/")
print(f"✓ Experiment   : {EXPERIMENT_NAME}")

# ============================================================
# LOAD DATASET
# ============================================================
print("\n[2/8] Loading dataset...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Pastikan sudah menjalankan Notebook preprocessing.")

df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop("Personality", axis=1)
y = df["Personality"]

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
print("\n[3/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================================
# SCALING
# ============================================================
print("\n[4/8] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved: {scaler_path}")

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
print("\n[5/8] Starting hyperparameter tuning...")
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
print("\n[6/8] Evaluating model...")
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
# GENERATE ARTIFACTS TAMBAHAN (Sesuai Gambar Skilled)
# ============================================================
print("\n[7/8] Generating extra artifacts...")

# 1. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
cm_path = "training_confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
print(f"✓ Confusion Matrix created: {cm_path}")

# ============================================================
# MLFLOW LOGGING
# ============================================================
# ============================================================
# MLFLOW LOGGING (STRUKTUR RAPI)
# ============================================================
print("\n[8/8] Logging to MLflow...")

with mlflow.start_run(run_name="RandomForest_Skilled_Run") as run:
    
    # 1. Log Parameters & Metrics (Tetap sama)
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score_weighted", f1)
    
    # 2. Log Model (Ini akan membuat folder 'model' di root artifacts)
    # Di dalamnya otomatis ada MLmodel, conda.yaml, model.pkl, dll
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model", 
        registered_model_name="RandomForest_Personality_Classifier"
    )
    
    # 3. Log file lainnya KE DALAM folder 'model' agar terlihat menyatu
    # Kita arahkan artifact_path ke "model" juga
    mlflow.log_artifact(scaler_path, artifact_path="model")
    mlflow.log_artifact(cm_path, artifact_path="model") 
    
    print(f"✓ Semua artefak telah dikirim ke folder 'model' di MLflow")
    
    # Save model locally (opsional, backup)
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    
    print(f"✓ Model logged to MLflow Run ID: {run.info.run_id}")
    print(f"✓ Artifacts sent to server: http://127.0.0.1:5000/")

# Bersihkan file temporary gambar jika perlu
if os.path.exists(cm_path):
    os.remove(cm_path)

print("\n" + "="*60)
print("✅ TRAINING & LOGGING COMPLETED!")
print("Sekarang buka browser di http://127.0.0.1:5000/ untuk cek hasil.")
print("="*60)