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
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACT_DIR, "train_preprocessed.csv")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# MLflow Experiment Name
EXPERIMENT_NAME = "Personality_Classification_Tuning"

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
# HYPERPARAMETER TUNING dengan GridSearchCV
# ------------------------------------------------------------
print("\nStarting Hyperparameter Tuning...")

# Model yang akan di-tuning: Random Forest
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
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Best model dari GridSearch
best_model = grid_search.best_estimator_

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# ------------------------------------------------------------
# EVALUASI MODEL
# ------------------------------------------------------------
y_pred = best_model.predict(X_test_scaled)

# Metrics (sama dengan autolog)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# MLFLOW MANUAL LOGGING (SKILLED)
# Metriks yang sama dengan autolog
# ------------------------------------------------------------
with mlflow.start_run(run_name="RandomForest_Tuned"):
    
    # 1. Log parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # 2. Log metrics (SAMA DENGAN AUTOLOG)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_score_weighted", f1)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # 3. Log model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="RandomForest_Personality_Classifier"
    )
    
    # 4. Log scaler sebagai artifact
    mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
    
    # 5. Simpan model ke local artifacts (untuk CI/CD)
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    
    print(f"\n✅ Model logged to MLflow successfully!")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"✅ Model saved to: {model_path}")

print("\n" + "="*50)
print("Training completed successfully!")
print("="*50)