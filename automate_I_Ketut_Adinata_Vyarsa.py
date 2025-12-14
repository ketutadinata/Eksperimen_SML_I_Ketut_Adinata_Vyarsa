# ============================================================
# automate_I_Ketut_Adinata_Vyarsa.py
# Pipeline otomatis: Load data -> Preprocessing -> Training
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# Konfigurasi path
# ------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACT_DIR, "train_preprocessed.csv")

SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load dataset hasil preprocessing
# ------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# Split fitur dan target
# ------------------------------------------------------------
X = df.drop("Personality", axis=1)
y = df["Personality"]

# ------------------------------------------------------------
# Train-test split (ulang secara eksplisit untuk reproducibility)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Scaling (fit hanya pada data train)
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan scaler sebagai artefak
joblib.dump(scaler, SCALER_PATH)

# ------------------------------------------------------------
# Training model (baseline: Logistic Regression)
# ------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------
# Evaluasi model
# ------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(report)

# ------------------------------------------------------------
# Simpan model
# ------------------------------------------------------------
joblib.dump(model, MODEL_PATH)

print("\nPipeline selesai.")
print(f"Scaler disimpan di : {SCALER_PATH}")
print(f"Model disimpan di  : {MODEL_PATH}")
