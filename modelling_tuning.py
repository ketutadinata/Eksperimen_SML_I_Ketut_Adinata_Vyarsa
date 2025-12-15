import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import dagshub

# ----------------------------
# INIT DAGSHUB + MLFLOW
# ----------------------------
dagshub.init(
    repo_owner='ketutadinata1811',
    repo_name='my-first-repo',
    mlflow=True
)

# ----------------------------
# CONFIG
# ----------------------------
TRAIN_PATH = "artifacts/train_preprocessed.csv"
TEST_PATH = "artifacts/test_preprocessed.csv"
TARGET_COL = "Personality"

os.makedirs("artifacts", exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
print("Loading preprocessed data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(TARGET_COL, axis=1)
y_train = train_df[TARGET_COL]

X_test = test_df.drop(TARGET_COL, axis=1)
y_test = test_df[TARGET_COL]

# ----------------------------
# MODEL 1: Logistic Regression + Tuning
# ----------------------------
print("Training Logistic Regression with GridSearchCV...")

logreg = LogisticRegression(max_iter=1000)

logreg_param = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

logreg_grid = GridSearchCV(
    logreg,
    logreg_param,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

logreg_grid.fit(X_train, y_train)
best_logreg = logreg_grid.best_estimator_

y_pred_lr = best_logreg.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average="macro")

# ----------------------------
# MLflow logging
# ----------------------------
mlflow.log_param("model_1", "LogisticRegression")
mlflow.log_params(logreg_grid.best_params_)
mlflow.log_metric("logreg_accuracy", lr_acc)
mlflow.log_metric("logreg_f1_macro", lr_f1)

mlflow.sklearn.log_model(best_logreg, artifact_path="logistic_regression_model")

with open("artifacts/logreg_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred_lr))
mlflow.log_artifact("artifacts/logreg_report.txt")

# ----------------------------
# MODEL 2: Random Forest
# ----------------------------
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average="macro")

mlflow.log_param("model_2", "RandomForest")
mlflow.log_metric("rf_accuracy", rf_acc)
mlflow.log_metric("rf_f1_macro", rf_f1)

mlflow.sklearn.log_model(rf, artifact_path="random_forest_model")

with open("artifacts/rf_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred_rf))
mlflow.log_artifact("artifacts/rf_report.txt")

# ----------------------------
# SAVE BEST MODEL
# ----------------------------
if rf_f1 > lr_f1:
    best_model = rf
    best_name = "RandomForest"
else:
    best_model = best_logreg
    best_name = "LogisticRegression"

joblib.dump(best_model, "artifacts/best_model.pkl")
mlflow.log_param("best_model", best_name)
mlflow.log_artifact("artifacts/best_model.pkl")

# ----------------------------
# ARTIFACT TAMBAHAN
# ----------------------------
# Confusion Matrix Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix RF")
plt.savefig("artifacts/confusion_matrix_rf.png")
mlflow.log_artifact("artifacts/confusion_matrix_rf.png")
plt.close()

# Feature Importance Random Forest
fi_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=fi_df)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("artifacts/rf_feature_importance.png")
mlflow.log_artifact("artifacts/rf_feature_importance.png")
plt.close()

# Dataset snapshot
df_snapshot = pd.concat([X_test, y_test], axis=1)
df_snapshot.to_csv("artifacts/dataset_snapshot.csv", index=False)
mlflow.log_artifact("artifacts/dataset_snapshot.csv")

# ----------------------------
# FINAL OUTPUT
# ----------------------------
print("=== TRAINING SELESAI ===")
print(f"LogReg  | Acc: {lr_acc:.4f} | F1: {lr_f1:.4f}")
print(f"RF      | Acc: {rf_acc:.4f} | F1: {rf_f1:.4f}")
print(f"BEST MODEL: {best_name}")

import mlflow