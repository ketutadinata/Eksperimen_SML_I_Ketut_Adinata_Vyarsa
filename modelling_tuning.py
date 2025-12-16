import os
import joblib
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import (
    load_raw_data_from_kaggle,
    clean_and_encode,
    split_and_scale,
    save_artifacts
)

# =====================================================
# DAGSHUB + MLFLOW INIT
# =====================================================
dagshub.init(
    repo_owner="ketutadinata1811",
    repo_name="my-first-repo",
    mlflow=True
)

mlflow.set_experiment("eksperimen-mlflow")

os.makedirs("artifacts", exist_ok=True)

# =====================================================
# START RUN
# =====================================================
with mlflow.start_run():

    # =============================
    # PREPROCESSING (END-TO-END)
    # =============================
    df_raw = load_raw_data_from_kaggle()
    df_clean = clean_and_encode(df_raw)

    X_train, X_test, y_train, y_test, scaler, cols = split_and_scale(df_clean)

    train_df, test_df = save_artifacts(
        X_train, X_test, y_train, y_test, cols, scaler
    )

    mlflow.log_artifact("artifacts/train_preprocessed.csv")
    mlflow.log_artifact("artifacts/test_preprocessed.csv")
    mlflow.log_artifact("artifacts/scaler.pkl")

    X_train = train_df.drop("Personality", axis=1)
    y_train = train_df["Personality"]
    X_test = test_df.drop("Personality", axis=1)
    y_test = test_df["Personality"]

    # =============================
    # LOGISTIC REGRESSION
    # =============================
    logreg = LogisticRegression(max_iter=1000)
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }

    grid = GridSearchCV(
        logreg,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_lr = grid.best_estimator_

    lr_pred = best_lr.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred, average="macro")

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("logreg_f1_macro", lr_f1)
    mlflow.sklearn.log_model(best_lr, "logreg_model")

    # =============================
    # RANDOM FOREST
    # =============================
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")

    mlflow.log_metric("rf_f1_macro", rf_f1)
    mlflow.sklearn.log_model(rf, "rf_model")

    # =============================
    # BEST MODEL
    # =============================
    best_model = rf if rf_f1 > lr_f1 else best_lr
    best_name = "RandomForest" if rf_f1 > lr_f1 else "LogisticRegression"

    joblib.dump(best_model, "artifacts/best_model.pkl")
    mlflow.log_param("best_model", best_name)
    mlflow.log_artifact("artifacts/best_model.pkl")

    print("RUN SELESAI (END-TO-END)")
