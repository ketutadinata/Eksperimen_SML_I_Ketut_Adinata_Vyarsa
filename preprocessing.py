# preprocessing.py
import os
import pandas as pd
import joblib
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub


def load_raw_data_from_kaggle() -> pd.DataFrame:
    """
    Download & load dataset from KaggleHub
    """
    path = kagglehub.dataset_download(
        "rakeshkapilavai/extrovert-vs-introvert-behavior-data"
    )
    return pd.read_csv(path + "/personality_dataset.csv")


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_cols = [
        "Stage_fear",
        "Drained_after_socializing",
        "Personality"
    ]

    for col in categorical_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.capitalize()
        )

    df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
    df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
    df["Personality"] = df["Personality"].map({"Introvert": 0, "Extrovert": 1})

    return df.dropna()


def split_and_scale(
    df: pd.DataFrame,
    target_col="Personality",
    test_size=0.2,
    random_state=42
):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def save_artifacts(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_cols,
    scaler,
    output_dir="artifacts"
):
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    test_df = pd.DataFrame(X_test, columns=feature_cols)

    train_df["Personality"] = y_train.values
    test_df["Personality"] = y_test.values

    train_df.to_csv(f"{output_dir}/train_preprocessed.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_preprocessed.csv", index=False)

    return train_df, test_df
