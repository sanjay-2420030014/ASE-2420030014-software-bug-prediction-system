"""CM1 dataset preprocessing for the AI-Based Smart Bug Prediction System."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/bug_dataset.csv")
TARGET_COLUMN = "defects"


def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)


def preprocess_data(df, test_size=0.20, random_state=42):
    df = df.copy()

    # CM1 uses a boolean defects target; convert it to 0/1.
    if df[TARGET_COLUMN].dtype == bool:
        y = df[TARGET_COLUMN].astype(int)
    else:
        y = df[TARGET_COLUMN].astype(str).str.lower().map(
            {"false": 0, "true": 1}
        )
        if y.isna().any():
            raise ValueError("Unexpected values in defects target column.")

    X = df.drop(columns=[TARGET_COLUMN])

    # CM1 predictor columns are expected to be numeric.
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    if X.isna().any().any():
        raise ValueError("Missing/non-numeric predictor values detected.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Fit only on training data to prevent data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)


if __name__ == "__main__":
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    print("Dataset shape:", df.shape)
    print("Number of features:", len(feature_names))
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training class distribution:")
    print(y_train.value_counts().sort_index())
    print("Testing class distribution:")
    print(y_test.value_counts().sort_index())
    print("Preprocessing completed successfully.")
