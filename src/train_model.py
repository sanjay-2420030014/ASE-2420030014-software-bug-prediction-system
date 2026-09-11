"""Train and compare ML models for the AI-Based Smart Bug Prediction System.

Expected repository structure:
    data/bug_dataset.csv
    src/preprocessing.py
    src/train_model.py
    results/

Run from the repository root:
    python src/train_model.py
"""

from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Make the src folder importable even when this script is run from the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing import load_dataset, preprocess_data


RESULTS_DIR = ROOT / "results"
METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
MODEL_FILE = RESULTS_DIR / "bug_prediction_model.joblib"


def train_and_evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(ROOT / "data" / "bug_dataset.csv")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # CM1 is imbalanced, so balanced class weights help the models pay
    # appropriate attention to defective modules.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
    }

    rows = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(
                    y_test, y_pred, pos_label=1, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, pos_label=1, zero_division=0
                ),
                "F1-Score": f1_score(
                    y_test, y_pred, pos_label=1, zero_division=0
                ),
            }
        )
        fitted_models[name] = model

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values(
        by=["F1-Score", "Recall", "Accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    metrics_df.to_csv(METRICS_FILE, index=False)

    # Select the best model primarily by defective-class F1-score.
    best_name = metrics_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "feature_names": feature_names,
            "target_column": "defects",
            "best_model": best_name,
        },
        MODEL_FILE,
    )

    print("Model training completed successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model: {best_name}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Best model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_and_evaluate()
