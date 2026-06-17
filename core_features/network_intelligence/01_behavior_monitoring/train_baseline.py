# train_baseline.py
# ─────────────────────────────────────────────────────────────────────────────
# Trains the Isolation Forest on the TrustNet behavior dataset.
# Run this ONCE to generate models/isolation_forest.pkl and models/scaler.pkl.
# Re-run whenever you get new training data or want to retrain.
#
# Usage:
#   python train_baseline.py                          # uses trustnet_behavior_dataset.csv
#   python train_baseline.py /path/to/your_data.csv  # uses custom dataset
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import joblib
import json
import os
import sys
from datetime import datetime

from config import (
    MODEL_DIR, MODEL_PATH, SCALER_PATH, METADATA_PATH,
    DATASET_PATH, FEATURE_COLUMNS,
    IF_N_ESTIMATORS, IF_CONTAMINATION, IF_RANDOM_STATE, IF_N_JOBS
)
from logger import get_logger

logger = get_logger("train_baseline")


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the dataset CSV and returns features (X) and labels (y).
    Validates that all required columns exist.
    """
    logger.info(f"Loading dataset from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            f"Make sure trustnet_behavior_dataset.csv is in the same folder as train_baseline.py."
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Validate required columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing columns: {missing_cols}\n"
            f"Expected columns: {FEATURE_COLUMNS}"
        )

    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column (0=normal, 1=fraud)")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    # Drop any rows with NaN values
    before = len(X)
    X = X.dropna()
    y = y[X.index]
    if len(X) < before:
        logger.warning(f"Dropped {before - len(X):,} rows with NaN values")

    logger.info(f"Training data: {len(X):,} rows | Normal: {(y==0).sum():,} | Fraud: {(y==1).sum():,}")
    logger.info(f"Fraud ratio: {(y==1).sum()/len(y)*100:.2f}%")

    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None):
    """
    Fits a StandardScaler on training data.
    Isolation Forest works better with scaled features —
    prevents high-magnitude features (like distance_from_branch_km)
    from dominating low-magnitude features (like is_weekend).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return scaler, X_train_scaled, X_test_scaled

    return scaler, X_train_scaled


def train_model(X_train_scaled: np.ndarray) -> IsolationForest:
    """Trains the Isolation Forest model."""
    logger.info(
        f"Training Isolation Forest: "
        f"n_estimators={IF_N_ESTIMATORS}, "
        f"contamination={IF_CONTAMINATION}, "
        f"n_jobs={IF_N_JOBS}"
    )

    model = IsolationForest(
        n_estimators  = IF_N_ESTIMATORS,
        contamination = IF_CONTAMINATION,
        random_state  = IF_RANDOM_STATE,
        n_jobs        = IF_N_JOBS,
    )
    model.fit(X_train_scaled)
    logger.info("Model training complete")
    return model


def evaluate_model(
    model: IsolationForest,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluates model on the test set.
    Returns a dict of metrics for model_metadata.json.
    """
    X_test_scaled = scaler.transform(X_test)

    # Isolation Forest: predict returns -1 (anomaly) or 1 (normal)
    # Convert to 0/1 to match our labels
    raw_predictions = model.predict(X_test_scaled)
    predictions = (raw_predictions == -1).astype(int)

    # decision_function scores (lower = more anomalous)
    scores = model.decision_function(X_test_scaled)

    # Convert to probability-like values for ROC AUC
    # Negate because lower score = more anomalous = higher fraud probability
    fraud_probs = -scores

    metrics = {
        "test_samples":     int(len(y_test)),
        "fraud_samples":    int((y_test == 1).sum()),
        "normal_samples":   int((y_test == 0).sum()),
        "accuracy":         float(round((predictions == y_test).mean(), 4)),
        "fraud_recall":     float(round((predictions[y_test == 1] == 1).mean(), 4)),
        "false_positive_rate": float(round((predictions[y_test == 0] == 1).mean(), 4)),
        "f1_score":         float(round(f1_score(y_test, predictions, zero_division=0), 4)),
        "roc_auc":          float(round(roc_auc_score(y_test, fraud_probs), 4)),
    }

    logger.info("── Evaluation Results ──────────────────")
    logger.info(f"  Accuracy:           {metrics['accuracy']*100:.1f}%")
    logger.info(f"  Fraud Recall:       {metrics['fraud_recall']*100:.1f}%  (% of fraud caught)")
    logger.info(f"  False Positive Rate:{metrics['false_positive_rate']*100:.1f}%  (% of legit flagged)")
    logger.info(f"  F1 Score:           {metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
    logger.info("────────────────────────────────────────")

    # Full classification report
    print("\n── Classification Report ──")
    print(classification_report(
        y_test, predictions,
        target_names=["Normal (0)", "Fraud (1)"],
        zero_division=0
    ))

    print("── Confusion Matrix ──")
    cm = confusion_matrix(y_test, predictions)
    print(f"              Predicted Normal  Predicted Fraud")
    print(f"Actual Normal      {cm[0][0]:6d}           {cm[0][1]:6d}")
    print(f"Actual Fraud       {cm[1][0]:6d}           {cm[1][1]:6d}")

    return metrics


def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    metrics: dict,
    feature_columns: list,
    n_training_samples: int,
    dataset_path: str,
):
    """Saves model, scaler, and metadata to models/ directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved: {MODEL_PATH}")

    # Save scaler (MUST save alongside model — required for consistent predictions)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved: {SCALER_PATH}")

    # Save metadata (for RBI model governance + audit trail)
    metadata = {
        "trained_at":          datetime.utcnow().isoformat() + "Z",
        "model_type":          "IsolationForest",
        "n_estimators":        IF_N_ESTIMATORS,
        "contamination":       IF_CONTAMINATION,
        "n_training_samples":  n_training_samples,
        "dataset_source":      os.path.basename(dataset_path),
        "feature_columns":     feature_columns,
        "n_features":          len(feature_columns),
        "evaluation_metrics":  metrics,
        "target_false_positive_rate": 0.05,
        "actual_false_positive_rate": metrics.get("false_positive_rate"),
        "model_version":       "1.0.0",
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {METADATA_PATH}")


def main(dataset_path: str = None):
    dataset_path = dataset_path or DATASET_PATH

    logger.info("═" * 50)
    logger.info("TrustNet — Behavior Monitoring Model Training")
    logger.info("═" * 50)

    # 1. Load data
    X, y = load_dataset(dataset_path)

    # 2. Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # 3. Scale features
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # 4. Train
    model = train_model(X_train_scaled)

    # 5. Evaluate
    metrics = evaluate_model(model, scaler, X_test, y_test)

    # 6. Save everything
    save_artifacts(
        model            = model,
        scaler           = scaler,
        metrics          = metrics,
        feature_columns  = FEATURE_COLUMNS,
        n_training_samples = len(X_train),
        dataset_path     = dataset_path,
    )

    logger.info("Training complete. Run main.py to test predictions.")
    return model, scaler, metrics


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)