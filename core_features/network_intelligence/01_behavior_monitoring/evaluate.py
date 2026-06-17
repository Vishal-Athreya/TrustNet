# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluates the trained model on the dataset.
# Generates accuracy metrics, ROC curve, confusion matrix.
# Run this after train_baseline.py to verify model quality.
#
# Usage:
#   python evaluate.py
#   python evaluate.py /path/to/dataset.csv
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    average_precision_score,
)

sys.path.insert(0, os.path.dirname(__file__))

from config import DATASET_PATH, FEATURE_COLUMNS, METADATA_PATH
from isolation_forest import load_model
from logger import get_logger

logger = get_logger("evaluate")


def load_test_data(csv_path: str) -> tuple:
    """Loads dataset and returns features and labels."""
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = df[FEATURE_COLUMNS].dropna()
    y = df.loc[X.index, "label"]
    return X, y


def run_evaluation(csv_path: str = None):
    """
    Full evaluation pipeline:
    1. Load model + test data
    2. Generate predictions
    3. Print all metrics
    4. Save results to model_metadata.json
    """
    csv_path = csv_path or DATASET_PATH

    print("\n" + "═" * 60)
    print("  TrustNet — Behavior Model Evaluation")
    print("═" * 60)

    # ── Load ──────────────────────────────────────────────────────────
    model, scaler, metadata = load_model()
    X, y = load_test_data(csv_path)

    print(f"\nDataset   : {csv_path}")
    print(f"Total rows: {len(X):,}")
    print(f"Normal    : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"Fraud     : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    # ── Predict ───────────────────────────────────────────────────────
    X_scaled = scaler.transform(X)
    raw_preds = model.predict(X_scaled)         # -1=anomaly, 1=normal
    predictions = (raw_preds == -1).astype(int) # convert to 0/1

    # decision_function: lower = more anomalous = higher fraud probability
    scores = model.decision_function(X_scaled)
    fraud_probs = -scores  # negate so higher = more fraud

    # ── Core Metrics ──────────────────────────────────────────────────
    accuracy    = (predictions == y).mean()
    fraud_recall = (predictions[y == 1] == 1).mean()  # % of fraud caught
    fpr_val      = (predictions[y == 0] == 1).mean()  # % of legit wrongly flagged
    f1           = f1_score(y, predictions, zero_division=0)
    roc_auc      = roc_auc_score(y, fraud_probs)
    avg_precision = average_precision_score(y, fraud_probs)

    print("\n── Core Metrics ──────────────────────────────────────────")
    print(f"  Accuracy              : {accuracy*100:.2f}%")
    print(f"  Fraud Detection Rate  : {fraud_recall*100:.2f}%  (% of fraud caught)")
    print(f"  False Positive Rate   : {fpr_val*100:.2f}%  (% of legit sessions flagged)")
    print(f"  F1 Score              : {f1:.4f}")
    print(f"  ROC AUC               : {roc_auc:.4f}")
    print(f"  Avg Precision (PR AUC): {avg_precision:.4f}")

    # ── Target vs Actual ──────────────────────────────────────────────
    print("\n── Production Targets ────────────────────────────────────")
    targets = {
        "Fraud Detection Rate > 90%": (fraud_recall >= 0.90, f"{fraud_recall*100:.1f}%"),
        "False Positive Rate < 5%":   (fpr_val <= 0.05,      f"{fpr_val*100:.1f}%"),
        "ROC AUC > 0.85":             (roc_auc >= 0.85,      f"{roc_auc:.4f}"),
    }
    for target, (met, value) in targets.items():
        status = "✅ PASS" if met else "❌ FAIL"
        print(f"  {status}  {target} → actual: {value}")

    # ── Classification Report ─────────────────────────────────────────
    print("\n── Classification Report ──────────────────────────────────")
    print(classification_report(
        y, predictions,
        target_names=["Normal (0)", "Fraud (1)"],
        zero_division=0
    ))

    # ── Confusion Matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y, predictions)
    tn, fp, fn, tp = cm.ravel()
    print("── Confusion Matrix ────────────────────────────────────────")
    print(f"                    Predicted Normal    Predicted Fraud")
    print(f"  Actual Normal     {tn:8,}            {fp:8,}   ← false positives (legit flagged)")
    print(f"  Actual Fraud      {fn:8,}            {tp:8,}   ← true positives (fraud caught)")

    # ── Per Fraud Type Breakdown ──────────────────────────────────────
    try:
        df_full = pd.read_csv(csv_path)
        df_full = df_full.loc[X.index].copy()
        df_full["predicted"] = predictions

        fraud_df = df_full[df_full["label"] == 1]
        print("\n── Fraud Detection by Type ─────────────────────────────────")
        for fraud_type in fraud_df["fraud_type"].unique():
            subset = fraud_df[fraud_df["fraud_type"] == fraud_type]
            caught = (subset["predicted"] == 1).sum()
            total  = len(subset)
            rate   = caught / total * 100
            print(f"  {fraud_type:<30} {caught:4d}/{total:4d} caught ({rate:.1f}%)")
    except Exception as e:
        logger.warning(f"Could not compute per-type breakdown: {e}")

    # ── ROC Curve (text version) ──────────────────────────────────────
    fpr_curve, tpr_curve, thresholds = roc_curve(y, fraud_probs)
    print("\n── ROC Curve (selected thresholds) ─────────────────────────")
    print(f"  {'FPR':>8}  {'TPR (Recall)':>14}  {'Threshold':>12}")
    step = max(1, len(fpr_curve) // 8)
    for i in range(0, len(fpr_curve), step):
        print(f"  {fpr_curve[i]:8.4f}  {tpr_curve[i]:14.4f}  {thresholds[i]:12.4f}")

    # ── Score Distribution ────────────────────────────────────────────
    risk_scores = np.clip((0.5 - scores) * 100, 0, 100)
    normal_scores = risk_scores[y == 0]
    fraud_scores  = risk_scores[y == 1]

    print("\n── Risk Score Distribution ─────────────────────────────────")
    print(f"  Normal sessions  → mean: {normal_scores.mean():.1f}  "
          f"median: {np.median(normal_scores):.1f}  "
          f"max: {normal_scores.max():.1f}")
    print(f"  Fraud sessions   → mean: {fraud_scores.mean():.1f}  "
          f"median: {np.median(fraud_scores):.1f}  "
          f"min: {fraud_scores.min():.1f}")

    # ── Save updated metrics to metadata ─────────────────────────────
    eval_metrics = {
        "evaluated_at":         pd.Timestamp.utcnow().isoformat() + "Z",
        "dataset_size":         int(len(X)),
        "accuracy":             float(round(accuracy, 4)),
        "fraud_detection_rate": float(round(fraud_recall, 4)),
        "false_positive_rate":  float(round(fpr_val, 4)),
        "f1_score":             float(round(f1, 4)),
        "roc_auc":              float(round(roc_auc, 4)),
        "avg_precision":        float(round(avg_precision, 4)),
        "true_positives":       int(tp),
        "false_positives":      int(fp),
        "true_negatives":       int(tn),
        "false_negatives":      int(fn),
        "all_targets_met":      all(met for met, _ in targets.values()),
    }

    if metadata and os.path.exists(METADATA_PATH):
        metadata["last_evaluation"] = eval_metrics
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetrics saved to {METADATA_PATH}")

    print("\n" + "═" * 60)
    return eval_metrics


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(path)