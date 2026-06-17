# isolation_forest.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads the trained Isolation Forest + scaler and scores new sessions.
# Model and scaler are loaded once and cached in memory for performance.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import joblib
import json
import os
import pandas as pd

from config import MODEL_PATH, SCALER_PATH, METADATA_PATH
from exceptions import ModelNotFoundError, ModelLoadError, ScoringError
from logger import get_logger

logger = get_logger("isolation_forest")

# ── Module-level cache — loaded once per process ──────────────────────────
_model   = None
_scaler  = None
_metadata = None


def load_model():
    """
    Loads model + scaler from disk into memory.
    Cached after first call — subsequent calls return instantly.
    Raises ModelNotFoundError if pkl files don't exist.
    Raises ModelLoadError if files are corrupted.
    """
    global _model, _scaler, _metadata

    if _model is not None:
        return _model, _scaler, _metadata

    # Check files exist before attempting to load
    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Fix: run `python train_baseline.py` to train and save the model."
        )
    if not os.path.exists(SCALER_PATH):
        raise ModelNotFoundError(
            f"Scaler file not found: {SCALER_PATH}\n"
            f"Fix: run `python train_baseline.py` to regenerate both model and scaler."
        )

    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Model and scaler loaded successfully")
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load model/scaler: {type(e).__name__}: {e}\n"
            f"The file may be corrupted. Re-run train_baseline.py."
        )

    # Load metadata if available
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
        logger.info(
            f"Model metadata: trained_at={_metadata.get('trained_at')}, "
            f"version={_metadata.get('model_version')}, "
            f"n_samples={_metadata.get('n_training_samples'):,}"
        )

    return _model, _scaler, _metadata


def score_session(feature_df: pd.DataFrame) -> dict:
    """
    Scores a single session using the trained model.
    Works with both RandomForestClassifier and IsolationForest.
    """
    try:
        model, scaler, _ = load_model()

        feature_scaled = scaler.transform(feature_df)

        # ── Detect model type and score accordingly ────────────────────
        model_type = type(model).__name__

        if model_type == "RandomForestClassifier":
            # Random Forest uses predict_proba — returns fraud probability directly
            fraud_prob  = float(model.predict_proba(feature_scaled)[0][1])
            prediction  = int(model.predict(feature_scaled)[0])
            risk_score  = int(fraud_prob * 100)
            is_anomaly  = prediction == 1
            raw_score   = fraud_prob

        else:
            # Isolation Forest uses decision_function
            raw_score  = float(model.decision_function(feature_scaled)[0])
            prediction = int(model.predict(feature_scaled)[0])
            risk_score = int(np.clip((0.5 - raw_score) * 100, 0, 100))
            is_anomaly = (prediction == -1)

        logger.debug(
            f"Model: {model_type} | "
            f"risk_score={risk_score} | "
            f"is_anomaly={is_anomaly}"
        )

        return {
            "risk_score": risk_score,
            "is_anomaly": is_anomaly,
            "raw_score":  raw_score,
        }

    except (ModelNotFoundError, ModelLoadError):
        raise
    except Exception as e:
        raise ScoringError(
            f"Scoring failed: {type(e).__name__}: {e}"
        )

def get_model_info() -> dict:
    """
    Returns model metadata — used by the admin dashboard
    and for RBI model governance reporting.
    """
    _, _, metadata = load_model()
    return metadata or {"status": "metadata not available"}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from feature_engineering import build_feature_vector, validate_session_data
    from device_fingerprint import extract_device_features

    # Test 1: Normal session
    normal = {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "ip_address": "49.36.20.10",
        "hour_of_day": 14,
        "day_of_week": 2,
        "request_frequency": 2.0,
        "failed_attempts": 0,
        "session_duration": 300.0,
        "form_fill_speed": 2.5,
        "mouse_movement_score": 0.78,
    }

    # Test 2: Suspicious session
    suspicious = {
        "user_agent": "python-requests/2.31.0",
        "ip_address": "185.220.101.5",
        "hour_of_day": 2,
        "day_of_week": 6,
        "request_frequency": 28.0,
        "failed_attempts": 6,
        "session_duration": 7.0,
        "ip_is_vpn": 1,
        "ip_is_tor": 1,
        "ip_reputation_score": 92.0,
        "form_fill_speed": 200.0,
        "mouse_movement_score": 0.0,
        "copy_paste_detected": 1,
        "applications_same_ip": 18,
        "pan_seen_before": 1,
    }

    for name, sess in [("Normal", normal), ("Suspicious", suspicious)]:
        print(f"\n── {name} Session ──")
        validate_session_data(sess)
        dev = extract_device_features(sess)
        df  = build_feature_vector(sess, dev)
        result = score_session(df)
        print(f"  risk_score : {result['risk_score']}")
        print(f"  is_anomaly : {result['is_anomaly']}")
        print(f"  raw_score  : {result['raw_score']:.4f}")