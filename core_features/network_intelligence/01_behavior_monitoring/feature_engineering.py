# feature_engineering.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts raw session_data dict → clean 25-feature vector for the model.
# Keeps main.py and isolation_forest.py thin — all messy data cleaning
# and type coercion happens here in one place.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from datetime import datetime
from exceptions import FeatureEngineeringError, InvalidSessionDataError
from config import FEATURE_COLUMNS
from logger import get_logger

logger = get_logger("feature_engineering")

# ── Required keys that MUST be in session_data ────────────────────────────
REQUIRED_KEYS = [
    "user_agent",
    "ip_address",
    "hour_of_day",
    "failed_attempts",
    "session_duration",
]

# ── Safe defaults for optional keys ───────────────────────────────────────
DEFAULTS = {
    "day_of_week":             datetime.utcnow().weekday(),
    "is_weekend":              int(datetime.utcnow().weekday() >= 5),
    "num_devices_used":        1,
    "ip_is_vpn":               0,
    "ip_is_tor":               0,
    "ip_is_proxy":             0,
    "ip_reputation_score":     0.0,
    "is_headless_browser":     0,
    "user_agent_entropy":      3.5,
    "form_fill_speed":         2.5,
    "mouse_movement_score":    0.7,
    "copy_paste_detected":     0,
    "tab_switches":            1,
    "time_on_each_field":      5.0,
    "geolocation_mismatch":    0,
    "distance_from_branch_km": 10.0,
    "previous_applications":   0,
    "applications_same_ip":    1,
    "applications_same_device":1,
    "pan_seen_before":         0,
    "aadhaar_seen_before":     0,
    "velocity_score":          50.0,
}


def validate_session_data(session_data: dict):
    """
    Checks that all required keys are present.
    Raises InvalidSessionDataError with a clear message if any are missing.
    """
    missing = [k for k in REQUIRED_KEYS if k not in session_data]
    if missing:
        raise InvalidSessionDataError(
            f"session_data is missing required keys: {missing}. "
            f"Required keys are: {REQUIRED_KEYS}"
        )

    # Type validation for critical numeric fields
    numeric_fields = ["hour_of_day", "failed_attempts", "session_duration"]
    for field in numeric_fields:
        try:
            float(session_data[field])
        except (ValueError, TypeError):
            raise InvalidSessionDataError(
                f"Field '{field}' must be numeric, got: {session_data[field]!r}"
            )

    # Range validation
    hour = int(session_data["hour_of_day"])
    if not (0 <= hour <= 23):
        raise InvalidSessionDataError(
            f"hour_of_day must be 0-23, got: {hour}"
        )


def build_feature_vector(session_data: dict, device_features: dict) -> pd.DataFrame:
    """
    Merges session_data + device_features into a single-row DataFrame
    with exactly the 25 columns the model expects, in the correct order.

    Parameters:
        session_data:    Raw dict from the API request
        device_features: Output of device_fingerprint.extract_device_features()

    Returns:
        Single-row pd.DataFrame with columns matching FEATURE_COLUMNS
    """
    try:
        row = {}

        # ── Time features ──────────────────────────────────────────────
        row["hour_of_day"]   = int(session_data["hour_of_day"])
        row["day_of_week"]   = int(session_data.get(
            "day_of_week", DEFAULTS["day_of_week"]
        ))
        row["is_weekend"]    = int(session_data.get(
            "is_weekend", int(row["day_of_week"] >= 5)
        ))

        # ── Session behavior features ──────────────────────────────────
        row["failed_attempts"]    = int(session_data["failed_attempts"])
        row["session_duration"]   = float(session_data["session_duration"])
        row["num_devices_used"]   = int(session_data.get(
            "num_devices_used", DEFAULTS["num_devices_used"]
        ))

        # ── IP/Network features ────────────────────────────────────────
        row["ip_is_vpn"]= int(session_data.get("ip_is_vpn",    DEFAULTS["ip_is_vpn"]))
        row["ip_is_tor"]= int(session_data.get("ip_is_tor",    DEFAULTS["ip_is_tor"]))
        row["ip_is_proxy"]= int(session_data.get("ip_is_proxy",  DEFAULTS["ip_is_proxy"]))
        row["ip_reputation_score"] = float(session_data.get(
            "ip_reputation_score", DEFAULTS["ip_reputation_score"]
        ))

        # ── Device features (from device_fingerprint.py) ───────────────
        row["is_headless_browser"] = int(device_features.get(
            "is_headless_browser", DEFAULTS["is_headless_browser"]
        ))
        row["user_agent_entropy"]  = float(device_features.get(
            "user_agent_entropy", DEFAULTS["user_agent_entropy"]
        ))

        # ── Form interaction features ──────────────────────────────────
        row["form_fill_speed"]     = float(session_data.get(
            "form_fill_speed", DEFAULTS["form_fill_speed"]
        ))
        row["mouse_movement_score"]= float(session_data.get(
            "mouse_movement_score", DEFAULTS["mouse_movement_score"]
        ))
        row["copy_paste_detected"] = int(session_data.get(
            "copy_paste_detected", DEFAULTS["copy_paste_detected"]
        ))
        row["tab_switches"]        = int(session_data.get(
            "tab_switches", DEFAULTS["tab_switches"]
        ))
        row["time_on_each_field"]  = float(session_data.get(
            "time_on_each_field", DEFAULTS["time_on_each_field"]
        ))

        # ── Geolocation features ───────────────────────────────────────
        row["geolocation_mismatch"]    = int(session_data.get(
            "geolocation_mismatch", DEFAULTS["geolocation_mismatch"]
        ))
        row["distance_from_branch_km"] = float(session_data.get(
            "distance_from_branch_km", DEFAULTS["distance_from_branch_km"]
        ))

        # ── Historical/cross-reference features ───────────────────────
        row["previous_applications"]    = int(session_data.get(
            "previous_applications", DEFAULTS["previous_applications"]
        ))
        row["applications_same_ip"]     = int(session_data.get(
            "applications_same_ip", DEFAULTS["applications_same_ip"]
        ))
        row["applications_same_device"] = int(session_data.get(
            "applications_same_device", DEFAULTS["applications_same_device"]
        ))
        row["pan_seen_before"]          = int(session_data.get(
            "pan_seen_before", DEFAULTS["pan_seen_before"]
        ))
        row["aadhaar_seen_before"]      = int(session_data.get(
            "aadhaar_seen_before", DEFAULTS["aadhaar_seen_before"]
        ))
        row["velocity_score"] = float(session_data.get(
        "velocity_score", DEFAULTS["velocity_score"]
        ))

        # Build DataFrame with exact column order the model expects
        df = pd.DataFrame([row])[FEATURE_COLUMNS]
        logger.debug(f"Feature vector built: {row}")
        return df

    except (InvalidSessionDataError, FeatureEngineeringError):
        raise
    except Exception as e:
        raise FeatureEngineeringError(
            f"Failed to build feature vector: {type(e).__name__}: {e}"
        )


if __name__ == "__main__":
    from device_fingerprint import extract_device_features

    test_session = {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "ip_address": "49.36.20.10",
        "hour_of_day": 14,
        "request_frequency": 2.0,
        "failed_attempts": 0,
        "session_duration": 300.0,
    }

    device_feats = extract_device_features(test_session)
    df = build_feature_vector(test_session, device_feats)
    print("Feature vector shape:", df.shape)
    print(df.to_string())