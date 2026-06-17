# config.py
# ─────────────────────────────────────────────────────────────────────────────
# All configurable thresholds and settings for behavior monitoring.
# Change values here — they apply everywhere automatically.
# ─────────────────────────────────────────────────────────────────────────────

# ── Risk Score Thresholds ─────────────────────────────────────────────────
HIGH_RISK_THRESHOLD  = 65   # score >= 65 → HIGH RISK  (red)
SUSPICIOUS_THRESHOLD = 40   # score >= 40 → SUSPICIOUS (amber)
CLEAN_THRESHOLD      = 40   # score <  40 → CLEAN      (green)

# ── Isolation Forest Settings ─────────────────────────────────────────────
IF_N_ESTIMATORS  = 200
IF_CONTAMINATION = 0.03
IF_RANDOM_STATE  = 42
IF_N_JOBS        = -1

# ── Feature Thresholds (used by flag_engine.py) ───────────────────────────
MAX_SAFE_HOUR_START     = 6     # before 6 AM is suspicious
MAX_SAFE_HOUR_END       = 22    # after 10 PM is suspicious
MAX_SAFE_FREQ           = 10    # requests/min above this = suspicious
MAX_SAFE_FAILED         = 3     # failed attempts above this = suspicious
MIN_SAFE_DURATION       = 30    # session under 30 seconds = too fast
MAX_SAFE_DEVICES        = 1     # more than 1 device = suspicious
MAX_SAFE_IP_REPUTATION  = 40    # reputation score above 40 = suspicious IP
MIN_SAFE_UA_ENTROPY     = 2.5   # below 2.5 = likely fake user agent
MAX_SAFE_FILL_SPEED     = 10    # chars/sec above 10 = likely bot
MIN_SAFE_MOUSE_SCORE    = 0.2   # below 0.2 = no real mouse movement
MAX_SAFE_SAME_IP_APPS   = 5     # more than 5 apps from same IP = fraud ring
MAX_SAFE_DISTANCE_KM    = 500   # applying from more than 500km = suspicious

# ── File Paths ────────────────────────────────────────────────────────────
import os
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
LOG_DIR       = os.path.join(BASE_DIR, "logs")
MODEL_PATH    = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
LOG_PATH      = os.path.join(LOG_DIR,   "behavior.log")
DATASET_PATH  = os.path.join(BASE_DIR,  "trustnet_behavior_dataset.csv")

# ── The 25 features the model uses ───────────────────────────────────────
# NEW — matches what the model was trained on
FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "failed_attempts",
    "session_duration",
    "num_devices_used",
    "ip_is_vpn",
    "ip_is_tor",
    "ip_is_proxy",
    "ip_reputation_score",
    "is_headless_browser",
    "user_agent_entropy",
    "form_fill_speed",
    "mouse_movement_score",
    "copy_paste_detected",
    "tab_switches",
    "time_on_each_field",
    "geolocation_mismatch",
    "distance_from_branch_km",
    "previous_applications",
    "applications_same_ip",
    "applications_same_device",
    "pan_seen_before",
    "aadhaar_seen_before",
    "velocity_score",          # ← add this
]