# flag_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts raw feature values + model score into human-readable flags.
# Each flag maps to one sentence that appears on the bank officer's dashboard.
# All thresholds come from config.py — tune there, not here.
# ─────────────────────────────────────────────────────────────────────────────

from config import (
    MAX_SAFE_HOUR_START, MAX_SAFE_HOUR_END,
    MAX_SAFE_FREQ, MAX_SAFE_FAILED,
    MIN_SAFE_DURATION, MAX_SAFE_DEVICES,
    MAX_SAFE_IP_REPUTATION, MIN_SAFE_UA_ENTROPY,
    MAX_SAFE_FILL_SPEED, MIN_SAFE_MOUSE_SCORE,
    MAX_SAFE_SAME_IP_APPS, MAX_SAFE_DISTANCE_KM,
    HIGH_RISK_THRESHOLD, SUSPICIOUS_THRESHOLD,
)
from logger import get_logger

logger = get_logger("flag_engine")


# ── Flag definitions ────────────────────────────────────────────────────────
# Each entry: (flag_key, dashboard_message, severity)
# severity: "HIGH" | "MEDIUM" | "LOW"

FLAG_DEFINITIONS = {
    "off_hours_submission": {
        "message": "Application submitted outside business hours (before 6 AM or after 10 PM).",
        "severity": "MEDIUM",
    },
    "high_request_frequency": {
        "message": "Unusually high number of requests per minute from this IP — consistent with automated/bot activity.",
        "severity": "HIGH",
    },
    "multiple_failed_attempts": {
        "message": "Multiple failed login attempts before this session — possible credential stuffing.",
        "severity": "HIGH",
    },
    "suspiciously_fast_submission": {
        "message": "Entire application submitted in under 30 seconds — impossible for a human to complete manually.",
        "severity": "HIGH",
    },
    "multiple_devices_detected": {
        "message": "Multiple distinct devices detected within the same session — possible shared fraud operation.",
        "severity": "MEDIUM",
    },
    "vpn_detected": {
        "message": "IP address belongs to a known VPN provider — applicant is masking their real location.",
        "severity": "HIGH",
    },
    "tor_detected": {
        "message": "IP address is a known Tor exit node — very high anonymization, strongly associated with fraud.",
        "severity": "HIGH",
    },
    "proxy_detected": {
        "message": "IP address is a known proxy server — applicant is masking their real location.",
        "severity": "MEDIUM",
    },
    "suspicious_ip_reputation": {
        "message": "IP address has a high abuse reputation score — previously flagged for malicious activity.",
        "severity": "HIGH",
    },
    "headless_browser_detected": {
        "message": "Headless browser or automation tool detected (Selenium/Puppeteer/curl) — not a real user.",
        "severity": "HIGH",
    },
    "low_user_agent_entropy": {
        "message": "User-Agent string appears fake or auto-generated — inconsistent with a real browser.",
        "severity": "MEDIUM",
    },
    "bot_typing_speed": {
        "message": "Form fill speed is superhuman — fields were filled programmatically, not typed by a human.",
        "severity": "HIGH",
    },
    "no_mouse_movement": {
        "message": "No meaningful mouse movement detected during session — consistent with automated form submission.",
        "severity": "HIGH",
    },
    "copy_paste_all_fields": {
        "message": "Copy-paste detected across form fields — possible pre-filled or scripted submission.",
        "severity": "LOW",
    },
    "geolocation_mismatch": {
        "message": "IP address geolocation does not match the address provided in the application.",
        "severity": "MEDIUM",
    },
    "high_distance_from_branch": {
        "message": "Applicant is submitting from a location more than 500 km from the nearest bank branch.",
        "severity": "MEDIUM",
    },
    "fraud_ring_ip": {
        "message": "More than 5 loan applications have been submitted from this IP address — possible fraud ring.",
        "severity": "HIGH",
    },
    "pan_reuse": {
        "message": "This PAN has been seen in a previous application — possible identity reuse.",
        "severity": "HIGH",
    },
    "aadhaar_reuse": {
        "message": "This Aadhaar hash has been seen in a previous application — possible identity reuse.",
        "severity": "HIGH",
    },
    "isolation_forest_anomaly": {
        "message": "Session behavior is statistically anomalous compared to 97,000 legitimate sessions in the training baseline.",
        "severity": "HIGH",
    },
}


def generate_flags(features: dict, model_result: dict, device_features: dict) -> list[dict]:
    """
    Evaluates all feature thresholds and returns a list of triggered flags.

    Parameters:
        features:       Raw feature values (from feature_engineering.py)
        model_result:   Output of isolation_forest.score_session()
        device_features:Output of device_fingerprint.extract_device_features()

    Returns:
        List of dicts: [{"flag": str, "message": str, "severity": str}, ...]
        Sorted by severity: HIGH first, then MEDIUM, then LOW.
    """
    triggered = []

    def add_flag(flag_key: str):
        defn = FLAG_DEFINITIONS.get(flag_key)
        if defn:
            triggered.append({
                "flag":     flag_key,
                "message":  defn["message"],
                "severity": defn["severity"],
            })

    # ── Time ──────────────────────────────────────────────────────────────
    hour = features.get("hour_of_day", 12)
    if hour < MAX_SAFE_HOUR_START or hour > MAX_SAFE_HOUR_END:
        add_flag("off_hours_submission")

    # ── Session behavior ───────────────────────────────────────────────────
    if features.get("request_frequency", 0) > MAX_SAFE_FREQ:
        add_flag("high_request_frequency")

    if features.get("failed_attempts", 0) >= MAX_SAFE_FAILED:
        add_flag("multiple_failed_attempts")

    if features.get("session_duration", 999) < MIN_SAFE_DURATION:
        add_flag("suspiciously_fast_submission")

    if features.get("num_devices_used", 1) > MAX_SAFE_DEVICES:
        add_flag("multiple_devices_detected")

    # ── IP/Network ─────────────────────────────────────────────────────────
    if features.get("ip_is_vpn", 0):
        add_flag("vpn_detected")

    if features.get("ip_is_tor", 0):
        add_flag("tor_detected")

    if features.get("ip_is_proxy", 0):
        add_flag("proxy_detected")

    if features.get("ip_reputation_score", 0) > MAX_SAFE_IP_REPUTATION:
        add_flag("suspicious_ip_reputation")

    # ── Device ─────────────────────────────────────────────────────────────
    if device_features.get("is_headless_browser", 0):
        add_flag("headless_browser_detected")

    if device_features.get("user_agent_entropy", 5.0) < MIN_SAFE_UA_ENTROPY:
        add_flag("low_user_agent_entropy")

    # ── Form interaction ───────────────────────────────────────────────────
    if features.get("form_fill_speed", 0) > MAX_SAFE_FILL_SPEED:
        add_flag("bot_typing_speed")

    if features.get("mouse_movement_score", 1.0) < MIN_SAFE_MOUSE_SCORE:
        add_flag("no_mouse_movement")

    if features.get("copy_paste_detected", 0):
        add_flag("copy_paste_all_fields")

    # ── Geolocation ────────────────────────────────────────────────────────
    if features.get("geolocation_mismatch", 0):
        add_flag("geolocation_mismatch")

    if features.get("distance_from_branch_km", 0) > MAX_SAFE_DISTANCE_KM:
        add_flag("high_distance_from_branch")

    # ── Historical ─────────────────────────────────────────────────────────
    if features.get("applications_same_ip", 0) > MAX_SAFE_SAME_IP_APPS:
        add_flag("fraud_ring_ip")

    if features.get("pan_seen_before", 0):
        add_flag("pan_reuse")

    if features.get("aadhaar_seen_before", 0):
        add_flag("aadhaar_reuse")

    # ── Model verdict ──────────────────────────────────────────────────────
    if model_result.get("is_anomaly"):
        add_flag("isolation_forest_anomaly")

    # Sort: HIGH → MEDIUM → LOW
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    triggered.sort(key=lambda f: severity_order.get(f["severity"], 3))

    logger.debug(f"Flags triggered: {[f['flag'] for f in triggered]}")
    return triggered


def determine_risk_level(score: int) -> str:
    """Maps numeric score to a risk level label for the dashboard."""
    if score >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif score >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    else:
        return "CLEAN"
    
def score_vpn(session_data: dict) -> int:
    """
    VPN alone is not fraud.
    VPN + other signals = fraud.
    """
    if not session_data.get("ip_is_vpn"):
        return 0  # no VPN, no concern

    # VPN but applying during business hours
    # with normal speed = probably IT employee
    hour     = session_data.get("hour_of_day", 12)
    speed    = session_data.get("form_fill_speed", 2.0)
    mouse    = session_data.get("mouse_movement_score", 0.5)
    duration = session_data.get("session_duration", 300)

    if (6 <= hour <= 22 and
        speed < 10 and
        mouse > 0.3 and
        duration > 60):
        return 10   # LOW concern — likely IT professional

    return 40       # HIGH concern — VPN + other bad signals


if __name__ == "__main__":
    # Manual test
    test_features = {
        "hour_of_day": 2,
        "request_frequency": 25,
        "failed_attempts": 5,
        "session_duration": 8,
        "num_devices_used": 3,
        "ip_is_vpn": 1,
        "ip_is_tor": 0,
        "ip_is_proxy": 1,
        "ip_reputation_score": 80,
        "form_fill_speed": 150,
        "mouse_movement_score": 0.01,
        "copy_paste_detected": 1,
        "geolocation_mismatch": 1,
        "distance_from_branch_km": 800,
        "applications_same_ip": 12,
        "pan_seen_before": 1,
        "aadhaar_seen_before": 0,
    }
    test_device = {"is_headless_browser": 1, "user_agent_entropy": 0.8}
    test_model  = {"is_anomaly": True}

    flags = generate_flags(test_features, test_model, test_device)
    print(f"Flags triggered ({len(flags)}):")
    for f in flags:
        print(f"  [{f['severity']}] {f['flag']}: {f['message']}")