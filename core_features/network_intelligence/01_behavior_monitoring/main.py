# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point for behavior monitoring.
# This is the ONLY function the backend (network_layer/pipeline.py) calls.
# Everything else is internal to this module.
#
# Usage from backend:
#   from core_features.network_intelligence.behavior_monitoring.main import run_behavior_analysis
#   result = run_behavior_analysis(session_data)
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from config import HIGH_RISK_THRESHOLD, SUSPICIOUS_THRESHOLD
from exceptions import (
    BehaviorMonitoringError,
    InvalidSessionDataError,
    ModelNotFoundError,
)
from logger import get_logger, log_analysis_result, log_error
from device_fingerprint import extract_device_features
from feature_engineering import validate_session_data, build_feature_vector
from isolation_forest import score_session
from flag_engine import generate_flags, determine_risk_level
from vpn_detector import check_ip_before_upload

logger = get_logger("main")


def run_behavior_analysis(session_data: dict) -> dict:
    """
    Main entry point — called by network_layer/pipeline.py.

    What this does (in order):
      1. Validates session_data has required keys
      2. Extracts device fingerprint + browser/OS info
      3. Builds the 25-feature vector the model expects
      4. Scores with Isolation Forest → risk_score (0-100)
      5. Generates human-readable flags for the dashboard
      6. Determines risk level (HIGH / SUSPICIOUS / CLEAN)
      7. Logs the result for the audit trail
      8. Returns one clean dict

    Parameters:
        session_data (dict): Raw session information. Required keys:
            - user_agent        (str)   Browser User-Agent string
            - ip_address        (str)   Client IP address
            - hour_of_day       (int)   0-23
            - request_frequency (float) Requests per minute from this IP
            - failed_attempts   (int)   Failed logins before this session
            - session_duration  (float) Seconds spent on the form

        Optional keys (safe defaults used if missing):
            - screen_resolution      (str)   e.g. "1920x1080"
            - accept_language        (str)   e.g. "en-IN,hi;q=0.9"
            - timezone               (str)   e.g. "Asia/Kolkata"
            - day_of_week            (int)   0=Monday, 6=Sunday
            - is_weekend             (int)   0 or 1
            - num_devices_used       (int)   Distinct devices in session
            - ip_is_vpn              (int)   0 or 1   (overwritten by live check below)
            - ip_is_tor              (int)   0 or 1   (overwritten by live check below)
            - ip_is_proxy            (int)   0 or 1   (overwritten by live check below)
            - ip_reputation_score    (float) 0-100    (overwritten by live check below)
            - form_fill_speed        (float) Characters per second
            - mouse_movement_score   (float) 0.0-1.0
            - copy_paste_detected    (int)   0 or 1
            - tab_switches           (int)   Number of tab switches
            - time_on_each_field     (float) Avg seconds per field
            - geolocation_mismatch   (int)   0 or 1
            - distance_from_branch_km(float) KM from nearest branch
            - previous_applications  (int)   Prior applications count
            - applications_same_ip   (int)   Apps from this IP
            - applications_same_device(int)  Apps from this device
            - pan_seen_before        (int)   0 or 1
            - aadhaar_seen_before    (int)   0 or 1

    Returns:
        {
            "score":              int,        # 0-100, higher = more suspicious
            "risk_level":         str,        # "HIGH" | "SUSPICIOUS" | "CLEAN"
            "is_anomaly":         bool,       # Isolation Forest verdict
            "flags":              list[dict], # Human-readable flags for dashboard
            "device_fingerprint": str,        # SHA-256 hash for Fraud Memory
            "browser_type":       str,        # Detected browser
            "os_type":            str,        # Detected OS
            "is_headless_browser":bool,       # True if bot/automation detected
            "analyzed_at":        str,        # UTC ISO timestamp
            "details": {
                "raw_isolation_forest_score": float,
                "features_used":              dict,
            }
        }

    Raises:
        InvalidSessionDataError  if required keys are missing or invalid
        ModelNotFoundError       if model hasn't been trained yet
        BehaviorMonitoringError  for all other internal errors
    """
    session_id = session_data.get("session_id", f"unknown_{datetime.now().timestamp()}")
    logger.info(f"Analysis started | session_id={session_id}")

    try:
        # ── Step 0: VPN / Proxy / Tor check — block before anything else
        ip_check = check_ip_before_upload(session_data)
        if not ip_check["allowed"]:
            logger.warning(f"Upload BLOCKED for session_id={session_id} | reason={ip_check['action']}")
            return {
                "score":               100,
                "risk_level":          "HIGH",
                "is_anomaly":          True,
                "blocked":             True,
                "block_reason":        ip_check["message"],
                "flags": [{
                    "flag":     "upload_blocked_vpn_tor_proxy",
                    "message":  ip_check["message"],
                    "severity": "HIGH",
                }],
                "device_fingerprint":  "",
                "browser_type":        "Unknown",
                "os_type":             "Unknown",
                "is_headless_browser": False,
                "analyzed_at":         datetime.now(timezone.utc).isoformat(),
                "details":             {"vpn_check": ip_check["log_entry"]},
            }

        # Not hard-blocked, but still feed the LIVE detection result into
        # session_data before building the feature vector. Without this,
        # a WARN-level proxy/datacenter IP (score 20-39) never reaches the
        # Isolation Forest model — it would silently keep whatever
        # ip_is_vpn / ip_is_proxy / ip_reputation_score values the caller
        # happened to pass in (often just 0 / 0.0 placeholders).
        session_data["ip_is_vpn"]           = int(ip_check["is_vpn"])
        session_data["ip_is_tor"]           = int(ip_check["is_tor"])
        session_data["ip_is_proxy"]         = int(ip_check["is_proxy"])
        session_data["ip_reputation_score"] = float(ip_check["threat_score"])

        # ── Step 1: Validate input ─────────────────────────────────────
        validate_session_data(session_data)

        # ── Step 2: Extract device fingerprint and browser info ────────
        device_features = extract_device_features(session_data)

        # ── Step 3: Build 25-feature vector ───────────────────────────
        feature_df = build_feature_vector(session_data, device_features)

        # Keep a plain dict copy for flag engine + details output
        features_dict = feature_df.iloc[0].to_dict()

        # ── Step 4: Score with Isolation Forest ────────────────────────
        model_result = score_session(feature_df)

        # ── Step 5: Generate human-readable flags ──────────────────────
        flags = generate_flags(features_dict, model_result, device_features)

        # ── Step 6: Determine risk level ───────────────────────────────
        risk_level = determine_risk_level(model_result["risk_score"])

        # ── Step 7: Assemble result ────────────────────────────────────
        result = {
            "score":               model_result["risk_score"],
            "risk_level":          risk_level,
            "is_anomaly":          model_result["is_anomaly"],
            "flags":               flags,
            "device_fingerprint":  device_features["device_fingerprint"],
            "browser_type":        device_features["browser_type"],
            "os_type":             device_features["os_type"],
            "is_headless_browser": bool(device_features["is_headless_browser"]),
            "analyzed_at":         datetime.now(timezone.utc).isoformat(),
            "details": {
                "raw_isolation_forest_score": model_result["raw_score"],
                "features_used":              features_dict,
                "vpn_check":                  ip_check["log_entry"],
            },
        }

        # ── Step 8: Log for audit trail ────────────────────────────────
        log_analysis_result(logger, session_id, result)
        logger.info(
            f"Analysis complete | session_id={session_id} | "
            f"score={result['score']} | risk_level={risk_level} | "
            f"flags={len(flags)}"
        )

        return result

    except (InvalidSessionDataError, ModelNotFoundError):
        # Re-raise known errors — backend handles these specifically
        raise

    except BehaviorMonitoringError as e:
        log_error(logger, session_id, e)
        raise

    except Exception as e:
        # Wrap unexpected errors so backend always gets a typed exception
        log_error(logger, session_id, e)
        raise BehaviorMonitoringError(
            f"Unexpected error during behavior analysis: {type(e).__name__}: {e}"
        ) from e


if __name__ == "__main__":
    # ── Quick demo ─────────────────────────────────────────────────────

    test_cases = [
        {
            "name": "Normal Rural Applicant (Bihar, Kisan Credit Card)",
            "data": {
                "session_id": "TEST-001",
                "user_agent": "Mozilla/5.0 (Linux; Android 10; Redmi 9) AppleWebKit/537.36 Chrome/96.0.4664.45 Mobile Safari/537.36",
                "ip_address": "49.36.20.10",
                "screen_resolution": "720x1560",
                "accept_language": "hi-IN,hi;q=0.9,en-IN;q=0.8",
                "timezone": "Asia/Kolkata",
                "hour_of_day": 9,
                "day_of_week": 1,
                "is_weekend": 0,
                "request_frequency": 1.2,
                "failed_attempts": 0,
                "session_duration": 520.0,
                "form_fill_speed": 1.1,
                "mouse_movement_score": 0.65,
                "copy_paste_detected": 0,
                "tab_switches": 2,
                "time_on_each_field": 8.5,
                "geolocation_mismatch": 0,
                "distance_from_branch_km": 12.0,
                "previous_applications": 0,
                "applications_same_ip": 1,
                "applications_same_device": 1,
                "pan_seen_before": 0,
                "aadhaar_seen_before": 0,
            },
        },
        {
            "name": "Normal Urban Applicant (Mumbai, Personal Loan)",
            "data": {
                "session_id": "TEST-002",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "ip_address": "122.161.45.20",
                "screen_resolution": "1920x1080",
                "accept_language": "en-IN,en;q=0.9",
                "timezone": "Asia/Kolkata",
                "hour_of_day": 13,
                "day_of_week": 3,
                "is_weekend": 0,
                "request_frequency": 2.1,
                "failed_attempts": 0,
                "session_duration": 280.0,
                "form_fill_speed": 2.8,
                "mouse_movement_score": 0.82,
                "copy_paste_detected": 1,
                "tab_switches": 3,
                "time_on_each_field": 4.5,
                "geolocation_mismatch": 0,
                "distance_from_branch_km": 3.2,
                "previous_applications": 1,
                "applications_same_ip": 1,
                "applications_same_device": 1,
                "pan_seen_before": 0,
                "aadhaar_seen_before": 0,
            },
        },
        {
            "name": "Bot Attack (Automated Submission)",
            "data": {
                "session_id": "TEST-003",
                "user_agent": "python-requests/2.31.0",
                "ip_address": "185.220.101.5",
                "screen_resolution": "unknown",
                "hour_of_day": 2,
                "day_of_week": 6,
                "is_weekend": 1,
                "request_frequency": 28.0,
                "failed_attempts": 0,
                "session_duration": 7.0,
                "ip_is_vpn": 1,
                "ip_is_tor": 1,
                "ip_is_proxy": 1,
                "ip_reputation_score": 92.0,
                "form_fill_speed": 210.0,
                "mouse_movement_score": 0.0,
                "copy_paste_detected": 1,
                "tab_switches": 0,
                "time_on_each_field": 0.03,
                "geolocation_mismatch": 1,
                "distance_from_branch_km": 1200.0,
                "previous_applications": 4,
                "applications_same_ip": 22,
                "applications_same_device": 15,
                "pan_seen_before": 0,
                "aadhaar_seen_before": 0,
            },
        },
        {
            "name": "Fraud Ring (Same IP, Multiple Identities)",
            "data": {
                "session_id": "TEST-004",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                "ip_address": "103.76.45.89",
                "screen_resolution": "1366x768",
                "hour_of_day": 11,
                "day_of_week": 2,
                "is_weekend": 0,
                "request_frequency": 14.0,
                "failed_attempts": 1,
                "session_duration": 42.0,
                "ip_is_vpn": 1,
                "ip_is_proxy": 1,
                "ip_reputation_score": 65.0,
                "form_fill_speed": 7.0,
                "mouse_movement_score": 0.38,
                "copy_paste_detected": 1,
                "tab_switches": 1,
                "time_on_each_field": 1.5,
                "geolocation_mismatch": 1,
                "distance_from_branch_km": 380.0,
                "previous_applications": 6,
                "applications_same_ip": 19,
                "applications_same_device": 8,
                "pan_seen_before": 1,
                "aadhaar_seen_before": 1,
            },
        },
        {
            "name": "Identity Theft (Stolen PAN/Aadhaar, VPN)",
            "data": {
                "session_id": "TEST-005",
                "user_agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/121.0",
                "ip_address": "194.165.16.100",
                "screen_resolution": "1280x720",
                "hour_of_day": 3,
                "day_of_week": 5,
                "is_weekend": 1,
                "request_frequency": 5.5,
                "failed_attempts": 4,
                "session_duration": 110.0,
                "ip_is_vpn": 1,
                "ip_is_tor": 0,
                "ip_is_proxy": 1,
                "ip_reputation_score": 78.0,
                "form_fill_speed": 9.0,
                "mouse_movement_score": 0.28,
                "copy_paste_detected": 1,
                "tab_switches": 7,
                "time_on_each_field": 2.0,
                "geolocation_mismatch": 1,
                "distance_from_branch_km": 620.0,
                "previous_applications": 2,
                "applications_same_ip": 4,
                "applications_same_device": 3,
                "pan_seen_before": 1,
                "aadhaar_seen_before": 1,
            },
        },
        {
            "name": "Bot-like Typing/Mouse Pattern via CLEAN IP (isolates behavioral model)",
            "data": {
                "session_id": "TEST-006",
                "user_agent": "python-requests/2.31.0",
                "ip_address": "49.36.20.10",   # safe Jio range — won't get blocked at Step 0
                "screen_resolution": "unknown",
                "hour_of_day": 2,
                "day_of_week": 6,
                "is_weekend": 1,
                "request_frequency": 28.0,
                "failed_attempts": 0,
                "session_duration": 7.0,
                "form_fill_speed": 210.0,        # impossibly fast — no human types this
                "mouse_movement_score": 0.0,     # zero movement — scripted submission
                "copy_paste_detected": 1,
                "tab_switches": 0,
                "time_on_each_field": 0.03,      # impossibly fast field-filling
                "geolocation_mismatch": 0,
                "distance_from_branch_km": 5.0,
                "previous_applications": 0,
                "applications_same_ip": 1,
                "applications_same_device": 1,
                "pan_seen_before": 0,
                "aadhaar_seen_before": 0,
            },
        },
        {
            "name": "Borderline Suspicious Human via CLEAN IP (slightly rushed, no hard signal)",
            "data": {
                "session_id": "TEST-007",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                "ip_address": "122.161.50.10",   # safe Airtel range
                "screen_resolution": "1920x1080",
                "hour_of_day": 23,
                "day_of_week": 4,
                "is_weekend": 0,
                "request_frequency": 4.0,
                "failed_attempts": 1,
                "session_duration": 35.0,
                "form_fill_speed": 8.5,          # fast for a human, not impossible
                "mouse_movement_score": 0.15,    # low but nonzero movement
                "copy_paste_detected": 1,
                "tab_switches": 1,
                "time_on_each_field": 1.2,
                "geolocation_mismatch": 0,
                "distance_from_branch_km": 8.0,
                "previous_applications": 0,
                "applications_same_ip": 2,
                "applications_same_device": 1,
                "pan_seen_before": 0,
                "aadhaar_seen_before": 0,
            },
        },
    ]

    print("\n" + "═" * 60)
    print("  TrustNet — Behavior Monitoring Demo")
    print("═" * 60)

    for tc in test_cases:
        print(f"\n── {tc['name']} ──")
        try:
            result = run_behavior_analysis(tc["data"])
            print(f"  Score:      {result['score']}/100")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Anomaly:    {result['is_anomaly']}")
            print(f"  Browser:    {result['browser_type']} on {result['os_type']}")
            print(f"  Headless:   {result['is_headless_browser']}")
            print(f"  Fingerprint:{result['device_fingerprint'][:20]}...")
            print(f"  Flags ({len(result['flags'])}):")
            for f in result["flags"]:
                print(f"    [{f['severity']}] {f['flag']}")
        except ModelNotFoundError:
            print("  ERROR: Model not found. Run: python train_baseline.py")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "═" * 60)