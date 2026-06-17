# logger.py
import logging
import os
import json
from datetime import datetime
from config import LOG_DIR, LOG_PATH

os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str = "behavior_monitoring") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Simple readable format — time and message only
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%I:%M %p"
    )

    # File handler — overwrites every run so log stays clean
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler — shows in terminal while running
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_analysis_result(logger: logging.Logger, session_id: str, result: dict):
    """
    Writes a clean, readable summary of each session analysis to the log file.
    Easy to read — no JSON, no technical noise.
    """
    score       = result.get("score")
    risk_level  = result.get("risk_level")
    flags       = result.get("flags", [])
    fingerprint = result.get("device_fingerprint", "")
    is_anomaly  = result.get("is_anomaly", False)
    details     = result.get("details", {})
    device_info = details.get("device_info", {})

    # Risk level emoji for quick visual scan
    risk_icon = {
        "HIGH":       "🔴 HIGH RISK",
        "SUSPICIOUS": "🟡 SUSPICIOUS",
        "CLEAN":      "🟢 CLEAN",
    }.get(risk_level, risk_level)

    lines = [
        "",
        "=" * 60,
        f"  SESSION ID   : {session_id}",
        f"  TIME         : {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        f"  RESULT       : {risk_icon}",
        f"  SCORE        : {score} / 100",
        f"  ANOMALY      : {'YES — Behavior is abnormal' if is_anomaly else 'NO — Behavior looks normal'}",
        f"  BROWSER      : {device_info.get('browser', 'Unknown')}",
        f"  OS           : {device_info.get('os_type', 'Unknown')}",
        f"  HEADLESS BOT : {'YES — Not a real browser' if device_info.get('is_headless') else 'NO'}",
        f"  DEVICE ID    : {fingerprint[:24]}...",
        "-" * 60,
    ]

    if flags:
        # Count by severity
        high_flags   = [f for f in flags if f["severity"] == "HIGH"]
        medium_flags = [f for f in flags if f["severity"] == "MEDIUM"]
        low_flags    = [f for f in flags if f["severity"] == "LOW"]

        lines.append(
            f"  RED FLAGS    : {len(flags)} found "
            f"({len(high_flags)} HIGH, {len(medium_flags)} MEDIUM, {len(low_flags)} LOW)"
        )
        lines.append("")

        if high_flags:
            lines.append("  🔴 HIGH SEVERITY:")
            for f in high_flags:
                lines.append(f"     • {f['message']}")

        if medium_flags:
            lines.append("")
            lines.append("  🟡 MEDIUM SEVERITY:")
            for f in medium_flags:
                lines.append(f"     • {f['message']}")

        if low_flags:
            lines.append("")
            lines.append("  🟢 LOW SEVERITY:")
            for f in low_flags:
                lines.append(f"     • {f['message']}")
    else:
        lines.append("  RED FLAGS    : NONE — Session looks completely clean.")

    lines.append("=" * 60)
    lines.append("")

    for line in lines:
        logger.info(line)


def log_error(logger: logging.Logger, session_id: str, error: Exception):
    """Logs errors in plain English."""
    logger.error(
        f"\n"
        f"  ERROR in session {session_id}\n"
        f"  Problem : {type(error).__name__}\n"
        f"  Details : {str(error)}\n"
    )