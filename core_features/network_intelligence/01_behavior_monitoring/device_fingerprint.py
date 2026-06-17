# device_fingerprint.py
import hashlib
from exceptions import InvalidSessionDataError
from logger import get_logger

logger = get_logger("device_fingerprint")


def _parse_user_agent(user_agent: str) -> dict:
    ua = user_agent.lower()

    headless_patterns = [
        "headlesschrome", "phantomjs", "selenium", "puppeteer",
        "playwright", "webdriver", "python-requests", "curl",
        "scrapy", "httpx", "aiohttp", "go-http-client",
    ]
    is_headless = any(p in ua for p in headless_patterns)

    if "chrome" in ua and "edg" not in ua:
        browser = "Chrome"
    elif "firefox" in ua:
        browser = "Firefox"
    elif "safari" in ua and "chrome" not in ua:
        browser = "Safari"
    elif "edg" in ua:
        browser = "Edge"
    elif "samsung" in ua:
        browser = "Samsung Internet"
    elif "curl" in ua:
        browser = "curl"
    elif "python" in ua:
        browser = "Python-requests"
    elif "scrapy" in ua:
        browser = "Scrapy"
    else:
        browser = "Unknown"

    if "android" in ua:
        os_type = "Android"
    elif "iphone" in ua or "ipad" in ua:
        os_type = "iOS"
    elif "windows" in ua:
        os_type = "Windows"
    elif "linux" in ua:
        os_type = "Linux"
    elif "mac" in ua:
        os_type = "MacOS"
    else:
        os_type = "Unknown"

    return {
        "is_headless": is_headless,
        "browser":     browser,
        "os_type":     os_type,
    }


def compute_user_agent_entropy(user_agent: str) -> float:
    if not user_agent:
        return 0.0

    from collections import Counter
    import math

    counts = Counter(user_agent)
    total  = len(user_agent)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )
    return round(entropy, 3)


def generate_fingerprint(session_data: dict) -> str:
    """
    Combines device/browser/network signals into a single SHA-256 hash.
    Takes the full session_data dict — extracts what it needs internally.
    """
    user_agent        = session_data.get("user_agent",        "unknown")
    ip_address        = session_data.get("ip_address",        "unknown")
    screen_resolution = session_data.get("screen_resolution", "unknown")
    accept_language   = session_data.get("accept_language",   "unknown")
    timezone          = session_data.get("timezone",          "unknown")

    if not user_agent or not ip_address:
        raise InvalidSessionDataError(
            "user_agent and ip_address are required to generate a device fingerprint"
        )

    raw_string = "|".join([
        user_agent.strip().lower(),
        ip_address.strip(),
        screen_resolution.strip().lower(),
        accept_language.strip().lower(),
        timezone.strip(),
    ])

    fingerprint = hashlib.sha256(raw_string.encode("utf-8")).hexdigest()
    logger.debug(f"Fingerprint generated for IP {ip_address[:8]}*** → {fingerprint[:16]}...")
    return fingerprint


def extract_device_features(session_data: dict) -> dict:
    """
    Extracts all device-related features from raw session data.
    Called by feature_engineering.py to build the full feature vector.
    """
    user_agent = session_data.get("user_agent", "unknown")

    fingerprint = generate_fingerprint(session_data)
    ua_info     = _parse_user_agent(user_agent)
    entropy     = compute_user_agent_entropy(user_agent)

    return {
        "device_fingerprint":  fingerprint,
        "is_headless_browser": int(ua_info["is_headless"]),
        "browser_type":        ua_info["browser"],
        "os_type":             ua_info["os_type"],
        "user_agent_entropy":  entropy,
    }


if __name__ == "__main__":
    test_cases = [
        {
            "name":       "Normal Chrome user",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "ip_address": "49.36.20.10",
            "screen_resolution": "1920x1080",
        },
        {
            "name":       "Python bot",
            "user_agent": "python-requests/2.31.0",
            "ip_address": "185.220.101.5",
        },
        {
            "name":       "Headless Chrome",
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 HeadlessChrome/120.0.0.0",
            "ip_address": "103.45.67.89",
        },
    ]

    for tc in test_cases:
        print(f"\n── {tc['name']} ──")
        features = extract_device_features(tc)
        for k, v in features.items():
            print(f"  {k}: {v}")