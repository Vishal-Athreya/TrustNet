# vpn_detector.py
# =============================================================================
# Automatic VPN / Proxy / Tor Detection — 3-Layer System
#
# Layer 1: MaxMind GeoIP2 Anonymous IP database (.mmdb file) — most accurate
#          NOTE: This is a PAID MaxMind product. Free accounts cannot
#          download it. If you ever obtain a licensed copy and drop it at
#          data/GeoIP2-Anonymous-IP.mmdb, this layer activates automatically.
# Layer 2: ASN / Organization name matching — catches unknown VPN IPs
#          (uses the FREE GeoLite2-ASN database, downloadable below)
# Layer 3: Hardcoded known-bad IP prefix list — fast fallback
#
# All 3 layers run OFFLINE — no internet needed once database is downloaded.
# Falls back gracefully if database file is missing (uses Layers 2 + 3 only).
# =============================================================================

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import struct
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("vpn_detector")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MMDB_PATH  = os.path.join(BASE_DIR, "data", "GeoIP2-Anonymous-IP.mmdb")
ASN_DB     = os.path.join(BASE_DIR, "data", "GeoLite2-ASN.mmdb")

# =============================================================================
# LAYER 3 — Hardcoded known-bad IP prefixes (fast, offline, always available)
# Covers major VPN providers, Tor exit nodes, datacenter proxies
# =============================================================================

TOR_EXIT_RANGES = [
    "185.220.", "23.129.", "171.25.", "199.249.",
    "204.13.", "66.220.", "51.222.", "193.11.",
    "62.102.", "89.234.", "95.142.", "176.10.",
    "198.98.", "199.195.", "209.141.",
]

VPN_RANGES = [
    # NordVPN
    "45.142.", "194.165.",
    # ExpressVPN
    "103.76.", "198.54.",
    # ProtonVPN
    "37.120.", "185.159.",
    # Mullvad
    "51.75.", "146.70.", "91.108.", "193.138.", "138.199.",
    # Surfshark
    "147.135.", "149.102.",
    # Private Internet Access
    "156.146.",
    # CyberGhost
    "212.102.",
    # IPVanish
    "45.86.", "45.89.", "45.91.",
    # HideMyAss
    "217.138.",
    # VyprVPN
    "5.183.",
    # Hotspot Shield
    "188.241.",
    # TunnelBear
    "74.82.",
    # Windscribe
    "64.145.", "104.238.",
    # AirVPN
    "10.4.", "10.5.",
    # Astrill
    "103.216.",
    # Common datacenter / hosting ranges used by VPNs
    "205.147.", "104.223.", "45.55.", "159.65.",
    "167.99.", "139.59.", "157.230.",
]

PROXY_RANGES = [
    "103.21.", "194.61.", "185.229.", "185.195.",
    "91.109.", "45.155.", "79.137.", "195.206.",
    "185.181.", "193.239.", "194.116.", "185.176.",
    "79.141.", "45.137.",
]

# Datacenter / cloud provider ranges — high risk when applying from these
DATACENTER_RANGES = [
    "35.180.", "35.181.", "52.", "54.",          # AWS
    "40.74.", "40.80.", "40.112.",               # Azure
    "34.64.", "34.65.", "35.184.",               # Google Cloud
    "104.16.", "104.17.", "104.18.", "104.19.",  # Cloudflare
    "162.158.", "172.68.", "172.69.",            # Cloudflare
]

# Safe Indian ISP ranges — never block these
SAFE_INDIAN_ISP_RANGES = [
    # Jio
    "49.36.", "49.37.", "49.40.", "49.44.",
    "49.45.", "49.46.", "49.47.", "49.48.",
    "115.240.", "115.241.", "115.242.", "115.243.",
    "223.228.", "223.229.", "223.230.", "223.231.",
    "182.64.", "182.65.", "182.66.", "182.67.",
    # Airtel
    "1.186.", "1.187.", "1.188.", "1.189.",
    "117.192.", "117.193.", "117.194.", "117.195.",
    "122.160.", "122.161.", "122.162.", "122.163.",
    "182.68.", "182.69.", "182.70.", "182.71.",
    # BSNL
    "117.196.", "117.197.", "117.198.", "117.199.",
    "117.200.", "117.201.", "117.202.", "117.203.",
    "59.88.", "59.89.", "59.90.", "59.91.",
    "125.16.", "125.17.", "125.18.", "125.19.",
    # Vodafone Idea (Vi)
    "49.249.", "49.250.", "182.64.", "103.23.",
    "103.27.", "103.59.",
    # ACT Fibernet
    "49.206.", "49.207.", "49.208.",
    "103.224.", "103.225.", "103.226.",
    # Hathway
    "59.144.", "59.145.", "59.148.",
    "106.215.", "106.216.",
    # Tata / TTML
    "61.1.", "61.2.", "61.3.",
    "122.252.", "122.253.",
]

# Known VPN / anonymous service ASN organization name keywords
VPN_ORG_KEYWORDS = [
    "vpn", "mullvad", "nordvpn", "expressvpn", "protonvpn",
    "surfshark", "cyberghost", "pia", "private internet",
    "windscribe", "tunnelbear", "ipvanish", "hidemyass",
    "astrill", "airvpn", "vypr", "hotspot shield",
    "tor project", "torservers", "anonymous", "anonymizer",
    "proxy", "hide.me", "ivacy", "perfect privacy",
    "zenmate", "strongvpn", "privatevpn", "goose",
    "datacamp", "m247", "tzulo", "psychz",
    "hostpapa", "vultr", "linode", "digitalocean",
    "hostwinds", "quadranet", "cogent",
    # Common dedicated-server / datacenter hosts frequently used to run
    # VPN exit nodes and proxies (real bank applicants never connect from
    # rented server IPs, so these are treated the same as a known VPN org)
    "worldstream", "ovh", "hetzner", "leaseweb", "scaleway",
    "contabo", "softlayer", "rackspace", "choopa",
    "datacenter", "data center", "dedicated server", "colocation",
    # Generic naming pattern many hosting providers use in reverse-DNS
    # records (e.g. "185-182-194-254.hosted-by-worldstream.net") —
    # catches providers not explicitly listed above
    "hosted-by",
]


# =============================================================================
# LAYER 1 — MaxMind GeoIP2 database lookup (most accurate)
# =============================================================================

def _check_maxmind(ip: str) -> dict:
    """
    Uses MaxMind GeoIP2 Anonymous IP database.
    Returns detection result or None if database not available.

    NOTE: GeoIP2 Anonymous IP is a PAID MaxMind product — the free
    `--download` flag in this script will NOT fetch it. This layer simply
    stays dormant (available: False) unless you manually place a licensed
    copy at data/GeoIP2-Anonymous-IP.mmdb.
    """
    if not os.path.exists(MMDB_PATH):
        return {"available": False}

    try:
        import geoip2.database
        import geoip2.errors

        with geoip2.database.Reader(MMDB_PATH) as reader:
            response = reader.anonymous_ip(ip)
            return {
                "available":  True,
                "is_vpn":     response.is_anonymous_vpn,
                "is_tor":     response.is_tor_exit_node,
                "is_proxy":   response.is_public_proxy,
                "is_hosting": response.is_hosting_provider,
                "is_anon":    response.is_anonymous,
                "source":     "MaxMind GeoIP2",
            }
    except Exception:
        return {"available": False}


# =============================================================================
# LAYER 2 — ASN / Organization name matching
# =============================================================================

def _check_asn(ip: str) -> dict:
    """
    Looks up the organization name that owns the IP block.
    VPN providers have recognizable org names (e.g. 'Mullvad VPN AB').
    Works offline if GeoLite2-ASN.mmdb is present (this one IS free and
    downloadable via --download).
    Falls back to socket reverse-DNS if not.
    """
    org_name = ""

    # Try MaxMind ASN database first
    if os.path.exists(ASN_DB):
        try:
            import geoip2.database
            with geoip2.database.Reader(ASN_DB) as reader:
                response = reader.asn(ip)
                org_name = (response.autonomous_system_organization or "").lower()
        except Exception:
            pass

    # Fallback: reverse DNS lookup (works for most IPs)
    if not org_name:
        try:
            hostname = socket.gethostbyaddr(ip)[0].lower()
            org_name = hostname
        except Exception:
            org_name = ""

    if not org_name:
        return {"is_vpn_org": False, "org_name": "unknown", "matched_keyword": None}

    matched = None
    for keyword in VPN_ORG_KEYWORDS:
        if keyword in org_name:
            matched = keyword
            break

    return {
        "is_vpn_org":      matched is not None,
        "org_name":        org_name,
        "matched_keyword": matched,
    }


# =============================================================================
# LAYER 3 — IP prefix matching (always available, no dependencies)
# =============================================================================

def _check_prefix(ip: str) -> dict:
    """Fast offline prefix matching against known bad IP ranges."""

    # Check safe Indian ISPs first — these are NEVER blocked
    is_safe_indian = any(ip.startswith(r) for r in SAFE_INDIAN_ISP_RANGES)

    is_tor     = any(ip.startswith(r) for r in TOR_EXIT_RANGES)
    is_vpn     = any(ip.startswith(r) for r in VPN_RANGES)
    is_proxy   = any(ip.startswith(r) for r in PROXY_RANGES)
    is_hosting = any(ip.startswith(r) for r in DATACENTER_RANGES)

    return {
        "is_safe_indian": is_safe_indian,
        "is_tor":         is_tor,
        "is_vpn":         is_vpn,
        "is_proxy":       is_proxy,
        "is_hosting":     is_hosting,
    }


# =============================================================================
# CORE — Combine all 3 layers into one verdict
# =============================================================================

def _is_private_ip(ip: str) -> bool:
    """Returns True if IP is private/local — never run VPN check on these."""
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def check_ip_before_upload(session_data: dict) -> dict:
    """
    Main entry point — run before accepting any document upload.

    Automatically detects:
    - VPN connections (all major providers)
    - Tor exit nodes
    - Proxy servers
    - Datacenter / hosting IPs (unusual for real bank applicants)

    Uses 3 detection layers in order:
    1. MaxMind GeoIP2 database (if installed) — most accurate
    2. ASN organization name lookup — catches unknown VPN providers
    3. IP prefix matching — fast fallback, always works

    Returns:
    {
        "allowed":       bool,       # True = let upload proceed
        "action":        str,        # "allow" / "warn" / "block"
        "threat_score":  int,        # 0-100
        "is_vpn":        bool,
        "is_tor":        bool,
        "is_proxy":      bool,
        "is_hosting":    bool,
        "is_safe_indian": bool,
        "detection_method": str,     # which layer caught it
        "org_name":      str,        # ISP / org that owns the IP
        "message":       str,        # shown to user if blocked
        "log_entry":     dict,       # written to audit log
    }
    """
    ip         = session_data.get("ip_address", "")
    session_id = session_data.get("session_id", "unknown")

    # ── Handle private/local IPs (localhost testing) ───────────────────────
    if not ip or _is_private_ip(ip):
        return _build_result(
            allowed=True, action="allow", threat_score=0,
            is_vpn=False, is_tor=False, is_proxy=False,
            is_hosting=False, is_safe_indian=True,
            detection_method="private_ip",
            org_name="localhost",
            message="Local connection — no network check performed.",
            session_id=session_id, ip=ip,
        )

    # ── Layer 1: MaxMind GeoIP2 ────────────────────────────────────────────
    maxmind = _check_maxmind(ip)
    detection_method = "prefix_matching"   # default

    if maxmind.get("available"):
        detection_method = "MaxMind_GeoIP2"
        mm_vpn   = maxmind.get("is_vpn", False)
        mm_tor   = maxmind.get("is_tor", False)
        mm_proxy = maxmind.get("is_proxy", False) or maxmind.get("is_hosting", False)
        mm_anon  = maxmind.get("is_anon", False)
    else:
        mm_vpn = mm_tor = mm_proxy = mm_anon = False

    # ── Layer 2: ASN / org name ────────────────────────────────────────────
    asn = _check_asn(ip)
    if asn.get("is_vpn_org"):
        detection_method = f"ASN_org_match:{asn['matched_keyword']}"

    # ── Layer 3: Prefix matching ───────────────────────────────────────────
    prefix = _check_prefix(ip)

    # ── Combine all layers ─────────────────────────────────────────────────
    is_safe_indian = prefix["is_safe_indian"]

    # If any layer says yes → it's detected
    is_tor     = mm_tor   or prefix["is_tor"]
    is_vpn     = mm_vpn   or prefix["is_vpn"]   or asn.get("is_vpn_org", False)
    is_proxy   = mm_proxy or prefix["is_proxy"]
    is_hosting = prefix["is_hosting"]

    # ── Compute threat score ───────────────────────────────────────────────
    threat_score = 0
    if is_tor:          threat_score += 55    # Tor = highest risk
    if is_vpn:          threat_score += 40
    if is_proxy:        threat_score += 30
    if is_hosting:      threat_score += 20
    if mm_anon:         threat_score += 10    # extra signal from MaxMind
    threat_score = min(threat_score, 100)

    # Safe Indian ISP overrides hosting flag (banks have their own servers)
    if is_safe_indian:
        threat_score = 0
        is_vpn = is_tor = is_proxy = is_hosting = False

    # ── Decide action ──────────────────────────────────────────────────────
    if threat_score >= 40:
        action  = "block"
        allowed = False
        if is_tor:
            message = (
                "Your connection is routed through the Tor network. "
                "For security reasons, document uploads via Tor are not permitted. "
                "Please connect using a regular internet connection and try again."
            )
        elif is_vpn:
            message = (
                "A VPN connection has been detected on your network. "
                "Please disable your VPN and reconnect using your regular "
                "internet connection to proceed with your application."
            )
        else:
            message = (
                "Your connection has been flagged as high-risk (proxy/anonymous network). "
                "Please use a regular internet connection to submit your application."
            )

    elif threat_score >= 20:
        action  = "warn"
        allowed = True    # let through but flag for manual review
        message = (
            "Your connection appears to be using an anonymizing service. "
            "Your application has been flagged for additional manual review."
        )

    else:
        action  = "allow"
        allowed = True
        message = "Connection verified. You may proceed with your document upload."

    return _build_result(
        allowed=allowed, action=action, threat_score=threat_score,
        is_vpn=is_vpn, is_tor=is_tor, is_proxy=is_proxy,
        is_hosting=is_hosting, is_safe_indian=is_safe_indian,
        detection_method=detection_method,
        org_name=asn.get("org_name", "unknown"),
        message=message,
        session_id=session_id, ip=ip,
    )


def _build_result(
    allowed, action, threat_score,
    is_vpn, is_tor, is_proxy, is_hosting, is_safe_indian,
    detection_method, org_name, message, session_id, ip,
) -> dict:
    """Builds the standardised result dict and writes the audit log entry."""

    # Mask last octet of IP for privacy in logs
    parts = ip.split(".")
    masked_ip = ".".join(parts[:3]) + ".***" if len(parts) == 4 else ip

    log_entry = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "session_id":       session_id,
        "ip_masked":        masked_ip,
        "action":           action,
        "threat_score":     threat_score,
        "is_vpn":           is_vpn,
        "is_tor":           is_tor,
        "is_proxy":         is_proxy,
        "is_hosting":       is_hosting,
        "is_safe_indian":   is_safe_indian,
        "detection_method": detection_method,
        "org_name":         org_name,
    }

    # Write to audit log
    _write_log(log_entry, action, message, masked_ip)

    return {
        "allowed":          allowed,
        "action":           action,
        "threat_score":     threat_score,
        "is_vpn":           is_vpn,
        "is_tor":           is_tor,
        "is_proxy":         is_proxy,
        "is_hosting":       is_hosting,
        "is_safe_indian":   is_safe_indian,
        "detection_method": detection_method,
        "org_name":         org_name,
        "message":          message,
        "log_entry":        log_entry,
    }


def _write_log(entry: dict, action: str, message: str, masked_ip: str):
    """Writes a clean, human-readable entry to the behavior log."""

    action_icons = {
        "allow": "✅ ALLOWED",
        "warn":  "⚠️  WARNING",
        "block": "🚫 BLOCKED",
    }
    icon = action_icons.get(action, action.upper())

    lines = [
        "",
        "=" * 60,
        "  VPN / PROXY / TOR CHECK",
        f"  SESSION ID      : {entry['session_id']}",
        f"  IP ADDRESS      : {masked_ip}  (masked for privacy)",
        f"  TIME            : {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        f"  RESULT          : {icon}",
        f"  THREAT SCORE    : {entry['threat_score']} / 100",
        f"  DETECTION METHOD: {entry['detection_method']}",
        f"  NETWORK / ORG   : {entry['org_name']}",
        "-" * 60,
        f"  Tor Detected    : {'🔴 YES — Tor exit node' if entry['is_tor']     else '✅ No'}",
        f"  VPN Detected    : {'🔴 YES — VPN active'    if entry['is_vpn']     else '✅ No'}",
        f"  Proxy Detected  : {'🔴 YES — Proxy server'  if entry['is_proxy']   else '✅ No'}",
        f"  Datacenter IP   : {'⚠️  YES — Hosting IP'   if entry['is_hosting'] else '✅ No'}",
        f"  Safe Indian ISP : {'✅ YES'                  if entry['is_safe_indian'] else '❌ No'}",
        "-" * 60,
        f"  MESSAGE         : {message}",
        "=" * 60,
        "",
    ]

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "behavior.log")

    log = logging.getLogger("vpn_detector")
    if not log.handlers:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

    for line in lines:
        log.info(line)


# =============================================================================
# DATABASE DOWNLOADER — run once to get the FREE MaxMind ASN database
# =============================================================================

def download_maxmind_database():
    """
    Downloads the free MaxMind GeoLite2-ASN database (powers Layer 2).

    IMPORTANT: GeoIP2 Anonymous IP (the dedicated VPN/Tor/proxy flag
    database) is a PAID MaxMind product. Free accounts cannot download it,
    and any attempt always returns HTTP 401 regardless of how correct your
    credentials are. This function does NOT attempt that download — it
    only fetches the free GeoLite2-ASN database.

    You need a free MaxMind account:
    1. Go to https://www.maxmind.com/en/geolite2/signup
    2. Sign up (free, no credit card)
    3. Get your Account ID + License Key from
       Account menu > Manage License Keys
    4. Run: python vpn_detector.py --download YOUR_ACCOUNT_ID YOUR_LICENSE_KEY

    Note: newly generated license keys can take a few minutes to activate
    on MaxMind's side. If you get a 401 immediately after creating a key,
    wait ~10 minutes and try again.
    """
    import sys
    if len(sys.argv) < 4:
        print("\nUsage: python vpn_detector.py --download ACCOUNT_ID LICENSE_KEY")
        print("\nGet free credentials at:")
        print("  https://www.maxmind.com/en/geolite2/signup\n")
        return

    account_id  = sys.argv[2]
    license_key = sys.argv[3]

    import urllib.request
    import urllib.error
    import tarfile
    import base64

    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    db_name     = "GeoLite2-ASN"
    output_file = "GeoLite2-ASN.mmdb"
    output_path = os.path.join(data_dir, output_file)

    url = (
        f"https://download.maxmind.com/geoip/databases/"
        f"{db_name}/download?suffix=tar.gz"
    )
    print(f"Downloading {db_name} (free database)...")

    creds = base64.b64encode(f"{account_id}:{license_key}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {creds}"})

    # IMPORTANT: MaxMind responds with a 302 redirect to a signed Cloudflare
    # R2 / S3-style URL that already carries its own auth in the query
    # string. Python's urllib, unlike curl, auto-follows redirects AND
    # forwards the original Authorization header to the new host — which
    # makes the storage layer reject the request with a 401 even though
    # your MaxMind credentials were fine. We intercept the redirect here
    # and issue a clean second request with NO Authorization header.
    class _NoAutoRedirect(urllib.request.HTTPRedirectHandler):
        def http_error_302(self, req, fp, code, msg, headers):
            fp.redirect_location = headers.get("Location")
            return fp
        http_error_301 = http_error_303 = http_error_307 = http_error_302

    opener = urllib.request.build_opener(_NoAutoRedirect)

    success = False
    try:
        response = opener.open(req, timeout=30)
        redirect_location = getattr(response, "redirect_location", None)

        if redirect_location:
            # Second hop: plain request, no Authorization header
            with urllib.request.urlopen(redirect_location, timeout=60) as final:
                data = final.read()
        else:
            data = response.read()

        tar_path = os.path.join(data_dir, f"{db_name}.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(data)

        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".mmdb"):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, data_dir, filter="data")
                    success = True
                    break

        os.remove(tar_path)

    except urllib.error.HTTPError as e:
        print(f"  Error: HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("  -> Account ID / License Key was rejected by MaxMind.")
            print("     - Double-check both values for typos")
            print("     - If the key was just created, wait ~10 minutes and retry")
            print("       (new keys take a little time to activate)")
            print("     - Confirm the key shows 'Active' under")
            print("       Account > Manage License Keys on maxmind.com")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    if success and os.path.exists(output_path):
        print(f"Done! Saved data/{output_file}")
        print("Layer 2 (ASN org-name matching) now uses official MaxMind data")
        print("instead of the reverse-DNS fallback.")
    else:
        print("Download FAILED — data/GeoLite2-ASN.mmdb was NOT saved.")
        print("Layers 2 (reverse-DNS fallback) and 3 (prefix list) still")
        print("work fine without it, just slightly less precise.")
    print()
    print("Reminder: GeoIP2 Anonymous IP is a paid-only MaxMind product and")
    print("is intentionally NOT downloaded by this script.")


# =============================================================================
# TEST — run this file directly to test all 3 detection layers
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        download_maxmind_database()
        sys.exit(0)

    print("\n" + "=" * 60)
    print("  TrustNet — VPN / Proxy / Tor Detection Test")
    print(f"  MaxMind Anonymous-IP DB (Layer 1): {'✅ ACTIVE' if os.path.exists(MMDB_PATH) else '⚠️  Not installed (paid-only, using Layers 2+3)'}")
    print(f"  MaxMind ASN DB (Layer 2 boost)    : {'✅ ACTIVE' if os.path.exists(ASN_DB) else '⚠️  Not downloaded yet'}")
    print("=" * 60)

    test_cases = [
        # (description, ip, should_be_blocked)
        ("Jio India (safe)",          "49.36.20.10",    False),
        ("Airtel India (safe)",        "122.161.50.10",  False),
        ("BSNL India (safe)",          "117.196.20.5",   False),
        ("Tor exit node",              "185.220.101.5",  True),
        ("ProtonVPN",                  "37.120.200.1",   True),
        ("NordVPN",                    "45.142.212.100", True),
        ("Mullvad VPN",                "51.75.33.10",    True),
        ("Surfshark",                  "147.135.10.5",   True),
        ("Proxy server",               "103.21.58.100",  True),
        ("AWS datacenter",             "52.14.100.200",  True),
        ("Localhost (testing)",        "127.0.0.1",      False),
    ]

    blocked = 0
    allowed = 0

    for name, ip, should_block in test_cases:
        result = check_ip_before_upload({
            "session_id": f"TEST-{ip}",
            "ip_address": ip,
        })

        action = result["action"].upper()
        score  = result["threat_score"]
        method = result["detection_method"]
        org    = result["org_name"][:35] if result["org_name"] else "unknown"

        icon = "🚫" if action == "BLOCK" else ("⚠️ " if action == "WARN" else "✅")
        match = "✓" if (result["action"] in ("block","warn")) == should_block else "✗ WRONG"

        print(f"\n  {name}")
        print(f"    {icon} {action:8s} | Score: {score:3d}/100 | {match}")
        print(f"    Method: {method}")
        print(f"    Org:    {org}")

        if result["action"] in ("block", "warn"):
            blocked += 1
        else:
            allowed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {allowed} ALLOWED  |  {blocked} BLOCKED/WARNED")
    print(f"  Log written to: logs/behavior.log")

    if not os.path.exists(ASN_DB):
        print("\n  💡 To strengthen Layer 2 with official MaxMind ASN data:")
        print("     python vpn_detector.py --download ACCOUNT_ID LICENSE_KEY")
        print("     (Free MaxMind account: maxmind.com/en/geolite2/signup)")
    print("=" * 60 + "\n")