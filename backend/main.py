# backend/main.py
# ─────────────────────────────────────────────────────────────────────────────
# TrustNet — Backend Orchestrator
#
# This is the single FastAPI server that the frontend talks to.
# It wires every TrustNet module together in the correct order:
#
#   POST /analyse
#     ├── 1. Quantum-encrypt PAN / Aadhaar before they touch memory/disk
#     ├── 2. Behavior monitoring  (VPN + device + Isolation Forest)
#     ├── 3. Benford's Law        (numeric integrity per document)
#     ├── 4. Metadata Forensics   (creation date / software tampering)
#     ├── 5. NLP Semantics        (BERT entity extraction + consistency)
#     ├── 6. GNN Cross-Document   (fraud-ring entity graph)
#     ├── 7. Federated Learning   (global cross-bank risk score)
#     └── 8. Aggregate verdict    (APPROVE / REVIEW / REJECT)
#
#   GET  /health   — liveness probe
#   GET  /keys     — confirm quantum keypair is loaded (never returns private key)
#
# Run:
#   pip install -r requirements.txt
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import logging
import tempfile
import traceback
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ── Resolve paths to sibling feature modules ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # repo root
NET  = ROOT / "core_features" / "network_intelligence"
DOC  = ROOT / "core_features" / "document_intelligence"

# NOTE: sys.path manipulation removed — we load submodules by file path
# using importlib to avoid circular imports (all submodules also have a
# file called main.py, which Python was resolving to THIS file).


# ── importlib loader ────────────────────────────────────────────────────────

def _load_submodule(directory: Path, filename: str = "main"):
    """
    Load `filename`.py from `directory` as an isolated module.

    Uses importlib.util.spec_from_file_location so Python never confuses
    the submodule's main.py with backend/main.py.  Each submodule gets a
    unique name in sys.modules (trustnet_<stem>_<filename>) so repeated
    imports are cached correctly.
    """
    file_path = directory / f"{filename}.py"
    module_name = f"trustnet_{directory.name}_{filename}"

    # Return cached module if already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None:
        raise ImportError(f"Cannot find {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod          # register before exec to handle internal relative imports
    spec.loader.exec_module(mod)
    return mod


# ── Import every module's public entry point ────────────────────────────────
# Each import is wrapped individually so a missing dependency in one module
# doesn't kill the entire backend — the module just reports unavailable.

MODULES = {}   # tracks which modules loaded successfully

try:
    _bm = _load_submodule(NET / "01_behavior_monitoring")
    run_behavior_analysis = _bm.run_behavior_analysis
    MODULES["behavior"] = True
except Exception as e:
    logging.warning(f"[startup] behavior_monitoring unavailable: {e}")
    MODULES["behavior"] = False
    run_behavior_analysis = None

try:
    _qc = _load_submodule(NET / "03_quantum_crypto")
    load_or_create_keypair = _qc.load_or_create_keypair
    encrypt_text           = _qc.encrypt_text
    encrypt_data           = _qc.encrypt_data
    MODULES["crypto"] = True
except Exception as e:
    logging.warning(f"[startup] quantum_crypto unavailable: {e}")
    MODULES["crypto"] = False
    load_or_create_keypair = encrypt_text = encrypt_data = None

try:
    _bl = _load_submodule(DOC / "02_benfords_law")
    analyze_benford_from_file = _bl.analyze_benford_from_file
    MODULES["benfords"] = True
except Exception as e:
    logging.warning(f"[startup] benfords_law unavailable: {e}")
    MODULES["benfords"] = False
    analyze_benford_from_file = None

try:
    _mf = _load_submodule(DOC / "03_metadata_forensics")
    analyse_metadata = _mf.analyse_metadata
    MODULES["metadata"] = True
except Exception as e:
    logging.warning(f"[startup] metadata_forensics unavailable: {e}")
    MODULES["metadata"] = False
    analyse_metadata = None

try:
    _nlp = _load_submodule(DOC / "04_nlp_semantics")
    analyse_nlp = _nlp.analyse_nlp
    MODULES["nlp"] = True
except Exception as e:
    logging.warning(f"[startup] nlp_semantics unavailable: {e}")
    MODULES["nlp"] = False
    analyse_nlp = None

try:
    _gnn = _load_submodule(DOC / "05_graph_neural_network")
    analyze_cross_document_graph = _gnn.analyze_cross_document_graph
    MODULES["gnn"] = True
except Exception as e:
    logging.warning(f"[startup] graph_neural_network unavailable: {e}")
    MODULES["gnn"] = False
    analyze_cross_document_graph = None

try:
    fl_dir = NET / "02_federated_learning"
    if str(fl_dir) not in sys.path:
        sys.path.insert(0, str(fl_dir))  # must be BEFORE behavior monitoring
    _fl = _load_submodule(fl_dir, filename="scoring")
    score_records = _fl.score_records
    MODULES["federated"] = True
except Exception as e:
    logging.warning(f"[startup] federated_learning unavailable: {e}")
    MODULES["federated"] = False
    score_records = None

# ── Quantum keypair — loaded ONCE at startup, reused for every request ──────
CRYPTO_PUBLIC_KEY  = None
CRYPTO_PRIVATE_KEY = None
if MODULES["crypto"]:
    try:
        CRYPTO_PUBLIC_KEY, CRYPTO_PRIVATE_KEY = load_or_create_keypair()
        logging.info("[startup] Quantum keypair loaded ✓")
    except Exception as e:
        logging.warning(f"[startup] Could not load quantum keypair: {e}")
        MODULES["crypto"] = False


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrustNet API",
    description="Real-time AI fraud intelligence for Indian banks and NBFCs",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trustnet.backend")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_run(module_name: str, fn, *args, **kwargs):
    """Calls fn(*args, **kwargs), returns (result, error_str).
    Never raises — any exception is caught and returned as an error string."""
    if not MODULES.get(module_name):
        return None, f"{module_name} module not available"
    if fn is None:
        return None, f"{module_name} function not loaded"
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        logger.warning(f"[{module_name}] Error: {e}")
        return None, str(e)


def _aggregate_verdict(scores: dict) -> dict:
    """
    Combines individual module scores into one final decision.

    Score weights (tuned for Indian bank KYC fraud context):
      - Behavior monitoring    30% (live session signals + VPN detection)
      - NLP Semantics          25% (document text authenticity)
      - Metadata Forensics     20% (file-level tampering)
      - Benford's Law          15% (numeric integrity)
      - Federated Learning     10% (cross-bank global signal)
      - GNN: anomaly bonus     +15 points if fraud ring detected (flat penalty)

    Final score 0-100 → APPROVE (<35) / REVIEW (35-64) / REJECT (≥65)
    """
    weights = {
        "behavior":  0.30,
        "nlp":       0.25,
        "metadata":  0.20,
        "benfords":  0.15,
        "federated": 0.10,
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        score = scores.get(key)
        if score is not None:
            weighted_sum  += score * weight
            total_weight  += weight

    # Normalise in case some modules weren't available
    combined = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # GNN fraud ring — flat penalty on top
    if scores.get("gnn_fraud_ring"):
        combined = min(100.0, combined + 15.0)

    combined = round(combined, 1)

    if combined >= 65:
        verdict = "REJECT"
    elif combined >= 35:
        verdict = "REVIEW"
    else:
        verdict = "APPROVE"

    return {"final_score": combined, "verdict": verdict}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": MODULES,
    }


@app.get("/keys")
async def keys_status():
    return {
        "quantum_crypto": MODULES.get("crypto", False),
        "public_key_loaded": CRYPTO_PUBLIC_KEY is not None,
        "note": "Private key is never returned via API.",
    }


@app.post("/analyse")
async def analyse(
    request: Request,

    # ── Session / behavioral data ────────────────────────────────────────
    session_id:              str   = Form(default=""),
    user_agent:              str   = Form(default="unknown"),
    screen_resolution:       str   = Form(default="unknown"),
    timezone_str:            str   = Form(default="unknown"),
    session_duration:        float = Form(default=300.0),
    mouse_movement_score:    float = Form(default=0.5),
    copy_paste_detected:     int   = Form(default=0),
    form_fill_speed:         float = Form(default=2.5),
    time_on_each_field:      float = Form(default=5.0),
    tab_switches:            int   = Form(default=0),

    # ── Applicant PII — encrypted immediately, never stored plain ───────
    applicant_name:          str   = Form(default=""),
    pan_number:              str   = Form(default=""),
    aadhaar_number:          str   = Form(default=""),

    # ── Document type hint for NLP and metadata modules ──────────────────
    document_type:           str   = Form(default=""),

    # ── Uploaded documents (up to 5) ────────────────────────────────────
    file1: Optional[UploadFile] = None,
    file2: Optional[UploadFile] = None,
    file3: Optional[UploadFile] = None,
    file4: Optional[UploadFile] = None,
    file5: Optional[UploadFile] = None,
):
    """
    Main analysis endpoint — called once per loan/account application.

    Accepts:
      - Session behavioral signals (mouse, typing, tab switches, VPN)
      - Applicant PAN + Aadhaar (encrypted quantum-safe before anything else)
      - Up to 5 uploaded documents (PDF, image, DOCX)

    Returns a complete fraud intelligence report with one final verdict.
    """
    now = datetime.now()
    sid = session_id or f"APP-{now.strftime('%H%M%S%f')}"
    logger.info(f"[{sid}] Analysis started")

    # ── Resolve real client IP ────────────────────────────────────────────
    client_ip = request.headers.get("X-Forwarded-For", "") or request.client.host
    if client_ip in ("127.0.0.1", "::1", "localhost"):
        import urllib.request as _req
        try:
            client_ip = _req.urlopen("https://api.ipify.org", timeout=3).read().decode()
        except Exception:
            client_ip = "127.0.0.1"

    result = {
        "session_id":  sid,
        "analysed_at": now.isoformat(),
        "modules":     {},
        "encrypted_pii": {},
        "verdict":     {},
        "errors":      [],
    }

    # ── Step 1: Encrypt PAN + Aadhaar immediately ─────────────────────────
    if MODULES["crypto"] and CRYPTO_PUBLIC_KEY is not None:
        if pan_number:
            try:
                result["encrypted_pii"]["pan"] = encrypt_text(pan_number, CRYPTO_PUBLIC_KEY)
            except Exception as e:
                result["errors"].append(f"PAN encryption failed: {e}")

        if aadhaar_number:
            try:
                result["encrypted_pii"]["aadhaar"] = encrypt_text(aadhaar_number, CRYPTO_PUBLIC_KEY)
            except Exception as e:
                result["errors"].append(f"Aadhaar encryption failed: {e}")

        if result["encrypted_pii"]:
            logger.info(f"[{sid}] PII encrypted with ML-KEM-768")
    else:
        result["errors"].append("Quantum crypto unavailable — PII not encrypted")

    # ── Step 2: Behavior monitoring (VPN + device + Isolation Forest) ─────
    session_data = {
        "session_id":              sid,
        "user_agent":              user_agent,
        "ip_address":              client_ip,
        "screen_resolution":       screen_resolution,
        "timezone":                timezone_str,
        "hour_of_day":             now.hour,
        "day_of_week":             now.weekday(),
        "is_weekend":              int(now.weekday() >= 5),
        "request_frequency":       2.0,
        "failed_attempts":         0,
        "session_duration":        session_duration,
        "form_fill_speed":         form_fill_speed,
        "mouse_movement_score":    mouse_movement_score,
        "copy_paste_detected":     copy_paste_detected,
        "tab_switches":            tab_switches,
        "time_on_each_field":      time_on_each_field,
        "geolocation_mismatch":    0,
        "distance_from_branch_km": 10.0,
        "previous_applications":   0,
        "applications_same_ip":    1,
        "applications_same_device":1,
        "pan_seen_before":         0,
        "aadhaar_seen_before":     0,
        "num_devices_used":        1,
        "ip_is_vpn":               0,
        "ip_is_tor":               0,
        "ip_is_proxy":             0,
        "ip_reputation_score":     0.0,
        "velocity_score":          50.0,
    }

    behavior_result, behavior_err = _safe_run("behavior", run_behavior_analysis, session_data)
    if behavior_result:
        result["modules"]["behavior"] = {
            "score":        behavior_result["score"],
            "risk_level":   behavior_result["risk_level"],
            "is_anomaly":   behavior_result["is_anomaly"],
            "is_headless":  behavior_result["is_headless_browser"],
            "browser":      behavior_result["browser_type"],
            "os":           behavior_result["os_type"],
            "blocked":      behavior_result.get("blocked", False),
            "block_reason": behavior_result.get("block_reason", ""),
            "flags":        behavior_result["flags"],
        }
        behavior_score = behavior_result["score"]

        # Hard-block: VPN/Tor detected — return immediately
        if behavior_result.get("blocked"):
            result["verdict"] = {
                "final_score": 100,
                "verdict": "REJECT",
                "reason": behavior_result.get("block_reason", "VPN/Tor/Proxy detected"),
            }
            logger.warning(f"[{sid}] Hard-blocked: {behavior_result.get('block_reason')}")
            return JSONResponse(result)
    else:
        behavior_score = 50.0   # neutral fallback if module failed
        result["errors"].append(f"behavior: {behavior_err}")

    # ── Steps 3-6: Document analysis ─────────────────────────────────────
    # Collect all uploaded files as (filename, bytes) pairs
    uploaded_files = []
    for upload in [file1, file2, file3, file4, file5]:
        if upload is not None:
            try:
                file_bytes = await upload.read()
                uploaded_files.append((upload.filename, file_bytes))
            except Exception as e:
                result["errors"].append(f"File read error ({upload.filename}): {e}")

    benfords_score = None
    metadata_score = None
    nlp_score      = None
    gnn_fraud_ring = False

    temp_file_paths = []   # track temp files for cleanup

    for filename, file_bytes in uploaded_files:
        logger.info(f"[{sid}] Analysing document: {filename} ({len(file_bytes):,} bytes)")

        # Write to a temp file for modules that need a file path
        suffix = Path(filename).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()
        temp_file_paths.append(tmp.name)

        # Step 3: Benford's Law
        benford_result, benford_err = _safe_run(
            "benfords", analyze_benford_from_file, tmp.name
        )
        if benford_result:
            # integrity_score 100 = clean, 0 = anomalous — invert for risk
            doc_benford_risk = 100.0 - benford_result.get("integrity_score", 100.0)
            benfords_score   = max(benfords_score or 0, doc_benford_risk)
            result["modules"].setdefault("benfords", []).append({
                "file":             filename,
                "anomaly_detected": benford_result.get("anomaly_detected"),
                "integrity_score":  benford_result.get("integrity_score"),
                "explanation":      benford_result.get("explanation"),
            })
        elif benford_err:
            result["errors"].append(f"benfords ({filename}): {benford_err}")

        # Step 4: Metadata Forensics
        meta_result, meta_err = _safe_run(
            "metadata", analyse_metadata,
            file_bytes, filename,
            None,
            document_type or None,
        )
        if meta_result:
            try:
                meta_dict = meta_result.model_dump() if hasattr(meta_result, "model_dump") else dict(meta_result)
            except Exception:
                meta_dict = {"raw": str(meta_result)}

            doc_meta_score  = meta_dict.get("risk_score", 0)
            metadata_score  = max(metadata_score or 0, doc_meta_score)
            result["modules"].setdefault("metadata", []).append({
                "file":       filename,
                "risk_score": doc_meta_score,
                "risk_level": meta_dict.get("risk_level", "UNKNOWN"),
                "flags":      meta_dict.get("flags", []),
            })
        elif meta_err:
            result["errors"].append(f"metadata ({filename}): {meta_err}")

        # Step 5: NLP Semantics
        nlp_result, nlp_err = _safe_run(
            "nlp", analyse_nlp,
            file_bytes, filename,
            document_type or None,
            False,
            applicant_name or None,
            pan_number or None,
            aadhaar_number or None,
        )
        if nlp_result:
            try:
                nlp_dict = nlp_result.model_dump() if hasattr(nlp_result, "model_dump") else dict(nlp_result)
            except Exception:
                nlp_dict = {"raw": str(nlp_result)}

            doc_nlp_score = nlp_dict.get("risk_score", 0)
            nlp_score     = max(nlp_score or 0, doc_nlp_score)
            result["modules"].setdefault("nlp", []).append({
                "file":       filename,
                "risk_score": doc_nlp_score,
                "risk_level": nlp_dict.get("risk_level", "UNKNOWN"),
                "flags":      nlp_dict.get("semantic_flags", []),
                "entities":   nlp_dict.get("entities", {}),
                "identity_verification": nlp_dict.get("identity_verification"),
            })
        elif nlp_err:
            result["errors"].append(f"nlp ({filename}): {nlp_err}")

    # Step 6: GNN Cross-Document Graph (needs all files together)
    if temp_file_paths and MODULES.get("gnn"):
        gnn_result, gnn_err = _safe_run(
            "gnn", analyze_cross_document_graph, temp_file_paths
        )
        if gnn_result:
            gnn_fraud_ring = gnn_result.get("anomaly_detected", False)
            result["modules"]["gnn"] = {
                "fraud_ring_detected": gnn_fraud_ring,
                "integrity_score":     gnn_result.get("integrity_score", 100),
                "anomalies":           gnn_result.get("anomalies", []),
            }
        elif gnn_err:
            result["errors"].append(f"gnn: {gnn_err}")

    # Cleanup temp files
    for path in temp_file_paths:
        try:
            os.unlink(path)
        except Exception:
            pass

    # Step 7: Federated Learning (global cross-bank score)
    federated_score = None
    if MODULES.get("federated"):
        fl_record = {
            "session_duration":        session_data["session_duration"],
            "mouse_movement_score":    session_data["mouse_movement_score"],
            "copy_paste_detected":     session_data["copy_paste_detected"],
            "form_fill_speed":         session_data["form_fill_speed"],
            "tab_switches":            session_data["tab_switches"],
            "time_on_each_field":      session_data["time_on_each_field"],
            "hour_of_day":             session_data["hour_of_day"],
            "is_weekend":              session_data["is_weekend"],
            "failed_attempts":         session_data["failed_attempts"],
        }
        fl_result, fl_err = _safe_run("federated", score_records, [fl_record])
        if fl_result and fl_result.get("predictions"):
            federated_score = fl_result["predictions"][0]["risk_score"]
            result["modules"]["federated"] = {
                "risk_score":    federated_score,
                "model_version": fl_result.get("model_version", "unknown"),
            }
        elif fl_err:
            result["errors"].append(f"federated: {fl_err}")

    # ── Step 8: Aggregate final verdict ──────────────────────────────────
    module_scores = {
        "behavior":  behavior_score,
        "benfords":  benfords_score,
        "metadata":  metadata_score,
        "nlp":       nlp_score,
        "federated": federated_score,
        "gnn_fraud_ring": gnn_fraud_ring,
    }
    result["verdict"] = _aggregate_verdict(module_scores)
    result["module_scores"] = {k: v for k, v in module_scores.items() if v is not None}

    logger.info(
        f"[{sid}] Done — score={result['verdict']['final_score']} "
        f"verdict={result['verdict']['verdict']} "
        f"errors={len(result['errors'])}"
    )
    return JSONResponse(result)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    hostname  = socket.gethostname()
    local_ip  = socket.gethostbyname(hostname)
    print(f"\n  TrustNet backend running")
    print(f"  Local  : http://localhost:8000")
    print(f"  Network: http://{local_ip}:8000")
    print(f"  Docs   : http://localhost:8000/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)