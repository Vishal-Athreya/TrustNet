# test_server.py
# Simple test server — run this, open on phone, upload a file
# Tests the full layer including VPN detection

from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(__file__))
from main import run_behavior_analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple upload page — open this on your phone"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>TrustNet — Document Upload Test</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: Arial, sans-serif;
                background: #f0f4f8;
                padding: 20px;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 24px;
                max-width: 480px;
                margin: 20px auto;
                box-shadow: 0 2px 12px rgba(0,0,0,0.1);
            }
            h1 {
                color: #1a365d;
                font-size: 22px;
                margin-bottom: 6px;
            }
            p {
                color: #666;
                font-size: 14px;
                margin-bottom: 20px;
            }
            .text-field {
                width: 100%;
                padding: 12px;
                border: 1px solid #cbd5e0;
                border-radius: 8px;
                font-size: 14px;
                margin-bottom: 16px;
            }
            .text-field:focus {
                outline: none;
                border-color: #4299e1;
            }
            .upload-box {
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                padding: 30px;
                text-align: center;
                margin-bottom: 16px;
                cursor: pointer;
                transition: border-color 0.2s;
            }
            .upload-box:hover { border-color: #4299e1; }
            .upload-box input { display: none; }
            .upload-box label {
                cursor: pointer;
                color: #4299e1;
                font-weight: bold;
            }
            .file-name {
                margin-top: 8px;
                font-size: 13px;
                color: #555;
            }
            button {
                width: 100%;
                padding: 14px;
                background: #2b6cb0;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 8px;
            }
            button:disabled {
                background: #a0aec0;
                cursor: not-allowed;
            }
            #result {
                margin-top: 20px;
                display: none;
            }
            .result-card {
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
            }
            .HIGH    { background: #fff5f5; border-left: 4px solid #e53e3e; }
            .SUSPICIOUS { background: #fffff0; border-left: 4px solid #d69e2e; }
            .CLEAN   { background: #f0fff4; border-left: 4px solid #38a169; }
            .BLOCKED { background: #fff5f5; border-left: 4px solid #e53e3e; }
            .score {
                font-size: 36px;
                font-weight: bold;
                margin: 8px 0;
            }
            .label {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 4px;
            }
            .flag {
                background: #edf2f7;
                border-radius: 6px;
                padding: 8px 12px;
                margin-top: 8px;
                font-size: 13px;
            }
            .flag.HIGH-flag   { border-left: 3px solid #e53e3e; }
            .flag.MEDIUM-flag { border-left: 3px solid #d69e2e; }
            .flag.LOW-flag    { border-left: 3px solid #38a169; }
            .info-row {
                display: flex;
                justify-content: space-between;
                font-size: 13px;
                color: #555;
                padding: 4px 0;
                border-bottom: 1px solid #eee;
            }
            .loader {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .debug-row {
                font-size: 11px;
                color: #999;
                padding: 2px 0;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🏦 TrustNet</h1>
            <p>Upload any document to test fraud detection</p>

            <input type="text" id="nameInput" class="text-field" placeholder="Applicant full name">

            <div class="upload-box">
                <input type="file" id="fileInput" accept=".pdf,.jpg,.jpeg,.png,.doc,.docx">
                <label for="fileInput">📎 Choose a file</label>
                <div class="file-name" id="fileName">No file chosen</div>
            </div>

            <button id="uploadBtn" onclick="uploadFile()" disabled>
                Analyse Document
            </button>

            <div id="result"></div>
        </div>

        <script>
            // Show file name when selected
            document.getElementById('fileInput').addEventListener('change', function() {
                const name = this.files[0]?.name || 'No file chosen';
                document.getElementById('fileName').textContent = name;
                document.getElementById('uploadBtn').disabled = !this.files[0];
            });

            // ── Real behavior tracking ──────────────────────────────────
            let mouseMovements    = 0;
            let copyPasteCount    = 0;
            let startTime         = Date.now();
            let keystrokeTimes    = [];   // real keydown timestamps in nameInput
            let nameFieldFocusAt  = null;
            let nameFieldTotalSec = 0;
            let tabSwitchCount    = 0;

            document.addEventListener('mousemove', () => mouseMovements++);
            document.addEventListener('touchmove', () => mouseMovements++);
            document.addEventListener('paste',     () => copyPasteCount++);

            // Real tab-switch detection
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) tabSwitchCount++;
            });

            const nameInput = document.getElementById('nameInput');

            // Real keystroke timing — this is what makes form_fill_speed real
            nameInput.addEventListener('keydown', () => {
                keystrokeTimes.push(Date.now());
            });

            nameInput.addEventListener('focus', () => {
                nameFieldFocusAt = Date.now();
            });

            nameInput.addEventListener('blur', () => {
                if (nameFieldFocusAt) {
                    nameFieldTotalSec += (Date.now() - nameFieldFocusAt) / 1000;
                    nameFieldFocusAt = null;
                }
            });

            async function uploadFile() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) return;

                // Close out an open focus timer if the field is still focused
                if (nameFieldFocusAt) {
                    nameFieldTotalSec += (Date.now() - nameFieldFocusAt) / 1000;
                    nameFieldFocusAt = null;
                }

                const btn = document.getElementById('uploadBtn');
                btn.disabled  = true;
                btn.textContent = 'Analysing...';

                document.getElementById('result').innerHTML = `
                    <div class="loader">⏳ Checking your connection and analysing...</div>
                `;
                document.getElementById('result').style.display = 'block';

                const duration = (Date.now() - startTime) / 1000;

                // Real typing speed from actual keystrokes. If text exists
                // in the field but ZERO keydown events were ever recorded,
                // the value was injected via script/console/paste rather
                // than typed — that's flagged as maximally suspicious
                // (999 chars/sec, physically impossible for a human).
                const nameValue = nameInput.value || "";
                let formFillSpeed;
                if (keystrokeTimes.length >= 2) {
                    const elapsedSec = (keystrokeTimes[keystrokeTimes.length - 1] - keystrokeTimes[0]) / 1000;
                    formFillSpeed = elapsedSec > 0 ? keystrokeTimes.length / elapsedSec : 999;
                } else if (nameValue.length > 0 && keystrokeTimes.length === 0) {
                    formFillSpeed = 999;
                } else {
                    formFillSpeed = 0;
                }

                const formData = new FormData();
                formData.append('file', file);
                formData.append('session_duration',      duration.toString());
                formData.append('mouse_movement_score',  Math.min(mouseMovements / 200, 1.0).toString());
                formData.append('copy_paste_detected',   (copyPasteCount > 0 ? 1 : 0).toString());
                formData.append('user_agent',            navigator.userAgent);
                formData.append('screen_resolution',     screen.width + 'x' + screen.height);
                formData.append('timezone',              Intl.DateTimeFormat().resolvedOptions().timeZone);
                formData.append('form_fill_speed',       formFillSpeed.toString());
                formData.append('time_on_each_field',    nameFieldTotalSec.toString());
                formData.append('tab_switches',          tabSwitchCount.toString());

                try {
                    const response = await fetch('/analyse', {
                        method: 'POST',
                        body: formData,
                    });
                    const result = await response.json();
                    showResult(result);
                } catch (err) {
                    document.getElementById('result').innerHTML = `
                        <div class="result-card BLOCKED">
                            <div class="label">❌ Connection Error</div>
                            <div>${err.message}</div>
                        </div>
                    `;
                }

                btn.disabled  = false;
                btn.textContent = 'Analyse Document';
            }

            function showResult(r) {
                let html = '';

                if (r.blocked) {
                    html = `
                        <div class="result-card BLOCKED">
                            <div class="label">🚫 Upload Blocked</div>
                            <p style="margin-top:8px;font-size:14px;">${r.block_reason}</p>
                        </div>
                    `;
                } else {
                    const icon = {
                        HIGH: '🔴', SUSPICIOUS: '🟡', CLEAN: '🟢'
                    }[r.risk_level] || '';

                    html = `
                        <div class="result-card ${r.risk_level}">
                            <div class="label">${icon} ${r.risk_level}</div>
                            <div class="score">${r.score}/100</div>
                            <div class="info-row">
                                <span>Browser</span>
                                <span>${r.browser_type} on ${r.os_type}</span>
                            </div>
                            <div class="info-row">
                                <span>Headless Bot</span>
                                <span>${r.is_headless_browser ? '⚠️ YES' : '✅ No'}</span>
                            </div>
                            <div class="info-row">
                                <span>Anomaly</span>
                                <span>${r.is_anomaly ? '⚠️ YES' : '✅ No'}</span>
                            </div>
                        </div>
                    `;

                    if (r.flags && r.flags.length > 0) {
                        html += `<div style="font-weight:bold;margin:12px 0 8px;font-size:14px;">
                            🚩 ${r.flags.length} Flag(s) Found:
                        </div>`;
                        r.flags.forEach(f => {
                            html += `
                                <div class="flag ${f.severity}-flag">
                                    <strong>[${f.severity}]</strong> ${f.message}
                                </div>
                            `;
                        });
                    } else {
                        html += `
                            <div class="flag">
                                ✅ No suspicious flags found
                            </div>
                        `;
                    }
                }

                document.getElementById('result').innerHTML = html;
                document.getElementById('result').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """


@app.post("/analyse")
async def analyse(request: Request, file: UploadFile):
    """Receives file upload + session data, runs full analysis"""

    # Get form data
    form           = await request.form()
    duration       = float(form.get("session_duration",     300.0))
    mouse          = float(form.get("mouse_movement_score", 0.7))
    paste          = int(form.get("copy_paste_detected",    0))
    ua             = form.get("user_agent",       "unknown")
    screen         = form.get("screen_resolution","unknown")
    tz             = form.get("timezone",         "unknown")
    form_speed     = float(form.get("form_fill_speed",      2.5))
    time_per_field = float(form.get("time_on_each_field",   5.0))
    tab_switches   = int(float(form.get("tab_switches",     0)))

    # Get real client IP
    client_ip = request.headers.get("X-Forwarded-For", "")
    if not client_ip:
        client_ip = request.client.host

    # When testing locally, 127.0.0.1 means "this machine" —
    # fetch the real public IP instead so VPN detection works correctly
    if client_ip in ("127.0.0.1", "::1", "localhost"):
        import urllib.request
        try:
            client_ip = urllib.request.urlopen(
                "https://api.ipify.org", timeout=3
            ).read().decode()
        except Exception:
            client_ip = "127.0.0.1"

    now = datetime.datetime.now()

    session_data = {
        "session_id":              f"MOBILE-{now.strftime('%H%M%S')}",
        "user_agent":              ua,
        "ip_address":              client_ip,
        "screen_resolution":       screen,
        "timezone":                tz,
        "hour_of_day":             now.hour,
        "day_of_week":             now.weekday(),
        "is_weekend":              int(now.weekday() >= 5),
        "failed_attempts":         0,
        "session_duration":        duration,
        "num_devices_used":        1,
        "ip_is_vpn":               0,
        "ip_is_tor":               0,
        "ip_is_proxy":             0,
        "ip_reputation_score":     0.0,
        "form_fill_speed":         form_speed,
        "mouse_movement_score":    mouse,
        "copy_paste_detected":     paste,
        "tab_switches":            tab_switches,
        "time_on_each_field":      time_per_field,
        "geolocation_mismatch":    0,
        "distance_from_branch_km": 10.0,
        "previous_applications":   0,
        "applications_same_ip":    1,
        "applications_same_device":1,
        "pan_seen_before":         0,
        "aadhaar_seen_before":     0,
        "velocity_score":          50.0,
    }

    result = run_behavior_analysis(session_data)
    return JSONResponse(result)

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "="*55)
    print("  TrustNet Test Server Running")
    print("="*55)
    print(f"\n  Open on YOUR laptop  : http://localhost:8000")
    print(f"  Open on YOUR PHONE   : http://{local_ip}:8000")
    print(f"\n  Make sure phone and laptop are on SAME WiFi!")
    print("="*55 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)