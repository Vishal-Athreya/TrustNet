"""
main.py — Error Level Analysis (ELA)
TrustNet | document_intelligence / 01_error_level_analysis

What it does
────────────
ELA detects pixel-level image tampering and copy-paste artefacts in scanned
documents (salary slips, bank statements, land records, ID cards).

How it works
────────────
JPEG compression is lossy and consistent: every region of an image that has
been compressed the same number of times will have a similar error level when
re-saved at a fixed quality.  A region that was *edited* — even slightly —
will have been re-compressed at least once more than the surrounding pixels,
so when you subtract a freshly-saved reference copy from the original, the
edited region sticks out as a bright patch of high error.

ELA amplifies those differences and computes four statistics per page:
  • mean_error       — baseline error level of the whole page
  • max_error        — brightest single-pixel anomaly
  • std_error        — spread; localised tampering produces high std
  • high_error_ratio — fraction of pixels with unusually high error

Each page gets a 0-100 risk score.  The document score is the max across pages.
A score ≥ 60 → HIGH RISK; 30–59 → SUSPICIOUS; < 30 → CLEAN.

Inputs / Outputs
────────────────
Input : path to a PDF file (str | Path)
Output: ELAResult dataclass with per-page details and a final verdict dict

Usage (standalone test)
───────────────────────
  python main.py path/to/document.pdf
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF ≥ 1.26.1 required for Python 3.13

from utils import (
    pdf_page_to_pil,
    compute_ela_array,
    region_stats,
    ela_to_heatmap_bgr,
    resize_for_report,
)


# ── Scoring thresholds ────────────────────────────────────────────────────────

# Mean ELA error below this → page is likely untampered
CLEAN_MEAN_THRESHOLD: float = 8.0

# Mean ELA error above this → page is likely tampered
TAMPER_MEAN_THRESHOLD: float = 18.0

# High-error pixel ratio above this → localised tampering patch suspected
RATIO_HIGH_RISK: float = 0.12

# Std deviation above this → uneven error distribution (tampering signature)
STD_HIGH_RISK: float = 22.0

# Final document risk bands
RISK_HIGH = "HIGH RISK"
RISK_SUSPICIOUS = "SUSPICIOUS"
RISK_CLEAN = "CLEAN"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PageELAResult:
    page_number: int          # 1-indexed
    mean_error: float
    max_error: float
    std_error: float
    high_error_ratio: float
    page_score: float         # 0-100
    flags: list[str] = field(default_factory=list)


@dataclass
class ELAResult:
    file_path: str
    total_pages: int
    pages: list[PageELAResult] = field(default_factory=list)
    document_score: float = 0.0        # max page score
    risk_level: str = RISK_CLEAN
    summary_flags: list[str] = field(default_factory=list)
    analysed_at: str = ""
    error: str | None = None


# ── Scoring logic ─────────────────────────────────────────────────────────────

def score_page(stats: dict[str, float]) -> tuple[float, list[str]]:
    """
    Convert region statistics to a 0-100 risk score and a list of plain-English
    flag strings.  This function is the single source of truth for how stats
    map to scores — change the weights here, nowhere else.

    Scoring breakdown (max 100):
      • mean_error          → up to 40 points
      • std_error           → up to 30 points
      • high_error_ratio    → up to 30 points
    """
    flags: list[str] = []
    score: float = 0.0

    mean = stats["mean_error"]
    std = stats["std_error"]
    ratio = stats["high_error_ratio"]
    max_err = stats["max_error"]

    # ── Mean error component (0-40) ──────────────────────────────────────────
    if mean >= TAMPER_MEAN_THRESHOLD:
        mean_score = 40.0
        flags.append(
            f"High average ELA error ({mean:.1f}) — document may have been re-saved "
            f"after editing."
        )
    elif mean >= CLEAN_MEAN_THRESHOLD:
        # Linear interpolation between the two thresholds
        mean_score = 40.0 * (mean - CLEAN_MEAN_THRESHOLD) / (
            TAMPER_MEAN_THRESHOLD - CLEAN_MEAN_THRESHOLD
        )
    else:
        mean_score = 0.0

    # ── Std-deviation component (0-30) ───────────────────────────────────────
    if std >= STD_HIGH_RISK:
        std_score = 30.0
        flags.append(
            f"Uneven error distribution (std={std:.1f}) — localised tampering patch "
            f"suspected (copy-paste or text overlay)."
        )
    else:
        std_score = min(30.0, (std / STD_HIGH_RISK) * 30.0)

    # ── High-error-ratio component (0-30) ────────────────────────────────────
    if ratio >= RATIO_HIGH_RISK:
        ratio_score = 30.0
        flags.append(
            f"{ratio * 100:.1f}% of pixels have abnormally high error — "
            f"large tampered area detected."
        )
    else:
        ratio_score = min(30.0, (ratio / RATIO_HIGH_RISK) * 30.0)

    # ── Max error bonus flag (informational, no extra points) ────────────────
    if max_err > 200.0:
        flags.append(
            f"Extreme single-pixel error ({max_err:.0f}/255) — "
            f"possible hard edge introduced by splicing."
        )

    score = mean_score + std_score + ratio_score
    return round(min(score, 100.0), 2), flags


def risk_level_from_score(score: float) -> str:
    if score >= 60.0:
        return RISK_HIGH
    if score >= 30.0:
        return RISK_SUSPICIOUS
    return RISK_CLEAN


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_document(file_path: str | Path) -> ELAResult:
    """
    Run full ELA on every page of a PDF and return an ELAResult.

    Parameters
    ----------
    file_path : str | Path
        Absolute or relative path to the target PDF.

    Returns
    -------
    ELAResult
        Populated dataclass.  On any unrecoverable error the `.error` field
        is set and `.risk_level` is returned as RISK_CLEAN so downstream
        scoring is not inflated by analysis failures.
    """
    file_path = Path(file_path)
    result = ELAResult(
        file_path=str(file_path),
        total_pages=0,
        analysed_at=datetime.now(timezone.utc).isoformat(),
    )

    if not file_path.exists():
        result.error = f"File not found: {file_path}"
        return result

    if file_path.suffix.lower() != ".pdf":
        result.error = f"Unsupported file type '{file_path.suffix}' — only PDFs accepted."
        return result

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        result.error = f"Could not open PDF: {exc}"
        return result

    result.total_pages = len(doc)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        try:
            pil_img = pdf_page_to_pil(page)
            ela_arr = compute_ela_array(pil_img)
            stats = region_stats(ela_arr)
            page_score, flags = score_page(stats)

            page_result = PageELAResult(
                page_number=page_number,
                mean_error=stats["mean_error"],
                max_error=stats["max_error"],
                std_error=stats["std_error"],
                high_error_ratio=stats["high_error_ratio"],
                page_score=page_score,
                flags=flags,
            )
            result.pages.append(page_result)

        except Exception as exc:
            # Log page-level error but continue processing remaining pages
            result.pages.append(
                PageELAResult(
                    page_number=page_number,
                    mean_error=0.0,
                    max_error=0.0,
                    std_error=0.0,
                    high_error_ratio=0.0,
                    page_score=0.0,
                    flags=[f"Page {page_number} analysis failed: {exc}"],
                )
            )

    doc.close()

    # ── Aggregate to document-level score ────────────────────────────────────
    if result.pages:
        result.document_score = max(p.page_score for p in result.pages)
        result.risk_level = risk_level_from_score(result.document_score)

        # Collect all unique flags from pages that contributed to the final score
        worst_page = max(result.pages, key=lambda p: p.page_score)
        result.summary_flags = worst_page.flags or ["No anomalies detected."]

    return result


# ── Output formatter ──────────────────────────────────────────────────────────

def format_verdict(result: ELAResult) -> dict[str, Any]:
    """
    Produce the standard TrustNet verdict dictionary consumed by the backend
    aggregator and the frontend dashboard.

    Schema
    ------
    {
        "module"        : "error_level_analysis",
        "file_path"     : str,
        "total_pages"   : int,
        "document_score": float,          # 0-100
        "risk_level"    : str,            # CLEAN | SUSPICIOUS | HIGH RISK
        "summary_flags" : list[str],
        "pages"         : list[dict],     # per-page detail
        "analysed_at"   : str,            # ISO-8601 UTC
        "error"         : str | None
    }
    """
    verdict = {
        "module": "error_level_analysis",
        "file_path": result.file_path,
        "total_pages": result.total_pages,
        "document_score": result.document_score,
        "risk_level": result.risk_level,
        "summary_flags": result.summary_flags,
        "pages": [
            {
                "page_number": p.page_number,
                "page_score": p.page_score,
                "mean_error": p.mean_error,
                "max_error": p.max_error,
                "std_error": p.std_error,
                "high_error_ratio": p.high_error_ratio,
                "flags": p.flags,
            }
            for p in result.pages
        ],
        "analysed_at": result.analysed_at,
        "error": result.error,
    }
    return verdict


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"\n[TrustNet ELA] Analysing: {pdf_path}\n")

    result = analyse_document(pdf_path)
    verdict = format_verdict(result)

    print(json.dumps(verdict, indent=2))

    # Print a clean human-readable summary to stdout
    print("\n" + "─" * 60)
    print(f"  RISK LEVEL    : {verdict['risk_level']}")
    print(f"  DOCUMENT SCORE: {verdict['document_score']} / 100")
    print(f"  PAGES ANALYSED: {verdict['total_pages']}")
    if verdict["error"]:
        print(f"  ERROR         : {verdict['error']}")
    if verdict["summary_flags"]:
        print("\n  KEY FLAGS:")
        for flag in verdict["summary_flags"]:
            print(f"    • {flag}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
