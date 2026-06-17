"""
TrustNet — Layer 1 / Sub-layer 03: Metadata Forensics
======================================================
What this does (plain English):
    Every digital file carries hidden data beyond its visible content — creation
    timestamps, editing software, author names, GPS coordinates, revision history.
    Fraudsters who forge documents rarely clean this metadata perfectly.

    This module cross-examines that hidden layer against what the document *claims*:
      • A salary slip "issued on 01-Jan-2024" but last edited on 15-Mar-2026? Red flag.
      • A government-stamped PDF created in Microsoft Word? Red flag.
      • An Aadhaar card whose EXIF says it was photographed with an iPhone in 2026?
        Red flag.
      • A "scanned" document with zero camera metadata and perfect DPI? Red flag.

Supported file types:
    PDF, DOCX, XLSX, PPTX, JPEG, PNG, TIFF

Output contract (MetadataForensicsResult):
    Every caller receives a typed Pydantic model — safe to JSON-serialise and pass
    directly to the TrustNet scoring engine.

Author:  TrustNet Core Team — Document Intelligence Module
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import piexif
from dateutil import parser as dateutil_parser
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ──────────────────────────────────────────────────────────────────────────────
# ENUMS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


# Software signatures we expect from legitimate government/bank document issuers.
# If a "bank statement" was created in GIMP, that's a red flag.
LEGITIMATE_PDF_PRODUCERS: set[str] = {
    "oracle", "adobe", "microsoft", "libreoffice", "sap", "finacle",
    "temenos", "infosys", "tcs", "oracle financials", "flexcube",
    "bankmaster", "nucleus", "creditsights",
}

SUSPICIOUS_PDF_PRODUCERS: set[str] = {
    "gimp", "inkscape", "canva", "photoshop", "paint", "corel",
    "foxit phantom", "pdf-xchange", "nitro", "smallpdf", "ilovepdf",
    "img2pdf",  # converts image → PDF, classic forgery trick
}

# Documents that should NEVER have been created by a photo app
DOCUMENT_TYPES_WITH_EXPECTED_SOFTWARE: dict[str, list[str]] = {
    "salary_slip": ["tally", "oracle", "sap", "finacle", "hrms", "payroll"],
    "bank_statement": ["finacle", "flexcube", "temenos", "oracle", "bankmaster"],
    "aadhaar": ["uidai", "digilocker", "protean"],
    "pan_card": ["protean", "utiitsl", "nsdl"],
    "land_record": ["gov", "nic", "e-district", "bhulekh"],
    "itr": ["traces", "efiling", "income tax"],
}

# Maximum tolerable gap between "document date" and metadata creation date (days).
# A salary slip for March 2024 should not have been created in 2026.
MAX_DATE_GAP_DAYS = 90


# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC OUTPUT MODELS
# ──────────────────────────────────────────────────────────────────────────────

class DateAnomalyDetail(BaseModel):
    """A single date inconsistency found in metadata."""
    field_name: str = Field(description="Which metadata field (e.g. 'creation_date')")
    metadata_value: str | None = Field(description="What the metadata says")
    claimed_value: str | None = Field(description="What the document claims (if extractable)")
    gap_days: float | None = Field(description="Absolute difference in days")
    explanation: str = Field(description="Human-readable explanation of why this is suspicious")


class SoftwareAnomalyDetail(BaseModel):
    """A software/tool inconsistency."""
    tool_found: str | None
    tool_expected_pattern: list[str]
    explanation: str


class MetadataSnapshot(BaseModel):
    """Raw metadata extracted — the audit evidence."""
    creation_date: str | None = None
    modification_date: str | None = None
    author: str | None = None
    producer: str | None = None           # PDF-specific: what generated the PDF
    creator_tool: str | None = None       # XMP: xmp:CreatorTool
    last_modified_by: str | None = None   # DOCX/XLSX: last editor
    revision_count: int | None = None     # DOCX: number of revisions
    page_count: int | None = None
    file_size_bytes: int | None = None
    mime_type_declared: str | None = None
    mime_type_actual: str | None = None
    sha256: str | None = None
    # EXIF-specific (images)
    camera_make: str | None = None
    camera_model: str | None = None
    gps_latitude: float | None = None
    gps_longitude: float | None = None
    image_dpi: tuple[int, int] | None = None
    color_space: str | None = None
    # Office-specific
    company: str | None = None
    template_used: str | None = None


class MetadataForensicsResult(BaseModel):
    """
    Final output of the Metadata Forensics module.
    Consumed by the TrustNet scoring engine.
    """
    file_name: str
    file_type: str                        # 'pdf', 'docx', 'image', etc.
    risk_level: RiskLevel
    risk_score: float = Field(
        ge=0.0, le=100.0,
        description="0 = clean, 100 = definite forgery indicator",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Plain-English red flags for the officer's report",
    )
    date_anomalies: list[DateAnomalyDetail] = Field(default_factory=list)
    software_anomaly: SoftwareAnomalyDetail | None = None
    metadata: MetadataSnapshot
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_exiftool_output: dict[str, Any] | None = Field(
        default=None,
        description="Full ExifTool JSON — stored for audit trail",
    )


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: SHA-256 of file bytes
# ──────────────────────────────────────────────────────────────────────────────

def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Run ExifTool (system binary) on a temp file
# ExifTool is the gold standard — it reads 200+ metadata formats.
# ──────────────────────────────────────────────────────────────────────────────

def _run_exiftool(file_path: str) -> dict[str, Any] | None:
    """
    Runs ExifTool and returns parsed JSON.
    Returns None if ExifTool is not installed (graceful degradation).
    """
    try:
        result = subprocess.run(
            ["exiftool", "-json", "-all", file_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            return data[0] if data else None
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.warning("ExifTool not available or timed out — using Python-only fallback.")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Safe date parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return dateutil_parser.parse(str(value))
    except (ValueError, OverflowError):
        return None


def _date_gap_days(dt1: datetime | None, dt2: datetime | None) -> float | None:
    """Return absolute gap in days between two datetimes, timezone-aware."""
    if dt1 is None or dt2 is None:
        return None
    # Strip timezone for comparison if mixed
    d1 = dt1.replace(tzinfo=None) if dt1.tzinfo else dt1
    d2 = dt2.replace(tzinfo=None) if dt2.tzinfo else dt2
    return abs((d1 - d2).total_seconds() / 86400)


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTOR: PDF
# ──────────────────────────────────────────────────────────────────────────────

def _extract_pdf_metadata(data: bytes) -> MetadataSnapshot:
    """
    Extract metadata from a PDF using PyMuPDF (primary) + pypdf (fallback).
    PyMuPDF reads XMP streams; pypdf reads the /Info dictionary.
    """
    snap = MetadataSnapshot()
    snap.mime_type_declared = "application/pdf"

    # ── PyMuPDF pass ──────────────────────────────────────────────────────────
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        meta = doc.metadata or {}

        snap.creation_date = meta.get("creationDate")
        snap.modification_date = meta.get("modDate")
        snap.author = meta.get("author")
        snap.producer = meta.get("producer")
        snap.creator_tool = meta.get("creator")
        snap.page_count = doc.page_count

        # XMP metadata (richer than /Info)
        xmp_raw = doc.get_xml_metadata()
        if xmp_raw:
            # Extract xmp:ModifyDate — more reliable than /Info modDate
            m = re.search(r"<xmp:ModifyDate>(.*?)</xmp:ModifyDate>", xmp_raw)
            if m:
                snap.modification_date = snap.modification_date or m.group(1).strip()
            m = re.search(r"<xmp:CreateDate>(.*?)</xmp:CreateDate>", xmp_raw)
            if m:
                snap.creation_date = snap.creation_date or m.group(1).strip()
            m = re.search(r"<xmp:CreatorTool>(.*?)</xmp:CreatorTool>", xmp_raw)
            if m:
                snap.creator_tool = snap.creator_tool or m.group(1).strip()

        doc.close()
    except Exception as exc:
        logger.warning(f"PyMuPDF extraction failed: {exc}")

    # ── pypdf fallback for /Info dictionary ───────────────────────────────────
    try:
        reader = PdfReader(io.BytesIO(data))
        info = reader.metadata
        if info:
            snap.author = snap.author or info.get("/Author")
            snap.producer = snap.producer or info.get("/Producer")
            snap.creator_tool = snap.creator_tool or info.get("/Creator")
            if not snap.creation_date:
                snap.creation_date = str(info.get("/CreationDate", ""))
            if not snap.modification_date:
                snap.modification_date = str(info.get("/ModDate", ""))
    except Exception as exc:
        logger.warning(f"pypdf extraction failed: {exc}")

    return snap


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTOR: Office Documents (DOCX / XLSX / PPTX)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_office_metadata(data: bytes, extension: str) -> MetadataSnapshot:
    """
    Extract core properties from Office Open XML formats.
    All three (docx, xlsx, pptx) share the same core_properties interface.
    """
    snap = MetadataSnapshot()

    try:
        if extension == "docx":
            from docx import Document
            doc = Document(io.BytesIO(data))
            props = doc.core_properties
            snap.mime_type_declared = (
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            )
        elif extension == "xlsx":
            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(data), read_only=True)
            props = wb.properties
            snap.mime_type_declared = (
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            )
        elif extension == "pptx":
            from pptx import Presentation
            prs = Presentation(io.BytesIO(data))
            props = prs.core_properties
            snap.mime_type_declared = (
                "application/vnd.openxmlformats-officedocument"
                ".presentationml.presentation"
            )
        else:
            return snap

        snap.author = str(props.author) if props.author else None
        snap.last_modified_by = (
            str(props.last_modified_by) if props.last_modified_by else None
        )
        snap.company = str(getattr(props, "company", "") or "")
        snap.revision_count = getattr(props, "revision", None)
        if props.created:
            snap.creation_date = props.created.isoformat()
        if props.modified:
            snap.modification_date = props.modified.isoformat()

    except Exception as exc:
        logger.warning(f"Office metadata extraction failed ({extension}): {exc}")

    return snap


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTOR: Image files (JPEG, PNG, TIFF)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_image_metadata(data: bytes) -> MetadataSnapshot:
    """
    Extract EXIF data from images.
    Key checks: GPS (why does an income certificate have GPS coordinates?),
    camera model (this came from an iPhone, not a scanner), DPI.
    """
    snap = MetadataSnapshot()

    try:
        img = Image.open(io.BytesIO(data))
        snap.image_dpi = img.info.get("dpi")
        snap.color_space = img.mode
        snap.mime_type_declared = Image.MIME.get(img.format, "image/unknown")

        # ── EXIF via piexif ───────────────────────────────────────────────────
        raw_exif = img.info.get("exif")
        if raw_exif:
            exif_dict = piexif.load(raw_exif)

            zeroth = exif_dict.get("0th", {})
            snap.camera_make = _decode_bytes(zeroth.get(piexif.ImageIFD.Make))
            snap.camera_model = _decode_bytes(zeroth.get(piexif.ImageIFD.Model))

            # DateTime from EXIF (when photo was taken)
            dt_str = _decode_bytes(zeroth.get(piexif.ImageIFD.DateTime))
            if dt_str:
                # EXIF datetime: "2024:03:15 10:30:00"
                snap.creation_date = dt_str.replace(":", "-", 2)

            # GPS
            gps = exif_dict.get("GPS", {})
            if gps:
                snap.gps_latitude = _dms_to_decimal(
                    gps.get(piexif.GPSIFD.GPSLatitude),
                    gps.get(piexif.GPSIFD.GPSLatitudeRef),
                )
                snap.gps_longitude = _dms_to_decimal(
                    gps.get(piexif.GPSIFD.GPSLongitude),
                    gps.get(piexif.GPSIFD.GPSLongitudeRef),
                )
    except Exception as exc:
        logger.warning(f"Image metadata extraction failed: {exc}")

    return snap


def _decode_bytes(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00").strip()
    return str(value).strip() if value else None


def _dms_to_decimal(
    dms: tuple | None, ref: bytes | None
) -> float | None:
    """Convert GPS degrees/minutes/seconds to decimal degrees."""
    if not dms or len(dms) < 3:
        return None
    try:
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
        decimal = degrees + minutes / 60 + seconds / 3600
        ref_str = ref.decode("utf-8").strip() if isinstance(ref, bytes) else str(ref)
        if ref_str in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except (ZeroDivisionError, TypeError, IndexError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSER: Date anomalies
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_dates(
    snap: MetadataSnapshot,
    claimed_document_date: str | None,
) -> list[DateAnomalyDetail]:
    """
    Check for temporal inconsistencies.
    The key insight: if a document was 'issued' in 2022 but its file metadata
    shows it was *created* or *last edited* in 2025/2026, it was forged.
    """
    anomalies: list[DateAnomalyDetail] = []

    creation_dt = _parse_date(snap.creation_date)
    modification_dt = _parse_date(snap.modification_date)
    claimed_dt = _parse_date(claimed_document_date)
    now = datetime.now()

    # ── Check 1: Modification date is AFTER creation date by an unusual margin ──
    gap = _date_gap_days(creation_dt, modification_dt)
    if gap is not None and gap > 1:
        # Small edits (metadata, compression) are common but large gaps are suspicious.
        anomalies.append(DateAnomalyDetail(
            field_name="modification_date vs creation_date",
            metadata_value=snap.modification_date,
            claimed_value=snap.creation_date,
            gap_days=gap,
            explanation=(
                f"Document was modified {gap:.0f} days after it was initially created. "
                f"Legitimate issued documents (salary slips, bank statements) "
                f"should not require post-creation edits."
            ),
        ))

    # ── Check 2: Claimed document date vs metadata creation date ──────────────
    if claimed_dt and creation_dt:
        gap_claimed = _date_gap_days(claimed_dt, creation_dt)
        if gap_claimed is not None and gap_claimed > MAX_DATE_GAP_DAYS:
            anomalies.append(DateAnomalyDetail(
                field_name="claimed_date vs metadata_creation_date",
                metadata_value=snap.creation_date,
                claimed_value=claimed_document_date,
                gap_days=gap_claimed,
                explanation=(
                    f"Document claims to be dated '{claimed_document_date}' "
                    f"but was digitally created {gap_claimed:.0f} days "
                    f"{'before' if creation_dt < claimed_dt else 'after'} that date. "
                    f"This exceeds the {MAX_DATE_GAP_DAYS}-day tolerance and suggests "
                    f"a backdated or post-dated forgery."
                ),
            ))

    # ── Check 3: Modification date is in the future ───────────────────────────
    if modification_dt and modification_dt > now:
        anomalies.append(DateAnomalyDetail(
            field_name="modification_date",
            metadata_value=snap.modification_date,
            claimed_value=None,
            gap_days=_date_gap_days(modification_dt, now),
            explanation=(
                f"Modification date '{snap.modification_date}' is set in the future. "
                f"This indicates clock manipulation during forgery."
            ),
        ))

    # ── Check 4: Creation date is in the future ───────────────────────────────
    if creation_dt and creation_dt > now:
        anomalies.append(DateAnomalyDetail(
            field_name="creation_date",
            metadata_value=snap.creation_date,
            claimed_value=None,
            gap_days=_date_gap_days(creation_dt, now),
            explanation=(
                f"Creation date '{snap.creation_date}' is set in the future — "
                f"impossible for a legitimately issued historical document."
            ),
        ))

    # ── Check 5: Very recent edit on an old document ──────────────────────────
    if claimed_dt and modification_dt:
        if (now - claimed_dt).days > 365 and (now - modification_dt).days < 30:
            anomalies.append(DateAnomalyDetail(
                field_name="recent_edit_on_old_document",
                metadata_value=snap.modification_date,
                claimed_value=claimed_document_date,
                gap_days=(now - modification_dt).days,
                explanation=(
                    f"A document claimed to be from "
                    f"{claimed_document_date} was edited within the last 30 days. "
                    f"This is the classic pattern of a forged backdated document."
                ),
            ))

    return anomalies


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSER: Software anomalies
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_software(
    snap: MetadataSnapshot,
    document_type: str | None,
) -> SoftwareAnomalyDetail | None:
    """
    Cross-checks the creating software against what we'd expect for this document type.
    A PAN card created in GIMP is not a PAN card.
    """
    # Gather all software-related metadata into one string for matching
    tool_indicators = " ".join(filter(None, [
        snap.producer or "",
        snap.creator_tool or "",
        snap.camera_make or "",
        snap.camera_model or "",
    ])).lower()

    if not tool_indicators.strip():
        return None

    # ── Check: Known suspicious tools ────────────────────────────────────────
    for suspicious in SUSPICIOUS_PDF_PRODUCERS:
        if suspicious in tool_indicators:
            return SoftwareAnomalyDetail(
                tool_found=tool_indicators,
                tool_expected_pattern=list(LEGITIMATE_PDF_PRODUCERS),
                explanation=(
                    f"Document was created or processed by '{suspicious}' — "
                    f"a photo/design editing tool, not a legitimate document "
                    f"issuance system. This is a strong indicator of manual forgery."
                ),
            )

    # ── Check: Document-type-specific expected software ───────────────────────
    if document_type and document_type.lower() in DOCUMENT_TYPES_WITH_EXPECTED_SOFTWARE:
        expected = DOCUMENT_TYPES_WITH_EXPECTED_SOFTWARE[document_type.lower()]
        found_match = any(e in tool_indicators for e in expected)
        if not found_match:
            return SoftwareAnomalyDetail(
                tool_found=tool_indicators or None,
                tool_expected_pattern=expected,
                explanation=(
                    f"For document type '{document_type}', we expect the issuing "
                    f"software to match one of: {expected}. "
                    f"Found '{tool_indicators}' instead. "
                    f"This document may not have originated from a legitimate source."
                ),
            )

    return None


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSER: Image-specific checks
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_image_flags(snap: MetadataSnapshot) -> list[str]:
    """
    Additional flags specific to image documents.
    """
    flags: list[str] = []

    # GPS in a financial document is suspicious
    if snap.gps_latitude is not None:
        flags.append(
            f"EXIF GPS coordinates found ({snap.gps_latitude}, {snap.gps_longitude}). "
            f"Legitimate scanned documents do not embed GPS data — this image was "
            f"captured on a mobile device, not scanned from a physical document."
        )

    # Camera make/model in a "scanned" document is suspicious
    if snap.camera_make or snap.camera_model:
        tool = f"{snap.camera_make or ''} {snap.camera_model or ''}".strip()
        flags.append(
            f"EXIF camera data: '{tool}'. This document was photographed on a device, "
            f"not scanned from a printer. Phone photographs of documents are a common "
            f"substitution method."
        )

    # Suspicious DPI — legitimate scans are 150–600 DPI
    if snap.image_dpi:
        dpi_x, dpi_y = snap.image_dpi
        if dpi_x < 72 or dpi_y < 72:
            flags.append(
                f"Image DPI is {dpi_x}x{dpi_y} — extremely low for a document scan. "
                f"Legitimate bank/government document scans are 150–600 DPI."
            )
        if dpi_x != dpi_y:
            flags.append(
                f"Asymmetric DPI ({dpi_x}x{dpi_y}) — may indicate resampling or "
                f"manipulation after scanning."
            )

    return flags


# ──────────────────────────────────────────────────────────────────────────────
# SCORE CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────

def _calculate_score(
    date_anomalies: list[DateAnomalyDetail],
    software_anomaly: SoftwareAnomalyDetail | None,
    extra_flags: list[str],
    exiftool_data: dict | None,
) -> tuple[float, RiskLevel]:
    """
    Convert findings into a 0–100 risk score.
    Scoring philosophy: additive, capped at 100.
    """
    score = 0.0

    # ── Date anomaly scoring ──────────────────────────────────────────────────
    for anomaly in date_anomalies:
        gap = anomaly.gap_days or 0
        if "recent_edit_on_old_document" in anomaly.field_name:
            score += 40.0   # Strongest signal
        elif "claimed_date vs metadata_creation_date" in anomaly.field_name:
            # Scale with the gap — bigger gap = more likely forged
            score += min(35.0, 10 + gap * 0.05)
        elif "future" in anomaly.explanation.lower():
            score += 30.0
        elif "modification" in anomaly.field_name:
            score += min(20.0, 5 + gap * 0.02)

    # ── Software anomaly scoring ──────────────────────────────────────────────
    if software_anomaly:
        tool = (software_anomaly.tool_found or "").lower()
        if any(s in tool for s in SUSPICIOUS_PDF_PRODUCERS):
            score += 45.0   # Photo editor created a "government document" — huge red flag
        else:
            score += 20.0   # Wrong-but-not-obviously-malicious software

    # ── Image flags ───────────────────────────────────────────────────────────
    for flag in extra_flags:
        if "gps" in flag.lower():
            score += 15.0
        elif "camera" in flag.lower():
            score += 20.0
        elif "dpi" in flag.lower():
            score += 10.0

    # ── ExifTool-specific: check if metadata was deliberately stripped ─────────
    if exiftool_data:
        all_fields = json.dumps(exiftool_data).lower()
        if "pdf:createdate" not in all_fields and "createdate" not in all_fields:
            score += 10.0   # Stripped metadata is itself suspicious

    score = min(score, 100.0)

    # ── Map to RiskLevel ──────────────────────────────────────────────────────
    if score >= 60:
        level = RiskLevel.HIGH_RISK
    elif score >= 25:
        level = RiskLevel.SUSPICIOUS
    else:
        level = RiskLevel.CLEAN

    return round(score, 2), level


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def analyse_metadata(
    file_bytes: bytes,
    file_name: str,
    claimed_document_date: str | None = None,
    document_type: str | None = None,
) -> MetadataForensicsResult:
    """
    Main entry point. Call this from the TrustNet backend or as a standalone module.

    Args:
        file_bytes:             Raw bytes of the uploaded file.
        file_name:              Original file name (used to infer extension).
        claimed_document_date:  The date the document *claims* to be from
                                (e.g. "01 Jan 2024" from the visible text).
                                Pass None if unknown.
        document_type:          Semantic type: 'salary_slip', 'bank_statement',
                                'aadhaar', 'pan_card', 'land_record', 'itr', etc.
                                Pass None for generic analysis.

    Returns:
        MetadataForensicsResult — fully typed, JSON-serialisable.
    """
    extension = Path(file_name).suffix.lstrip(".").lower()
    sha256 = _sha256_of_bytes(file_bytes)

    logger.info(f"[MetadataForensics] Analysing '{file_name}' ({len(file_bytes):,} bytes)")

    # ── Step 1: Extract raw metadata ─────────────────────────────────────────
    if extension == "pdf":
        snap = _extract_pdf_metadata(file_bytes)
        file_type = "pdf"
    elif extension in ("docx", "doc"):
        snap = _extract_office_metadata(file_bytes, "docx")
        file_type = "docx"
    elif extension in ("xlsx", "xls"):
        snap = _extract_office_metadata(file_bytes, "xlsx")
        file_type = "xlsx"
    elif extension in ("pptx", "ppt"):
        snap = _extract_office_metadata(file_bytes, "pptx")
        file_type = "pptx"
    elif extension in ("jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"):
        snap = _extract_image_metadata(file_bytes)
        file_type = "image"
    else:
        # Unknown type — minimal extraction via ExifTool only
        snap = MetadataSnapshot()
        file_type = extension or "unknown"

    snap.sha256 = sha256
    snap.file_size_bytes = len(file_bytes)

    # ── Step 2: ExifTool (if available) — run on temp file ───────────────────
    exiftool_data: dict | None = None
    with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        exiftool_data = _run_exiftool(tmp_path)
        if exiftool_data:
            # Backfill any fields ExifTool found that Python libraries missed
            snap.creation_date = snap.creation_date or exiftool_data.get("CreateDate")
            snap.modification_date = snap.modification_date or exiftool_data.get("ModifyDate")
            snap.producer = snap.producer or exiftool_data.get("Producer")
            snap.creator_tool = snap.creator_tool or exiftool_data.get("CreatorTool")
            snap.author = snap.author or exiftool_data.get("Author")
    finally:
        os.unlink(tmp_path)

    # ── Step 3: Run anomaly analysers ─────────────────────────────────────────
    date_anomalies = _analyse_dates(snap, claimed_document_date)
    software_anomaly = _analyse_software(snap, document_type)
    image_flags = _analyse_image_flags(snap) if file_type == "image" else []

    # ── Step 4: Assemble all plain-English flags ──────────────────────────────
    all_flags: list[str] = []
    for anomaly in date_anomalies:
        all_flags.append(f"DATE ANOMALY: {anomaly.explanation}")
    if software_anomaly:
        all_flags.append(f"SOFTWARE ANOMALY: {software_anomaly.explanation}")
    all_flags.extend(image_flags)

    # ── Step 5: Calculate score and risk level ────────────────────────────────
    risk_score, risk_level = _calculate_score(
        date_anomalies, software_anomaly, image_flags, exiftool_data
    )

    logger.info(
        f"[MetadataForensics] Result: {risk_level.value} "
        f"(score={risk_score}) | {len(all_flags)} flags"
    )

    return MetadataForensicsResult(
        file_name=file_name,
        file_type=file_type,
        risk_level=risk_level,
        risk_score=risk_score,
        flags=all_flags,
        date_anomalies=date_anomalies,
        software_anomaly=software_anomaly,
        metadata=snap,
        raw_exiftool_output=exiftool_data,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI / STANDALONE USAGE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path> [claimed_date] [document_type]")
        print("Example: python main.py salary_slip.pdf '2024-01-01' salary_slip")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    claimed = sys.argv[2] if len(sys.argv) > 2 else None
    doc_type = sys.argv[3] if len(sys.argv) > 3 else None

    result = analyse_metadata(
        file_bytes=path.read_bytes(),
        file_name=path.name,
        claimed_document_date=claimed,
        document_type=doc_type,
    )

    print("\n" + "═" * 70)
    print(f"  TrustNet Metadata Forensics — {result.file_name}")
    print("═" * 70)
    print(f"  Risk Level : {result.risk_level.value}")
    print(f"  Risk Score : {result.risk_score}/100")
    print(f"  Flags      : {len(result.flags)}")
    print()

    for i, flag in enumerate(result.flags, 1):
        print(f"  [{i}] {flag}")
        print()

    print("  Metadata Snapshot:")
    for k, v in result.metadata.model_dump().items():
        if v is not None:
            print(f"    {k:25s} : {v}")

    print("\n  Full JSON output:")
    print(result.model_dump_json(indent=2))