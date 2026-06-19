"""
TrustNet — Layer 1 / Sub-layer 04: NLP Semantics
=================================================
Added in v2.1: Identity Cross-Verification
  - Extracts name, PAN, Aadhaar from document text
  - Cross-checks against what the applicant claimed on the form
  - Mismatch → HIGH severity flag → score jumps to REVIEW/REJECT
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from transformers import AutoModel, AutoTokenizer

# Optional spaCy — graceful degradation if model not downloaded
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except (ImportError, OSError):
    _NLP = None
    _SPACY_AVAILABLE = False
    logger.warning("spaCy 'en_core_web_sm' not available — NER will be regex-based.")


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


DOCUMENT_KEYWORDS: dict[str, list[str]] = {
    "salary_slip": [
        "basic pay", "hra", "da", "pf", "professional tax", "gross salary",
        "net pay", "epf", "employee id", "month", "year", "deductions",
    ],
    "bank_statement": [
        "account number", "ifsc", "balance", "debit", "credit", "transaction",
        "branch", "closing balance", "opening balance", "statement period",
    ],
    "aadhaar": [
        "uidai", "unique identification", "government of india", "dob",
        "enrolment", "aadhaar",
    ],
    "pan_card": [
        "income tax department", "permanent account number", "government of india",
        "date of birth",
    ],
    "land_record": [
        "survey number", "khasra", "khata", "khatauni", "patta",
        "registration", "sub-registrar", "revenue", "taluka", "district",
    ],
    "itr": [
        "assessment year", "pan", "gross total income", "tax payable",
        "tds", "refund", "filing date", "income tax return",
    ],
}

CURRENCY_PATTERN = re.compile(
    r"(?:₹|Rs\.?\s*|INR\s*)?([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
    r"|\b(\d{1,2}\s+\w+\s+\d{4})"
    r"|\b(\w+\s+\d{4})\b"
    r"|\b(\d{4}[-]\d{2}[-]\d{2})\b",
    re.IGNORECASE,
)
PAN_PATTERN    = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
IFSC_PATTERN   = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")

BERT_MODEL_NAME = "distilbert-base-uncased"
_tokenizer: Any = None
_bert_model: Any = None


def _load_bert() -> tuple[Any, Any]:
    global _tokenizer, _bert_model
    if _tokenizer is None:
        logger.info(f"[NLP] Loading BERT model: {BERT_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        _bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        _bert_model.eval()
        logger.info("[NLP] BERT model loaded.")
    return _tokenizer, _bert_model


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC OUTPUT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    persons: list[str] = Field(default_factory=list)
    organisations: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    monetary_amounts: list[float] = Field(default_factory=list)
    pan_numbers: list[str] = Field(default_factory=list)
    aadhaar_numbers: list[str] = Field(default_factory=list)
    ifsc_codes: list[str] = Field(default_factory=list)
    account_numbers: list[str] = Field(default_factory=list)


class SemanticFlag(BaseModel):
    check_name: str
    severity: str   # 'HIGH', 'MEDIUM', 'LOW'
    description: str
    evidence: str | None = None


class NumericalCheck(BaseModel):
    check_name: str
    passed: bool
    expected: float | None = None
    found: float | None = None
    discrepancy: float | None = None
    description: str


class BertEmbeddingResult(BaseModel):
    document_type_matched: str | None
    similarity_score: float | None = None
    structural_deviation: float | None = None
    explanation: str


class IdentityVerificationResult(BaseModel):
    """Result of cross-checking form-submitted identity vs document text."""
    name_claimed: str | None = None
    pan_claimed: str | None = None
    aadhaar_claimed: str | None = None

    name_found_in_doc: str | None = None
    pan_found_in_doc: str | None = None
    aadhaar_found_in_doc: str | None = None

    name_match: bool | None = None      # None = could not verify
    pan_match: bool | None = None
    aadhaar_match: bool | None = None

    name_match_score: float | None = None   # 0-100 fuzzy score
    flags: list[str] = Field(default_factory=list)


class NLPSemanticsResult(BaseModel):
    file_name: str
    document_type: str | None
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=100.0)
    flags: list[str] = Field(default_factory=list)
    extracted_text_length: int = 0
    ocr_used: bool = False
    entities: ExtractedEntities
    semantic_flags: list[SemanticFlag] = Field(default_factory=list)
    numerical_checks: list[NumericalCheck] = Field(default_factory=list)
    bert_result: BertEmbeddingResult | None = None
    keyword_coverage: float | None = None
    cross_document_contradictions: list[str] = Field(default_factory=list)
    identity_verification: IdentityVerificationResult | None = None
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(data: bytes) -> tuple[str, bool]:
    ocr_used = False
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        native_text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        if len(native_text.strip()) >= 50:
            return native_text, ocr_used
        logger.info("[NLP] Native PDF text insufficient — falling back to EasyOCR.")
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        doc = fitz.open(stream=data, filetype="pdf")
        ocr_pages: list[str] = []
        for page in doc:
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            results = reader.readtext(img_cv, detail=0)
            ocr_pages.append(" ".join(results))
        doc.close()
        ocr_used = True
        return "\n".join(ocr_pages), ocr_used
    except Exception as exc:
        logger.warning(f"PDF text extraction failed: {exc}")
        return "", False


def _extract_text_from_image(data: bytes) -> tuple[str, bool]:
    try:
        import easyocr
        img_array = np.frombuffer(data, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(img_cv, detail=0)
        return " ".join(results), True
    except Exception as exc:
        logger.warning(f"Image OCR failed: {exc}")
        return "", True


def extract_text(file_bytes: bytes, file_name: str) -> tuple[str, bool]:
    ext = Path(file_name).suffix.lstrip(".").lower()
    if ext == "pdf":
        return _extract_text_from_pdf(file_bytes)
    elif ext in ("jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"):
        return _extract_text_from_image(file_bytes)
    else:
        logger.warning(f"Unsupported file type for NLP: {ext}")
        return "", False


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> ExtractedEntities:
    entities = ExtractedEntities()

    raw_amounts = CURRENCY_PATTERN.findall(text)
    entities.monetary_amounts = sorted(set(
        float(amt.replace(",", ""))
        for amt in raw_amounts
        if amt and amt.replace(",", "").replace(".", "").isdigit()
    ))

    date_matches = DATE_PATTERN.findall(text)
    flat_dates = [d for group in date_matches for d in group if d]
    entities.dates = list(dict.fromkeys(flat_dates))

    entities.pan_numbers = list(set(PAN_PATTERN.findall(text.upper())))

    raw_aadhaars = AADHAAR_PATTERN.findall(text)
    entities.aadhaar_numbers = [
        a.replace(" ", "")[:4] + "XXXXXXXX"
        for a in raw_aadhaars
    ]

    # Store raw Aadhaar for verification (unmasked, used internally only)
    entities._raw_aadhaar_numbers = [a.replace(" ", "") for a in raw_aadhaars]  # type: ignore

    entities.ifsc_codes = list(set(IFSC_PATTERN.findall(text.upper())))

    account_pattern = re.compile(
        r"(?:account\s*(?:number|no\.?|#)?\s*:?\s*)(\d{9,18})",
        re.IGNORECASE,
    )
    entities.account_numbers = list(set(account_pattern.findall(text)))

    if _SPACY_AVAILABLE and _NLP and len(text) > 0:
        chunk = text[:100_000]
        doc = _NLP(chunk)
        entities.persons = list(dict.fromkeys(
            ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"
        ))
        entities.organisations = list(dict.fromkeys(
            ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"
        ))
    else:
        org_pattern = re.compile(
            r"\b([A-Z][A-Za-z\s&.]+(?:Pvt\.?\s*Ltd\.?|Limited|Corp|Inc|LLP|LLC|Bank))\b"
        )
        entities.organisations = list(dict.fromkeys(org_pattern.findall(text)))

    return entities


# ─────────────────────────────────────────────────────────────────────────────
# ★ NEW: IDENTITY CROSS-VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_name(name: str) -> str:
    """Lowercase, strip titles, collapse spaces."""
    name = name.lower().strip()
    for title in ["mr.", "mrs.", "ms.", "dr.", "shri", "smt.", "kumari"]:
        name = name.replace(title, "")
    return " ".join(name.split())


def _find_name_in_text(claimed_name: str, text: str, entities: ExtractedEntities) -> tuple[str | None, float]:
    """
    Try to find the claimed name in the document.
    Returns (best_match_found, fuzzy_score 0-100).
    Checks spaCy persons first, then raw text sliding window.
    """
    if not claimed_name or not text:
        return None, 0.0

    norm_claimed = _normalise_name(claimed_name)
    best_match = None
    best_score = 0.0

    # Check spaCy-extracted persons
    for person in entities.persons:
        norm_person = _normalise_name(person)
        score = fuzz.token_sort_ratio(norm_claimed, norm_person)
        if score > best_score:
            best_score = score
            best_match = person

    # Also scan raw text for the name (handles cases spaCy misses)
    # Look in lines that might contain "Employee:", "Name:", "Account Holder:" etc.
    name_context_pattern = re.compile(
        r"(?:employee|name|account\s*holder|applicant|customer|beneficiary)"
        r"\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})",
        re.IGNORECASE,
    )
    for match in name_context_pattern.finditer(text):
        candidate = match.group(1).strip()
        norm_candidate = _normalise_name(candidate)
        score = fuzz.token_sort_ratio(norm_claimed, norm_candidate)
        if score > best_score:
            best_score = score
            best_match = candidate

    # Direct substring check (handles ALL CAPS documents)
    if norm_claimed in text.lower():
        best_score = max(best_score, 100.0)
        if best_match is None:
            best_match = claimed_name

    return best_match, best_score


def verify_identity(
    text: str,
    entities: ExtractedEntities,
    claimed_name: str | None,
    claimed_pan: str | None,
    claimed_aadhaar: str | None,
) -> tuple[IdentityVerificationResult, list[SemanticFlag]]:
    """
    Cross-check form-submitted identity values against extracted document content.

    Returns (IdentityVerificationResult, list[SemanticFlag])
    Flags are HIGH severity for hard mismatches, MEDIUM for soft/unverifiable.
    """
    result = IdentityVerificationResult(
        name_claimed=claimed_name,
        pan_claimed=claimed_pan,
        aadhaar_claimed=claimed_aadhaar,
    )
    flags: list[SemanticFlag] = []

    # ── PAN verification ──────────────────────────────────────────────────
    if claimed_pan:
        claimed_pan_clean = claimed_pan.strip().upper().replace(" ", "")
        doc_pans = entities.pan_numbers  # already upper from regex

        if doc_pans:
            if claimed_pan_clean in doc_pans:
                result.pan_found_in_doc = claimed_pan_clean
                result.pan_match = True
            else:
                result.pan_found_in_doc = doc_pans[0]
                result.pan_match = False
                flags.append(SemanticFlag(
                    check_name="pan_mismatch",
                    severity="HIGH",
                    description=(
                        f"PAN MISMATCH: Applicant claimed PAN '{claimed_pan_clean}' "
                        f"but document contains '{doc_pans[0]}'. "
                        f"This is a strong indicator of identity fraud."
                    ),
                    evidence=f"Claimed: {claimed_pan_clean} | Found in doc: {', '.join(doc_pans)}",
                ))
        else:
            result.pan_match = None   # PAN not found in document — can't verify
            flags.append(SemanticFlag(
                check_name="pan_not_found_in_doc",
                severity="MEDIUM",
                description=(
                    f"PAN '{claimed_pan_clean}' could not be located in the uploaded document. "
                    f"A genuine KYC document should contain the applicant's PAN."
                ),
                evidence="No PAN pattern found in document text.",
            ))

    # ── Aadhaar verification ──────────────────────────────────────────────
    if claimed_aadhaar:
        claimed_aadhaar_clean = re.sub(r"\s+", "", claimed_aadhaar.strip())
        raw_aadhaars = getattr(entities, '_raw_aadhaar_numbers', [])

        if raw_aadhaars:
            if claimed_aadhaar_clean in raw_aadhaars:
                result.aadhaar_found_in_doc = claimed_aadhaar_clean[:4] + "XXXXXXXX"
                result.aadhaar_match = True
            else:
                result.aadhaar_found_in_doc = raw_aadhaars[0][:4] + "XXXXXXXX"
                result.aadhaar_match = False
                flags.append(SemanticFlag(
                    check_name="aadhaar_mismatch",
                    severity="HIGH",
                    description=(
                        f"AADHAAR MISMATCH: Applicant claimed Aadhaar starting with "
                        f"'{claimed_aadhaar_clean[:4]}' but document shows a different "
                        f"Aadhaar number. Possible identity substitution."
                    ),
                    evidence=f"Claimed first 4: {claimed_aadhaar_clean[:4]} | Doc first 4: {raw_aadhaars[0][:4]}",
                ))
        else:
            result.aadhaar_match = None
            # Only flag if this is an Aadhaar/KYC document type
            flags.append(SemanticFlag(
                check_name="aadhaar_not_found_in_doc",
                severity="LOW",
                description=(
                    "Aadhaar number could not be located in document text. "
                    "This may be expected for salary slips or bank statements."
                ),
                evidence="No 12-digit Aadhaar pattern found.",
            ))

    # ── Name verification ─────────────────────────────────────────────────
    if claimed_name and claimed_name.strip():
        name_found, name_score = _find_name_in_text(claimed_name, text, entities)
        result.name_found_in_doc = name_found
        result.name_match_score = round(name_score, 1)

        if name_score >= 80:
            result.name_match = True
        elif name_score >= 50:
            result.name_match = False
            flags.append(SemanticFlag(
                check_name="name_partial_mismatch",
                severity="MEDIUM",
                description=(
                    f"NAME PARTIAL MISMATCH: Applicant's name '{claimed_name}' has only "
                    f"{name_score:.0f}% similarity to the best name match "
                    f"'{name_found}' found in the document. "
                    f"Could be a nickname, typo, or different person."
                ),
                evidence=f"Claimed: '{claimed_name}' | Best doc match: '{name_found}' | Score: {name_score:.0f}/100",
            ))
        elif name_found is not None:
            result.name_match = False
            flags.append(SemanticFlag(
                check_name="name_mismatch",
                severity="HIGH",
                description=(
                    f"NAME MISMATCH: Applicant claims to be '{claimed_name}' but the "
                    f"document belongs to '{name_found}' (match score: {name_score:.0f}/100). "
                    f"This document likely does not belong to this applicant."
                ),
                evidence=f"Claimed: '{claimed_name}' | Found in doc: '{name_found}' | Score: {name_score:.0f}/100",
            ))
        else:
            result.name_match = None
            flags.append(SemanticFlag(
                check_name="name_not_found_in_doc",
                severity="MEDIUM",
                description=(
                    f"IDENTITY UNVERIFIABLE: The name '{claimed_name}' could not be "
                    f"located anywhere in the uploaded document. "
                    f"The document may belong to a different person."
                ),
                evidence="Name not found via NER or context pattern search.",
            ))

    return result, flags


# ─────────────────────────────────────────────────────────────────────────────
# BERT SEMANTIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

_REFERENCE_TEMPLATES: dict[str, str] = {
    "salary_slip": (
        "Employee Salary Slip. Employee ID Employee Name Designation Department. "
        "Basic Pay HRA DA Special Allowance Gross Salary. "
        "Provident Fund Professional Tax Income Tax Total Deductions. "
        "Net Pay in words. Bank Account IFSC Code Month Year."
    ),
    "bank_statement": (
        "Account Statement. Account Number Account Holder Name IFSC Branch. "
        "Statement Period Opening Balance Closing Balance. "
        "Date Description Reference Debit Credit Balance. "
        "Total Credits Total Debits."
    ),
    "aadhaar": (
        "Government of India Unique Identification Authority UIDAI. "
        "Name Date of Birth Gender Aadhaar Number Address. "
        "Your Aadhaar."
    ),
}


def _get_bert_embedding(text: str, tokenizer: Any, model: Any) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyse_bert_semantics(text: str, document_type: str | None) -> BertEmbeddingResult:
    if len(text.strip()) < 20:
        return BertEmbeddingResult(
            document_type_matched=None, similarity_score=None,
            structural_deviation=None,
            explanation="Insufficient text extracted for semantic analysis.",
        )
    try:
        tokenizer, model = _load_bert()
    except Exception as exc:
        return BertEmbeddingResult(
            document_type_matched=None, similarity_score=None,
            structural_deviation=None, explanation=f"BERT model unavailable: {exc}",
        )

    doc_embedding = _get_bert_embedding(text[:5000], tokenizer, model)
    best_match, best_score = None, -1.0

    templates_to_check = (
        {document_type: _REFERENCE_TEMPLATES[document_type]}
        if document_type and document_type in _REFERENCE_TEMPLATES
        else _REFERENCE_TEMPLATES
    )

    for dtype, template_text in templates_to_check.items():
        score = _cosine_similarity(doc_embedding, _get_bert_embedding(template_text, tokenizer, model))
        if score > best_score:
            best_score, best_match = score, dtype

    deviation = 1.0 - best_score
    if best_score >= 0.80:
        explanation = f"Document structure closely matches '{best_match}' (similarity={best_score:.2f})."
    elif best_score >= 0.60:
        explanation = f"Moderate structural match to '{best_match}' (similarity={best_score:.2f})."
    else:
        explanation = (
            f"Low structural similarity to all known templates "
            f"(best: '{best_match}', score={best_score:.2f})."
        )

    return BertEmbeddingResult(
        document_type_matched=best_match,
        similarity_score=round(best_score, 4),
        structural_deviation=round(deviation, 4),
        explanation=explanation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NUMERICAL CONSISTENCY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _check_numerical_consistency(text: str, document_type: str | None) -> list[NumericalCheck]:
    checks: list[NumericalCheck] = []

    def extract_field(label_patterns: list[str]) -> float | None:
        for pattern in label_patterns:
            m = re.search(
                rf"{pattern}\s*:?\s*(?:₹|Rs\.?\s*|INR\s*)?([\d,]+(?:\.\d{{1,2}})?)",
                text, re.IGNORECASE,
            )
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except ValueError:
                    continue
        return None

    if document_type == "salary_slip":
        basic = extract_field([r"basic\s*pay", "basic salary", "basic"])
        hra   = extract_field(["hra", r"house\s*rent\s*allowance"])
        da    = extract_field(["da", r"dearness\s*allowance"])
        gross = extract_field([r"gross\s*salary", r"gross\s*pay", r"gross\s*earnings"])
        net   = extract_field([r"net\s*pay", r"net\s*salary", r"take\s*home"])
        total_deductions = extract_field([r"total\s*deductions", r"deductions\s*total"])

        components = [v for v in [basic, hra, da] if v is not None]
        if components and gross is not None:
            component_sum = sum(components)
            discrepancy = abs(component_sum - gross)
            tolerance = gross * 0.02
            checks.append(NumericalCheck(
                check_name="gross_salary_vs_components",
                passed=discrepancy <= tolerance or discrepancy <= 500,
                expected=gross, found=component_sum, discrepancy=discrepancy,
                description=f"Basic+HRA+DA ({component_sum:,.2f}) vs Gross ({gross:,.2f}).",
            ))

        if gross is not None and total_deductions is not None and net is not None:
            expected_net = gross - total_deductions
            discrepancy = abs(expected_net - net)
            checks.append(NumericalCheck(
                check_name="net_salary_arithmetic",
                passed=discrepancy <= 100,
                expected=expected_net, found=net, discrepancy=discrepancy,
                description=f"Gross - Deductions = {expected_net:,.2f}, found Net = {net:,.2f}.",
            ))

    elif document_type == "bank_statement":
        opening       = extract_field([r"opening\s*balance"])
        closing       = extract_field([r"closing\s*balance"])
        total_credits = extract_field([r"total\s*credits?", r"total\s*cr"])
        total_debits  = extract_field([r"total\s*debits?", r"total\s*dr"])

        if all(v is not None for v in [opening, closing, total_credits, total_debits]):
            expected_closing = opening + total_credits - total_debits
            discrepancy = abs(expected_closing - closing)
            checks.append(NumericalCheck(
                check_name="bank_balance_arithmetic",
                passed=discrepancy <= 1.0,
                expected=expected_closing, found=closing, discrepancy=discrepancy,
                description=f"Opening + Credits - Debits = {expected_closing:,.2f}, Closing = {closing:,.2f}.",
            ))

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _check_keyword_coverage(text: str, document_type: str | None) -> tuple[float | None, list[SemanticFlag]]:
    if not document_type or document_type not in DOCUMENT_KEYWORDS:
        return None, []

    expected = DOCUMENT_KEYWORDS[document_type]
    text_lower = text.lower()
    found   = [kw for kw in expected if kw in text_lower]
    missing = [kw for kw in expected if kw not in text_lower]
    coverage = len(found) / len(expected)

    flags: list[SemanticFlag] = []
    if coverage < 0.5:
        flags.append(SemanticFlag(
            check_name="keyword_coverage", severity="HIGH",
            description=(
                f"Only {coverage:.0%} of expected keywords for '{document_type}' found. "
                f"Missing: {', '.join(missing[:5])}."
            ),
            evidence=f"Missing: {', '.join(missing)}",
        ))
    elif coverage < 0.75:
        flags.append(SemanticFlag(
            check_name="keyword_coverage", severity="MEDIUM",
            description=(
                f"Only {coverage:.0%} of expected '{document_type}' keywords found. "
                f"Missing: {', '.join(missing[:3])}."
            ),
            evidence=f"Missing: {', '.join(missing)}",
        ))
    return coverage, flags


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-DOCUMENT CONTRADICTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def check_cross_document_consistency(results: list[NLPSemanticsResult]) -> list[str]:
    contradictions: list[str] = []
    if len(results) < 2:
        return contradictions

    all_pans     = [(r.file_name, set(r.entities.pan_numbers)) for r in results]
    all_orgs     = [(r.file_name, r.entities.organisations)    for r in results]
    all_accounts = [(r.file_name, set(r.entities.account_numbers)) for r in results]

    pan_sets = [(n, p) for n, p in all_pans if p]
    if len(pan_sets) >= 2:
        for i in range(len(pan_sets)):
            for j in range(i + 1, len(pan_sets)):
                if not (pan_sets[i][1] & pan_sets[j][1]):
                    contradictions.append(
                        f"PAN MISMATCH: '{pan_sets[i][0]}' has PAN {pan_sets[i][1]} "
                        f"but '{pan_sets[j][0]}' has {pan_sets[j][1]}."
                    )

    org_sets = [(n, o) for n, o in all_orgs if o]
    if len(org_sets) >= 2:
        for i in range(len(org_sets)):
            for j in range(i + 1, len(org_sets)):
                matched = any(
                    fuzz.partial_ratio(a.lower(), b.lower()) >= 80
                    for a in org_sets[i][1][:5]
                    for b in org_sets[j][1][:5]
                )
                if not matched and org_sets[i][1] and org_sets[j][1]:
                    contradictions.append(
                        f"EMPLOYER MISMATCH: '{org_sets[i][0]}' shows '{org_sets[i][1][0]}' "
                        f"but '{org_sets[j][0]}' shows '{org_sets[j][1][0]}'."
                    )

    acct_sets = [(n, a) for n, a in all_accounts if a]
    if len(acct_sets) >= 2:
        for i in range(len(acct_sets)):
            for j in range(i + 1, len(acct_sets)):
                if not (acct_sets[i][1] & acct_sets[j][1]):
                    contradictions.append(
                        f"ACCOUNT MISMATCH: Account numbers differ between "
                        f"'{acct_sets[i][0]}' and '{acct_sets[j][0]}'."
                    )

    return contradictions


# ─────────────────────────────────────────────────────────────────────────────
# SCORE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_score(
    semantic_flags: list[SemanticFlag],
    numerical_checks: list[NumericalCheck],
    bert_result: BertEmbeddingResult | None,
    keyword_coverage: float | None,
    text_length: int,
) -> tuple[float, RiskLevel]:
    score = 0.0

    if text_length < 30:
        score += 40.0

    if bert_result and bert_result.structural_deviation is not None:
        dev = bert_result.structural_deviation
        if dev > 0.40:
            score += 35.0
        elif dev > 0.25:
            score += 20.0
        elif dev > 0.15:
            score += 10.0

    if keyword_coverage is not None:
        if keyword_coverage < 0.40:
            score += 30.0
        elif keyword_coverage < 0.65:
            score += 15.0

    severity_weights = {"HIGH": 25.0, "MEDIUM": 12.0, "LOW": 5.0}
    for flag in semantic_flags:
        score += severity_weights.get(flag.severity, 5.0)

    for check in numerical_checks:
        if not check.passed:
            score += 35.0

    score = min(score, 100.0)

    if score >= 60:
        level = RiskLevel.HIGH_RISK
    elif score >= 25:
        level = RiskLevel.SUSPICIOUS
    else:
        level = RiskLevel.CLEAN

    return round(score, 2), level


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyse_nlp(
    file_bytes: bytes,
    file_name: str,
    document_type: str | None = None,
    run_bert: bool = True,
    # ★ NEW: identity fields from the application form
    applicant_name: str | None = None,
    applicant_pan: str | None = None,
    applicant_aadhaar: str | None = None,
) -> NLPSemanticsResult:
    """
    Main entry point for NLP Semantics analysis.

    Args:
        file_bytes:         Raw bytes of the uploaded file.
        file_name:          Original file name.
        document_type:      'salary_slip', 'bank_statement', 'aadhaar', etc.
        run_bert:           Set False to skip BERT.
        applicant_name:     Full name as entered on the application form.
        applicant_pan:      PAN as entered on the form (e.g. ABCDE1234F).
        applicant_aadhaar:  Aadhaar as entered on the form (12 digits).

    Returns:
        NLPSemanticsResult — fully typed, JSON-serialisable.
    """
    logger.info(f"[NLP] Analysing '{file_name}' (type: {document_type})")

    text, ocr_used = extract_text(file_bytes, file_name)
    text_length = len(text.strip())
    logger.info(f"[NLP] Extracted {text_length} chars. OCR used: {ocr_used}")

    entities = extract_entities(text)

    bert_result: BertEmbeddingResult | None = None
    if run_bert and text_length >= 20:
        bert_result = analyse_bert_semantics(text, document_type)

    keyword_coverage, keyword_flags = _check_keyword_coverage(text, document_type)
    numerical_checks = _check_numerical_consistency(text, document_type)

    all_semantic_flags: list[SemanticFlag] = []
    all_semantic_flags.extend(keyword_flags)

    for check in numerical_checks:
        if not check.passed:
            all_semantic_flags.append(SemanticFlag(
                check_name=f"numerical_{check.check_name}",
                severity="HIGH",
                description=check.description,
                evidence=(
                    f"Expected={check.expected:.2f}, Found={check.found:.2f}, "
                    f"Discrepancy=₹{check.discrepancy:.2f}"
                    if check.expected and check.found and check.discrepancy else None
                ),
            ))

    if bert_result and bert_result.structural_deviation is not None:
        if bert_result.structural_deviation > 0.25:
            all_semantic_flags.append(SemanticFlag(
                check_name="bert_structural_anomaly",
                severity="HIGH" if bert_result.structural_deviation > 0.40 else "MEDIUM",
                description=bert_result.explanation,
                evidence=f"Structural deviation: {bert_result.structural_deviation:.2f}",
            ))

    # ★ Identity cross-verification
    identity_result: IdentityVerificationResult | None = None
    if any([applicant_name, applicant_pan, applicant_aadhaar]):
        identity_result, identity_flags = verify_identity(
            text=text,
            entities=entities,
            claimed_name=applicant_name,
            claimed_pan=applicant_pan,
            claimed_aadhaar=applicant_aadhaar,
        )
        all_semantic_flags.extend(identity_flags)
        logger.info(
            f"[NLP] Identity check: name_match={identity_result.name_match} "
            f"pan_match={identity_result.pan_match} "
            f"aadhaar_match={identity_result.aadhaar_match}"
        )

    plain_flags: list[str] = [
        f"NLP [{sf.severity}]: {sf.description}"
        for sf in all_semantic_flags
    ]

    risk_score, risk_level = _calculate_score(
        all_semantic_flags, numerical_checks, bert_result, keyword_coverage, text_length,
    )

    logger.info(f"[NLP] Result: {risk_level.value} (score={risk_score}) | {len(plain_flags)} flags")

    return NLPSemanticsResult(
        file_name=file_name,
        document_type=document_type,
        risk_level=risk_level,
        risk_score=risk_score,
        flags=plain_flags,
        extracted_text_length=text_length,
        ocr_used=ocr_used,
        entities=entities,
        semantic_flags=all_semantic_flags,
        numerical_checks=numerical_checks,
        bert_result=bert_result,
        keyword_coverage=keyword_coverage,
        identity_verification=identity_result,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path> [document_type] [--no-bert]")
        sys.exit(1)
    path = Path(sys.argv[1])
    doc_type = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    result = analyse_nlp(
        file_bytes=path.read_bytes(),
        file_name=path.name,
        document_type=doc_type,
        run_bert="--no-bert" not in sys.argv,
    )
    print(result.model_dump_json(indent=2))