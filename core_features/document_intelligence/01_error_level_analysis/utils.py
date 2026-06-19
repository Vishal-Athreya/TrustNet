"""
utils.py — OpenCV helper functions for Error Level Analysis
TrustNet | document_intelligence / 01_error_level_analysis
"""

import io
import numpy as np
import cv2
from PIL import Image


# ── JPEG quality used when re-encoding for ELA ───────────────────────────────
ELA_RECOMPRESS_QUALITY: int = 90


def pdf_page_to_pil(page) -> Image.Image:
    """
    Render a PyMuPDF page object to a PIL Image at 2× resolution (150 DPI).
    The higher DPI makes compression artefacts more visible for ELA.
    """
    mat = page.get_pixmap(dpi=150)
    img_bytes = mat.tobytes("png")
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ndarray back to a PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def recompress_jpeg(img: Image.Image, quality: int = ELA_RECOMPRESS_QUALITY) -> Image.Image:
    """
    Save a PIL image to an in-memory JPEG buffer at `quality`, then reload it.
    This introduces the standard JPEG re-compression used by ELA to expose
    regions that were previously edited (and therefore re-compressed more than once).
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compute_ela_array(original: Image.Image, quality: int = ELA_RECOMPRESS_QUALITY) -> np.ndarray:
    """
    Core ELA computation.

    Steps:
      1. Re-save the original at `quality` to create a reference version.
      2. Subtract the reference from the original pixel-by-pixel.
      3. Amplify the difference so it is human/algorithm visible.

    Returns a float32 ndarray (H, W, 3) with values in [0, 255].
    Untampered regions produce near-zero differences; edited patches stand out.
    """
    recompressed = recompress_jpeg(original, quality=quality)

    orig_arr = np.array(original, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    diff = np.abs(orig_arr - recomp_arr)

    # Amplify: scale so the max difference fills 0-255.
    # Clamp to avoid divide-by-zero on perfectly clean images.
    max_diff = diff.max()
    if max_diff < 1e-6:
        return diff  # flat — image is pristine or already heavily compressed

    amplified = (diff / max_diff) * 255.0
    return amplified.astype(np.float32)


def ela_to_heatmap_bgr(ela_arr: np.ndarray) -> np.ndarray:
    """
    Convert an ELA float32 array to a JET heatmap (BGR uint8).
    Useful for visual inspection and report thumbnails.
    Cold colours (blue) = low error = untampered.
    Hot colours (red) = high error = potentially tampered.
    """
    gray = cv2.cvtColor(ela_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def region_stats(ela_arr: np.ndarray) -> dict:
    """
    Compute summary statistics over the ELA difference array.
    These feed directly into the scoring logic in main.py.

    Returns:
        mean_error   — average pixel-level difference (float)
        max_error    — brightest single pixel (float)
        std_error    — spread of errors; high std = localised tampering
        high_error_ratio — fraction of pixels above the 95th-percentile
                           threshold (proxy for tampered-region coverage)
    """
    flat = ela_arr.flatten().astype(np.float32)

    mean_error = float(np.mean(flat))
    max_error = float(np.max(flat))
    std_error = float(np.std(flat))

    p95 = float(np.percentile(flat, 95))
    high_error_ratio = float(np.sum(flat > p95) / len(flat)) if len(flat) > 0 else 0.0

    return {
        "mean_error": round(mean_error, 4),
        "max_error": round(max_error, 4),
        "std_error": round(std_error, 4),
        "high_error_ratio": round(high_error_ratio, 4),
    }


def resize_for_report(img: Image.Image, max_width: int = 800) -> Image.Image:
    """
    Proportionally resize a PIL image so its width does not exceed `max_width`.
    Used to keep report thumbnails at a reasonable size.
    """
    w, h = img.size
    if w <= max_width:
        return img
    scale = max_width / w
    return img.resize((max_width, int(h * scale)), Image.LANCZOS)
