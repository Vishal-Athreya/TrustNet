import re
import os
import numpy as np
from scipy.stats import chisquare
from pypdf import PdfReader

# Ideal Benford distribution for digits 1-9
BENFORD_DISTRIBUTION = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]

def extract_text_from_file(file_path: str) -> str:
    """Reads the physical file and extracts text."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += (page.extract_text() or "") + " "
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def analyze_benford_from_file(file_path: str, alpha: float = 0.05) -> dict:
    """
    Pipeline: Reads document -> Extracts Numbers -> Runs Benford test.
    This function processes a single document to verify its mathematical integrity.
    """
    text = extract_text_from_file(file_path)
    
    # Extract numbers (integers and floats)
    raw_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    valid_numbers = [num for num in raw_numbers if float(num) >= 10]
    
    if len(valid_numbers) < 10: 
        return {
            "feature": "Benford's Law",
            "anomaly_detected": False,
            "integrity_score": 100.0,
            "explanation": "Insufficient numeric data to perform statistical distribution analysis."
        }
    
    leading_digits = [int(num.replace('.', '').lstrip('0')[0]) for num in valid_numbers if num.replace('.', '').lstrip('0')]
    observed_counts = [leading_digits.count(d) for d in range(1, 10)]
    total_observed = sum(observed_counts)
    
    expected_counts = [max(p * total_observed, 0.0001) for p in BENFORD_DISTRIBUTION]
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    
    anomaly_detected = p_value < alpha
    integrity_score = min(100.0, max(0.0, float(p_value * 100 * (1 / alpha)))) if anomaly_detected else 100.0

    return {
        "feature": "Benford's Law",
        "anomaly_detected": bool(anomaly_detected),
        "integrity_score": round(integrity_score, 2),
        "metrics": {"sample_size": total_observed, "p_value": round(float(p_value), 5)},
        "explanation": "Anomalous numerical distribution detected. Numbers appear manually fabricated." if anomaly_detected else "Numerical trends match natural statistical distributions."
    }
