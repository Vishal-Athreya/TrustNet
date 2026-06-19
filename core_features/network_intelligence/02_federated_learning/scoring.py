import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from constants import DEFAULT_MODEL_DIR
from client_training import sigmoid
from storage import load_global_model, write_json
from validation import parse_float


def score_records(
    records: Iterable[dict[str, Any]],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Score records with the current global model."""
    model = load_global_model(Path(model_dir), required=True)
    feature_names = list(model["feature_names"])
    weights = np.array(model["weights"], dtype=np.float64)
    bias = float(model["bias"])

    predictions = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} must be a dictionary.")
        row = []
        for feature in feature_names:
            if feature not in record:
                raise ValueError(f"Record at index {index} is missing feature: {feature}")
            row.append(parse_float(record[feature], f"record[{index}].{feature}"))
        probability = float(sigmoid(np.array(row, dtype=np.float64) @ weights + bias))
        predictions.append(
            {
                "index": index,
                "fraud_probability": round(probability, 6),
                "risk_score": int(round(probability * 100)),
            }
        )

    return {
        "model_version": model["model_version"],
        "model_hash": model["model_hash"],
        "schema_hash": model["schema_hash"],
        "predictions": predictions,
    }


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Input CSV must contain a header row.")
        return list(reader)


def score_csv(input_csv: str | Path, output_path: str | Path, model_dir: str | Path) -> dict[str, Any]:
    result = score_records(read_csv_records(Path(input_csv)), model_dir)
    write_json(Path(output_path), result)
    return result