import csv
import math
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from security import file_sha256, reject_raw_data_keys, schema_hash, sign_payload
from storage import load_model_from_path, utc_now, write_json
from validation import clean_feature_names, float_list, validate_client_update


def train_local_update(
    csv_path: str | Path,
    target_column: str,
    client_id: str,
    round_id: str,
    output_path: str | Path | None = None,
    feature_columns: Optional[Sequence[str]] = None,
    global_model_path: str | Path | None = None,
    epochs: int = 200,
    learning_rate: float = 0.01,
    l2: float = 0.001,
    hmac_secret: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Train a local binary logistic model from a real CSV and create a client
    update. Raw CSV rows stay local and are not written to the update JSON.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if l2 < 0:
        raise ValueError("l2 must be zero or greater.")

    starting_model = load_model_from_path(global_model_path) if global_model_path else None
    if starting_model is not None:
        model_features = list(starting_model["feature_names"])
        if feature_columns is None:
            feature_columns = model_features
        elif list(feature_columns) != model_features:
            raise ValueError(
                "feature_columns must match the global model feature order exactly."
            )

    feature_names, x_matrix, y_vector = load_csv_training_data(
        csv_path=csv_path,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    if starting_model is not None:
        initial_weights = np.array(starting_model["weights"], dtype=np.float64)
        initial_bias = float(starting_model["bias"])
    else:
        initial_weights = np.zeros(x_matrix.shape[1], dtype=np.float64)
        initial_bias = 0.0

    weights, bias, training_summary = fit_logistic_regression(
        x_matrix=x_matrix,
        y_vector=y_vector,
        initial_weights=initial_weights,
        initial_bias=initial_bias,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )

    probabilities = sigmoid(x_matrix @ weights + bias)
    metrics = classification_metrics(y_vector, probabilities)
    metrics.update(training_summary)

    update_metadata = {
        "created_at": utc_now(),
        "algorithm": "local_logistic_regression_gradient_descent",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "target_column": target_column,
        "source_file_sha256": file_sha256(csv_path),
        "started_from_model_hash": (
            starting_model.get("model_hash") if starting_model is not None else None
        ),
    }
    if metadata:
        reject_raw_data_keys(metadata)
        update_metadata.update(metadata)

    update_payload = {
        "client_id": str(client_id).strip(),
        "round_id": str(round_id).strip(),
        "sample_count": int(y_vector.shape[0]),
        "feature_names": feature_names,
        "schema_hash": schema_hash(feature_names),
        "weights": float_list(weights),
        "bias": float(bias),
        "metrics": metrics,
        "metadata": update_metadata,
    }

    secret = hmac_secret or os.environ.get("TRUSTNET_FL_HMAC_SECRET")
    if secret:
        update_payload["signature"] = sign_payload(update_payload, secret)

    validate_client_update(
        update_payload,
        expected_feature_names=feature_names,
        expected_round_id=round_id,
        hmac_secret=secret,
    )

    if output_path is not None:
        write_json(Path(output_path), update_payload)

    return update_payload


def load_csv_training_data(
    csv_path: Path,
    target_column: str,
    feature_columns: Optional[Sequence[str]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Training CSV must contain a header row.")
        rows = list(reader)

    if not rows:
        raise ValueError("Training CSV contains no data rows.")
    if target_column not in reader.fieldnames:
        raise ValueError(f"target_column not found in CSV header: {target_column}")

    if feature_columns is None:
        feature_names = infer_numeric_feature_columns(rows, reader.fieldnames, target_column)
    else:
        feature_names = clean_feature_names(feature_columns)
        missing = [name for name in feature_names if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Feature columns missing from CSV: {missing}")

    x_rows: list[list[float]] = []
    y_values: list[int] = []

    for row_number, row in enumerate(rows, start=2):
        y_values.append(parse_binary_label(row.get(target_column), row_number, target_column))
        x_row = []
        for feature in feature_names:
            value = row.get(feature)
            if value is None or str(value).strip() == "":
                x_row.append(float("nan"))
                continue
            try:
                x_row.append(float(value))
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in feature '{feature}' at CSV row {row_number}: {value!r}"
                ) from exc
        x_rows.append(x_row)

    x_matrix = np.array(x_rows, dtype=np.float64)
    y_vector = np.array(y_values, dtype=np.float64)
    column_means = np.nanmean(x_matrix, axis=0)

    if np.any(np.isnan(column_means)):
        empty_columns = [
            feature_names[index]
            for index, value in enumerate(column_means)
            if math.isnan(float(value))
        ]
        raise ValueError(f"Feature columns contain no numeric values: {empty_columns}")

    nan_rows, nan_columns = np.where(np.isnan(x_matrix))
    if len(nan_rows):
        x_matrix[nan_rows, nan_columns] = column_means[nan_columns]

    return feature_names, x_matrix, y_vector


def infer_numeric_feature_columns(
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
    target_column: str,
) -> list[str]:
    numeric_columns: list[str] = []

    for column in fieldnames:
        if column == target_column:
            continue
        observed = [row.get(column, "").strip() for row in rows if row.get(column, "").strip()]
        if not observed:
            continue
        all_numeric = True
        for value in observed:
            try:
                float(value)
            except ValueError:
                all_numeric = False
                break
        if all_numeric:
            numeric_columns.append(column)

    if not numeric_columns:
        raise ValueError(
            "No numeric feature columns could be inferred. Pass --feature-columns explicitly."
        )

    return clean_feature_names(numeric_columns)


def fit_logistic_regression(
    x_matrix: np.ndarray,
    y_vector: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    if x_matrix.ndim != 2:
        raise ValueError("x_matrix must be two-dimensional.")
    if y_vector.ndim != 1:
        raise ValueError("y_vector must be one-dimensional.")
    if x_matrix.shape[0] != y_vector.shape[0]:
        raise ValueError("x_matrix and y_vector row counts do not match.")
    if x_matrix.shape[1] != initial_weights.shape[0]:
        raise ValueError("Initial weight count does not match feature count.")

    weights = np.array(initial_weights, dtype=np.float64)
    bias = float(initial_bias)
    sample_count = float(x_matrix.shape[0])
    initial_loss = log_loss(y_vector, sigmoid(x_matrix @ weights + bias), weights, l2)

    for _ in range(epochs):
        logits = x_matrix @ weights + bias
        probabilities = sigmoid(logits)
        errors = probabilities - y_vector
        gradient_weights = (x_matrix.T @ errors) / sample_count + l2 * weights
        gradient_bias = float(np.sum(errors) / sample_count)
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    final_loss = log_loss(y_vector, sigmoid(x_matrix @ weights + bias), weights, l2)
    return weights, bias, {
        "initial_loss": round(float(initial_loss), 6),
        "loss": round(float(final_loss), 6),
    }


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.float64)
    positives = y_true == 1
    negatives = y_true == 0

    true_positive = float(np.sum((predictions == 1) & positives))
    true_negative = float(np.sum((predictions == 0) & negatives))
    false_positive = float(np.sum((predictions == 1) & negatives))
    false_negative = float(np.sum((predictions == 0) & positives))
    total = float(len(y_true))

    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "positive_rate": round(float(np.mean(y_true)), 6),
    }


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    data_loss = -np.mean(
        y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities)
    )
    regularization = 0.5 * l2 * float(np.sum(weights**2))
    return float(data_loss + regularization)


def parse_binary_label(value: Any, row_number: int, column: str) -> int:
    normalized = str(value).strip().lower()
    positive = {"1", "true", "yes", "y", "fraud", "fraudulent", "positive"}
    negative = {"0", "false", "no", "n", "genuine", "legit", "legitimate", "negative"}

    if normalized in positive:
        return 1
    if normalized in negative:
        return 0

    try:
        numeric = float(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid binary label in column '{column}' at CSV row {row_number}: {value!r}"
        ) from exc

    if numeric == 1:
        return 1
    if numeric == 0:
        return 0
    raise ValueError(
        f"Invalid binary label in column '{column}' at CSV row {row_number}: {value!r}"
    )