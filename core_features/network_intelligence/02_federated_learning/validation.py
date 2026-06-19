import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from exceptions import InvalidFederatedUpdateError
from schemas import ClientUpdate
from security import reject_raw_data_keys, schema_hash, sha256_json, verify_signature


def validate_client_update(
    raw_update: Any,
    expected_feature_names: Optional[Sequence[str]],
    expected_round_id: Optional[str],
    hmac_secret: Optional[str],
) -> ClientUpdate:
    if not isinstance(raw_update, dict):
        raise InvalidFederatedUpdateError("Each client update must be a dictionary.")

    reject_raw_data_keys(raw_update)

    if hmac_secret:
        verify_signature(raw_update, hmac_secret)

    client_id = str(raw_update.get("client_id", "")).strip()
    if not client_id:
        raise InvalidFederatedUpdateError("client_id is required.")
    if len(client_id) > 128:
        raise InvalidFederatedUpdateError("client_id is too long.")

    round_id = str(raw_update.get("round_id", "")).strip()
    if not round_id:
        raise InvalidFederatedUpdateError("round_id is required.")
    if expected_round_id is not None and round_id != expected_round_id:
        raise InvalidFederatedUpdateError(
            f"round_id mismatch: expected {expected_round_id}, got {round_id}."
        )

    try:
        sample_count = int(raw_update.get("sample_count"))
    except (TypeError, ValueError) as exc:
        raise InvalidFederatedUpdateError("sample_count must be an integer.") from exc
    if sample_count <= 0:
        raise InvalidFederatedUpdateError("sample_count must be greater than zero.")

    feature_names = clean_feature_names(raw_update.get("feature_names"))
    if expected_feature_names is not None and feature_names != list(expected_feature_names):
        raise InvalidFederatedUpdateError(
            "feature_names do not match the global schema or first accepted update."
        )

    provided_schema_hash = raw_update.get("schema_hash")
    if provided_schema_hash and str(provided_schema_hash) != schema_hash(feature_names):
        raise InvalidFederatedUpdateError("schema_hash does not match feature_names.")

    weights = parse_weight_vector(raw_update.get("weights"), len(feature_names))
    bias = parse_float(raw_update.get("bias"), "bias")
    metrics = parse_metrics(raw_update.get("metrics") or {})
    metadata = raw_update.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise InvalidFederatedUpdateError("metadata must be a dictionary when provided.")

    unsigned_update = dict(raw_update)
    unsigned_update.pop("signature", None)

    return ClientUpdate(
        client_id=client_id,
        round_id=round_id,
        sample_count=sample_count,
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        metrics=metrics,
        metadata=metadata,
        update_hash=sha256_json(unsigned_update),
    )


def clean_feature_names(feature_names: Any) -> list[str]:
    if isinstance(feature_names, str):
        feature_names = [part.strip() for part in feature_names.split(",") if part.strip()]
    if not isinstance(feature_names, Sequence):
        raise InvalidFederatedUpdateError("feature_names must be a list or comma-separated string.")

    cleaned = [str(name).strip() for name in feature_names]
    if not cleaned or any(not name for name in cleaned):
        raise InvalidFederatedUpdateError("feature_names must contain at least one non-empty name.")
    if len(set(cleaned)) != len(cleaned):
        raise InvalidFederatedUpdateError("feature_names must not contain duplicates.")
    if any(len(name) > 128 for name in cleaned):
        raise InvalidFederatedUpdateError("feature_names must be 128 characters or shorter.")
    return cleaned


def parse_weight_vector(raw_weights: Any, expected_length: int) -> np.ndarray:
    if not isinstance(raw_weights, Sequence) or isinstance(raw_weights, (str, bytes)):
        raise InvalidFederatedUpdateError("weights must be a numeric list.")
    if len(raw_weights) != expected_length:
        raise InvalidFederatedUpdateError(
            f"weights length mismatch: expected {expected_length}, got {len(raw_weights)}."
        )
    return np.array(
        [parse_float(value, f"weights[{index}]") for index, value in enumerate(raw_weights)],
        dtype=np.float64,
    )


def parse_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidFederatedUpdateError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise InvalidFederatedUpdateError(f"{field_name} must be finite.")
    return parsed


def parse_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise InvalidFederatedUpdateError("metrics must be a dictionary when provided.")

    parsed: dict[str, float] = {}
    for name, value in metrics.items():
        metric_name = str(name).strip()
        if not metric_name:
            raise InvalidFederatedUpdateError("Metric names cannot be empty.")
        parsed[metric_name] = parse_float(value, f"metrics.{metric_name}")
    return parsed


def float_list(values: np.ndarray | Sequence[float]) -> list[float]:
    return [round(float(value), 12) for value in values]