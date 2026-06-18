# aggregation.py
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from constants import DEFAULT_MODEL_DIR
from exceptions import GlobalModelNotFoundError, InvalidFederatedUpdateError
from schemas import ClientUpdate, FederatedConfig
from security import reject_raw_data_keys
from storage import (
    build_next_model,
    empty_model,
    global_model_path,
    load_global_model,
    save_global_model,
    write_round_audit,
)
from validation import clean_feature_names, validate_client_update


def run_federated(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate one federated learning round using weighted FedAvg.

    The server accepts only client model updates. It rejects raw records, rows,
    labels, feature matrices, and similar raw-data keys.
    """
    if not isinstance(payload, dict):
        raise InvalidFederatedUpdateError("Federated payload must be a dictionary.")

    reject_raw_data_keys(payload)

    round_id = str(payload.get("round_id") or default_round_id()).strip()
    if not round_id:
        raise InvalidFederatedUpdateError("round_id cannot be empty.")

    config = parse_config(payload.get("config") or {})
    raw_updates = payload.get("client_updates")
    if not isinstance(raw_updates, list) or not raw_updates:
        raise InvalidFederatedUpdateError("client_updates must be a non-empty list.")

    existing_model = load_global_model(config.model_dir, required=False)
    expected_features: Optional[list[str]]
    if existing_model is not None:
        expected_features = list(existing_model["feature_names"])
    elif config.require_existing_model:
        raise GlobalModelNotFoundError(
            f"Global model not found at {global_model_path(config.model_dir)}."
        )
    else:
        expected_features = extract_first_update_feature_names(raw_updates)

    accepted: list[ClientUpdate] = []
    rejected: list[dict[str, str]] = []
    seen_clients: set[str] = set()

    for raw_update in raw_updates:
        client_hint = client_hint_from(raw_update)
        try:
            update = validate_client_update(
                raw_update,
                expected_feature_names=expected_features,
                expected_round_id=round_id,
                hmac_secret=config.hmac_secret,
            )
            if update.client_id in seen_clients:
                raise InvalidFederatedUpdateError(
                    f"Duplicate client_id in this round: {update.client_id}"
                )
            if update.sample_count < config.min_samples_per_client:
                raise InvalidFederatedUpdateError(
                    "sample_count is below min_samples_per_client "
                    f"({update.sample_count} < {config.min_samples_per_client})."
                )
            accepted.append(update)
            seen_clients.add(update.client_id)
        except Exception as exc:
            rejected.append({"client_id": client_hint, "reason": str(exc)})

    if len(accepted) < config.min_clients:
        raise InvalidFederatedUpdateError(
            "Not enough valid client updates for aggregation: "
            f"accepted={len(accepted)}, required={config.min_clients}, "
            f"rejected={len(rejected)}."
        )

    feature_names = accepted[0].feature_names
    if existing_model is None:
        existing_model = empty_model(feature_names)

    previous_weights = np.array(existing_model["weights"], dtype=np.float64)
    previous_bias = float(existing_model["bias"])

    clipped_updates = [
        clip_client_update(update, previous_weights, previous_bias, config.max_update_norm)
        for update in accepted
    ]

    new_weights, new_bias = fedavg(
        clipped_updates,
        previous_weights=previous_weights,
        previous_bias=previous_bias,
        server_learning_rate=config.server_learning_rate,
    )

    total_samples = int(sum(update.sample_count for update in clipped_updates))
    weighted_metrics = weighted_average_metrics(clipped_updates)
    previous_model_hash = existing_model.get("model_hash")
    new_model = build_next_model(
        previous_model=existing_model,
        feature_names=feature_names,
        weights=new_weights,
        bias=new_bias,
        round_id=round_id,
        total_samples=total_samples,
        weighted_metrics=weighted_metrics,
        accepted_clients=len(clipped_updates),
    )

    model_path = save_global_model(config.model_dir, new_model)
    audit_path = write_round_audit(
        model_dir=config.model_dir,
        round_id=round_id,
        previous_model_hash=previous_model_hash,
        new_model=new_model,
        accepted_updates=clipped_updates,
        rejected_updates=rejected,
        config=config,
    )

    return {
        "status": "success",
        "round_id": round_id,
        "aggregation_algorithm": "FedAvg",
        "model_version": new_model["model_version"],
        "model_hash": new_model["model_hash"],
        "schema_hash": new_model["schema_hash"],
        "accepted_clients": len(clipped_updates),
        "rejected_clients": len(rejected),
        "total_samples": total_samples,
        "weighted_metrics": weighted_metrics,
        "global_model_path": str(model_path),
        "audit_log_path": str(audit_path),
        "rejected_updates": rejected,
    }


def parse_config(raw_config: dict[str, Any]) -> FederatedConfig:
    if not isinstance(raw_config, dict):
        raise InvalidFederatedUpdateError("config must be a dictionary when provided.")

    model_dir = Path(raw_config.get("model_dir") or DEFAULT_MODEL_DIR)
    min_clients = int(raw_config.get("min_clients", 2))
    min_samples = int(raw_config.get("min_samples_per_client", 1))
    max_update_norm = float(raw_config.get("max_update_norm", 25.0))
    server_learning_rate = float(raw_config.get("server_learning_rate", 1.0))
    require_existing_model = bool(raw_config.get("require_existing_model", False))
    hmac_secret = raw_config.get("hmac_secret") or os.environ.get("TRUSTNET_FL_HMAC_SECRET")

    if min_clients <= 0:
        raise InvalidFederatedUpdateError("min_clients must be greater than zero.")
    if min_samples <= 0:
        raise InvalidFederatedUpdateError("min_samples_per_client must be greater than zero.")
    if max_update_norm <= 0 or not math.isfinite(max_update_norm):
        raise InvalidFederatedUpdateError("max_update_norm must be a positive finite number.")
    if not (0 < server_learning_rate <= 1):
        raise InvalidFederatedUpdateError(
            "server_learning_rate must be greater than 0 and less than or equal to 1."
        )

    return FederatedConfig(
        model_dir=model_dir,
        min_clients=min_clients,
        min_samples_per_client=min_samples,
        max_update_norm=max_update_norm,
        server_learning_rate=server_learning_rate,
        require_existing_model=require_existing_model,
        hmac_secret=str(hmac_secret) if hmac_secret else None,
    )


def clip_client_update(
    update: ClientUpdate,
    previous_weights: np.ndarray,
    previous_bias: float,
    max_update_norm: float,
) -> ClientUpdate:
    delta_weights = update.weights - previous_weights
    delta_bias = update.bias - previous_bias
    delta_norm = float(np.sqrt(np.sum(delta_weights**2) + delta_bias**2))

    if delta_norm <= max_update_norm:
        return ClientUpdate(
            client_id=update.client_id,
            round_id=update.round_id,
            sample_count=update.sample_count,
            feature_names=update.feature_names,
            weights=update.weights,
            bias=update.bias,
            metrics=update.metrics,
            metadata=update.metadata,
            update_hash=update.update_hash,
            clipped=False,
            original_delta_norm=delta_norm,
        )

    scale = max_update_norm / delta_norm
    clipped_weights = previous_weights + delta_weights * scale
    clipped_bias = previous_bias + delta_bias * scale

    return ClientUpdate(
        client_id=update.client_id,
        round_id=update.round_id,
        sample_count=update.sample_count,
        feature_names=update.feature_names,
        weights=clipped_weights,
        bias=float(clipped_bias),
        metrics=update.metrics,
        metadata=update.metadata,
        update_hash=update.update_hash,
        clipped=True,
        original_delta_norm=delta_norm,
    )


def fedavg(
    updates: Sequence[ClientUpdate],
    previous_weights: np.ndarray,
    previous_bias: float,
    server_learning_rate: float,
) -> tuple[np.ndarray, float]:
    total_samples = float(sum(update.sample_count for update in updates))
    averaged_weights = sum(
        update.weights * (update.sample_count / total_samples) for update in updates
    )
    averaged_bias = sum(update.bias * (update.sample_count / total_samples) for update in updates)

    new_weights = previous_weights + server_learning_rate * (averaged_weights - previous_weights)
    new_bias = previous_bias + server_learning_rate * (averaged_bias - previous_bias)
    return np.array(new_weights, dtype=np.float64), float(new_bias)


def weighted_average_metrics(updates: Sequence[ClientUpdate]) -> dict[str, float]:
    metric_names = sorted({name for update in updates for name in update.metrics})
    total_samples = float(sum(update.sample_count for update in updates))
    output: dict[str, float] = {}

    for name in metric_names:
        numerator = 0.0
        denominator = 0.0
        for update in updates:
            if name not in update.metrics:
                continue
            numerator += update.metrics[name] * update.sample_count
            denominator += update.sample_count
        if denominator > 0:
            output[name] = round(numerator / denominator, 6)

    output["participating_clients"] = float(len(updates))
    output["total_samples"] = total_samples
    return output


def extract_first_update_feature_names(raw_updates: Sequence[Any]) -> list[str]:
    for raw_update in raw_updates:
        if isinstance(raw_update, dict) and raw_update.get("feature_names"):
            return clean_feature_names(raw_update["feature_names"])
    raise InvalidFederatedUpdateError(
        "No existing global model was found, and no update supplied feature_names."
    )


def client_hint_from(raw_update: Any) -> str:
    if isinstance(raw_update, dict):
        return str(raw_update.get("client_id") or "unknown")
    return "unknown"


def default_round_id() -> str:
    return "round_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
