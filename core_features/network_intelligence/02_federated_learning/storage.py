import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from constants import GLOBAL_MODEL_FILENAME, MODEL_FORMAT_VERSION
from exceptions import GlobalModelNotFoundError
from schemas import ClientUpdate, FederatedConfig
from security import model_hash, safe_metadata, schema_hash
from validation import clean_feature_names, float_list, parse_float, parse_weight_vector


def initialize_global_model(
    feature_columns: Sequence[str],
    model_dir: str | Path,
    created_by: str = "trustnet",
) -> dict[str, Any]:
    feature_names = clean_feature_names(feature_columns)
    model = empty_model(feature_names)
    model["created_by"] = created_by
    model["model_hash"] = model_hash(model)
    save_global_model(Path(model_dir), model)
    return model


def empty_model(feature_names: Sequence[str]) -> dict[str, Any]:
    cleaned_features = clean_feature_names(feature_names)
    now = utc_now()
    model = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_id": str(uuid.uuid4()),
        "model_version": 0,
        "created_at": now,
        "updated_at": now,
        "feature_names": cleaned_features,
        "schema_hash": schema_hash(cleaned_features),
        "weights": [0.0 for _ in cleaned_features],
        "bias": 0.0,
        "rounds_completed": 0,
        "last_round_id": None,
        "aggregation_algorithm": "FedAvg",
        "last_round_summary": {},
    }
    model["model_hash"] = model_hash(model)
    return model


def build_next_model(
    previous_model: dict[str, Any],
    feature_names: list[str],
    weights,
    bias: float,
    round_id: str,
    total_samples: int,
    weighted_metrics: dict[str, float],
    accepted_clients: int,
) -> dict[str, Any]:
    now = utc_now()
    previous_version = int(previous_model.get("model_version", 0))
    model = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_id": previous_model.get("model_id") or str(uuid.uuid4()),
        "model_version": previous_version + 1,
        "created_at": previous_model.get("created_at") or now,
        "updated_at": now,
        "feature_names": feature_names,
        "schema_hash": schema_hash(feature_names),
        "weights": float_list(weights),
        "bias": float(bias),
        "rounds_completed": int(previous_model.get("rounds_completed", 0)) + 1,
        "last_round_id": round_id,
        "aggregation_algorithm": "FedAvg",
        "last_round_summary": {
            "total_samples": total_samples,
            "accepted_clients": accepted_clients,
            "weighted_metrics": weighted_metrics,
        },
    }
    model["model_hash"] = model_hash(model)
    return model


def load_global_model(model_dir: Path, required: bool) -> Optional[dict[str, Any]]:
    model_path = global_model_path(model_dir)
    if not model_path.exists():
        if required:
            raise GlobalModelNotFoundError(f"Global model not found: {model_path}")
        return None
    return validate_global_model(read_json(model_path), model_path)


def load_model_from_path(path: str | Path | None) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    model_path = Path(path)
    if model_path.is_dir():
        model_path = global_model_path(model_path)
    if not model_path.exists():
        raise GlobalModelNotFoundError(f"Global model not found: {model_path}")
    return validate_global_model(read_json(model_path), model_path)


def validate_global_model(model: Any, model_path: Path) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise GlobalModelNotFoundError(f"Invalid model file: {model_path}")
    feature_names = clean_feature_names(model.get("feature_names"))
    weights = parse_weight_vector(model.get("weights"), len(feature_names))
    bias = parse_float(model.get("bias"), "bias")
    model["feature_names"] = feature_names
    model["weights"] = float_list(weights)
    model["bias"] = bias
    model["schema_hash"] = schema_hash(feature_names)
    model.setdefault("model_version", 0)
    model.setdefault("model_id", str(uuid.uuid4()))
    model.setdefault("model_hash", model_hash(model))
    return model


def save_global_model(model_dir: Path, model: dict[str, Any]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = global_model_path(model_dir)
    write_json(model_path, model)
    return model_path


def write_round_audit(
    model_dir: Path,
    round_id: str,
    previous_model_hash: Optional[str],
    new_model: dict[str, Any],
    accepted_updates: Sequence[ClientUpdate],
    rejected_updates: Sequence[dict[str, str]],
    config: FederatedConfig,
) -> Path:
    audit = {
        "round_id": round_id,
        "completed_at": utc_now(),
        "aggregation_algorithm": "FedAvg",
        "previous_model_hash": previous_model_hash,
        "new_model_hash": new_model["model_hash"],
        "model_version": new_model["model_version"],
        "schema_hash": new_model["schema_hash"],
        "total_samples": int(sum(update.sample_count for update in accepted_updates)),
        "accepted_updates": [
            {
                "client_id": update.client_id,
                "sample_count": update.sample_count,
                "metrics": update.metrics,
                "metadata": safe_metadata(update.metadata),
                "update_hash": update.update_hash,
                "clipped": update.clipped,
                "original_delta_norm": round(update.original_delta_norm, 6),
            }
            for update in accepted_updates
        ],
        "rejected_updates": list(rejected_updates),
        "config": {
            "min_clients": config.min_clients,
            "min_samples_per_client": config.min_samples_per_client,
            "max_update_norm": config.max_update_norm,
            "server_learning_rate": config.server_learning_rate,
            "require_existing_model": config.require_existing_model,
            "hmac_required": bool(config.hmac_secret),
        },
    }

    rounds_dir = model_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    audit_path = rounds_dir / f"{safe_filename(round_id)}.json"
    write_json(audit_path, audit)
    return audit_path


def global_model_path(model_dir: Path) -> Path:
    return Path(model_dir) / GLOBAL_MODEL_FILENAME


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temp_path.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "round"
