import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Sequence

from constants import RAW_DATA_KEYS
from exceptions import InvalidFederatedUpdateError


def reject_raw_data_keys(value: Any, path: str = "$") -> None:
    """Reject payloads that try to send raw training data to the server."""
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in RAW_DATA_KEYS:
                raise InvalidFederatedUpdateError(
                    f"Raw training data key is not allowed in federated payload: {path}.{key}"
                )
            reject_raw_data_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_raw_data_keys(nested, f"{path}[{index}]")


def safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    reject_raw_data_keys(metadata)
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        elif isinstance(value, dict):
            safe[str(key)] = safe_metadata(value)
        else:
            safe[str(key)] = str(value)
    return safe


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    unsigned_payload = dict(payload)
    unsigned_payload.pop("signature", None)
    return hmac.new(
        secret.encode("utf-8"),
        canonical_json(unsigned_payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_signature(payload: dict[str, Any], secret: str) -> None:
    provided_signature = str(payload.get("signature", "")).strip()
    expected_signature = sign_payload(payload, secret)
    if not provided_signature:
        raise InvalidFederatedUpdateError("Missing HMAC signature.")
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise InvalidFederatedUpdateError("Invalid HMAC signature.")


def schema_hash(feature_names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def model_hash(model: dict[str, Any]) -> str:
    model_copy = dict(model)
    model_copy.pop("model_hash", None)
    return sha256_json(model_copy)


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
