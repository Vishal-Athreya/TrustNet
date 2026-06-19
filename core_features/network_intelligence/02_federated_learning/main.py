"""
TrustNet Federated Learning public entry point.

This file stays intentionally small. The feature is split by responsibility:
aggregation.py        server-side FedAvg
client_training.py    bank-side local CSV training
scoring.py            global model inference
storage.py            model and audit persistence
validation.py         payload/schema validation
security.py           HMAC, hashes, raw-data rejection
cli.py                command-line interface
"""

import sys
from pathlib import Path


FEATURE_DIR = Path(__file__).resolve().parent
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from aggregation import run_federated  # noqa: E402
from client_training import train_local_update  # noqa: E402
from cli import main  # noqa: E402
from scoring import score_records  # noqa: E402
from storage import initialize_global_model  # noqa: E402


__all__ = [
    "initialize_global_model",
    "run_federated",
    "score_records",
    "train_local_update",
]


if __name__ == "__main__":
    raise SystemExit(main())