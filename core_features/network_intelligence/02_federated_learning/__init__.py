import sys
from pathlib import Path


FEATURE_DIR = Path(__file__).resolve().parent
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from aggregation import run_federated  # noqa: E402
from client_training import train_local_update  # noqa: E402
from scoring import score_records  # noqa: E402
from storage import initialize_global_model  # noqa: E402


__all__ = [
    "initialize_global_model",
    "run_federated",
    "score_records",
    "train_local_update",
]
