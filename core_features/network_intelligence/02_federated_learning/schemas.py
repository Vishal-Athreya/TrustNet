from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    round_id: str
    sample_count: int
    feature_names: list[str]
    weights: np.ndarray
    bias: float
    metrics: dict[str, float]
    metadata: dict[str, Any]
    update_hash: str
    clipped: bool = False
    original_delta_norm: float = 0.0


@dataclass(frozen=True)
class FederatedConfig:
    model_dir: Path
    min_clients: int = 2
    min_samples_per_client: int = 1
    max_update_norm: float = 25.0
    server_learning_rate: float = 1.0
    require_existing_model: bool = False
    hmac_secret: Optional[str] = None
