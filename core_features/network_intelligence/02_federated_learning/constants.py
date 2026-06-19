from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models"
GLOBAL_MODEL_FILENAME = "global_model.json"
MODEL_FORMAT_VERSION = 1

RAW_DATA_KEYS = {
    "data",
    "dataset",
    "feature_matrix",
    "features_matrix",
    "labels",
    "raw_data",
    "raw_records",
    "raw_rows",
    "records",
    "rows",
    "samples",
    "target_values",
    "training_data",
    "training_records",
    "x",
    "y",
}