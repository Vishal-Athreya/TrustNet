# exceptions.py
# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions for behavior monitoring.
# Using specific exception types means the backend can catch and handle
# each failure mode differently instead of catching all exceptions blindly.
# ─────────────────────────────────────────────────────────────────────────────


class BehaviorMonitoringError(Exception):
    """Base exception for all behavior monitoring errors."""
    pass


class ModelNotFoundError(BehaviorMonitoringError):
    """
    Raised when the trained model file does not exist.
    Fix: run train_baseline.py to generate models/isolation_forest.pkl
    """
    pass


class ModelLoadError(BehaviorMonitoringError):
    """
    Raised when the model file exists but cannot be loaded
    (corrupted file, wrong joblib version, etc.)
    """
    pass


class InvalidSessionDataError(BehaviorMonitoringError):
    """
    Raised when session_data is missing required keys or has invalid values.
    The error message includes which keys are missing.
    """
    pass


class FeatureEngineeringError(BehaviorMonitoringError):
    """
    Raised when feature engineering fails
    (e.g. a value cannot be converted to the expected type).
    """
    pass


class ScoringError(BehaviorMonitoringError):
    """
    Raised when the Isolation Forest scoring step itself fails.
    """
    pass